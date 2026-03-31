# =============================================================================
# ensemble.py — Model 7: Weighted Ensemble
#
# Combines any subset of completed models by averaging their softmax
# probability outputs on the test set.
#
# Supported models (auto-detected by checkpoint presence):
#   model1_traditional  → SVM (uses predict_proba from saved .pkl)
#   model4_efficientnet → EfficientNetSER (.pth checkpoint)
#   model5_resnet18     → ResNet18DualSER (.pth checkpoint)
#   model2_cnn_bilstm_mel, model3_cnn_bilstm_mfcc, model6_fusion  (future)
#
# How it works:
#   1. For each model that has a saved checkpoint, run the full test set
#      through it and collect per-sample probability arrays.
#   2. Multiply each model's probabilities by its config weight.
#   3. Normalize the weights to sum to 1.0 (accounts for missing models).
#   4. Sum → final probability per class → argmax → prediction.
#
# No training happens here. This is pure inference + combination.
# =============================================================================

import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _detect_available_models() -> dict:
    """
    Scan OUTPUTS_DIR for completed model checkpoints.
    Returns a dict: {model_name: checkpoint_path_or_pkl_dir}

    A PyTorch model is "available" if best_model.pth exists in its output dir.
    Model 1 is "available" if svm_model.pkl and scaler.pkl exist.
    """
    available = {}

    for model_name in config.ENSEMBLE_WEIGHTS:
        out_dir = os.path.join(config.OUTPUTS_DIR, model_name)

        if model_name == "model1_traditional":
            svm_path    = os.path.join(out_dir, "svm_model.pkl")
            scaler_path = os.path.join(out_dir, "scaler.pkl")
            if os.path.isfile(svm_path) and os.path.isfile(scaler_path):
                available[model_name] = out_dir
        else:
            ckpt = os.path.join(out_dir, "best_model.pth")
            if os.path.isfile(ckpt):
                available[model_name] = ckpt

    return available


def _load_pytorch_model(model_name: str, ckpt_path: str):
    """Instantiate and load a PyTorch model from checkpoint."""
    if "efficientnet" in model_name:
        from models.efficientnet_b0 import EfficientNetSER
        model = EfficientNetSER()
    elif "resnet18" in model_name:
        from models.resnet18_dual import ResNet18DualSER
        model = ResNet18DualSER()
    elif "cnn_bilstm_mel" in model_name:
        from models.cnn_bilstm_mel import CNNBiLSTMMel
        model = CNNBiLSTMMel()
    elif "cnn_bilstm_mfcc" in model_name:
        from models.cnn_bilstm_mfcc import CNNBiLSTMMFCC
        model = CNNBiLSTMMFCC()
    elif "fusion" in model_name:
        from models.fusion import FusionSER
        model = FusionSER()
    else:
        raise ValueError(f"Unknown PyTorch model name: {model_name}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# PROBABILITY EXTRACTION
# ---------------------------------------------------------------------------

def get_pytorch_probs(model_name: str, ckpt_path: str,
                       test_loader, device: torch.device) -> np.ndarray:
    """
    Run a PyTorch model on the test set and return softmax probabilities.

    Returns:
        probs: np.ndarray of shape (N_test_samples, NUM_CLASSES)
    """
    model = _load_pytorch_model(model_name, ckpt_path)
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch, device)               # (B, NUM_CLASSES)
            probs  = F.softmax(logits, dim=1)           # (B, NUM_CLASSES)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)            # (N, NUM_CLASSES)


def get_sklearn_probs(out_dir: str) -> tuple:
    """
    Run the saved SVM on the test scalar features and return probabilities.

    Returns:
        probs:  np.ndarray of shape (N_test_samples, NUM_CLASSES)
        labels: np.ndarray of shape (N_test_samples,) — true labels
    """
    from models.traditional_ml import load_scalar_features

    svm_path    = os.path.join(out_dir, "svm_model.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    with open(svm_path,    "rb") as f: svm    = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    X_test, y_test = load_scalar_features("test")
    X_test_s = scaler.transform(X_test)

    # SVC with probability=True returns shape (N, NUM_CLASSES)
    probs = svm.predict_proba(X_test_s)

    # SVC predict_proba columns are ordered by svm.classes_ — ensure correct order
    # Build a (N, NUM_CLASSES) array with columns in 0..NUM_CLASSES-1 order
    ordered = np.zeros((probs.shape[0], config.NUM_CLASSES), dtype=np.float32)
    for col_idx, class_label in enumerate(svm.classes_):
        ordered[:, int(class_label)] = probs[:, col_idx]

    return ordered, y_test


# ---------------------------------------------------------------------------
# ENSEMBLE PREDICTOR
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Weighted ensemble of any combination of available models.

    Usage:
        ensemble = EnsemblePredictor()
        ensemble.build(device)                    # detect + load all available models
        probs, labels = ensemble.predict_test()   # run on test set
        ensemble.save_results(output_dir)
    """

    def __init__(self):
        self.available   = {}   # model_name → ckpt_path or out_dir
        self.raw_weights = {}   # model_name → raw weight from config
        self.norm_weights= {}   # model_name → normalized weight (sums to 1)
        self.probs_dict  = {}   # model_name → (N, NUM_CLASSES) ndarray
        self.true_labels = None

    def build(self, device: torch.device, test_loader=None):
        """
        Detect available models and collect their test-set probabilities.

        Args:
            device:      torch.device for PyTorch models
            test_loader: DataLoader for the test split (created internally if None)
        """
        self.available = _detect_available_models()

        if not self.available:
            raise RuntimeError(
                "No trained models found in outputs/.\n"
                "Run at least one training script before the ensemble."
            )

        print(f"Available models for ensemble: {list(self.available.keys())}")

        # Compute normalized weights (only over available models)
        total_w = sum(config.ENSEMBLE_WEIGHTS[m] for m in self.available)
        self.raw_weights  = {m: config.ENSEMBLE_WEIGHTS[m] for m in self.available}
        self.norm_weights = {m: w / total_w for m, w in self.raw_weights.items()}

        print("\nNormalized ensemble weights:")
        for m, w in self.norm_weights.items():
            print(f"  {m}: {w:.3f}")

        # Create test loader if not provided
        if test_loader is None:
            from data.dataset import get_dataloader
            test_loader = get_dataloader("test")

        # Collect true labels once (same for all models)
        true_labels_list = []
        for batch in test_loader:
            true_labels_list.extend(batch["label"].numpy())
        self.true_labels = np.array(true_labels_list)

        # Get probabilities from each available model
        print()
        for model_name, ckpt_or_dir in self.available.items():
            print(f"  Running inference: {model_name}...")
            if model_name == "model1_traditional":
                probs, _ = get_sklearn_probs(ckpt_or_dir)
            else:
                probs = get_pytorch_probs(
                    model_name, ckpt_or_dir, test_loader, device
                )
            self.probs_dict[model_name] = probs
            print(f"    Probs shape: {probs.shape}")

    def predict_test(self) -> tuple:
        """
        Compute the weighted average of all model probabilities.

        Returns:
            final_probs: (N, NUM_CLASSES) — weighted average probabilities
            true_labels: (N,) — ground truth
        """
        if not self.probs_dict:
            raise RuntimeError("Call build() first.")

        # Weighted sum of probability arrays
        final_probs = np.zeros(
            (len(self.true_labels), config.NUM_CLASSES), dtype=np.float32
        )
        for model_name, probs in self.probs_dict.items():
            final_probs += self.norm_weights[model_name] * probs

        return final_probs, self.true_labels

    def save_results(self, output_dir: str):
        """
        Run prediction, evaluate, and save all outputs.
        """
        from sklearn.metrics import accuracy_score, f1_score
        from evaluation.evaluate import (
            save_confusion_matrix, save_classification_report,
        )

        os.makedirs(output_dir, exist_ok=True)

        final_probs, true_labels = self.predict_test()
        preds = final_probs.argmax(axis=1)

        acc      = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)

        print(f"\nEnsemble test accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

        # Per-model individual accuracy for comparison
        print("\nPer-model individual accuracy on test set:")
        for model_name, probs in self.probs_dict.items():
            ind_preds = probs.argmax(axis=1)
            ind_acc   = accuracy_score(true_labels, ind_preds)
            print(f"  {model_name}: {ind_acc:.4f}")

        save_confusion_matrix(true_labels, preds, output_dir,
                              model_name="Ensemble")
        save_classification_report(true_labels, preds, output_dir,
                                   model_name="Ensemble")

        # Save probability arrays for potential downstream use
        np.save(os.path.join(output_dir, "ensemble_probs.npy"), final_probs)
        np.save(os.path.join(output_dir, "ensemble_labels.npy"), true_labels)

        # Save weights used
        weights_path = os.path.join(output_dir, "weights_used.txt")
        with open(weights_path, "w") as f:
            f.write("Models included in this ensemble run:\n")
            for m, w in self.norm_weights.items():
                raw_acc = accuracy_score(
                    true_labels, self.probs_dict[m].argmax(axis=1)
                )
                f.write(f"  {m}: weight={w:.3f}, individual_acc={raw_acc:.4f}\n")
            f.write(f"\nFinal ensemble accuracy: {acc:.4f}\n")
            f.write(f"Final ensemble macro F1: {macro_f1:.4f}\n")
        print(f"\nResults saved to: {output_dir}")
        print(f"Weights summary:  {weights_path}")

        return acc, macro_f1

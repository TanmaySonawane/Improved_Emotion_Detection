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
#   model2_cnn_bilstm_mel
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
        from models.cnn_bilstm_mel import CNNBiLSTMMelSER
        model = CNNBiLSTMMelSER()
    elif "cnn_bilstm_mfcc" in model_name:
        from models.cnn_bilstm_mfcc import CNNBiLSTMMFCCSER
        model = CNNBiLSTMMFCCSER()
    elif "multifeature" in model_name:
        from models.multifeature_cnn_bilstm import MultiFeatureCNNBiLSTMSER
        model = MultiFeatureCNNBiLSTMSER()
    elif "wav2vec2" in model_name:
        from models.wav2vec2_ser import Wav2Vec2SER
        model = Wav2Vec2SER()
    else:
        raise ValueError(f"Unknown PyTorch model name: {model_name}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# PROBABILITY EXTRACTION
# ---------------------------------------------------------------------------

def _tta_augment(batch: dict) -> list:
    """
    Generate 3 views of each mel for Test-Time Augmentation:
      - original (no change)
      - mild time mask (15 frames)
      - mild freq mask (10 bins)

    Returns list of 3 batch dicts sharing all tensors except mel.
    Only the mel key is altered — all other features stay identical.
    """
    mel = batch["mel"]   # (B, N_MELS, N_FRAMES)
    views = [batch]      # view 0: original

    # View 1: time mask
    b1 = {k: v for k, v in batch.items()}
    m1 = mel.clone()
    t  = 15
    t0 = (mel.shape[2] - t) // 2   # mask the center region
    m1[:, :, t0:t0+t] = 0.0
    b1["mel"] = m1
    views.append(b1)

    # View 2: freq mask
    b2 = {k: v for k, v in batch.items()}
    m2 = mel.clone()
    f  = 10
    f0 = (mel.shape[1] - f) // 2
    m2[:, f0:f0+f, :] = 0.0
    b2["mel"] = m2
    views.append(b2)

    return views


def get_pytorch_probs(model_name: str, ckpt_path: str,
                       test_loader, device: torch.device,
                       use_tta: bool = True) -> np.ndarray:
    """
    Run a PyTorch model on the test set and return softmax probabilities.
    With TTA, averages probabilities over 3 augmented views per sample.

    Returns:
        probs: np.ndarray of shape (N_test_samples, NUM_CLASSES)
    """
    model = _load_pytorch_model(model_name, ckpt_path)
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            if use_tta:
                views = _tta_augment(batch)
                view_probs = []
                for view in views:
                    logits = model(view, device)
                    view_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                # Average across 3 views
                probs = np.mean(view_probs, axis=0)   # (B, NUM_CLASSES)
            else:
                logits = model(batch, device)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)           # (N, NUM_CLASSES)


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
        self.available    = {}   # model_name → ckpt_path or out_dir
        self.raw_weights  = {}   # model_name → raw weight from config
        self.norm_weights = {}   # model_name → normalized weight (sums to 1)
        self.opt_weights  = {}   # model_name → val-optimized weight (set by optimize_weights)
        self.probs_dict   = {}   # model_name → (N_test, NUM_CLASSES) ndarray
        self.val_probs    = {}   # model_name → (N_val,  NUM_CLASSES) ndarray
        self.true_labels  = None
        self.val_labels   = None

    def build(self, device: torch.device, test_loader=None, val_loader=None):
        """
        Detect available models and collect their test-set probabilities.

        Args:
            device:      torch.device for PyTorch models
            test_loader: DataLoader for the test split (created internally if None)
            val_loader:  DataLoader for the val split (created internally if None)
                         Used by optimize_weights() — not strictly required.
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

        # Create loaders if not provided
        if test_loader is None or val_loader is None:
            from data.dataset import get_dataloader
            if test_loader is None:
                test_loader = get_dataloader("test")
            if val_loader is None:
                val_loader = get_dataloader("val")

        # Collect true labels for test and val
        true_labels_list = []
        for batch in test_loader:
            true_labels_list.extend(batch["label"].numpy())
        self.true_labels = np.array(true_labels_list)

        val_labels_list = []
        for batch in val_loader:
            val_labels_list.extend(batch["label"].numpy())
        self.val_labels = np.array(val_labels_list)

        # Get probabilities from each available model (test + val)
        print()
        for model_name, ckpt_or_dir in self.available.items():
            print(f"  Running inference: {model_name}...")
            if model_name == "model1_traditional":
                probs, _ = get_sklearn_probs(ckpt_or_dir)
                # val probs for model1
                from models.traditional_ml import load_scalar_features
                import pickle
                svm_path    = os.path.join(ckpt_or_dir, "svm_model.pkl")
                scaler_path = os.path.join(ckpt_or_dir, "scaler.pkl")
                with open(svm_path,    "rb") as f: svm    = pickle.load(f)
                with open(scaler_path, "rb") as f: scaler = pickle.load(f)
                X_val, _ = load_scalar_features("val")
                X_val_s  = scaler.transform(X_val)
                val_probs_raw = svm.predict_proba(X_val_s)
                val_probs = np.zeros((val_probs_raw.shape[0], config.NUM_CLASSES), dtype=np.float32)
                for col_idx, cls in enumerate(svm.classes_):
                    val_probs[:, int(cls)] = val_probs_raw[:, col_idx]
            else:
                probs     = get_pytorch_probs(model_name, ckpt_or_dir, test_loader, device)
                val_probs = get_pytorch_probs(model_name, ckpt_or_dir, val_loader,  device)
            self.probs_dict[model_name] = probs
            self.val_probs[model_name]  = val_probs
            print(f"    Probs shape: {probs.shape}")

    def optimize_weights(self):
        """
        Find ensemble weights that maximize validation set accuracy using
        scipy.optimize. Stores results in self.opt_weights.

        The search is over a (N_models,) simplex: weights >= 0, sum = 1.
        Objective: minimize negative validation accuracy.
        """
        from scipy.optimize import minimize
        from sklearn.metrics import accuracy_score

        if not self.val_probs:
            print("  [Ensemble] No val probs available — using normalized config weights.")
            self.opt_weights = self.norm_weights.copy()
            return

        model_names = list(self.val_probs.keys())
        n = len(model_names)

        def neg_val_acc(w):
            w = np.array(w)
            w = np.clip(w, 0, None)
            total = w.sum()
            if total < 1e-8:
                return 1.0  # worst possible (0% accuracy)
            w = w / total
            combined = sum(w[i] * self.val_probs[model_names[i]] for i in range(n))
            preds = combined.argmax(axis=1)
            return -accuracy_score(self.val_labels, preds)

        # Start from current normalized weights
        w0 = np.array([self.norm_weights[m] for m in model_names])

        result = minimize(
            neg_val_acc,
            w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-6},
        )

        opt_w = np.clip(result.x, 0, None)
        opt_w = opt_w / opt_w.sum()

        self.opt_weights = {model_names[i]: float(opt_w[i]) for i in range(n)}

        print("\nOptimized weights (val-set accuracy maximization):")
        for m, w in self.opt_weights.items():
            print(f"  {m}: {w:.3f}")
        print(f"  Val accuracy at optimized weights: {-result.fun:.4f}")

    def predict_test(self, use_optimized: bool = True) -> tuple:
        """
        Compute the weighted average of all model probabilities.

        Args:
            use_optimized: If True and optimize_weights() has been called,
                           use val-optimized weights. Falls back to norm_weights.
        Returns:
            final_probs: (N, NUM_CLASSES) — weighted average probabilities
            true_labels: (N,) — ground truth
        """
        if not self.probs_dict:
            raise RuntimeError("Call build() first.")

        weights = self.opt_weights if (use_optimized and self.opt_weights) else self.norm_weights

        final_probs = np.zeros(
            (len(self.true_labels), config.NUM_CLASSES), dtype=np.float32
        )
        for model_name, probs in self.probs_dict.items():
            final_probs += weights[model_name] * probs

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

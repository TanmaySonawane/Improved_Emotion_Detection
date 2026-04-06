# =============================================================================
# evaluate.py
#
# Evaluation utilities used by ALL training scripts and the ensemble.
# Produces:
#   - Confusion matrix plot (PNG)
#   - Classification report (TXT + CSV)
#   - Training curves plot  (PNG)
#   - Per-model summary table
#
# Functions here are imported by every train_*.py script.
# Also has a standalone __main__ mode to re-evaluate all saved models at once.
#
# Run all:  python Scripts/evaluation/evaluate.py --all
# Run one:  python Scripts/evaluation/evaluate.py --model model4_efficientnet
# =============================================================================

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on Windows without display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------

def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                           output_dir: str, model_name: str = "Model"):
    """
    Plot and save a normalized confusion matrix as a PNG.

    The matrix is normalized by true class so each row sums to 1.
    This makes it easy to spot which emotions are confused with which,
    regardless of class imbalance.
    """
    emotion_names = [config.EMOTIONS[i] for i in sorted(config.EMOTIONS.keys())]
    cm = confusion_matrix(labels, preds, labels=sorted(config.EMOTIONS.keys()))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Normalized ---
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=emotion_names, yticklabels=emotion_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name}\nNormalized Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # --- Raw counts ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=emotion_names, yticklabels=emotion_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name}\nRaw Counts")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------------------------

def save_classification_report(labels: np.ndarray, preds: np.ndarray,
                                output_dir: str, model_name: str = "Model"):
    """
    Save sklearn's classification_report as both .txt and .csv.

    The report shows per-class precision, recall, F1 and support,
    plus macro and weighted averages.
    """
    emotion_names = [config.EMOTIONS[i] for i in sorted(config.EMOTIONS.keys())]

    # Text report
    report_str = classification_report(
        labels, preds,
        labels=sorted(config.EMOTIONS.keys()),
        target_names=emotion_names,
        digits=4,
        zero_division=0,
    )
    txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_str)
    print(f"  Classification report saved: {txt_path}")

    # CSV report (easier to compare across models later)
    report_dict = classification_report(
        labels, preds,
        labels=sorted(config.EMOTIONS.keys()),
        target_names=emotion_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report_dict).T
    csv_path = os.path.join(output_dir, "classification_report.csv")
    df.to_csv(csv_path)
    print(f"  Classification report CSV: {csv_path}")

    # Print summary to console
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"  Test accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(f"\n{report_str}")

    return report_str


# ---------------------------------------------------------------------------
# TRAINING CURVES
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_dir: str, model_name: str = "Model"):
    """
    Plot training and validation loss + accuracy curves.

    Args:
        history: TrainingHistory object (from train_utils.py) OR path to history.json
    """
    # Accept either a TrainingHistory object or a JSON path
    if isinstance(history, str):
        with open(history) as f:
            data = json.load(f)
        train_loss = data["train_loss"]
        val_loss   = data["val_loss"]
        train_acc  = data["train_acc"]
        val_acc    = data["val_acc"]
    else:
        train_loss = history.train_loss
        val_loss   = history.val_loss
        train_acc  = history.train_acc
        val_acc    = history.val_acc

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_loss,   "r-o", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, val_acc,   "r-o", markersize=3, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CROSS-MODEL SUMMARY TABLE
# ---------------------------------------------------------------------------

def build_summary_table(output_base_dir: str = None) -> pd.DataFrame:
    """
    Scan all model output directories for classification_report.csv files
    and compile a single comparison table.

    Returns a DataFrame with one row per model showing test accuracy and F1.
    """
    if output_base_dir is None:
        output_base_dir = config.OUTPUTS_DIR

    rows = []
    model_dirs = sorted(os.listdir(output_base_dir))

    for model_dir in model_dirs:
        csv_path = os.path.join(output_base_dir, model_dir, "classification_report.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0)

        try:
            accuracy  = float(df.loc["accuracy",  "precision"]) if "accuracy"  in df.index else None
            macro_f1  = float(df.loc["macro avg", "f1-score"])  if "macro avg" in df.index else None
            weighted_f1 = float(df.loc["weighted avg", "f1-score"]) if "weighted avg" in df.index else None
        except Exception:
            accuracy = macro_f1 = weighted_f1 = None

        rows.append({
            "model":        model_dir,
            "accuracy":     accuracy,
            "macro_f1":     macro_f1,
            "weighted_f1":  weighted_f1,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("macro_f1", ascending=False)

    return summary


def print_summary_table():
    """Print formatted model comparison table to console."""
    table = build_summary_table()
    if table.empty:
        print("No completed model results found.")
        return
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(table.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60)

    # Save to file
    table_path = os.path.join(config.OUTPUTS_DIR, "model_comparison.csv")
    table.to_csv(table_path, index=False)
    print(f"Table saved: {table_path}")


# ---------------------------------------------------------------------------
# STANDALONE: Re-evaluate a specific model from saved checkpoint
# ---------------------------------------------------------------------------

def evaluate_from_checkpoint(model_name: str):
    """
    Load a saved model checkpoint and re-run evaluation on the test set.
    Useful if you want to regenerate plots without retraining.

    model_name: e.g., "model4_efficientnet", "model5_resnet18"
    """
    import torch
    from data.dataset import get_dataloader
    from training.train_utils import get_device, LabelSmoothingCrossEntropy, validate

    output_dir = os.path.join(config.OUTPUTS_DIR, model_name)
    ckpt_path  = os.path.join(output_dir, "best_model.pth")

    if not os.path.isfile(ckpt_path):
        print(f"No checkpoint found: {ckpt_path}")
        return

    device = get_device()

    # Instantiate the correct model class based on model_name
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
    else:
        print(f"Unknown model_name: {model_name}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test_loader = get_dataloader("test")
    criterion = LabelSmoothingCrossEntropy()
    _, test_acc, preds, labels = validate(model, test_loader, criterion, device)

    display_name = model_name.replace("_", " ").title()
    save_confusion_matrix(labels, preds, output_dir, model_name=display_name)
    save_classification_report(labels, preds, output_dir, model_name=display_name)

    # Re-plot training curves if history exists
    hist_path = os.path.join(output_dir, "history.json")
    if os.path.isfile(hist_path):
        plot_training_curves(hist_path, output_dir, model_name=display_name)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER Evaluation")
    parser.add_argument("--all",   action="store_true",
                        help="Print summary table of all completed models")
    parser.add_argument("--model", type=str, default=None,
                        help="Re-evaluate a specific model, e.g. --model model4_efficientnet")
    args = parser.parse_args()

    if args.all:
        print_summary_table()

    if args.model:
        print(f"Re-evaluating: {args.model}")
        evaluate_from_checkpoint(args.model)

    if not args.all and not args.model:
        print("Usage:")
        print("  python evaluate.py --all               # show comparison table")
        print("  python evaluate.py --model model4_efficientnet  # re-eval one model")

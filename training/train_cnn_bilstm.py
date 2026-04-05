# =============================================================================
# train_cnn_bilstm.py — Train Model 2 (mel) or Model 3 (mfcc)
#
# Usage:
#   python Scripts/training/train_cnn_bilstm.py --variant mel
#   python Scripts/training/train_cnn_bilstm.py --variant mfcc
#
# The --variant flag selects the model class and output directory.
# Everything else (optimizer, scheduler, early stopping) is shared.
# =============================================================================

import os
import sys
import argparse

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.dataset import get_dataloader
from training.train_utils import (
    get_device, compute_class_weights_tensor, LabelSmoothingCrossEntropy,
    train_one_epoch, validate, EarlyStopping, TrainingHistory,
)
from evaluation.evaluate import save_confusion_matrix, save_classification_report, plot_training_curves

# ---------------------------------------------------------------------------
# Hyperparameters (CNN+BiLSTM specific — intentionally not in config since
# they only apply to these two models)
# ---------------------------------------------------------------------------
_LR           = 1e-3
_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS   = 80
_ETA_MIN      = 1e-6  # CosineAnnealingLR minimum LR
_PATIENCE_ES  = 15    # EarlyStopping — stop after 15 no-improve epochs


def main(variant: str):
    assert variant in ("mel", "mfcc"), f"--variant must be 'mel' or 'mfcc', got '{variant}'"

    if variant == "mel":
        from models.cnn_bilstm_mel import CNNBiLSTMMelSER as ModelClass
        output_dir = os.path.join(config.OUTPUTS_DIR, "model2_cnn_bilstm_mel")
        model_label = "CNN+BiLSTM+Attention (Mel)"
    else:
        from models.cnn_bilstm_mfcc import CNNBiLSTMMFCCSER as ModelClass
        output_dir = os.path.join(config.OUTPUTS_DIR, "model3_cnn_bilstm_mfcc")
        model_label = "CNN+BiLSTM+Attention (MFCC)"

    print("=" * 60)
    print(f"Model — {model_label}")
    print("=" * 60)
    config.create_output_dirs()
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    model  = ModelClass().to(device)
    print(f"Total parameters: {model.count_params():,}")

    print("\nLoading data...")
    train_loader = get_dataloader("train")
    val_loader   = get_dataloader("val")
    test_loader  = get_dataloader("test")

    train_labels  = train_loader.dataset.get_labels()
    class_weights = compute_class_weights_tensor(train_labels, device)
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.LABEL_SMOOTHING,
        weight=class_weights,
    )

    optimizer  = optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_MAX_EPOCHS, eta_min=_ETA_MIN
    )
    scaler     = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path  = os.path.join(output_dir, "best_model.pth")
    early_stop = EarlyStopping(patience=_PATIENCE_ES, checkpoint_path=ckpt_path)
    history    = TrainingHistory()

    print(f"\nTraining for up to {_MAX_EPOCHS} epochs...")
    for epoch in range(1, _MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        scheduler.step()   # cosine annealing steps every epoch unconditionally

        print(f"  Epoch {epoch:3d}/{_MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if early_stop.step(val_loss, model):
            break

    print("\nLoading best checkpoint for test evaluation...")
    early_stop.load_best(model)

    _, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    history.save(os.path.join(output_dir, "history.json"))
    plot_training_curves(history, output_dir, model_name=model_label)
    save_confusion_matrix(labels, preds, output_dir, model_name=model_label)
    save_classification_report(labels, preds, output_dir, model_name=model_label)

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Best model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", required=True, choices=["mel", "mfcc"],
        help="Which CNN+BiLSTM model to train: 'mel' (Model 2) or 'mfcc' (Model 3)"
    )
    args = parser.parse_args()
    main(args.variant)

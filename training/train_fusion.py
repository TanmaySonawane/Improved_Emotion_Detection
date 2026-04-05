# =============================================================================
# train_fusion.py — Train Model 6: MFCC + Chroma Fusion
#
# Run: python Scripts/training/train_fusion.py
# =============================================================================

import os
import sys

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.fusion import FusionSER
from data.dataset import get_dataloader
from training.train_utils import (
    get_device, compute_class_weights_tensor, LabelSmoothingCrossEntropy,
    train_one_epoch, validate, EarlyStopping, TrainingHistory,
)
from evaluation.evaluate import save_confusion_matrix, save_classification_report, plot_training_curves

_LR           = 1e-3
_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS   = 80
_ETA_MIN      = 1e-6
_PATIENCE_ES  = 15

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "model6_fusion")


def main():
    print("=" * 60)
    print("Model 6 — MFCC + Chroma Fusion (CNN+BiLSTM+Attention)")
    print("=" * 60)
    config.create_output_dirs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    model  = FusionSER().to(device)
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
    ckpt_path  = os.path.join(OUTPUT_DIR, "best_model.pth")
    early_stop = EarlyStopping(patience=_PATIENCE_ES, checkpoint_path=ckpt_path)
    history    = TrainingHistory()

    print(f"\nTraining for up to {_MAX_EPOCHS} epochs...")
    for epoch in range(1, _MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{_MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if early_stop.step(val_loss, model):
            break

    print("\nLoading best checkpoint for test evaluation...")
    early_stop.load_best(model)

    _, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    history.save(os.path.join(OUTPUT_DIR, "history.json"))
    plot_training_curves(history, OUTPUT_DIR, model_name="Fusion (MFCC+Chroma)")
    save_confusion_matrix(labels, preds, OUTPUT_DIR, model_name="Fusion (MFCC+Chroma)")
    save_classification_report(labels, preds, OUTPUT_DIR, model_name="Fusion (MFCC+Chroma)")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Best model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

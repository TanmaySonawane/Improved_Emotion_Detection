# =============================================================================
# train_resnet18.py — Train Model 5: ResNet-18 Dual-Input
#
# Run: python Scripts/training/train_resnet18.py
# =============================================================================

import os
import sys

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.resnet18_dual import ResNet18DualSER
from data.dataset import get_dataloader
from training.train_utils import (
    get_device, compute_class_weights_tensor, LabelSmoothingCrossEntropy,
    train_one_epoch, validate, EarlyStopping, TrainingHistory,
)
from evaluation.evaluate import save_confusion_matrix, save_classification_report, plot_training_curves

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "model5_resnet18")


def main():
    print("=" * 60)
    print("Model 5 — ResNet-18 Dual-Input (Mel + Waveform)")
    print("=" * 60)
    config.create_output_dirs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    model  = ResNet18DualSER().to(device)
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

    optimizer  = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                             weight_decay=1e-4)
    # CosineAnnealingWarmRestarts restarts every T_0=20 epochs (doubling each time).
    # With T_max=100 and early stopping at epoch 22, plain CosineAnnealingLR never
    # reaches its low-LR fine-tuning phase. Warm restarts ensure each 20-epoch cycle
    # completes a full cosine sweep regardless of when early stopping fires.
    scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    scaler     = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path  = os.path.join(OUTPUT_DIR, "best_model.pth")
    early_stop = EarlyStopping(patience=12, checkpoint_path=ckpt_path)
    history    = TrainingHistory()

    print(f"\nTraining for up to {config.MAX_EPOCHS} epochs...")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if early_stop.step(val_loss, model):
            break

    print("\nLoading best checkpoint for test evaluation...")
    early_stop.load_best(model)

    test_loss, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    history.save(os.path.join(OUTPUT_DIR, "history.json"))
    plot_training_curves(history, OUTPUT_DIR, model_name="ResNet-18 Dual")
    save_confusion_matrix(labels, preds, OUTPUT_DIR, model_name="ResNet-18 Dual")
    save_classification_report(labels, preds, OUTPUT_DIR, model_name="ResNet-18 Dual")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

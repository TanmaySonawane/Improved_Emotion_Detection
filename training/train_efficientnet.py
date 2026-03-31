# =============================================================================
# train_efficientnet.py — Train Model 4: EfficientNet-B0
#
# Two-phase training:
#   Phase 1 (warmup):    Freeze backbone, train head only for WARMUP_EPOCHS
#   Phase 2 (fine-tune): Unfreeze top 3 MBConv blocks, lower LR, train with
#                        early stopping
#
# Run: python Scripts/training/train_efficientnet.py
# =============================================================================

import os
import sys

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.efficientnet_b0 import EfficientNetSER
from data.dataset import get_dataloader
from training.train_utils import (
    get_device, compute_class_weights_tensor, LabelSmoothingCrossEntropy,
    train_one_epoch, validate, EarlyStopping, TrainingHistory,
)
from evaluation.evaluate import save_confusion_matrix, save_classification_report, plot_training_curves

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "model4_efficientnet")


def main():
    print("=" * 60)
    print("Model 4 — EfficientNet-B0 (Transfer Learning)")
    print("=" * 60)
    config.create_output_dirs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    model  = EfficientNetSER().to(device)

    # DataLoaders (SpecAugment active for train)
    print("\nLoading data...")
    train_loader = get_dataloader("train")
    val_loader   = get_dataloader("val")
    test_loader  = get_dataloader("test")

    # Class weights from train labels
    train_labels = train_loader.dataset.get_labels()
    class_weights = compute_class_weights_tensor(train_labels, device)
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.LABEL_SMOOTHING,
        weight=class_weights,
    )

    history = TrainingHistory()
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    scaler = GradScaler()

    # ================================================================
    # PHASE 1: Warmup — freeze backbone, train head only
    # ================================================================
    print(f"\n--- Phase 1: Warmup ({config.EFFICIENTNET_WARMUP_EPOCHS} epochs, backbone frozen) ---")
    model.freeze_backbone()

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
    )

    for epoch in range(1, config.EFFICIENTNET_WARMUP_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer_p1, scaler, criterion, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        print(f"  Epoch {epoch:3d}/{config.EFFICIENTNET_WARMUP_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

    # ================================================================
    # PHASE 2: Fine-tune — unfreeze top 3 MBConv blocks
    # ================================================================
    print(f"\n--- Phase 2: Fine-tune (≤{config.EFFICIENTNET_FINETUNE_EPOCHS} epochs, top blocks unfrozen) ---")
    model.unfreeze_top_blocks(n_blocks=3)

    optimizer_p2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.EFFICIENTNET_FINETUNE_LR,   # lower LR to protect pretrained weights
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode="min", patience=5, factor=0.5, verbose=True
    )
    early_stop = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        checkpoint_path=ckpt_path,
    )

    for epoch in range(1, config.EFFICIENTNET_FINETUNE_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer_p2, scaler, criterion, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch:3d}/{config.EFFICIENTNET_FINETUNE_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if early_stop.step(val_loss, model):
            break

    # ================================================================
    # FINAL EVALUATION on test set (used ONCE at the end)
    # ================================================================
    print("\nLoading best checkpoint for test evaluation...")
    early_stop.load_best(model)

    test_loss, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save results
    history.save(os.path.join(OUTPUT_DIR, "history.json"))
    plot_training_curves(history, OUTPUT_DIR, model_name="EfficientNet-B0")
    save_confusion_matrix(labels, preds, OUTPUT_DIR, model_name="EfficientNet-B0")
    save_classification_report(labels, preds, OUTPUT_DIR, model_name="EfficientNet-B0")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Best model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

# =============================================================================
# train_wav2vec2.py — Train Model 9: Wav2Vec 2.0 Fine-tuned for SER
#
# Key differences from other training scripts:
#   - batch_size=8 (not 32) — wav2vec2 backbone is 94M parameters
#   - lr=1e-4 (not 1e-3) — fine-tuning pretrained model, not training from scratch
#   - use_mixup=False — waveform mixup not applied (mel mixup would be unused)
#   - CosineAnnealingWarmRestarts — full LR cycle every 20 epochs
#   - MAX_EPOCHS=40 — fine-tuning converges faster than training from scratch
#
# Requires: pip install transformers
#
# Run: py run_pipeline.py --step train9
#   or: python Scripts/training/train_wav2vec2.py
# =============================================================================

import os
import sys

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.wav2vec2_ser import Wav2Vec2SER
from data.dataset import get_dataloader
from training.train_utils import (
    get_device, compute_class_weights_tensor, LabelSmoothingCrossEntropy,
    train_one_epoch, validate, EarlyStopping, TrainingHistory,
)
from evaluation.evaluate import save_confusion_matrix, save_classification_report, plot_training_curves

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "model9_wav2vec2")

# ---------------------------------------------------------------------------
# Hyperparameters — tuned for fine-tuning a large pretrained model
# ---------------------------------------------------------------------------
_LR           = 1e-4    # lower than default 1e-3 — avoid destroying pretrained weights
_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS   = 40      # fine-tuning converges faster than training from scratch
_PATIENCE_ES  = 12
_BATCH_SIZE   = 8       # smaller batch to fit wav2vec2-base (94M params) in 4GB VRAM


def main():
    print("=" * 60)
    print("Model 9 — Wav2Vec 2.0 Fine-tuned for SER")
    print("  Pretrained: facebook/wav2vec2-base (960h LibriSpeech)")
    print("  Fine-tuning: top 6 of 12 transformer layers + classifier")
    print("=" * 60)
    config.create_output_dirs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()

    print("\nBuilding model (will download wav2vec2-base on first run ~360MB)...")
    model = Wav2Vec2SER(freeze_bottom_n_layers=6).to(device)
    total, trainable = model.count_params()
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}  ({100*trainable/total:.1f}%)")

    # -----------------------------------------------------------------------
    # Data — smaller batch size for VRAM budget
    # -----------------------------------------------------------------------
    print("\nLoading data...")
    train_loader = get_dataloader("train",  batch_size=_BATCH_SIZE)
    val_loader   = get_dataloader("val",    batch_size=_BATCH_SIZE)
    test_loader  = get_dataloader("test",   batch_size=_BATCH_SIZE)

    train_labels  = train_loader.dataset.get_labels()
    class_weights = compute_class_weights_tensor(train_labels, device)
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.LABEL_SMOOTHING,
        weight=class_weights,
    )

    # -----------------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------------
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=_LR,
        weight_decay=_WEIGHT_DECAY,
    )
    # CosineAnnealingWarmRestarts: first restart at epoch 20, then epoch 60.
    # Ensures at least one full cosine cycle completes before early stopping.
    scheduler  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    scaler     = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path  = os.path.join(OUTPUT_DIR, "best_model.pth")
    early_stop = EarlyStopping(patience=_PATIENCE_ES, checkpoint_path=ckpt_path)
    history    = TrainingHistory()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nFine-tuning for up to {_MAX_EPOCHS} epochs (batch_size={_BATCH_SIZE})...")
    print("  Note: first epoch is slower — torchaudio resamples on-the-fly.\n")

    for epoch in range(1, _MAX_EPOCHS + 1):
        # use_mixup=False: mixup mixes mel spectrograms; wav2vec2 reads waveform
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            use_mixup=False,
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history.record(train_loss, val_loss, train_acc, val_acc)
        scheduler.step(epoch - 1)  # WarmRestarts uses epoch-based stepping

        print(f"  Epoch {epoch:3d}/{_MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if early_stop.step(val_loss, model):
            break

    print("\nLoading best checkpoint for test evaluation...")
    early_stop.load_best(model)

    test_loss, test_acc, preds, labels = validate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    history.save(os.path.join(OUTPUT_DIR, "history.json"))
    plot_training_curves(history, OUTPUT_DIR, model_name="Wav2Vec2-SER")
    save_confusion_matrix(labels, preds, OUTPUT_DIR, model_name="Wav2Vec2-SER")
    save_classification_report(labels, preds, OUTPUT_DIR, model_name="Wav2Vec2-SER")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

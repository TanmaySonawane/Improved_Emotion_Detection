# =============================================================================
# train_utils.py
#
# Shared utilities for ALL PyTorch training scripts:
#   - GPU setup and verification
#   - Class weight computation (handles imbalanced datasets)
#   - Label Smoothing Cross Entropy loss
#   - AMP training loop (one epoch)
#   - Validation loop
#   - EarlyStopping with checkpoint saving
# =============================================================================

import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# GPU SETUP
# ---------------------------------------------------------------------------

def get_device(verbose: bool = True) -> torch.device:
    """
    Returns the best available device (CUDA GPU or CPU).
    Prints GPU info and a warning if GPU is not available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("WARNING: GPU not available, training on CPU (will be slow).")
            print("  Verify PyTorch CUDA installation: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    return device


# ---------------------------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------------------------

def compute_class_weights_tensor(labels: np.ndarray,
                                  device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights from label array and return as a CUDA/CPU tensor.

    Why? The 'neutral' class has far fewer samples than other emotions.
    Class weights give higher loss to mistakes on under-represented classes,
    preventing the model from ignoring them.

    The weights are derived dynamically from the actual label distribution —
    they are NOT hardcoded for 5 classes.
    """
    # np.unique returns only classes that actually appear in labels
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    # Build a full-size tensor of length NUM_CLASSES with correct indices
    weight_tensor = torch.ones(config.NUM_CLASSES, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        weight_tensor[int(cls)] = float(w)

    return weight_tensor.to(device)


# ---------------------------------------------------------------------------
# LABEL SMOOTHING CROSS ENTROPY
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Instead of training the model to output [0,0,1,0,0] for the correct class,
    we train it to output [(ε/K), (ε/K), (1-ε + ε/K), (ε/K), (ε/K)].
    This prevents over-confidence and improves calibration.

    smoothing=0.1 is the standard value used in most SER papers.

    Supports class weights (passed as weight= to the underlying NLLLoss).
    """

    def __init__(self, smoothing: float = 0.1,
                 weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.num_classes = config.NUM_CLASSES  # derived from config, not hardcoded

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, num_classes) — raw model outputs (before softmax)
            targets: (batch,) — integer class indices

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)   # (batch, num_classes)

        # Smooth targets: fill all classes with smoothing / num_classes,
        # then add (1 - smoothing) to the true class.
        smooth_targets = torch.full_like(log_probs,
                                         self.smoothing / self.num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs)  # (batch, num_classes)

        # Apply class weights if provided
        if self.weight is not None:
            w = self.weight[targets]  # (batch,)
            loss = loss.sum(dim=-1) * w
            return loss.mean()
        else:
            return loss.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# AMP TRAINING LOOP
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    criterion: nn.Module,
                    device: torch.device,
                    feature_key: str = "mel") -> tuple:
    """
    Run one training epoch with Automatic Mixed Precision (AMP).

    AMP uses float16 for most computations (faster, less VRAM) while keeping
    critical parts in float32 to prevent NaN gradients.

    Args:
        feature_key: Which key to extract from the batch dict as the primary
                     input. Can be "mel", "mfcc", etc. For multi-input models,
                     the model's forward() receives the full batch dict.

    Returns:
        (avg_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            # Pass the full batch dict to the model.
            # Each model's forward() picks what it needs.
            outputs = model(batch, device)   # (batch, num_classes) logits
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * len(labels)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device) -> tuple:
    """
    Run a validation/test pass (no gradient updates).

    Returns:
        (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            # No autocast during validation (not needed, saves complexity)
            outputs = model(batch, device)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * len(labels)
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors validation loss and stops training if it doesn't improve
    for `patience` consecutive epochs.

    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience: int = None, checkpoint_path: str = "best_model.pth",
                 verbose: bool = True):
        self.patience   = patience if patience is not None else config.EARLY_STOP_PATIENCE
        self.ckpt_path  = checkpoint_path
        self.verbose    = verbose
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call after each validation pass.

        Returns:
            True if training should stop, False to continue.
        """
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.ckpt_path)
            if self.verbose:
                print(f"    [EarlyStopping] New best val_loss={val_loss:.4f} → saved checkpoint")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"    [EarlyStopping] Stopping training.")
                return True
        return False

    def load_best(self, model: nn.Module):
        """Restore the best saved weights into `model`."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        elif os.path.isfile(self.ckpt_path):
            model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))


# ---------------------------------------------------------------------------
# TRAINING HISTORY
# ---------------------------------------------------------------------------

class TrainingHistory:
    """Tracks epoch-by-epoch metrics for later plotting."""

    def __init__(self):
        self.train_loss = []
        self.val_loss   = []
        self.train_acc  = []
        self.val_acc    = []

    def record(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

    def save(self, path: str):
        import json
        data = {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "train_acc":  self.train_acc,
            "val_acc":    self.val_acc,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Training history saved: {path}")

# =============================================================================
# dataset.py
#
# PyTorch Dataset and DataLoader factory for the SER project.
#
# Each item returns a dict of tensors:
#   mel, mfcc, chroma, spectral_contrast, waveform, scalar, label
#
# SpecAugment (time masking + frequency masking) is applied ONLINE to mel
# during training only — it is built into the Dataset, not the model.
#
# CRITICAL: num_workers=0 is enforced everywhere (Windows PyTorch requirement).
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# SPECAUGMENT
# Applied only when split == "train".
# ---------------------------------------------------------------------------

def spec_augment(mel: torch.Tensor,
                 time_mask_param: int = 30,
                 freq_mask_param: int = 20,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2) -> torch.Tensor:
    """
    Apply SpecAugment to a mel spectrogram tensor.

    Input shape: (n_mels, n_frames) or (n_mels, n_frames, 1)

    Time masking: randomly zero out a contiguous block of time frames.
    Frequency masking: randomly zero out a contiguous block of mel bins.

    Why does this help?
      During training, we randomly "hide" parts of the spectrogram.
      The model is forced to predict emotion without relying on any single
      part. This improves generalization to unseen audio.
    """
    # Work on a copy so we don't modify the original tensor
    mel = mel.clone()

    # Handle optional channel dimension
    squeezed = mel.ndim == 3
    if squeezed:
        mel = mel.squeeze(-1)  # (n_mels, n_frames)

    n_mels, n_frames = mel.shape

    # --- Time masking ---
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, n_frames))
        t0 = np.random.randint(0, n_frames - t + 1)
        mel[:, t0 : t0 + t] = 0.0

    # --- Frequency masking ---
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels))
        f0 = np.random.randint(0, n_mels - f + 1)
        mel[f0 : f0 + f, :] = 0.0

    if squeezed:
        mel = mel.unsqueeze(-1)
    return mel


# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class SERDataset(Dataset):
    """
    Speech Emotion Recognition Dataset.

    Reads rows from manifest.csv filtered by split, loads .npy features,
    and returns a dict of tensors ready for model input.

    Args:
        manifest_path: Path to manifest.csv
        split: "train", "val", or "test"
        apply_spec_augment: If True, apply SpecAugment to mel (train only recommended)
    """

    def __init__(self, manifest_path: str, split: str,
                 apply_spec_augment: bool = False):
        super().__init__()
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.apply_spec_augment = apply_spec_augment

        print(f"  SERDataset [{split}]: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load features from .npy files ---
        mel      = np.load(row["mel_path"])               # (N_MELS, N_FRAMES)
        mfcc     = np.load(row["mfcc_path"])              # (N_MFCC, N_FRAMES, 3)
        chroma   = np.load(row["chroma_path"])            # (N_CHROMA, N_FRAMES)
        sc       = np.load(row["spectral_contrast_path"]) # (7, N_FRAMES)
        waveform = np.load(row["waveform_path"])          # (N_SAMPLES,)
        scalar   = np.load(row["scalar_path"])            # (SCALAR_DIM,)
        label    = int(row["label"])

        # --- Convert to float32 tensors ---
        mel_t      = torch.tensor(mel,      dtype=torch.float32)
        mfcc_t     = torch.tensor(mfcc,     dtype=torch.float32)
        chroma_t   = torch.tensor(chroma,   dtype=torch.float32)
        sc_t       = torch.tensor(sc,       dtype=torch.float32)
        wave_t     = torch.tensor(waveform, dtype=torch.float32)
        scalar_t   = torch.tensor(scalar,   dtype=torch.float32)
        label_t    = torch.tensor(label,    dtype=torch.long)

        # --- Online SpecAugment (train only) ---
        if self.apply_spec_augment:
            mel_t = spec_augment(mel_t)

        return {
            "mel":               mel_t,       # (N_MELS, N_FRAMES)
            "mfcc":              mfcc_t,      # (N_MFCC, N_FRAMES, 3)
            "chroma":            chroma_t,    # (N_CHROMA, N_FRAMES)
            "spectral_contrast": sc_t,        # (7, N_FRAMES)
            "waveform":          wave_t,      # (N_SAMPLES,)
            "scalar":            scalar_t,    # (SCALAR_DIM,)
            "label":             label_t,     # scalar
        }

    def get_labels(self):
        """Return all labels as a numpy array (used for class weight computation)."""
        return self.df["label"].values.astype(np.int64)


# ---------------------------------------------------------------------------
# DATALOADER FACTORY
# ---------------------------------------------------------------------------

def get_dataloader(split: str, batch_size: int = None,
                   apply_spec_augment: bool = None,
                   shuffle: bool = None) -> DataLoader:
    """
    Create a DataLoader for a given split.

    Defaults:
        batch_size: config.BATCH_SIZE
        apply_spec_augment: True for train, False for val/test
        shuffle: True for train, False for val/test
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if apply_spec_augment is None:
        apply_spec_augment = (split == "train")
    if shuffle is None:
        shuffle = (split == "train")

    dataset = SERDataset(
        manifest_path=config.MANIFEST_PATH,
        split=split,
        apply_spec_augment=apply_spec_augment,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,  # CRITICAL: 0 on Windows
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),    # avoids tiny final batches during training
    )
    return loader


def get_all_loaders(batch_size: int = None):
    """Convenience: returns (train_loader, val_loader, test_loader)."""
    return (
        get_dataloader("train", batch_size=batch_size),
        get_dataloader("val",   batch_size=batch_size),
        get_dataloader("test",  batch_size=batch_size),
    )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing SERDataset...")

    train_loader = get_dataloader("train", batch_size=4)
    batch = next(iter(train_loader))

    print("\nBatch keys and shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

    # Verify expected shapes
    assert batch["mel"].shape      == (4, config.N_MELS, config.N_FRAMES), \
        f"Mel shape mismatch: {batch['mel'].shape}"
    assert batch["mfcc"].shape     == (4, config.N_MFCC, config.N_FRAMES, 3), \
        f"MFCC shape mismatch: {batch['mfcc'].shape}"
    assert batch["chroma"].shape   == (4, config.N_CHROMA, config.N_FRAMES), \
        f"Chroma shape mismatch: {batch['chroma'].shape}"
    assert batch["spectral_contrast"].shape == (4, config.N_SPECTRAL_CONTRAST_ROWS, config.N_FRAMES), \
        f"SC shape mismatch: {batch['spectral_contrast'].shape}"
    assert batch["waveform"].shape == (4, config.N_SAMPLES), \
        f"Waveform shape mismatch: {batch['waveform'].shape}"
    assert batch["scalar"].shape   == (4, config.SCALAR_DIM), \
        f"Scalar shape mismatch: {batch['scalar'].shape}"
    assert batch["label"].shape    == (4,), \
        f"Label shape mismatch: {batch['label'].shape}"
    assert batch["label"].max() < config.NUM_CLASSES, \
        f"Label out of range: {batch['label'].max()}"

    print("\nAll shape checks PASSED!")

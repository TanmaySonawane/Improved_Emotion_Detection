# =============================================================================
# cnn_bilstm_mfcc.py — Model 3: CNN + BiLSTM + Attention on MFCC Stack
#
# Input: MFCC stack (B, 40, 173, 3) — 3 channels are [raw, delta, delta-delta]
#
# Architecture:
#   mfcc (B, 40, 173, 3)
#   → permute → (B, 3, 40, 173)   [channels first for Conv2d]
#   → 3 CNN blocks (MaxPool on freq axis only) → (B, 128, 5, 173)
#   → mean over freq → (B, 128, 173)
#   → permute → (B, 173, 128)    [173 time steps × 128-dim feature]
#   → 2-layer BiLSTM(128 → 64 bi) → (B, 173, 128)
#   → self-attention → (B, 128)
#   → Dropout(0.3) → Linear(128, NUM_CLASSES)
#
# Why a separate MFCC model?
#   MFCC captures cepstral (vocal tract shape) features that mel spectrograms
#   de-emphasize. Delta/delta-delta channels explicitly encode rate of change —
#   a strong cue for dynamic emotions like fear (fast changes) vs sad (slow).
#   MFCC + mel together in the ensemble cover complementary feature spaces.
#
# Run: python Scripts/training/train_cnn_bilstm.py --variant mfcc
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Re-use the same building blocks — import to avoid duplication
from models.cnn_bilstm_mel import SelfAttention, _cnn_block


class CNNBiLSTMMFCCSER(nn.Module):
    """
    CNN + BiLSTM + Attention model for Speech Emotion Recognition.
    Input feature: per-row-normalized MFCC stack (B, N_MFCC, N_FRAMES, 3).

    3 CNN blocks are used (vs 4 for mel) because the frequency axis is only
    40 bins (MFCC) vs 128 bins (mel), so fewer pooling steps are needed.
    """

    def __init__(self):
        super().__init__()

        # 3 CNN blocks reduce freq: 40 → 20 → 10 → 5
        # Time dimension (173) is untouched throughout.
        # 3 input channels = raw MFCC + delta + delta-delta
        self.cnn = nn.Sequential(
            _cnn_block(3,   32),   # (B, 32,  20, 173)
            _cnn_block(32,  64),   # (B, 64,  10, 173)
            _cnn_block(64,  128),  # (B, 128,  5, 173)
        )

        # Spatial Dropout before BiLSTM input.
        # Unlike nn.Dropout (zeros random scalars), Dropout1d zeros entire
        # time steps. This forces the model to not rely on any single frame
        # for its decision — directly combats memorizing speaker-specific
        # temporal patterns that caused the 91% train / 59% test gap.
        self.spatial_drop = nn.Dropout1d(p=0.2)

        # 2-layer BiLSTM: 173 time steps, each 128-dim
        # Hidden reduced 64→32 to reduce capacity on this small feature space.
        # Output: (B, 173, 64) — 32 forward + 32 backward per step
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )

        self.attention  = SelfAttention(hidden_dim=64)  # 32×2 directions

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        mfcc = batch["mfcc"].to(device)        # (B, 40, 173, 3)
        x    = mfcc.permute(0, 3, 1, 2)        # (B, 3, 40, 173) — channels first

        x = self.cnn(x)                         # (B, 128, 5, 173)
        x = x.mean(dim=2)                       # (B, 128, 173) — collapse freq
        x = x.permute(0, 2, 1)                  # (B, 173, 128) — time-first
        x = self.spatial_drop(x)                # zero entire time steps (Spatial Dropout)

        x, _ = self.bilstm(x)                   # (B, 173, 64)
        x    = self.attention(x)                 # (B, 64)
        return self.classifier(x)               # (B, NUM_CLASSES)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

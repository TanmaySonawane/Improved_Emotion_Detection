# =============================================================================
# multifeature_cnn_bilstm.py — Model 8: Multi-Feature CNN+BiLSTM+Attention
#
# Combines ALL per-frame acoustic features into one joint representation:
#   mel (128) + MFCC+Δ+ΔΔ flat (40×3=120) + Chroma (12) + Spectral Contrast (7)
#   = 267 features per time frame
#
# Architecture:
#   Per-frame features stacked → (B, 267, 173)
#   → unsqueeze → (B, 1, 267, 173)
#   → 5 CNN blocks (MaxPool freq-only): 267→133→66→33→16→8
#   → mean over freq → (B, 256, 173)
#   → permute → (B, 173, 256)
#   → Dropout1d(0.2)  [spatial dropout]
#   → 2-layer BiLSTM(256→256 bi) → (B, 173, 512)
#   → self-attention → (B, 512)
#   → Dropout(0.3) → Linear(512, NUM_CLASSES)
#
# Why this matters (DCRF-BiLSTM paper insight):
#   Separate models (mel model, MFCC model, fusion model) each see only a
#   subset of acoustic information. Cross-feature relationships — e.g. high
#   pitch AND rapid ZCR change AND bright chroma together signal "happy" —
#   cannot be learned when features go to separate models. A single model
#   that sees all 267 dimensions simultaneously can learn these conjunctions.
#
#   The DCRF-BiLSTM paper achieves 97.83% on RAVDESS using this strategy.
#
# Run: py run_pipeline.py --step train8
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Re-use SE blocks and SelfAttention from cnn_bilstm_mel
from models.cnn_bilstm_mel import SelfAttention, _cnn_block


class MultiFeatureCNNBiLSTMSER(nn.Module):
    """
    Multi-Feature CNN+BiLSTM+Attention for Speech Emotion Recognition.

    Input: all per-frame features from the batch dict —
        mel              (B, 128, 173)
        mfcc             (B, 40, 173, 3)   → flattened to (B, 120, 173)
        chroma           (B, 12, 173)
        spectral_contrast(B, 7, 173)
        → concatenated → (B, 267, 173)

    Unlike Models 2, 3, and 6 which each see one or two feature types,
    this model processes all features in a single CNN+BiLSTM tower.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------
        # CNN: 5 blocks reduce freq dimension 267→133→66→33→16→8
        #      Time dimension (173) preserved throughout.
        #      SE blocks in each _cnn_block recalibrate channels after each conv.
        #      Channels: 1→32→64→128→256→256
        # ---------------------------------------------------------------
        self.cnn = nn.Sequential(
            _cnn_block(1,   32),   # (B, 32,  133, 173)
            _cnn_block(32,  64),   # (B, 64,   66, 173)
            _cnn_block(64,  128),  # (B, 128,  33, 173)
            _cnn_block(128, 256),  # (B, 256,  16, 173)
            _cnn_block(256, 256),  # (B, 256,   8, 173)
        )

        # Spatial Dropout: zeros entire time steps before BiLSTM.
        # Prevents the model from memorising speaker-specific temporal patterns.
        self.spatial_drop = nn.Dropout1d(p=0.2)

        # BiLSTM: hidden=256 per direction (larger than mel model's 128)
        # per the DCRF-BiLSTM paper recommendation for richer cross-feature sequences.
        # Output: (B, 173, 512) — 256 forward + 256 backward per step
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )

        self.attention = SelfAttention(hidden_dim=512)  # 256 × 2 directions

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        mel    = batch["mel"].to(device)                # (B, 128, T)
        mfcc   = batch["mfcc"].to(device)               # (B, 40, T, 3)
        chroma = batch["chroma"].to(device)             # (B, 12, T)
        sc     = batch["spectral_contrast"].to(device)  # (B, 7, T)

        # MFCC: (B, 40, T, 3) → permute → (B, 3, 40, T) → reshape → (B, 120, T)
        # Stacks raw, delta, delta-delta channels along the feature axis.
        mfcc_flat = mfcc.permute(0, 3, 1, 2).reshape(mfcc.size(0), -1, mfcc.size(2))

        # Concatenate all features along the frequency axis: (B, 267, T)
        x = torch.cat([mel, mfcc_flat, chroma, sc], dim=1)

        x = x.unsqueeze(1)       # (B, 1, 267, T) — single channel for Conv2d
        x = self.cnn(x)          # (B, 256, 8, T)
        x = x.mean(dim=2)        # (B, 256, T) — collapse residual freq bins
        x = x.permute(0, 2, 1)   # (B, T, 256) — time-first for LSTM
        x = self.spatial_drop(x) # zero entire time steps

        x, _ = self.bilstm(x)    # (B, T, 512)
        x    = self.attention(x) # (B, 512)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

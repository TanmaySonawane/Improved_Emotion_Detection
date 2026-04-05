# =============================================================================
# fusion.py — Model 6: MFCC + Chroma Fusion (Parallel CNN+BiLSTM+Attention)
#
# Architecture:
#   Branch A — MFCC stack (B, 40, 173, 3):
#     → permute → (B, 3, 40, 173)
#     → 3 CNN blocks (freq-only pool) → (B, 128, 5, 173)
#     → mean over freq → (B, 128, 173)
#     → permute → (B, 173, 128)
#     → 2-layer BiLSTM(128 → 64 bi) → (B, 173, 128)
#     → self-attention → (B, 128)
#
#   Branch B — Chroma (B, 12, 173):
#     → unsqueeze → (B, 1, 12, 173)
#     → 2 CNN blocks (freq-only pool) → (B, 64, 3, 173)
#     → mean over freq → (B, 64, 173)
#     → permute → (B, 173, 64)
#     → 2-layer BiLSTM(64 → 32 bi) → (B, 173, 64)
#     → self-attention → (B, 64)
#
#   Fusion:
#     cat([128, 64]) = 192 → Dropout(0.3) → Linear(192, NUM_CLASSES)
#
# Why MFCC + Chroma?
#   MFCC captures vocal tract resonance (timbre, loudness dynamics — anger vs calm).
#   Chroma captures harmonic/pitch content (happy speech uses wider pitch range
#   and brighter chroma bins). Together they cover complementary aspects of emotion:
#   MFCC = how the voice sounds, Chroma = what notes are being used.
#
# Run: python Scripts/training/train_fusion.py
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Re-use shared building blocks from cnn_bilstm_mel
from models.cnn_bilstm_mel import SelfAttention, _cnn_block


# ---------------------------------------------------------------------------
# BRANCH A: MFCC
# ---------------------------------------------------------------------------

class MFCCBranch(nn.Module):
    """
    3-channel MFCC stack → CNN → BiLSTM → attention → 128-dim embedding.
    Identical structure to CNNBiLSTMMFCCSER (Model 3).
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            _cnn_block(3,   32),   # (B, 32,  20, 173)
            _cnn_block(32,  64),   # (B, 64,  10, 173)
            _cnn_block(64,  128),  # (B, 128,  5, 173)
        )
        self.bilstm    = nn.LSTM(128, 64, num_layers=2, bidirectional=True,
                                  dropout=0.3, batch_first=True)
        self.attention = SelfAttention(hidden_dim=128)  # 64×2

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """mfcc: (B, 40, 173, 3) → embedding: (B, 128)"""
        x = mfcc.permute(0, 3, 1, 2)   # (B, 3, 40, 173)
        x = self.cnn(x)                  # (B, 128, 5, 173)
        x = x.mean(dim=2)               # (B, 128, 173)
        x = x.permute(0, 2, 1)          # (B, 173, 128)
        x, _ = self.bilstm(x)           # (B, 173, 128)
        return self.attention(x)         # (B, 128)


# ---------------------------------------------------------------------------
# BRANCH B: CHROMA
# ---------------------------------------------------------------------------

class ChromaBranch(nn.Module):
    """
    Single-channel chroma → CNN → BiLSTM → attention → 64-dim embedding.

    Chroma has only 12 frequency bins so we use 2 CNN blocks (not 3 or 4).
    Two MaxPool(2,1) steps: 12 → 6 → 3. Mean-pooling the 3 remaining rows
    gives a clean 64-dim time sequence for BiLSTM.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            _cnn_block(1,  32),   # (B, 32,  6, 173)
            _cnn_block(32, 64),   # (B, 64,  3, 173)
        )
        self.bilstm    = nn.LSTM(64, 32, num_layers=2, bidirectional=True,
                                  dropout=0.3, batch_first=True)
        self.attention = SelfAttention(hidden_dim=64)  # 32×2

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        """chroma: (B, 12, 173) → embedding: (B, 64)"""
        x = chroma.unsqueeze(1)   # (B, 1, 12, 173)
        x = self.cnn(x)            # (B, 64, 3, 173)
        x = x.mean(dim=2)         # (B, 64, 173)
        x = x.permute(0, 2, 1)    # (B, 173, 64)
        x, _ = self.bilstm(x)     # (B, 173, 64)
        return self.attention(x)   # (B, 64)


# ---------------------------------------------------------------------------
# FUSION MODEL
# ---------------------------------------------------------------------------

class FusionSER(nn.Module):
    """
    Model 6: Parallel MFCC + Chroma branches fused for SER.
    """

    def __init__(self):
        super().__init__()
        self.mfcc_branch   = MFCCBranch()    # → (B, 128)
        self.chroma_branch = ChromaBranch()  # → (B, 64)

        # 128 + 64 = 192
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(192, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        mfcc   = batch["mfcc"].to(device)    # (B, 40, 173, 3)
        chroma = batch["chroma"].to(device)  # (B, 12, 173)

        emb_mfcc   = self.mfcc_branch(mfcc)      # (B, 128)
        emb_chroma = self.chroma_branch(chroma)  # (B, 64)

        fused = torch.cat([emb_mfcc, emb_chroma], dim=1)  # (B, 192)
        return self.classifier(fused)                       # (B, NUM_CLASSES)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

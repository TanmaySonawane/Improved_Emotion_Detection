# =============================================================================
# cnn_bilstm_mel.py — Model 2: CNN + BiLSTM + Attention on Mel Spectrogram
#
# Architecture:
#   mel (B, 128, 173)
#   → unsqueeze → (B, 1, 128, 173)
#   → 4 CNN blocks (MaxPool on freq axis only, time preserved) → (B, 256, 8, 173)
#   → mean over freq → (B, 256, 173)
#   → permute → (B, 173, 256)   [173 time steps × 256-dim feature]
#   → 2-layer BiLSTM(256 → 128 bi) → (B, 173, 256)
#   → self-attention → (B, 256)
#   → Dropout(0.3) → Linear(256, NUM_CLASSES)
#
# Why CNN before BiLSTM?
#   Raw mel frames are noisy. CNN blocks learn local frequency–time patterns
#   (formants, onsets) before the BiLSTM captures their temporal evolution.
#
# Why attention?
#   Emotional peaks are short-lived. Attention lets the model focus on the
#   most discriminative time steps rather than uniformly weighting all 173.
#
# Run: python Scripts/training/train_cnn_bilstm.py --variant mel
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# BUILDING BLOCKS
# ---------------------------------------------------------------------------

class _SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.

    Squeeze:  GlobalAvgPool over (H, W) → scalar per channel
    Excite:   FC(C→C//16) → ReLU → FC(C//16→C) → Sigmoid → per-channel scale

    Why this helps: each CNN block produces C feature maps. Not all channels
    are equally informative for every frame. SE learns to upweight channels
    that encode discriminative frequency patterns (e.g. high-energy formants
    for angry, low-energy for neutral) and suppress uninformative ones.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)   # never go below 4 units
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        s = x.mean(dim=[2, 3])          # (B, C) — global avg pool
        s = self.fc(s)                   # (B, C) — channel weights 0-1
        return x * s.unsqueeze(2).unsqueeze(3)  # (B, C, H, W)


def _cnn_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv2d(in→out, 3×3) → BN → ReLU → SE → MaxPool(freq=2, time=1)

    Pooling only on the frequency axis keeps the full 173-frame time
    resolution available for the BiLSTM. SE recalibrates channels after
    each conv so the most emotion-relevant frequency patterns are amplified.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        _SEBlock(out_ch),
        nn.MaxPool2d(kernel_size=(2, 1)),   # halve freq, keep time
    )


class SelfAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over the time dimension.

    Learns a scalar importance score for each of the T=173 time steps,
    then returns a weighted sum of all hidden states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H) — BiLSTM hidden states over T time steps
        Returns:
            context: (B, H) — attention-weighted sum
        """
        scores  = self.proj(torch.tanh(x))      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1) — sum to 1 over time
        context = (weights * x).sum(dim=1)      # (B, H)
        return context


# ---------------------------------------------------------------------------
# MAIN MODEL
# ---------------------------------------------------------------------------

class CNNBiLSTMMelSER(nn.Module):
    """
    CNN + BiLSTM + Attention model for Speech Emotion Recognition.
    Input feature: mel spectrogram (B, N_MELS, N_FRAMES).
    """

    def __init__(self):
        super().__init__()

        # 4 CNN blocks reduce freq: 128 → 64 → 32 → 16 → 8
        # Time dimension (173) is untouched throughout
        self.cnn = nn.Sequential(
            _cnn_block(1,   64),   # (B, 64,  64, 173)
            _cnn_block(64,  128),  # (B, 128, 32, 173)
            _cnn_block(128, 256),  # (B, 256, 16, 173)
            _cnn_block(256, 256),  # (B, 256,  8, 173)
        )

        # Spatial Dropout before BiLSTM: zeros entire time steps (not random neurons).
        # Prevents the model from over-relying on any specific time frame,
        # improving generalization across unseen speakers.
        self.spatial_drop = nn.Dropout1d(p=0.2)

        # 2-layer BiLSTM: 173 time steps, each 256-dim
        # Output: (B, 173, 256) — 128 forward + 128 backward per step
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,       # applied between layers (not after final layer)
            batch_first=True,
        )

        self.attention  = SelfAttention(hidden_dim=256)  # 128×2 directions

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        mel = batch["mel"].to(device)   # (B, 128, 173)
        x   = mel.unsqueeze(1)          # (B, 1, 128, 173)

        x = self.cnn(x)                 # (B, 256, 8, 173)
        x = x.mean(dim=2)              # (B, 256, 173) — collapse freq with mean
        x = x.permute(0, 2, 1)         # (B, 173, 256) — time-first for LSTM
        x = self.spatial_drop(x)       # zero entire time steps

        x, _ = self.bilstm(x)          # (B, 173, 256)
        x    = self.attention(x)        # (B, 256)
        return self.classifier(x)       # (B, NUM_CLASSES)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

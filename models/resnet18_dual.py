# =============================================================================
# resnet18_dual.py — Model 5: ResNet-18 + BiLSTM (Mel Spectrogram)
#
# Architecture (mel-only, matches the configuration that previously achieved 72%):
#
#   mel (B, 128, 173)
#   → trainable 1→3 channel Conv2d
#   → bilinear resize to (3, 224, 224)
#   → ResNet-18 pretrained backbone (layer4 output: (B, 512, 7, 7))
#   → mean over freq axis → (B, 512, 7)
#   → permute → (B, 7, 512)   [7 time steps]
#   → 1-layer BiLSTM(512→128 bi) → final hidden → (B, 256)
#   → Dropout(0.3) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128, NUM_CLASSES)
#
# Why remove the waveform branch?
#   The raw waveform 1D-CNN branch caused repeated gradient explosions during
#   training (val_loss spikes to 2.5+), destabilising the mel branch and
#   producing 53% accuracy — worse than without the branch at all.
#   The BiLSTM on mel features alone matches what achieved 72% previously.
#
# DESIGN: NUM_CLASSES, N_MELS, N_FRAMES read from config — never hardcoded.
# =============================================================================

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# MEL BRANCH: ResNet-18 backbone + BiLSTM temporal modelling
# ---------------------------------------------------------------------------

class MelBranch(nn.Module):
    """
    Extracts a 256-dim emotion embedding from a mel spectrogram.

    Steps:
      1. Expand 1→3 channels with a learned 1×1 conv (9 params).
      2. Resize (128×173) → (224×224) for ResNet-18 expected input.
      3. ResNet-18 pretrained backbone up to layer4: (B, 512, 7, 7).
      4. Mean-pool over freq axis → (B, 512, 7) — 7 temporal steps.
      5. BiLSTM(512→128 bi) captures how features change over time.
      6. Concatenate last forward + backward hidden → (B, 256).
    """

    def __init__(self):
        super().__init__()

        self.chan_expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )  # output: (B, 512, 7, 7) for 224×224 input

        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 128, 173)  →  embedding: (B, 256)"""
        x = mel.unsqueeze(1)                                  # (B, 1, 128, 173)
        x = self.chan_expand(x)                               # (B, 3, 128, 173)
        x = F.interpolate(x, size=(224, 224),
                          mode="bilinear", align_corners=False)  # (B, 3, 224, 224)
        x = self.backbone(x)                                  # (B, 512, 7, 7)
        x = x.mean(dim=2)                                     # (B, 512, 7)  — avg over freq
        x = x.permute(0, 2, 1)                                # (B, 7, 512)  — time-first
        _, (h_n, _) = self.bilstm(x)                          # h_n: (2, B, 128)
        return torch.cat([h_n[0], h_n[1]], dim=1)             # (B, 256)


# ---------------------------------------------------------------------------
# FULL MODEL
# ---------------------------------------------------------------------------

class ResNet18DualSER(nn.Module):
    """
    ResNet-18 + BiLSTM model for Speech Emotion Recognition.
    (Name kept as 'Dual' for backwards compatibility with ensemble checkpoint loading.)
    """

    def __init__(self):
        super().__init__()
        self.mel_branch = MelBranch()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        mel = batch["mel"].to(device)      # (B, 128, 173)
        emb = self.mel_branch(mel)          # (B, 256)
        return self.classifier(emb)         # (B, NUM_CLASSES)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

# =============================================================================
# resnet18_dual.py — Model 5: ResNet-18 Dual-Input (PyTorch, GPU + AMP)
#
# Two parallel input streams:
#   Stream A: Mel spectrogram (image branch via ResNet-18 backbone)
#   Stream B: Raw waveform    (1D CNN branch, downsampled to 8 kHz)
#
# Why dual-input?
#   Mel captures WHAT frequencies are active and HOW they change over time.
#   Raw waveform captures fine-grained temporal details (micro-prosody, rhythm)
#   that are lost when computing spectrograms. Combining both gives the model
#   complementary views of the same emotion signal.
#
# Architecture:
#   Stream A: mel (128,173) → trainable 1→3 channel conv → resize (3,224,224)
#             → ResNet-18 (pretrained, fc removed) → embedding (512,)
#   Stream B: waveform (88200,) → resample to 8kHz (32000,)
#             → 1D CNN stack → embedding (256,)
#   Fusion: cat([A, B]) → (768,) → Linear(768,256) → ReLU → Dropout
#                                → Linear(256, NUM_CLASSES)
#
# Target accuracy: 82–90%
# DESIGN: NUM_CLASSES, N_MELS, N_FRAMES, N_SAMPLES all read from config.
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
# STREAM A: MEL BRANCH
# ---------------------------------------------------------------------------

class MelBranch(nn.Module):
    """
    ResNet-18 backbone that processes the mel spectrogram as an image.

    The mel spectrogram (1 channel, 128×173) is mapped to 3 channels via a
    small trainable Conv2d before being fed into ResNet-18.

    Why trainable 1→3 mapping instead of simply repeating the channel 3 times?
      A learned linear combination gives the network more flexibility to
      "pre-process" the mel before it hits the ResNet. In practice this adds
      only 9 parameters but can improve accuracy vs naive channel repetition.

    Output: embedding vector of shape (512,) per sample.
    """

    def __init__(self):
        super().__init__()

        # 1-channel → 3-channel expansion (9 learnable parameters)
        self.chan_expand = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # ResNet-18 with ImageNet pretrained weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final average pool and fully-connected layer —
        # we'll do our own pooling after to get a 512-dim embedding.
        # base.layer4 → (B, 512, H', W') where H', W' depend on input size.
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # Adaptive average pool to always get (B, 512, 1, 1) regardless of input size
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, N_MELS, N_FRAMES) = (B, 128, 173)

        Returns:
            embedding: (B, 512)
        """
        x = mel.unsqueeze(1)                             # (B, 1, 128, 173)
        x = self.chan_expand(x)                          # (B, 3, 128, 173)
        x = F.interpolate(x, size=(224, 224),
                          mode="bilinear",
                          align_corners=False)           # (B, 3, 224, 224)
        x = self.backbone(x)                             # (B, 512, 7, 7)
        x = self.pool(x)                                 # (B, 512, 1, 1)
        x = torch.flatten(x, 1)                          # (B, 512)
        return x


# ---------------------------------------------------------------------------
# STREAM B: WAVEFORM BRANCH
# ---------------------------------------------------------------------------

class WaveformBranch(nn.Module):
    """
    1D CNN that processes the raw waveform downsampled to 8 kHz.

    Why 8 kHz?
      The original waveform is 88200 samples (4s @ 22050 Hz). Processing that
      directly would require enormous memory. Downsampling to 8 kHz gives 32000
      samples while preserving all speech frequencies (speech is intelligible at
      8 kHz — phone quality). This fits comfortably within 4GB VRAM.

    The downsampling is done in the forward pass using torchaudio's Resample
    (initialized once, reused every call).

    Architecture: 4 × [Conv1d → BatchNorm → ReLU] with stride-based downsampling,
    then AdaptiveAvgPool1d(1) → flatten → 256-dim embedding.
    """

    def __init__(self):
        super().__init__()

        # Resampler: 22050 Hz → 8000 Hz
        # Import here to avoid top-level import failures if torchaudio is missing
        try:
            import torchaudio.transforms as T
            self.resample = T.Resample(
                orig_freq=config.SAMPLE_RATE,
                new_freq=config.RESAMPLE_HZ,
            )
        except ImportError:
            self.resample = None   # fallback handled in forward()

        # 1D CNN stack — output channels grow, sequence length shrinks via stride
        # Input: (B, 1, 32000) after resampling
        self.conv_stack = nn.Sequential(
            # Layer 1: (B, 1, 32000) → (B, 32, ~7985)
            nn.Conv1d(1,   32,  kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Layer 2: (B, 32, ~7985) → (B, 64, ~1990)
            nn.Conv1d(32,  64,  kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Layer 3: (B, 64, ~1990) → (B, 128, ~993)
            nn.Conv1d(64,  128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Layer 4: (B, 128, ~993) → (B, 256, ~494)
            nn.Conv1d(128, 256, kernel_size=8,  stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Pool to fixed size regardless of exact sequence length
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _do_resample(self, x: torch.Tensor) -> torch.Tensor:
        """Resample (B, T) waveform from 22050 Hz to 8000 Hz."""
        if self.resample is not None:
            self.resample = self.resample.to(x.device)
            return self.resample(x)
        else:
            # Fallback: simple linear interpolation if torchaudio is unavailable
            target_len = config.WAVEFORM_DS_SAMPLES  # 32000
            return F.interpolate(
                x.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, N_SAMPLES) = (B, 88200)

        Returns:
            embedding: (B, 256)
        """
        x = self._do_resample(waveform)   # (B, 32000)
        x = x.unsqueeze(1)                 # (B, 1, 32000) — add channel dim for Conv1d
        x = self.conv_stack(x)             # (B, 256, ~494)
        x = self.pool(x)                   # (B, 256, 1)
        x = torch.flatten(x, 1)            # (B, 256)
        return x


# ---------------------------------------------------------------------------
# DUAL-INPUT FUSION MODEL
# ---------------------------------------------------------------------------

class ResNet18DualSER(nn.Module):
    """
    Dual-input ResNet-18 model for Speech Emotion Recognition.

    Mel branch (A) produces a 512-dim embedding.
    Waveform branch (B) produces a 256-dim embedding.
    Concatenated → 768-dim → two FC layers → NUM_CLASSES logits.

    The mel branch carries more weight by design (larger embedding: 512 vs 256)
    consistent with the feature importance insight: mel >> raw waveform.
    """

    def __init__(self):
        super().__init__()
        self.mel_branch  = MelBranch()
        self.wave_branch = WaveformBranch()

        # Fusion head
        # 512 (mel) + 256 (waveform) = 768
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, config.NUM_CLASSES),  # NUM_CLASSES from config, not hardcoded
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch:  dict from SERDataset — must contain "mel" and "waveform"
            device: torch.device

        Returns:
            logits: (batch_size, NUM_CLASSES)
        """
        mel      = batch["mel"].to(device)       # (B, 128, 173)
        waveform = batch["waveform"].to(device)  # (B, 88200)

        emb_mel  = self.mel_branch(mel)           # (B, 512)
        emb_wave = self.wave_branch(waveform)     # (B, 256)

        combined = torch.cat([emb_mel, emb_wave], dim=1)  # (B, 768)
        logits   = self.fusion(combined)                   # (B, NUM_CLASSES)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

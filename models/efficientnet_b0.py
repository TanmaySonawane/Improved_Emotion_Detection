# =============================================================================
# efficientnet_b0.py — Model 4: EfficientNet-B0 (Transfer Learning, PyTorch)
#
# Input: 3-channel "image" built from mel + chroma + spectral_contrast
# Target accuracy: 80–88%
#
# Architecture:
#   Channel 0: Mel spectrogram    (128, 173) — already correct height
#   Channel 1: Chroma STFT        (12, 173)  → bilinear upsample → (128, 173)
#   Channel 2: Spectral contrast  (7, 173)   → bilinear upsample → (128, 173)
#   Stacked → (3, 128, 173) → resized to (3, 224, 224)
#   → EfficientNet-B0 pretrained on ImageNet
#   → Replace final classifier → Linear(1280, NUM_CLASSES)
#
# Two-phase training (controlled by train_efficientnet.py):
#   Phase 1: Freeze backbone, train head only (10 epochs, lr=1e-3)
#   Phase 2: Unfreeze last 3 MBConv blocks, fine-tune (≤40 epochs, lr=1e-4)
#
# DESIGN: NUM_CLASSES always read from config — never hardcoded as 5.
# =============================================================================

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def build_3channel_input(batch: dict, device: torch.device) -> torch.Tensor:
    """
    Build a (batch, 3, 224, 224) input tensor from mel + chroma + spectral_contrast.

    Why these three features?
      - Mel:              the primary "sound picture" — best overall emotion signal
      - Chroma:           which musical notes are active — helps separate happy vs sad
                          (happy speech tends to use brighter, higher pitch classes)
      - Spectral contrast: how "peaky" vs "flat" the spectrum is across bands —
                          angry/fearful voices have sharper harmonic peaks vs neutral

    Why bilinear upsampling for chroma and spectral contrast?
      All 3 channels must have the same spatial size to be stacked.
      Bilinear interpolation smoothly scales the smaller maps up to 128 rows
      without introducing harsh step artifacts.

    Steps:
      1. mel            (B, 128, 173) — no resize needed
      2. chroma         (B, 12,  173) → upsample to (B, 128, 173)
      3. spectral_cont  (B,  7,  173) → upsample to (B, 128, 173)
      4. Stack          → (B, 3, 128, 173)
      5. Resize         → (B, 3, 224, 224) for EfficientNet-B0
    """
    mel    = batch["mel"].to(device)                  # (B, 128, 173)
    chroma = batch["chroma"].to(device)               # (B, 12,  173)
    sc     = batch["spectral_contrast"].to(device)    # (B,  7,  173)

    target_h = config.N_MELS    # 128
    target_t = config.N_FRAMES  # 173

    # Upsample chroma: (B, 12, 173) → (B, 128, 173)
    # unsqueeze(1) adds a fake channel dim so F.interpolate treats (12, 173) as (H, W)
    chroma_up = F.interpolate(
        chroma.unsqueeze(1),
        size=(target_h, target_t),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)                       # (B, 128, 173)

    # Upsample spectral contrast: (B, 7, 173) → (B, 128, 173)
    sc_up = F.interpolate(
        sc.unsqueeze(1),
        size=(target_h, target_t),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)                       # (B, 128, 173)

    # Stack the three channels → (B, 3, 128, 173)
    combined = torch.stack([mel, chroma_up, sc_up], dim=1)

    # Resize to (B, 3, 224, 224) — EfficientNet-B0 was pretrained on 224×224 images
    resized = F.interpolate(
        combined,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    return resized  # (B, 3, 224, 224)


class EfficientNetSER(nn.Module):
    """
    EfficientNet-B0 adapted for Speech Emotion Recognition.

    Why does ImageNet pretraining help for spectrograms?
      EfficientNet's early layers learn to detect edges, textures, and gradient
      patterns. In a mel spectrogram, these correspond to:
        - Edges → onset/offset of sounds, consonant boundaries
        - Textures → harmonic patterns (vowel quality, voice timbre)
        - Gradients → pitch contours, energy changes over time
      These features are directly useful for emotion recognition, so the pretrained
      weights provide a much better starting point than random initialization.
    """

    def __init__(self):
        super().__init__()

        # Load EfficientNet-B0 with ImageNet pretrained weights
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Keep the convolutional feature extractor (9 MBConv blocks)
        # For (B, 3, 224, 224) input this outputs (B, 1280, 7, 7)
        self.features = base.features

        # Keep the adaptive average pooling layer → (B, 1280, 1, 1)
        self.avgpool = base.avgpool

        # Replace the original classifier (1000 ImageNet classes) with our 5-class head
        # NUM_CLASSES is read from config — never hardcoded as 5
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch:  dict from SERDataset — must contain "mel", "chroma", "spectral_contrast"
            device: torch.device (cuda or cpu)

        Returns:
            logits: (batch_size, NUM_CLASSES) — raw scores before softmax
                    The loss function (LabelSmoothingCrossEntropy) applies softmax internally.
        """
        x = build_3channel_input(batch, device)  # (B, 3, 224, 224)
        x = self.features(x)                      # (B, 1280, 7, 7)
        x = self.avgpool(x)                       # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)                   # (B, 1280)
        x = self.classifier(x)                    # (B, NUM_CLASSES)
        return x

    # ------------------------------------------------------------------
    # Phase control methods (called by train_efficientnet.py)
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """
        Phase 1: freeze all backbone weights so only the new classifier head trains.

        Why? The pretrained weights encode valuable ImageNet patterns. If we
        train the whole network from scratch with a small dataset, we'd destroy
        those patterns. Warming up the head first ensures it learns to "speak
        the language" of EfficientNet's features before we adjust the backbone.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print(f"  [EfficientNet] Backbone frozen. "
              f"Trainable params: {self.count_trainable_params():,}")

    def unfreeze_top_blocks(self, n_blocks: int = 3):
        """
        Phase 2: unfreeze the last n_blocks MBConv blocks for fine-tuning.

        EfficientNet-B0's features module has 9 children (MBConv blocks + stem).
        The later blocks learn high-level patterns — these benefit most from
        adapting to spectrogram inputs. Early blocks (edges, textures) transfer
        directly and don't need updating.
        """
        # Freeze everything first
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze the last n_blocks children
        children = list(self.features.children())
        for child in children[-n_blocks:]:
            for param in child.parameters():
                param.requires_grad = True

        # Classifier always stays trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"  [EfficientNet] Unfroze top {n_blocks} MBConv blocks. "
              f"Trainable params: {self.count_trainable_params():,}")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

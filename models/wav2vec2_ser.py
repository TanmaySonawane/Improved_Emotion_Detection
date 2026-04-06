# =============================================================================
# wav2vec2_ser.py — Model 9: Wav2Vec 2.0 Fine-tuned for Speech Emotion Recognition
#
# Uses facebook/wav2vec2-base pretrained on 960 hours of LibriSpeech audio.
# The model has already learned prosody, rhythm, phoneme boundaries, and
# vocal effort from raw waveforms — exactly the cues that distinguish emotions.
#
# Fine-tuning strategy:
#   - Freeze: CNN feature extractor + feature projection (always frozen)
#   - Freeze: bottom 6 of 12 transformer layers
#   - Train:  top 6 transformer layers + classifier head
#
# Why freeze the bottom layers?
#   Lower layers encode basic acoustic features (pitch, formants). These are
#   already well-learned and unlikely to change for emotion tasks. Training
#   them would waste gradients and risk destroying generalizable representations.
#   Top layers encode higher-level sequence patterns — these adapt to the
#   emotion classification objective.
#
# Architecture:
#   waveform (B, 88200) at 22050 Hz
#   → resample to 16000 Hz → (B, 64000)
#   → Wav2Vec2 CNN feature extractor → (B, 200, 512)
#   → Wav2Vec2 transformer (12 layers) → last_hidden_state (B, 200, 768)
#   → mean pool over time → (B, 768)
#   → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, NUM_CLASSES)
#
# VRAM: ~2-3 GB at batch_size=8 with AMP — safe on RTX 3050 Ti 4GB
#
# Requirements: pip install transformers
#
# Run: py run_pipeline.py --step train9
# =============================================================================

import os
import sys

import torch
import torch.nn as nn

try:
    import torchaudio.functional as TAF
except ImportError:
    raise ImportError(
        "torchaudio not installed.\n"
        "Run: pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )

try:
    from transformers import Wav2Vec2Model
except ImportError:
    raise ImportError(
        "transformers not installed.\n"
        "Run: pip install transformers"
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Wav2Vec2-base was trained at 16 kHz. Our audio is stored at 22050 Hz.
_ORIG_FREQ   = config.SAMPLE_RATE   # 22050 Hz
_TARGET_FREQ = 16000                # 16 kHz — wav2vec2-base requirement


class Wav2Vec2SER(nn.Module):
    """
    Wav2Vec 2.0 fine-tuned for 5-class Speech Emotion Recognition.

    The pretrained backbone provides a strong initialization for vocal
    emotion cues. Fine-tuning only the top transformer layers avoids
    catastrophic forgetting of lower-level acoustic representations.
    """

    PRETRAINED = "facebook/wav2vec2-base"

    def __init__(self, freeze_bottom_n_layers: int = 6):
        super().__init__()

        # Load pretrained backbone using safetensors when available.
        # This bypasses the torch 2.6+ strictness for PyTorch checkpoint loading.
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            self.PRETRAINED,
            use_safetensors=True,
        )

        # Always freeze the CNN feature extractor (7 conv layers) and
        # the feature projection that maps 512 → 768 transformer inputs.
        # These are low-level, well-converged — no benefit in updating them.
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False

        # Freeze the bottom N transformer encoder layers (default: 6 of 12).
        # Only the top 6 layers + classifier head will receive gradient updates.
        for i in range(freeze_bottom_n_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False

        # Classifier head on top of mean-pooled hidden states
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for wav2vec2-base
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        waveform = batch["waveform"].to(device)  # (B, 88200) at 22050 Hz

        # Resample to 16 kHz — wav2vec2-base requirement
        waveform = TAF.resample(waveform, orig_freq=_ORIG_FREQ, new_freq=_TARGET_FREQ)
        # Now: (B, 64000)

        # Wav2Vec2 internally normalizes the waveform per-sample.
        # Pass attention_mask=None (no padding — all samples are the same length).
        outputs     = self.wav2vec2(waveform)
        hidden      = outputs.last_hidden_state  # (B, ~200, 768)

        # Mean pool over time frames → fixed-size representation
        pooled = hidden.mean(dim=1)              # (B, 768)

        return self.classifier(pooled)

    def count_params(self) -> tuple:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

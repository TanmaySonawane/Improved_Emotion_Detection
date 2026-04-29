# Speech Emotion Recognition (SER)

Detecting human emotion from speech audio using deep learning — from classical machine learning baselines through fine-tuned transformer models.

**Live demo:** *(HuggingFace Space link (https://huggingface.co/spaces/T22S/ser-emotion-demo))*  
**Test accuracy:** 75.3% (Wav2Vec2) | 5-class | Speaker-independent

---

## Results

| Model | Architecture | Test Accuracy | Macro F1 |
|-------|-------------|:-------------:|:--------:|
| Model 1 | SVM on 260-dim scalar features | 52.6% | 0.517 |
| Model 1 | Random Forest on 260-dim scalar features | 52.0% | 0.513 |
| Model 2 | CNN + BiLSTM + Attention (Mel spectrogram) | 67.4% | 0.673 |
| Model 3 | EfficientNet-B0 (ImageNet pretrained) | 67.5% | 0.673 |
| Model 4 | ResNet-18 (ImageNet pretrained) | 62.1% | 0.619 |
| Model 5 | Multi-Feature CNN + BiLSTM (267-dim joint) | 70.7% | 0.705 |
| **Model 6** | **Wav2Vec2-base fine-tuned (960h speech pretrained)** | **75.3%** | **0.749** |
| Ensemble | SLSQP val-optimized weights + TTA | 75.2% | 0.751 |

---

## Emotion Classes

| Label | Class |
|-------|-------|
| 0 | Angry |
| 1 | Fear |
| 2 | Happy |
| 3 | Neutral |
| 4 | Sad |

---

## Datasets

| Dataset | Actors | Files Used | Notes |
|---------|--------|-----------|-------|
| [RAVDESS](https://zenodo.org/record/1188976) | 24 | ~864 | Studio quality, 8 emotions → 5 kept |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 91 | ~6,171 | Diverse backgrounds, ages, accents |
| **Combined** | **115** | **~7,035** | Speaker-independent 70/15/15 split |

All splits are **speaker-independent** — every file from a given actor stays in exactly one split. The code hard-crashes if any actor appears in two splits.

---

## Architecture Overview

```
Raw Audio (.wav, 22050 Hz, 4 seconds = 88,200 samples)
         │
         ├── Scalar (260-dim) ──────────────────────► SVM / Random Forest
         │
         ├── Mel (128 × 173) ───────────────────────► CNN + BiLSTM + Attention
         │
         ├── Mel + Chroma + Spectral Contrast ──────► EfficientNet-B0 (3-channel)
         │   (resized to 3 × 224 × 224)
         │
         ├── Mel + Waveform ────────────────────────► ResNet-18 Dual-Input
         │
         ├── Mel(128) + MFCC_flat(120)              ► Multi-Feature CNN + BiLSTM
         │   + Chroma(12) + SC(7) = 267 × 173
         │
         └── Raw Waveform (88,200) ─────────────────► Wav2Vec2-base fine-tuned
                  resample to 16kHz inside model
                  CNN extractor → 12 Transformer layers
                                              │
                                    Weighted Ensemble (SLSQP + TTA)
                                              │
                                    5-class emotion prediction
```

---

## Feature Types

| Feature | Shape | Captures |
|---------|-------|---------|
| Mel spectrogram | (128, 173) | Energy at each frequency over time |
| MFCC + Δ + ΔΔ | (40, 173, 3) | Vocal tract shape + rate of change |
| Chroma | (12, 173) | Pitch class (musical note) activity |
| Spectral Contrast | (7, 173) | Tonal vs noisy character per band |
| Waveform | (88,200,) | Raw air pressure samples |
| Scalar | (260,) | Statistical summaries of all above |

---

## Model Descriptions

### Model 2 — CNN + BiLSTM + Attention
- 4 CNN blocks with Squeeze-Excitation channel recalibration, pooling on frequency axis only
- 2-layer Bidirectional LSTM reads the preserved 173-frame time sequence
- Self-attention picks the most emotionally salient time steps
- Spatial Dropout1d prevents memorizing speaker-specific temporal patterns

### Model 4 — EfficientNet-B0
- Mel, chroma, and spectral contrast stacked as a 3-channel image (like RGB)
- Two-phase training: frozen backbone → full fine-tune at 5×10⁻⁵ LR

### Model 5 — ResNet-18 Dual-Input
- Stream A: mel spectrogram → ResNet-18 backbone (ImageNet pretrained)
- Stream B: raw waveform downsampled to 8kHz → 4-layer Conv1d
- Embeddings concatenated and classified jointly

### Model 8 — Multi-Feature CNN + BiLSTM
- All per-frame features (267 dimensions) stacked along the frequency axis
- Single CNN + BiLSTM tower sees cross-feature relationships simultaneously
- Higher BiLSTM hidden size (256) per DCRF-BiLSTM paper recommendation

### Model 9 — Wav2Vec2-base (best model)
- Pretrained on 960 hours of unlabeled speech via self-supervised masking objective
- Bottom 6 of 12 transformer layers frozen; top 6 + classifier fine-tuned
- Raw waveform resampled from 22050 Hz → 16000 Hz inside `forward()`
- Batch size 8 (VRAM constraint), LR 1×10⁻⁴, CosineAnnealingWarmRestarts

---

## Training Techniques

| Technique | Where applied |
|-----------|--------------|
| Offline augmentation 5× (noise, pitch, stretch, volume) | All models |
| Online SpecAugment (time + frequency masking) | All CNN/BiLSTM models |
| Mixup (Beta 0.4) on mel spectrogram | Models 2, 4, 5, 8 |
| Label smoothing (0.1) | All PyTorch models |
| Class-balanced loss weights | All PyTorch models |
| AdamW + weight decay (1×10⁻⁴) | All PyTorch models |
| Cosine Annealing LR | Models 2, 4, 5, 8 |
| CosineAnnealingWarmRestarts | Model 9 |
| Automatic Mixed Precision (AMP) | All PyTorch models |
| Early stopping (patience 12–15) | All PyTorch models |
| SLSQP ensemble weight optimization | Ensemble |
| Test-Time Augmentation (3 views) | Ensemble |

---

## Project Structure

```
Capstone3/Scripts/
├── config.py                    # Single source of truth — all hyperparameters
├── run_pipeline.py              # Master runner: --step or --from
├── app.py                       # Gradio demo app (local + HuggingFace Space)
│
├── data/
│   ├── parse_labels.py          # Filename → (actor_id, label, emotion, dataset)
│   ├── extract_features.py      # Saves 6 .npy files per audio clip
│   ├── build_manifest.py        # Speaker-independent splits → manifest.csv
│   ├── augment.py               # Offline augmentation (train only, 5× expansion)
│   └── dataset.py               # PyTorch Dataset + SpecAugment + DataLoader factory
│
├── models/
│   ├── traditional_ml.py        # Model 1: SVM + Random Forest
│   ├── cnn_bilstm_mel.py        # Model 2: CNN + BiLSTM + Attention
│   ├── efficientnet_b0.py       # Model 4: EfficientNet-B0
│   ├── resnet18_dual.py         # Model 5: ResNet-18 Dual-Input
│   ├── multifeature_cnn_bilstm.py  # Model 8: Multi-Feature CNN + BiLSTM
│   ├── wav2vec2_ser.py          # Model 9: Wav2Vec2-base fine-tuned
│   └── ensemble.py              # Model 7: Weighted Ensemble + SLSQP + TTA
│
├── training/
│   ├── train_utils.py           # AMP loop, label smoothing, early stopping
│   ├── train_traditional.py
│   ├── train_cnn_bilstm.py
│   ├── train_efficientnet.py
│   ├── train_resnet18.py
│   ├── train_multifeature.py
│   ├── train_wav2vec2.py
│   └── train_ensemble.py
│
├── evaluation/
│   └── evaluate.py              # Confusion matrix, classification report, curves
│
└── docs/
    ├── project_overview.md      # Full beginner-friendly project explanation
    ├── path_a_b_explained.md    # Deep dive on Models 8 and 9
    └── features_explained.md    # Feature engineering reference
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/TanmaySonawane/speech-emotion-recognition.git
cd speech-emotion-recognition/Scripts
```

### 2. Install PyTorch with CUDA (GPU required for Wav2Vec2)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies
```bash
pip install "numpy<2.4"
pip install -r requirements.txt
```

### 4. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Running the Pipeline

```bash
# Full pipeline from scratch (10–15 hours)
py run_pipeline.py --all

# Resume from a specific step
py run_pipeline.py --from train2

# Run a single step
py run_pipeline.py --step train9

# Available steps:
# env, extract, manifest, augment, verify
# train1, train2, train4, train5, train8, train9
# ensemble, evaluate
```

---

## Running the Demo App

```bash
python app.py
```

Then open `http://localhost:7860` — upload a `.wav` file or record directly in the browser.

---

## Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3050 Ti (4 GB VRAM) |
| RAM | 30 GB |
| OS | Windows 11 |
| Framework | PyTorch 2.x + CUDA 12.1 |
| Python | 3.10+ |

Individual model training: 1–8 hours. Full pipeline: ~15 hours overnight.

---

## Per-Emotion Performance (Wav2Vec2)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry | 0.83 | 0.89 | 0.86 |
| Neutral | 0.74 | 0.91 | 0.81 |
| Happy | 0.85 | 0.66 | 0.74 |
| Fear | 0.64 | 0.84 | 0.73 |
| **Sad** | 0.77 | **0.50** | 0.61 |

Sad is the hardest class — acoustically near-identical to neutral, separated only by subtle timbral differences that even humans struggle to label consistently.


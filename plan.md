# Capstone3 — Speech Emotion Recognition (SER) Implementation Plan

## Context

This is the 3rd attempt at this SER project. Previous versions had 7 critical bugs (hardcoded shapes, label hardcoding, MFCC not normalized, speaker leakage, no augmentation, CPU-only PyTorch, ignoring mel feature importance). This plan eliminates all of them by design. The goal is to predict 5 emotions (angry=0, fear=1, happy=2, neutral=3, sad=4) from RAVDESS + CREMA-D audio files.

**Environment:** Windows 11, RTX 3050 4GB, 30GB RAM. PyTorch GPU + AMP works. TensorFlow GPU does NOT work → all models use PyTorch.

---

## Dataset Facts

- RAVDESS: 1440 total files → ~864 usable (skip calm/disgust/surprised) across 24 actors
- CREMA-D: 7442 total files → ~6171 usable (skip DIS) across 91 actors
- Grand total: ~7035 files → train grows to ~16,883 rows after 3× augmentation

---

## Folder / File Structure

```
Capstone3/
├── Data/
│   ├── RAVDESS/          (exists: Actor_01 .. Actor_24)
│   ├── CREMA-D/          (exists: flat folder)
│   ├── Features/         (CREATED by Step 1)
│   │   ├── mel/          ← (128,173) float32 .npy
│   │   ├── mfcc/         ← (40,173,3) normalized float32 .npy
│   │   ├── chroma/       ← (12,173) .npy
│   │   ├── spectral_contrast/  ← (7,173) .npy
│   │   ├── waveform/     ← (88200,) .npy
│   │   └── scalar/       ← (262,) .npy
│   ├── Augmented/        (CREATED by Step 3 — train only)
│   │   └── (same subfolders as Features/)
│   └── manifest.csv      (CREATED by Step 2, expanded by Step 3)
│
└── Scripts/
    ├── config.py                  ← SINGLE SOURCE OF TRUTH
    ├── prompt.md
    ├── run_pipeline.py            ← master runner
    │
    ├── data/
    │   ├── __init__.py
    │   ├── parse_labels.py        ← filename → (actor_id, label, emotion, dataset)
    │   ├── extract_features.py    ← saves 6 .npy per audio file
    │   ├── build_manifest.py      ← speaker-independent splits + CSV
    │   ├── augment.py             ← offline augmentation (train only)
    │   └── dataset.py             ← PyTorch Dataset class
    │
    ├── models/
    │   ├── __init__.py
    │   ├── traditional_ml.py      ← Model 1: SVM + RF
    │   ├── cnn_bilstm_mel.py      ← Model 2: CNN+BiLSTM mel (SE + SpecAugment)
    │   ├── cnn_bilstm_mfcc.py     ← Model 3: CNN+BiLSTM MFCC
    │   ├── efficientnet_b0.py     ← Model 4: torchvision EfficientNet-B0
    │   ├── resnet18_dual.py       ← Model 5: ResNet-18 dual input
    │   ├── fusion.py              ← Model 6: MFCC+Chroma fusion
    │   └── ensemble.py            ← Model 7: weighted average
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train_utils.py         ← AMP loop, class weights, label smoothing, early stop
    │   ├── train_traditional.py
    │   ├── train_cnn_bilstm.py    ← shared for models 2 & 3 (--feature flag)
    │   ├── train_efficientnet.py  ← two-phase training
    │   ├── train_resnet18.py
    │   ├── train_fusion.py
    │   └── train_ensemble.py
    │
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluate.py            ← confusion matrix, classification report, curves
    │
    ├── deployment/
    │   ├── __init__.py
    │   ├── preprocess_live.py     ← same pipeline as extract_features, in-memory
    │   └── app.py                 ← Gradio app
    │
    ├── docs/
    │   ├── features_explained.md
    │   ├── augmentation_explained.md
    │   └── models_explained.md
    │
    └── outputs/                   (CREATED at runtime)
        ├── model1_traditional/
        ├── model2_cnn_bilstm_mel/
        ├── model3_cnn_bilstm_mfcc/
        ├── model4_efficientnet/
        ├── model5_resnet18/
        ├── model6_fusion/
        └── model7_ensemble/
```

---

## config.py — Key Settings

```python
# Audio
SAMPLE_RATE   = 22050
DURATION      = 4.0
N_SAMPLES     = 88200          # = SAMPLE_RATE * DURATION
N_MELS        = 128
N_MFCC        = 40
N_CHROMA      = 12
N_SPECTRAL_CONTRAST = 7
HOP_LENGTH    = 512
N_FFT         = 2048
N_FRAMES      = 173            # = ceil(N_SAMPLES / HOP_LENGTH)

# Labels — NEVER hardcode labels elsewhere
EMOTIONS            = {0:"angry", 1:"fear", 2:"happy", 3:"neutral", 4:"sad"}
NUM_CLASSES         = len(EMOTIONS)    # = 5
RAVDESS_EMOTION_MAP = {"01":3, "03":2, "04":4, "05":0, "06":1}
CREMAD_EMOTION_MAP  = {"ANG":0, "FEA":1, "HAP":2, "NEU":3, "SAD":4}

# Paths (absolute, raw strings)
DATA_ROOT     = r"C:\Users\Owner\...\Capstone3\Data"
FEATURES_DIR  = DATA_ROOT + r"\Features"
AUG_DIR       = DATA_ROOT + r"\Augmented"
MANIFEST_PATH = DATA_ROOT + r"\manifest.csv"
OUTPUTS_DIR   = r"C:\Users\Owner\...\Capstone3\Scripts\outputs"

# Splits
SPLIT_SEED    = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO derived as 1 - TRAIN_RATIO - VAL_RATIO

# Training
BATCH_SIZE          = 32
NUM_WORKERS         = 0        # NEVER change — Windows requirement
LEARNING_RATE       = 1e-3
LABEL_SMOOTHING     = 0.1
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS          = 100

# Augmentation
AUG_NOISE_FACTOR    = 0.005
AUG_PITCH_STEPS     = [-1, 1]

# Ensemble weights (mel-based models weighted higher)
ENSEMBLE_WEIGHTS    = {"model2": 0.20, "model4": 0.30, "model5": 0.30, "model3": 0.10, "model6": 0.10}

# Waveform branch downsampling
RESAMPLE_HZ         = 8000
```

---

## How Each of the 7 Bugs Is Fixed

| # | Bug | Fix |
|---|-----|-----|
| 1 | Hardcoded shapes | Every shape in model definitions reads from config (e.g., `config.N_MELS`, `config.NUM_CLASSES`) |
| 2 | Hardcoded labels | Only the dicts in config.py define label mapping. `parse_labels.py` does `config.RAVDESS_EMOTION_MAP.get(code, None)` |
| 3 | MFCC not normalized | Per-row normalization in `extract_features.py` (see below). Saved data has max abs ≤ 10. `build_manifest.py` asserts this |
| 4 | Speaker leakage | `build_manifest.py` shuffles actor IDs then assigns all files from one actor to one split. Post-split assertion checks zero intersection |
| 5 | No augmentation | `augment.py` creates 2 variants per train sample → 3× train size. SpecAugment built into model forward pass (train mode only) |
| 6 | CPU PyTorch | Every training script checks `torch.cuda.is_available()` and warns. AMP used in every PyTorch train loop. `run_pipeline.py` prints GPU name at startup |
| 7 | Ignoring mel importance | Models 4 & 5 use mel as primary input. Ensemble weights mel-based models higher. `features_explained.md` explains why |

---

## Execution Order

```
Step 0  — Environment check
           python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
           python -c "import numpy; print(numpy.__version__)"
           # Fix if needed: pip install "numpy<2.4"

Step 1  — Extract features          (30–60 min, skip-if-exists safe to re-run)
           python Scripts/data/extract_features.py
           → Creates ~42,210 .npy files in Data/Features/

Step 2  — Build manifest            (~2 min)
           python Scripts/data/build_manifest.py
           → Creates Data/manifest.csv (7035 rows)
           → Prints per-split class counts
           → CRASHES if any actor overlap found

Step 3  — Augment train set         (~20–40 min)
           python Scripts/data/augment.py
           → Appends ~9,848 rows to manifest (train only, is_augmented=True)

Step 4  — Train Model 1: Traditional ML    (~5 min)
           python Scripts/training/train_traditional.py

Step 5  — Train Model 4: EfficientNet-B0  (~1–2 hr)
           python Scripts/training/train_efficientnet.py

Step 6  — Train Model 5: ResNet-18        (~1–2 hr)
           python Scripts/training/train_resnet18.py

## Skip steps 7-9 for initial run
Step 7  — Train Model 2: CNN+BiLSTM Mel   (~1–2 hr)
           python Scripts/training/train_cnn_bilstm.py --feature mel

Step 8  — Train Model 3: CNN+BiLSTM MFCC  (~1 hr)
           python Scripts/training/train_cnn_bilstm.py --feature mfcc

Step 9  — Train Model 6: Fusion           (~1 hr)
           python Scripts/training/train_fusion.py

Step 10 — Ensemble                        (~5 min, no training)
           python Scripts/training/train_ensemble.py

Step 11 — Evaluate all models
           python Scripts/evaluation/evaluate.py --all

Step 12 — Deploy
           python Scripts/deployment/app.py
```

---

## Speaker-Independent Split Algorithm

```
1. Collect unique actor IDs per dataset independently
2. Shuffle each list with seed=42
3. RAVDESS (24 actors): train=17, val=3, test=4
   CREMA-D  (91 actors): train=63, val=14, test=14
4. Build actor_id → split lookup dict
5. Assign every file's split by looking up its actor_id
6. POST-SPLIT ASSERTION (hard crash if fails):
   assert set(train_actors) & set(val_actors) == set()
   assert set(train_actors) & set(test_actors) == set()
   assert set(val_actors)   & set(test_actors) == set()
```

Why shuffle each dataset separately? So both RAVDESS and CREMA-D are proportionally represented in all 3 splits, not dominated by CREMA-D's 91 actors.

---

## MFCC Per-Row Normalization

```python
# In extract_features.py
mfcc_raw    = librosa.feature.mfcc(y=y_padded, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512)
mfcc_delta  = librosa.feature.delta(mfcc_raw, order=1)
mfcc_delta2 = librosa.feature.delta(mfcc_raw, order=2)
mfcc_stack  = np.stack([mfcc_raw, mfcc_delta, mfcc_delta2], axis=-1)  # (40, 173, 3)

# Per-row normalization: each of 40 rows in each of 3 channels → mean=0, std=1
for c in range(3):
    for i in range(40):
        row = mfcc_stack[i, :, c]
        mfcc_stack[i, :, c] = (row - row.mean()) / (row.std() + 1e-8)

np.save(path, mfcc_stack.astype(np.float32))
```

Validation: `build_manifest.py` loads a sample mfcc.npy and asserts `np.abs(arr).max() <= 10`.

---

## 262-Dim Scalar Feature Vector

Computed on `y_orig` (unpadded) to avoid padding noise:

| Component | Dim | Notes |
|-----------|-----|-------|
| ZCR mean + std | 2 | on y_orig frames |
| RMS mean + std | 2 | on y_orig frames |
| Spectral centroid mean + std | 2 | |
| Spectral rolloff mean + std | 2 | |
| Spectral bandwidth mean + std | 2 | |
| Spectral flatness mean + std | 2 | |
| Pitch mean + std | 2 | librosa.yin, fmin=50, fmax=500 |
| MFCC means (40 coef) | 40 | from y_orig (NOT normalized) |
| MFCC stds  (40 coef) | 40 | |
| Chroma means (12 notes) | 12 | |
| Chroma stds  (12 notes) | 12 | |
| Spectral contrast means (7 bands) | 7 | |
| Spectral contrast stds  (7 bands) | 7 | |
| Mel means (128 bins) | 128 | compact spectral fingerprint |
| **TOTAL** | **262** | |

For Model 1 (sklearn): standardize with `StandardScaler` fit on train only → save scaler.pkl.

---

## ResNet-18 Dual-Input Architecture

```
Stream A: Mel branch
  mel .npy (128,173) → unsqueeze → (1,128,173)
  → Conv2d(1,3,1)      (trainable 1→3 channel expansion)
  → F.interpolate to (3,224,224)
  → ResNet-18 backbone (pretrained, final avgpool+fc removed)
  → AdaptiveAvgPool2d(1) → Flatten → embedding_A (512,)

Stream B: Waveform branch
  waveform .npy (88200,) → resample to 32000 samples at 8kHz
  → (1, 32000)
  → Conv1d(1→32, k=64, stride=4)  + BN + ReLU
  → Conv1d(32→64, k=32, stride=4) + BN + ReLU
  → Conv1d(64→128, k=16, stride=2) + BN + ReLU
  → Conv1d(128→256, k=8, stride=2) + BN + ReLU
  → AdaptiveAvgPool1d(1) → Flatten → embedding_B (256,)

Fusion:
  cat([A, B]) → (768,)
  → Linear(768, 256) + ReLU + Dropout(0.3)
  → Linear(256, NUM_CLASSES)    ← NUM_CLASSES from config
```

Training: AMP (`autocast` + `GradScaler`), batch_size=32, class weights, label smoothing.

---

## EfficientNet-B0 3-Channel Stacking

```
Channel 0: mel            (128, 173)   — no resize
Channel 1: chroma         (12, 173)    → F.interpolate to (128, 173) bilinear
Channel 2: spectral_cont  (7, 173)     → F.interpolate to (128, 173) bilinear

Stack → (3, 128, 173) → F.interpolate to (3, 224, 224)
→ EfficientNet-B0 (torchvision, pretrained=True)
→ Replace classifier: Linear(1280, NUM_CLASSES)

Phase 1 (10 epochs):  freeze backbone, train head only, lr=1e-3
Phase 2 (≤40 epochs): unfreeze last 3 MBConv blocks, lr=1e-4, ReduceLROnPlateau
```

---

## Augmentation Integration (No Val/Test Leakage)

```
augment.py reads manifest.csv
↓
Filters to split=="train" only
↓
For each original train row:
  aug_A = add noise (AUG_NOISE_FACTOR × max_abs amplitude)
  aug_B = pitch_shift ±1 semitone
  Run full 6-feature extraction on each
  Save to Data/Augmented/{type}/
  New row: same metadata, split="train", is_augmented=True
↓
Concat original manifest + new rows → overwrite manifest.csv

Verification (run anytime):
  assert manifest[manifest.split=="val"]["is_augmented"].sum() == 0
  assert manifest[manifest.split=="test"]["is_augmented"].sum() == 0
```

---

## Training Utilities (train_utils.py)

- `compute_class_weights(labels)` → tensor for `CrossEntropyLoss(weight=...)`
- `LabelSmoothingCrossEntropy(smoothing=0.1)` → custom nn.Module
- `train_one_epoch(model, loader, optimizer, scaler, criterion, device)` → AMP loop
- `validate(model, loader, criterion, device)` → returns val loss + accuracy
- `EarlyStopping(patience=10)` → saves best checkpoint, triggers stop

---

## Deployment (Gradio app.py)

- Input: audio upload or microphone (auto-resampled to 22050 Hz)
- Calls `preprocess_live.py` (same pipeline as extract_features, in-memory)
- Loads best checkpoint (ensemble preferred)
- If `softmax_confidence < 0.5`: "Uncertain — please try again"
- Else: emotion label + confidence bar chart

---

## Documentation Files

- **features_explained.md** — mel, MFCC, chroma, spectral contrast explained for a beginner
- **augmentation_explained.md** — explain to a 10-year-old with audio analogies; include "how do I know it works?" section (train vs val accuracy gap)
- **models_explained.md** — high-level: what each model does, why transfer learning works for spectrograms, what ensemble means

---

## Critical Files to Create (in order)

1. `Scripts/config.py`
2. `Scripts/data/__init__.py`
3. `Scripts/data/parse_labels.py`
4. `Scripts/data/extract_features.py`
5. `Scripts/data/build_manifest.py`
6. `Scripts/data/augment.py`
7. `Scripts/data/dataset.py`
8. `Scripts/models/__init__.py`
9. `Scripts/models/traditional_ml.py`
10. `Scripts/training/__init__.py`
11. `Scripts/training/train_utils.py`
12. `Scripts/training/train_traditional.py`
13. `Scripts/models/efficientnet_b0.py`
14. `Scripts/training/train_efficientnet.py`
15. `Scripts/models/resnet18_dual.py`
16. `Scripts/training/train_resnet18.py`
17. `Scripts/models/cnn_bilstm_mel.py`
18. `Scripts/models/cnn_bilstm_mfcc.py`
19. `Scripts/training/train_cnn_bilstm.py`
20. `Scripts/models/fusion.py`
21. `Scripts/training/train_fusion.py`
22. `Scripts/models/ensemble.py`
23. `Scripts/training/train_ensemble.py`
24. `Scripts/evaluation/__init__.py`
25. `Scripts/evaluation/evaluate.py`
26. `Scripts/deployment/__init__.py`
27. `Scripts/deployment/preprocess_live.py`
28. `Scripts/deployment/app.py`
29. `Scripts/run_pipeline.py`
30. `Scripts/docs/features_explained.md`
31. `Scripts/docs/augmentation_explained.md`
32. `Scripts/docs/models_explained.md`

---

## Verification Tests

After Steps 1–3, run these checks before training:

```python
import pandas as pd, numpy as np
m = pd.read_csv(MANIFEST_PATH)

# No actor leakage
train_a = set(m[m.split=="train"]["actor_id"])
val_a   = set(m[m.split=="val"]["actor_id"])
test_a  = set(m[m.split=="test"]["actor_id"])
assert train_a & val_a == set() and train_a & test_a == set() and val_a & test_a == set()

# No augmented data in val/test
assert m[m.split!="train"]["is_augmented"].sum() == 0

# MFCC normalization check
sample_mfcc = np.load(m.iloc[0]["mfcc_path"])
assert np.abs(sample_mfcc).max() <= 10.0

# Feature shapes
assert np.load(m.iloc[0]["mel_path"]).shape == (128, 173)
assert np.load(m.iloc[0]["mfcc_path"]).shape == (40, 173, 3)
assert np.load(m.iloc[0]["chroma_path"]).shape == (12, 173)
assert np.load(m.iloc[0]["spectral_contrast_path"]).shape == (7, 173)
assert np.load(m.iloc[0]["waveform_path"]).shape == (88200,)
assert np.load(m.iloc[0]["scalar_path"]).shape == (262,)

print("All verification checks passed.")
```

---

## Target Accuracy Summary

| Model | Architecture | Target |
|-------|-------------|--------|
| 1 | SVM + RF (scalar 262-dim) | 50–55% |
| 2 | CNN+BiLSTM Mel + SE + SpecAugment | 78–85% |
| 3 | CNN+BiLSTM MFCC stack | 60–70% |
| 4 | EfficientNet-B0 (mel+chroma+spectral) | 80–88% |
| 5 | ResNet-18 dual (mel + waveform) | 82–90% |
| 6 | MFCC+Chroma Fusion | 75–85% |
| 7 | Weighted Ensemble (best models) | 90%+ |

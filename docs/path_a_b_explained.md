# Path A and Path B — Architecture Explanation

## Why Two New Models?

After Session 4, the ensemble reached **70.06%** test accuracy. The remaining gap to
75%+ cannot be closed by tuning the existing models further — they've hit their
architectural ceiling. The two new models take fundamentally different approaches
that literature confirms can push past 75%.

---

## Path B — Model 8: Multi-Feature CNN+BiLSTM+Attention

**File:** `models/multifeature_cnn_bilstm.py`
**Training:** `training/train_multifeature.py`
**Output:** `outputs/model8_multifeature/`

### The problem with separate models
Models 2–6 each see only a subset of acoustic features:
- Model 2: mel only
- Model 3: MFCC only
When the model for "angry" detection only sees mel, it misses the joint signal:
**high pitch (mel) AND fast rate-of-change (MFCC delta) AND dissonant chroma** =
strong "angry" indicator. No individual model can learn this cross-feature pattern.

### What Model 8 does differently
All per-frame features are stacked along the frequency axis before the CNN sees them:

```
mel              (128 bins  × 173 frames)
MFCC+Δ+ΔΔ flat  (120 values × 173 frames)   ← 40 coeff × 3 channels, reshaped
Chroma           ( 12 notes × 173 frames)
Spectral Contrast(  7 bands × 173 frames)
─────────────────────────────────────────
Combined         (267 dims  × 173 frames)   → treated as a single "spectrogram"
```

The CNN+BiLSTM then processes this 267-dim input jointly:
- **5 CNN blocks** (vs 4 for mel model) reduce 267 → 8, keeping 173 time steps intact
- **SE channel attention** in each block learns which of the 267 features matter most
- **Spatial Dropout1d(0.2)** zeros entire time steps before BiLSTM
- **BiLSTM(hidden=256 bi)** — larger than mel model's 128 per the DCRF paper
- **Self-attention** focuses on emotionally-peaked moments
- **Total input to BiLSTM:** 256 channels → 512-dim output after bidirectional pass

### Evidence it works
The DCRF-BiLSTM paper (2023) achieves 97.83% on RAVDESS by feeding combined
features. Our version is smaller (no dense CRF, VRAM budget), but the core idea —
joint feature learning — is proven.

**Expected standalone accuracy:** 73–76%
**Expected VRAM usage:** ~2 GB (safe on RTX 3050 Ti)
**Expected training time:** ~1–2 hours at batch_size=32

---

## Path A — Model 9: Wav2Vec 2.0 Fine-tuned for SER

**File:** `models/wav2vec2_ser.py`
**Training:** `training/train_wav2vec2.py`
**Output:** `outputs/model9_wav2vec2/`
**Requires:** `pip install transformers`

### Why pretrained models change everything
Models 2–8 are trained from scratch on ~7,000 samples (augmented to ~24,000).
That's not much data for learning acoustic emotion.

`facebook/wav2vec2-base` was pretrained on **960 hours of LibriSpeech audio** using
a contrastive self-supervised objective. In the process, it learned:
- Prosody (speech rhythm and stress patterns)
- Phoneme boundaries
- Vocal effort and breathiness
- Pitch and energy dynamics

These are exactly the cues humans use to recognize emotion. The model didn't learn
them for emotion recognition — it learned them as general acoustic representations.
Fine-tuning then redirects these representations toward our 5-class task.

### Architecture

```
waveform (88200 samples @ 22050 Hz)
↓ resample to 16000 Hz → 64000 samples
↓ CNN feature extractor (7 conv layers, stride 320 total)
  → ~200 frames × 512-dim features
↓ Feature projection: 512 → 768
↓ Transformer encoder (12 layers × 768-dim)
  FROZEN: layers 0–5    (low-level: pitch, formants — already good)
  TRAINED: layers 6–11  (high-level: sequences, prosody patterns)
↓ last_hidden_state: (B, 200, 768)
↓ mean pool over 200 frames → (B, 768)
↓ Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 5)
```

Total parameters: ~94 million (wav2vec2) + small head
Trainable: ~45 million (top 6 layers + head ≈ 48% of params)

### Why freeze the bottom 6 layers?
Lower transformer layers encode basic acoustic structure (like a fixed feature
extractor). Training them on 7,000 samples would overfit these generic representations
to our small dataset. The top 6 layers encode higher-level sequential patterns that
need to shift from "speech understanding" to "emotion understanding."

### Key training differences from other models

| Setting | Other models | Wav2Vec2 (Model 9) | Why |
|---------|-------------|-------------------|-----|
| Batch size | 32 | **8** | 94M params need more VRAM |
| Learning rate | 1e-3 | **1e-4** | Pretrained weights — small updates only |
| Mixup | Enabled | **Disabled** | Waveform mixup not implemented; mel mixup unused |
| Epochs | 80–100 | **40** | Fine-tuning converges faster |
| Scheduler | CosineAnnealingLR | **CosineAnnealingWarmRestarts** | Ensures full LR cycle |

### Why resample to 16 kHz?
Wav2Vec2-base was pretrained on audio sampled at 16,000 Hz. Our audio is stored at
22,050 Hz (librosa's default). Feeding 22 kHz audio to a 16 kHz model would
misalign the learned temporal patterns. The resampling happens on-the-fly in the
model's `forward()` method — no new files needed.

### First-run download
On first run, `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")` downloads
the model weights (~360 MB) from HuggingFace. This requires internet access.
Subsequent runs use the local cache (~/.cache/huggingface/).

**Expected standalone accuracy:** 78–85%
**Expected VRAM usage:** ~2.5–3 GB at batch_size=8 with AMP
**Expected training time:** ~2–3 hours (larger model, smaller batches)

---

## How the Ensemble Combines All Models

After both models are trained, run:
```
py run_pipeline.py --step ensemble
```

The ensemble uses `scipy.optimize.minimize` (SLSQP) to find weights that maximize
validation accuracy. Initial weights in `config.ENSEMBLE_WEIGHTS`:

| Model | Initial weight | Why |
|-------|---------------|-----|
| model1_traditional | 0.05 | Weak baseline |
| model2_cnn_bilstm_mel | 0.15 | Good mel model |
| model4_efficientnet | 0.15 | Good transfer learning |
| model5_resnet18 | 0.20 | Mel + temporal |
| model8_multifeature | 0.20 | All features jointly |
| model9_wav2vec2 | **0.30** | Highest — pretrained on audio |

The SLSQP optimizer will revise these based on what actually works on the val set.
Wav2Vec2 will likely end up with even higher weight if it performs as expected.

**Expected ensemble accuracy: 78–82%** (from 70.06% baseline)

---

## Run Order for Overnight Training

```bash
# Step 1: Install transformers (one-time)
pip install transformers

# Step 2: Run Tier 1 fixes (Models 2, 3, 5) + new models + ensemble
py run_pipeline.py --from train2
```

`--from train2` will execute: train2 → train3 → train4 → train5 → train6 → train8 → train9 → ensemble → evaluate

Estimated total time: **6–10 hours** depending on GPU speed.

Models 8 and 9 add approximately 3–5 hours on top of the existing model retraining.

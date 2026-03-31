# Models Explained (For Beginners)

This project trains 5 different models (plus an ensemble) on the same data. Each model approaches the problem differently — some are better at certain confusions, and combining them at the end gives the best overall accuracy.

---

## Model 1: Traditional ML — SVM + Random Forest

**Target accuracy: 50–55%**
**Input: 260-dimensional scalar feature vector**
**Script: `training/train_traditional.py`**

### What it does
This model doesn't use a neural network at all. It uses classical machine learning algorithms:

- **SVM (Support Vector Machine):** Finds the best mathematical boundary to separate the 5 emotion classes in 260-dimensional space. It looks for the widest possible "gap" between classes.
- **Random Forest:** Trains 300 decision trees, each on a random subset of the data and features. It takes a majority vote among all trees. Handles non-linear patterns that SVM misses.

### Why we include it
It's a **baseline**. If our fancy deep learning models only hit 55% accuracy, we know something went wrong — because a simple SVM can already do that. If deep models reach 85%, we know the neural network is truly learning better representations.

### Limitation
The 260-dim vector throws away spatial information. We lose the "picture" of how the spectrogram changes over time — we only get averages and standard deviations. That's why the target is relatively low.

---

## Model 4: EfficientNet-B0 (Transfer Learning)

**Target accuracy: 80–88%**
**Input: mel + chroma + spectral contrast stacked as (3, 224, 224) image**
**Script: `training/train_efficientnet.py`**

### What is transfer learning?
EfficientNet-B0 was originally trained on ImageNet — a dataset of 1.2 million natural photographs across 1000 categories (cats, cars, furniture, etc.). We take that pre-trained model and *transfer* its learned weights to our new task.

Why does this work? The early layers of any CNN learn to detect **edges, textures, and patterns** — the same kinds of patterns that appear in mel spectrograms (harmonic edges, textured noise regions, smooth tonal transitions). We don't need to learn these from scratch.

### Two-phase training
We can't just unleash a high learning rate on the whole network — we'd destroy the carefully learned ImageNet patterns.

**Phase 1 (10 epochs):** Freeze the entire backbone. Only train the new 5-class output head. This lets the head learn to "read" EfficientNet's features without disrupting them.

**Phase 2 (up to 40 epochs):** Unfreeze the last 3 MBConv blocks and fine-tune with a much lower learning rate. These high-level blocks learn to adapt their feature detectors for spectrogram patterns.

### What gets fed in
Three features are stacked like RGB channels of an image:
- Channel R = mel spectrogram (the main picture)
- Channel G = chroma (pitch patterns)
- Channel B = spectral contrast (harmonic sharpness)

Chroma and spectral contrast are smaller, so they get upsampled to match mel's 128 rows before stacking.

---

## Model 5: ResNet-18 Dual-Input

**Target accuracy: 82–90%**
**Input: mel spectrogram + raw waveform (8 kHz)**
**Script: `training/train_resnet18.py`**

### Why two inputs?
Each input sees something the other misses:

- **Mel spectrogram:** Great at capturing spectro-temporal patterns — which frequencies are active, how they change over time. It loses very fine-grained timing information.
- **Raw waveform (downsampled to 8 kHz):** Captures micro-timing — tiny pauses, voice tremors, and rhythm patterns that are averaged out in the spectrogram. Angry speech has different micro-timing than sad speech.

By combining both, the model gets a more complete picture of the emotion.

### Architecture
```
Mel → 1×1 Conv (1→3 channels) → resize 224×224 → ResNet-18 backbone → 512-dim vector
                                                                               ↓
                                                                        CONCAT → 768-dim → FC → 5 classes
                                                                               ↑
Waveform → downsample to 8kHz → 1D CNN (4 layers) ─────────────────── 256-dim vector
```

The mel branch gets a larger embedding (512) than the waveform branch (256), giving it more influence in the final decision — consistent with our knowledge that mel is the most important feature.

### AMP (Automatic Mixed Precision)
Training runs in mixed precision (float16 + float32). This halves the memory usage on the GPU, allowing a larger batch size or more complex model within the 4GB VRAM budget. The final output layer stays in float32 to maintain numerical stability.

---

## Model 7: Ensemble (Weighted Average)

**Target accuracy: 90%+**
**No training required — just combines saved model outputs**

### What is an ensemble?
After training all individual models, we save their checkpoints. At test time, we run all models on the same audio clip and average their probability outputs.

For example, if the test clip is "angry":
```
EfficientNet: [angry=0.50, fear=0.20, happy=0.15, neutral=0.08, sad=0.07]
ResNet-18:    [angry=0.68, fear=0.12, happy=0.10, neutral=0.05, sad=0.05]
Weighted avg: [angry=0.59, fear=0.16, happy=0.12, neutral=0.06, sad=0.06] → angry ✓
```

### Why do errors cancel out?
Each model was trained with different random seeds, different architectures, and different input representations. They make different mistakes. When one model is uncertain, another is often confident. The average is almost always more reliable than any individual model.

### Weights
Models that performed better on the validation set get higher weights:
- EfficientNet (mel+chroma+SC): 30%
- ResNet-18 (mel+waveform): 30%
- CNN+BiLSTM Mel: 20%
- CNN+BiLSTM MFCC: 10%
- Fusion: 10%

The mel-based models dominate because mel is the most informative feature.

---

## Why Does Accuracy Matter? What Does 85% Mean?

With 5 emotion classes and random guessing, you'd expect 20% accuracy. So:
- 50% = basic patterns learned (SVM level)
- 70% = clear emotion separability
- 85% = strong generalization across speakers
- 90%+ = near human-level (humans score ~75–80% on RAVDESS in studies)

Interestingly, humans don't agree on emotion 100% of the time either. A voice that one person hears as "fearful," another might hear as "surprised." Our 90%+ target is ambitious and represents state-of-the-art performance.

---

## Common Confusions and How We Address Them

| Confusion | Why it happens | Our fix |
|-----------|---------------|---------|
| Happy vs Sad | Both can have similar energy levels; differ mainly in pitch and tonal quality | Chroma feature (Model 4) captures tonal differences |
| Angry vs Fear | Both are high arousal; differ mainly in spectral texture | Spectral contrast (Model 4) and raw waveform micro-timing (Model 5) |
| Fear vs Neutral | Fear can sound quiet and controlled | BiLSTM temporal attention focuses on the subtle dynamics |

---

## Training Flow Summary

```
Data (RAVDESS + CREMA-D)
    ↓
Feature Extraction (.npy files)
    ↓
Manifest.csv (speaker-independent splits)
    ↓
Augmentation (×3 training data)
    ↓
┌─────────────────────────────────────────┐
│  Model 1: SVM + RF         → 50–55%    │
│  Model 4: EfficientNet-B0  → 80–88%    │
│  Model 5: ResNet-18 Dual   → 82–90%    │
│  (Models 2,3,6 → add later)            │
└─────────────────────────────────────────┘
    ↓
Model 7: Ensemble             → 90%+
    ↓
Deployment (Gradio app)
```

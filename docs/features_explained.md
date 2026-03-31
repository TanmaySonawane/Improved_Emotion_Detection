# Audio Features Explained (For Beginners)

This document explains every feature type used in the project — what it is, why we compute it, and what its shape means.

---

## The Big Picture

When you speak, your voice creates sound waves. A computer can't "understand" a raw audio file very well, so we convert it into different **visual representations** — like taking multiple X-rays of the same thing from different angles. Each representation highlights different aspects of emotion.

Think of it this way:
- A **happy** voice tends to be higher-pitched, faster, and brighter in tone.
- A **sad** voice tends to be slower, quieter, and lower in pitch.
- An **angry** voice tends to be loud, fast, and harsh in the high frequencies.

Our features capture these differences mathematically.

---

## Step 0: Preprocessing (Before Any Features)

**Sample rate: 22050 Hz**
This means we store 22,050 numbers per second of audio — each number is a tiny measurement of air pressure at that instant.

**Duration: 4.0 seconds → 88,200 samples**
All clips are trimmed or padded to exactly 4 seconds. If a recording is shorter, we add silence symmetrically on both sides ("center padding"). If it's longer, we take the center 4 seconds.

Why 4 seconds? It's long enough to capture a full emotional utterance but short enough to be memory-efficient.

---

## Feature 1: Mel Spectrogram `(128, 173)` ← PRIMARY

**File: `*_mel.npy`**

### What is it?
A spectrogram is a "picture of sound" — the horizontal axis is time, the vertical axis is frequency (how high or low a sound is), and the brightness shows how loud each frequency is at each moment.

A **mel spectrogram** uses the **mel scale** instead of regular (linear) frequency. The mel scale mimics how humans actually hear: we're more sensitive to differences in low frequencies than high ones. For example, we can easily tell 100 Hz from 200 Hz, but 8000 Hz vs 9000 Hz sounds nearly the same to us.

### Shape: (128, 173)
- **128** = number of frequency bins (evenly spaced on the mel scale from ~0 to 11025 Hz)
- **173** = number of time frames (each frame covers ~23ms; 4 seconds / ~23ms ≈ 173)

### Why it's PRIMARY
Mel spectrograms preserve the full frequency information of the voice. Every harmonic, every vowel quality, every pitch contour is visible. In experiments, models trained on mel spectrograms consistently outperform those trained on any other single feature for speech emotion recognition.

---

## Feature 2: MFCC Stack `(40, 173, 3)`

**File: `*_mfcc.npy`**

### What is MFCC?
MFCC stands for **Mel-Frequency Cepstral Coefficients**. It's a compact "summary" of the voice's timbral qualities — kind of like describing a photo by its most important colors rather than storing every pixel.

The MFCC is computed by:
1. Taking the mel spectrogram
2. Taking the log of the power values
3. Applying the Discrete Cosine Transform (DCT) — this compresses the information

The result is 40 numbers per time frame that capture the overall "shape" of the spectrum without the pitch detail.

### Why the "stack" (3 channels)?
We use 3 versions stacked together — like stacking 3 related images:
- **Channel 0**: MFCC — the snapshot
- **Channel 1**: Delta MFCC — how the MFCC is *changing* (like velocity)
- **Channel 2**: Delta-Delta MFCC — how the *change* is changing (like acceleration)

Together they capture not just what the voice sounds like, but how it's moving over time.

### Critical fix: Per-Row Normalization
Raw MFCC values are wildly different in scale — coefficient 0 might be -400 while coefficient 39 is only -5. This 80× difference would confuse neural networks. We normalize each row independently to mean=0, std=1 so all 40 coefficients contribute equally to learning.

---

## Feature 3: Chroma STFT `(12, 173)`

**File: `*_chroma.npy`**

### What is it?
Chroma represents the 12 pitch classes in Western music (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). For each time frame, it shows how much energy is present at each of the 12 semitones of an octave — regardless of which octave.

Think of it as "what notes are being sung/spoken, ignoring whether they're high or low."

### Why it helps with happy/sad confusion
Happy speech tends to use higher, more varied pitch classes (brighter, more major-key-like patterns). Sad speech tends to stay on lower, narrower pitch ranges (darker, more minor-key-like). Chroma captures this tonal character in a compact 12-row representation.

---

## Feature 4: Spectral Contrast `(7, 173)`

**File: `*_spectral_contrast.npy`**

### What is it?
Spectral contrast measures the difference between the peaks and valleys of the spectrum within each of 7 frequency bands. A high contrast value means the voice has sharp, clear harmonic peaks (like a vowel sound). A low contrast value means the energy is spread evenly (like a whisper or fricative consonant like "s").

### Why it helps with angry/fear vs neutral
Angry and fearful voices tend to have very high contrast — strong, tense harmonic structure. Neutral speech is calmer and more "flat" spectrally. This feature directly encodes that distinction.

---

## Feature 5: Raw Waveform `(88200,)`

**File: `*_waveform.npy`**

### What is it?
The raw center-padded audio signal — 88,200 numbers representing the amplitude of the sound wave at each of the 88,200 time points.

### How it's used
Only the ResNet-18 model (Model 5) uses the raw waveform. It first downsamples it to 8 kHz (32,000 samples), then processes it with a 1D CNN. The raw waveform captures very fine-grained temporal details — micro-rhythm and pitch micro-variations — that spectrograms blur over.

---

## Feature 6: Scalar Vector `(260,)`

**File: `*_scalar.npy`**

### What is it?
A single fixed-length vector of 260 numbers summarizing the entire audio clip statistically. These are computed on the **original unpadded** signal to avoid contamination from the silence we added.

### Why unpadded?
If we computed ZCR (zero crossing rate) on a padded signal, the silent padding regions would artificially inflate the ZCR values with zero crossings at the pad boundaries. Computing on the original gives honest statistics.

### Components (in order):
| Component | Dimension | What it measures |
|-----------|-----------|-----------------|
| ZCR mean + std | 2 | How often the waveform crosses zero — noisy/fricative sounds cross more |
| RMS mean + std | 2 | Root-mean-square energy — how loud the voice is |
| Spectral centroid mean + std | 2 | The "center of mass" of the spectrum — bright voices are higher |
| Spectral rolloff mean + std | 2 | Frequency below which 85% of energy lies |
| Spectral bandwidth mean + std | 2 | How wide the spectrum is |
| Spectral flatness mean + std | 2 | How "noisy" vs "tonal" the voice is |
| Pitch mean + std | 2 | Fundamental frequency (F0) |
| MFCC means × 40 | 40 | Average shape of each MFCC coefficient |
| MFCC stds × 40 | 40 | Variability of each MFCC coefficient |
| Chroma means × 12 | 12 | Average energy per pitch class |
| Chroma stds × 12 | 12 | Variability per pitch class |
| Spectral contrast means × 7 | 7 | Average peak/valley contrast per band |
| Spectral contrast stds × 7 | 7 | Variability of contrast |
| Mel bin means × 128 | 128 | Average energy per mel frequency band |
| **TOTAL** | **260** | |

This vector is the input to Model 1 (SVM + Random Forest).

---

## Feature Summary Table

| Feature | Shape | Primary model(s) | What emotion aspect it captures |
|---------|-------|-----------------|--------------------------------|
| Mel | (128, 173) | Models 2, 4, 5 | Full spectral-temporal picture |
| MFCC stack | (40, 173, 3) | Models 3, 6 | Timbral texture + dynamics |
| Chroma | (12, 173) | Models 4, 6 | Pitch class patterns (happy vs sad) |
| Spectral contrast | (7, 173) | Models 4, 6 | Harmonic sharpness (angry vs neutral) |
| Waveform | (88200,) | Model 5 | Fine-grained temporal detail |
| Scalar | (260,) | Model 1 | Statistical summary for traditional ML |

---

## Why mel > MFCC?

The MFCC is computed *from* the mel spectrogram by applying a DCT compression step. That compression throws away information. The mel spectrogram retains everything — every harmonic, every pitch detail. The DCT was invented in the 1980s to make features computationally feasible for speech recognition on slow hardware. Modern deep learning doesn't need that compression — it can handle the full mel spectrogram directly and learns which parts matter.

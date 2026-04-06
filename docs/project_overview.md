# Speech Emotion Recognition — Full Project Overview
### A Complete Walkthrough for Anyone Starting from Zero

---

## 1. The Problem: Can a Computer Hear How You Feel?

When a person says "I'm fine," those two words can mean completely different things
depending on *how* they are said. Said slowly with a flat voice — it sounds sad.
Said quickly with a clipped tone — it sounds annoyed. Said brightly with a rising
pitch — it actually means fine. The emotion is carried not in the words, but in the
**sound** of the voice: pitch, energy, rhythm, and how quickly things change.

**Speech Emotion Recognition (SER)** is the task of building a computer system that
listens to a voice and predicts the speaker's emotional state automatically.

### Why does this matter?

| Application | What it could do |
|-------------|-----------------|
| Mental health monitoring | Detect early signs of depression by tracking vocal patterns over weeks |
| Customer service | Flag frustrated callers automatically before they hang up |
| Driver safety | Alert a drowsy or agitated driver before an accident |
| Voice assistants | Allow Siri/Alexa to respond with appropriate tone based on your mood |
| Education technology | Adapt a tutoring app's difficulty when it detects student frustration |

This is an active research area with real commercial applications. It is also a
genuinely hard problem — even humans only agree on the "correct" emotion label
for the same audio clip about 70–80% of the time, because emotions are continuous
and blurry, not discrete boxes.

---

## 2. The Datasets: Where the Audio Comes From

A machine learning project needs labeled examples — audio files where someone has
already marked what emotion the speaker was expressing. We use two public datasets.

### RAVDESS — Ryerson Audio-Visual Database
24 professional actors (12 male, 12 female) recorded scripted sentences under
controlled studio conditions. Each sentence was delivered in 8 different emotional
styles. We use 5 of them:

> **angry · fearful · happy · neutral · sad**

Studio-quality, consistent, clean — a good controlled benchmark. ~864 usable files.

### CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset
91 actors from diverse backgrounds (different ages, ethnicities, accents) recorded
the same 6 sentences in 6 emotional styles. We use 5 emotions. ~6,171 usable files.

More diverse and realistic than RAVDESS. Combined with RAVDESS: **~7,000 audio files.**

### The Most Important Design Decision: Speaker-Independent Splits

We divide the data into three groups:
- **Train** (70%) — what the model learns from
- **Validation** (15%) — what we use to tune the model during development
- **Test** (15%) — the final exam, touched only once at the very end

The critical rule: **every file from a given actor stays in one group only.**

If actor 7's voice appears in both training and test data, the model can recognize
that voice and associate it with emotions it saw during training — without actually
understanding the emotion at all. This is called **actor leakage**, and it inflates
accuracy scores by 10–20 percentage points in published papers that don't control
for it. Our code crashes with an error message if any actor appears in two splits.

---

## 3. What Is Sound, Digitally?

Before any machine learning, it helps to understand what audio actually looks like
to a computer.

When a microphone records sound, it measures air pressure thousands of times per
second. Our audio is sampled at **22,050 Hz** — 22,050 pressure measurements per
second. A 4-second clip is therefore a list of 88,200 numbers, each between −1 and
+1, representing the waveform.

This raw waveform is hard to learn from directly. The numbers change extremely fast
and the connection between individual sample values and emotion is not obvious. We
need to transform the audio into representations that make emotional patterns more
visible.

---

## 4. Feature Engineering: Turning Sound Into Something Learnable

### What is a Spectrogram?

A spectrogram answers the question: **"at each moment in time, how loud is each
frequency?"**

Imagine you are at a concert. At any given second, the bass guitar is loud at low
frequencies, the violin is loud at high frequencies, and the singer is loud somewhere
in the middle. A spectrogram is a 2D picture of this: time runs left to right,
frequency runs bottom to top, and brightness indicates loudness.

Emotionally, spectrograms are meaningful:
- Angry speech is loud and bright across many frequencies
- Sad speech is quiet and concentrated in low frequencies
- Happy speech shows energy moving rapidly across frequencies

### The Mel Scale

Human hearing is not linear. We are much more sensitive to frequency differences
at low frequencies than at high. A difference of 200 Hz is very noticeable near
500 Hz, but nearly inaudible near 5000 Hz.

The **mel scale** is a warped frequency axis that matches human perception. A mel
spectrogram has 128 rows (mel frequency bands, low to high) and 173 columns (time
frames, each ~23ms). This is our primary input feature for most neural models.

### MFCC — Mel Frequency Cepstral Coefficients

MFCCs compress the mel spectrogram into 40 numbers per time frame. Think of them
as a compact "fingerprint" of the vocal tract shape at each moment — they describe
*how the mouth and throat are shaped* to produce the current sound.

We compute not just the raw MFCC but also:
- **Delta (Δ)**: the rate of change — how fast is the vocal tract shape changing?
- **Delta-delta (ΔΔ)**: the acceleration — is that rate of change speeding up or slowing down?

Fearful speech changes rapidly; sad speech changes slowly. The delta channels
explicitly encode this dynamic information. Result: a 40 × 173 × 3 array.

### Chroma

A 12 × 173 map representing which of the 12 musical pitch classes (C, C#, D, ..., B)
are active over time. Happy speech tends to have clear harmonic structure; distressed
speech tends to have more noise. Captures a different dimension than MFCC.

### Spectral Contrast

Measures the difference between spectral peaks and valleys across 7 frequency bands.
High contrast = clear, tonal sound. Low contrast = noisy, breathy, or whispered.
A 7 × 173 map.

### Center Padding

All clips are padded or cropped to exactly 4 seconds (88,200 samples). We use
**center padding** — the real speech sits in the middle, and silence is added
symmetrically at both ends. This matters because one-sided padding (adding silence
only at the end) would make the model see silence in a consistent location, which
could be exploited as a spurious feature.

### The Traditional ML Feature Vector (260 numbers)

For the classical machine learning approach, all of the above is compressed into
a single **260-dimensional vector** per clip: statistics (mean, standard deviation)
of ZCR, RMS energy, spectral centroid, spectral rolloff, spectral bandwidth,
spectral flatness, pitch, plus MFCC means/stds, chroma means/stds, spectral
contrast means/stds, and mel band means.

---

## 5. First Approach: Traditional Machine Learning

With 260-dimensional vectors, we can use classical algorithms that do not require
special hardware or days of training.

### SVM — Support Vector Machine

An SVM tries to find the flat surface (hyperplane) in 260-dimensional space that
best separates, say, "angry" from "not angry." With multiple classes it does this
pairwise. The trick that makes SVMs powerful is the **kernel** — it can implicitly
map the data to an even higher-dimensional space where the classes become linearly
separable, without ever actually computing those high-dimensional coordinates.

### Random Forest

A Random Forest builds hundreds of decision trees, each trained on a random subset
of the data and a random subset of the 260 features. Each tree votes on the emotion.
The majority vote is the final prediction. Because each tree sees different data and
features, the errors they make tend to be different, and averaging them out reduces
mistakes.

### Result: 52.6% accuracy (SVM) / 52.0% (Random Forest)

Much better than random (20% for 5 classes). But limited because these 260 numbers
collapse the entire 4-second clip into a single row. The model cannot see *when*
a pitch spike happened, or how fast the energy ramped up. All temporal dynamics are
thrown away.

---

## 6. Data Augmentation: Making the Dataset Bigger

~7,000 samples is small for deep learning. **Data augmentation** creates modified
copies of training data that preserve the emotion but sound slightly different,
exposing the model to more variety.

**Critically: augmentation is applied only to training data, never to validation or
test.** If we tested on augmented data, we'd be testing on easy variants of training
samples, not genuinely unseen audio.

### Offline Augmentation (saved to disk, applied before training)

For each training clip we generate 4 variants:

**1. Gaussian Noise**
Adds a tiny amount of random hiss (0.5% of peak amplitude) to the audio. This
simulates real-world recording conditions — fan noise, distant traffic, microphone
self-noise. Forces the model to listen to emotion, not audio cleanliness.

**2. Pitch Shift (±1 semitone)**
Shifts the audio up or down by one semitone without changing the speed. Emotion is
carried by *relative* pitch contour — whether your pitch is rising or falling
relative to your own baseline — not by absolute Hz values. A high-pitched person
and a low-pitched person can both sound angry. Pitch shifting during training stops
the model from memorising "high Hz = happy" and instead learns the pattern.

**3. Time Stretch (0.9× or 1.1× speed)**
Slows down or speeds up the audio without changing pitch. Different people talk at
different rates. Excited people talk faster; sad people talk slower. But we don't
want the model to learn "fast = happy" — we want it to learn the pattern regardless
of pace. Time stretching trains robustness to speaking rate.

**4. Volume Perturbation (0.7× or 1.3× gain)**
Makes the clip quieter or louder. Microphone distance, room size, and individual
vocal power all affect amplitude. This teaches the model to focus on the shape of
energy changes, not the absolute level.

Result: 5× training data (original + 4 variants = 5 copies per clip).

### Online Augmentation: SpecAugment (applied during each training batch)

SpecAugment modifies the mel spectrogram on-the-fly during training, differently
each time:

**Time Masking**: randomly zero out a block of up to 30 consecutive time frames.
The model must predict the emotion with a chunk of time hidden. This simulates
interruptions and teaches the model to use the rest of the clip, not just one moment.

**Frequency Masking**: randomly zero out a block of mel frequency bins. Forces the
model to not over-rely on a single frequency range. If all high-frequency bins are
hidden, the model must still infer emotion from the low-frequency content.

Because SpecAugment is random and different each epoch, the model effectively sees
a new version of each sample on every training pass. This is one of the most
effective regularization techniques for audio.

---

## 7. What Is a Neural Network?

A neural network is a function that maps an input (e.g., a 128×173 mel spectrogram)
to an output (e.g., probabilities for 5 emotion classes). It is made of layers, each
of which performs a mathematical transformation. The network has millions of
adjustable numbers called **weights**. Training means showing it labeled examples
and adjusting the weights to reduce mistakes.

The core training loop:
1. Show the model a batch of audio clips with known labels
2. The model predicts emotion probabilities
3. Compute a **loss** — how wrong the predictions are
4. **Backpropagation**: compute how much each weight contributed to the error
5. **Gradient descent**: nudge each weight slightly in the direction that reduces error
6. Repeat millions of times

### What Is a Convolution?

A convolution slides a small filter (say, 3×3) over a 2D input. At each position,
it multiplies the filter by the local patch and sums the result. The output is a
new 2D map where each pixel represents how strongly that filter pattern was present
at that location.

In images, a 3×3 filter can detect vertical edges, horizontal edges, corners, etc.
In mel spectrograms, filters detect local frequency-time patterns: a formant
transition, an energy onset, a vibrato oscillation.

A **Convolutional Neural Network (CNN)** stacks many such layers. The first layer
detects simple patterns (edges, onsets). Later layers combine those into more complex
patterns (formant contours, harmonic structures). This hierarchical feature learning
is why CNNs work so well on 2D data.

### What Is Pooling?

After convolution, pooling reduces the spatial size. **Max pooling** takes the
maximum value in each small region. This makes the representation more compact and
translation-invariant — it doesn't matter if a feature appears slightly to the left
or right.

In our models, we pool only on the **frequency axis** and leave the time axis
untouched. This is deliberate: we want to preserve the full 173-time-step sequence
for the LSTM to read.

### What Is Batch Normalization?

After each convolution, BatchNorm normalizes the output of that layer across the
batch so it has mean≈0 and std≈1. This prevents one layer's outputs from growing
unboundedly large and destabilizing the layers that follow. It also acts as a mild
regularizer.

---

## 8. Models Trained from Scratch

### Model 2: CNN + BiLSTM + Attention on Mel Spectrogram — 67.4% test accuracy

This is the core architecture, purpose-built for temporal audio data.

**Stage 1 — CNN:** 4 convolutional blocks process the 128×173 mel spectrogram.
Each block: Conv2D → BatchNorm → ReLU → SE Block → MaxPool(freq only).
The mel goes from 128 rows down to 8 rows while remaining 173 frames wide.

**SE (Squeeze-Excitation) Block:** After each convolution, we have many feature maps
(channels). Not all are equally useful. The SE block does a tiny two-layer network
that looks at the average of each channel and outputs a weight between 0 and 1 for
each. Channels encoding emotion-relevant patterns get upweighted; uninformative
channels get suppressed. Adds less than 1% extra parameters but measurably improves
accuracy.

**Stage 2 — BiLSTM:** An LSTM (Long Short-Term Memory) is a type of recurrent
network designed for sequences. At each time step, it receives the current input and
a "memory" from the previous step, and outputs a hidden state. It learns to remember
what matters and forget what doesn't over the sequence.

A standard LSTM reads left-to-right. A **Bidirectional LSTM (BiLSTM)** also reads
right-to-left simultaneously. The two directions are concatenated. This matters
because the emotional resolution of a sentence (how it ends) helps interpret the
ambiguous beginning. Both directions inform each other.

Two LSTM layers are stacked — the second layer sees the hidden states from the first
as its input, allowing it to model patterns of patterns (higher-order temporal
structure).

**Stage 3 — Self-Attention:** The BiLSTM produces a hidden state at each of the 173
time steps. To get a single fixed-size representation, we don't simply average
everything — that treats a moment of emotional peak the same as a silent pause.

Self-attention learns a scalar importance score for each time step and returns a
weighted sum. The model can focus on the most discriminative moment — a sudden pitch
spike, a drawn-out vowel, a sharp onset.

**Spatial Dropout:** Before the BiLSTM, instead of standard dropout (which randomly
zeroes individual neurons), we use Dropout1d which zeroes entire time steps. This
forces the model to not over-rely on any specific moment, and directly combats
memorizing speaker-specific temporal patterns that don't generalize.

**Final output:** Dropout → Linear layer → 5 class probabilities (via softmax).

### Model 8: Multi-Feature CNN+BiLSTM (Path B) — 70.7% test accuracy
All per-frame features are stacked along the frequency axis before the CNN:
mel (128) + MFCC flat (120) + chroma (12) + spectral contrast (7) = 267 dims per frame.
A single larger CNN+BiLSTM processes all 267 dimensions jointly.

This matters because emotions often show up as combinations across feature types.
High pitch AND rapid MFCC delta AND bright chroma together strongly suggest "happy" —
but you can only detect that joint pattern if all three enter the same model at once.

### The Training Tricks That Make It Work

**Label Smoothing:** Instead of training the model to output exactly [0, 0, 1, 0, 0]
for "happy," we train it to output [0.02, 0.02, 0.92, 0.02, 0.02]. This prevents the
model from becoming over-confident and improves its calibration on real-world audio
where the correct label is not always crisp.

**Class Weights:** Some emotions are rarer in the dataset than others. Class weights
give a higher penalty to mistakes on rare classes, preventing the model from
ignoring them to optimize for the common ones.

**AdamW Optimizer:** The optimizer is the algorithm that adjusts weights based on
gradients. AdamW adds weight decay — a small penalty for having large weights — which
acts as regularization and helps prevent overfitting.

**Cosine Annealing:** The learning rate controls how large each weight update step is.
Too large and training is unstable; too small and it converges too slowly. Cosine
annealing starts at a moderately high rate and smoothly decays following a cosine
curve, reaching a very small value by the final epoch. This drives the model into a
sharper, more generalizable minimum.

**Gradient Clipping:** Occasionally during training, gradients become very large
("exploding gradients") and a single update step corrupts the weights. We clip
all gradients to a maximum norm of 1.0, preventing any single step from doing damage.

**AMP (Automatic Mixed Precision):** Modern GPUs have special hardware for half-
precision (float16) arithmetic. AMP automatically runs certain operations in float16
(2× memory, 2× throughput) and others in float32 (full precision where needed).
This lets us fit larger models in GPU memory and train faster.

**Early Stopping:** We monitor validation loss after each epoch. If it doesn't
improve for 15 consecutive epochs, training stops automatically and the best
checkpoint is restored. This prevents wasting time on epochs that only overfit.

---

## 9. Pretrained Models: Standing on the Shoulders of Giants

Training from scratch means the model must learn everything from ~7,000 samples.
That's not much data. Pretrained models have already learned general representations
from massive datasets — we then redirect those representations toward our specific task.

### What Is Transfer Learning?

Transfer learning is the idea that knowledge from one task is useful for another.
A model that has learned to recognize objects in millions of photos has learned
how to detect edges, textures, and shapes. These skills transfer to analyzing
spectrograms (which are also 2D images with local structure).

The process:
1. Take a large pretrained model
2. Replace the final layer with a new classifier for our 5 emotions
3. Fine-tune: train on our data, either keeping earlier layers frozen or updating everything

### Model 4: EfficientNet-B0 (pretrained on ImageNet) — 67.5% test accuracy

EfficientNet is a CNN architecture that was trained on ImageNet — 14 million labeled
images across 1,000 categories. We take the backbone (everything except the final
classifier) and attach a new classifier for 5 emotions.

Input: mel + chroma + spectral contrast stacked as a 3-channel "image" (like RGB),
resized to 224×224 for EfficientNet's expected input size.

Training uses two phases:
- **Phase 1**: freeze the backbone, train only the new classifier. This lets the
  classifier learn something useful before we start moving the pretrained weights.
- **Phase 2**: unfreeze all layers, train end-to-end at a much smaller learning rate
  (5×10⁻⁵ vs 1×10⁻³). Small updates preserve the pretrained representations while
  gradually steering them toward emotion recognition.

### Model 5: ResNet-18 + BiLSTM (pretrained on ImageNet) — 62.1% test accuracy

ResNet-18 is another ImageNet-pretrained CNN. Rather than using it purely as a
feature extractor, we attach a BiLSTM to its intermediate feature maps. ResNet
extracts spatial patterns; the BiLSTM then models how those patterns evolve over time.
This hybrid combines the spatial understanding of a vision model with the temporal
modeling of a sequence model.

### Model 9: Wav2Vec 2.0 — The Big Step (pretrained on 960 hours of speech) — 75.3% test accuracy

This model represents a fundamental shift from image-domain pretraining to
**audio-domain pretraining**.

`facebook/wav2vec2-base` is a **transformer** model — a different architecture from
CNNs, based on attention mechanisms that can model relationships between any two
positions in a sequence regardless of distance. It was trained on 960 hours of
unlabeled speech using a self-supervised objective: predict the correct audio
token from a context of surrounding frames, with some frames masked out. This is
analogous to BERT in NLP, which predicts masked words from context.

**Why does this matter for emotion?**
To predict a masked audio frame, the model had to learn prosody (speech rhythm),
phoneme boundaries, vocal effort, and intonation patterns — all the acoustic
properties that humans use to recognize emotion. It learned these representations
without ever seeing a single emotion label, just from learning the structure of
speech itself. Our fine-tuning then redirects these representations toward the
5-class emotion task.

**Architecture walkthrough:**
```
raw waveform (88,200 samples at 22,050 Hz)
  ↓ resample to 16,000 Hz (wav2vec2 was pretrained at this rate)
  ↓ CNN feature extractor (7 conv layers) — reduces to ~200 frames
  ↓ linear projection → 768-dimensional vectors
  ↓ 12 transformer layers (each with multi-head self-attention)
      layers 0–5:  FROZEN — low-level acoustic structure, already good
      layers 6–11: TRAINED — high-level temporal/prosodic patterns
  ↓ mean pool across 200 frames → one 768-dim vector per clip
  ↓ Linear(768 → 256) → ReLU → Dropout → Linear(256 → 5)
```

**What is a Transformer / Multi-Head Attention?**
At every layer, each of the 200 time positions looks at all other positions and
decides which ones are most relevant to understanding itself. This is the attention
mechanism: position 100 might find that position 10 (a stressed syllable at the start)
is highly informative for interpreting what's happening now. Unlike LSTM which passes
information through a sequential chain, transformers make all these connections
simultaneously, which makes them powerful at capturing long-range dependencies.

Multi-head attention does this with several parallel attention "heads," each looking
for different types of relationships. The results are combined.

**Training differences from other models:**
- Batch size 8 (not 32) — 94 million parameters need more GPU memory
- Learning rate 1×10⁻⁴ (10× lower) — pretrained weights need small nudges, not big rewrites
- 40 epochs max (pretrained models converge faster)
- Mixup disabled (mixup in this pipeline mixes mel spectrograms; this model reads raw waveforms)

---

## 10. The Ensemble: Combining All Models

No single model is best at everything. The ensemble takes every trained model,
runs all of them on the same test sample, and combines their predictions.

Each model outputs a probability vector of length 5 (one probability per emotion,
summing to 1.0). The ensemble takes a weighted average of all these vectors. The
final prediction is the emotion with the highest combined probability.

**How are the weights chosen?**
We use mathematical optimization (`scipy.optimize.minimize` with the SLSQP method)
to find the weights that maximize accuracy on the **validation set** — the held-out
data that was never used for training. This is done before ever touching the test set.
The SLSQP solver adjusts 8 weights (one per model) subject to the constraint that
they all be ≥ 0 and sum to 1.0.

**Test-Time Augmentation (TTA):**
For each test sample, we run every PyTorch model three times:
1. Original mel spectrogram
2. Center time-masked version (15 frames zeroed)
3. Center frequency-masked version (10 bins zeroed)

The three probability outputs are averaged before combining with the ensemble
weights. This is free — no retraining — and adds about 1–2% accuracy by reducing
the variance in individual predictions.

### Accuracy Progression

| Stage | Test Accuracy | What was added |
|-------|--------------|----------------|
| Traditional ML (SVM) | 52.6% | 260-dim scalar features |
| Traditional ML (Random Forest) | 52.0% | 260-dim scalar features |
| CNN+BiLSTM (Mel) — Model 2 | 67.4% | Mel spectrogram + SE + SpecAugment |
| EfficientNet-B0 — Model 4 | 67.5% | ImageNet pretraining + 3-channel mel stack |
| ResNet-18 Dual — Model 5 | 62.1% | ResNet + raw waveform branch |
| Multi-Feature CNN+BiLSTM — Model 8 | 70.7% | All features jointly (mel+MFCC+chroma+SC) |
| Wav2Vec2 Fine-tuned — Model 9 | **75.3%** | 960h speech pretraining |
| **Weighted Ensemble (all models)** | **75.2%** | SLSQP val-optimized weights + TTA |

---

## 11. Regularization: How We Fight Overfitting

**Overfitting** happens when a model memorises the training data instead of learning
general patterns. It scores very high on training data and much lower on unseen test
data. This is the central challenge in any small-dataset ML project.

Techniques used in this project to combat overfitting:

| Technique | What it does |
|-----------|-------------|
| Speaker-independent splits | Ensures the model never heard the test speakers during training |
| Data augmentation (5×) | More training variety, harder to memorize |
| SpecAugment | Randomly hides parts of spectrograms during training |
| Dropout | Randomly zeros a fraction of neuron outputs during training, preventing co-dependence |
| Spatial Dropout (Dropout1d) | Zeros entire time steps — directly prevents memorizing speaker-specific temporal patterns |
| Weight decay (AdamW) | Penalizes large weights, keeping the model "simple" |
| Label smoothing | Prevents over-confidence |
| Early stopping | Stops when val loss stops improving, saving the best checkpoint |

---

## 12. Why Is This Hard? (Honest Expectations)

Even humans only agree on emotion labels about 70–80% of the time for the same
recording, because emotion is subjective and continuous. Some recordings are
genuinely ambiguous — fear and excitement share many acoustic properties (both
involve raised pitch, increased energy, faster rate). The datasets themselves have
label noise.

Our results are on **acted speech** (professional actors performing emotions under
studio conditions), which is the easiest possible scenario. Real conversational speech
is much harder — emotions in natural conversation are subtle, mixed, and context-
dependent. A system trained here achieving 75% represents strong academic-level
performance, but would likely score 50–65% on real-world conversational audio.

The techniques used throughout — speaker-independent training, extensive augmentation,
pretrained models, ensemble methods — are exactly what state-of-the-art deployed
commercial systems use, scaled up to millions of hours of data and hundreds of actor
identities.

### Per-Emotion Difficulty — What the Results Reveal

The classification reports from our best model (Wav2Vec2, 75.3% overall) show that
not all emotions are equally difficult to predict:

| Emotion | Precision | Recall | F1 | Why |
|---------|-----------|--------|----|-----|
| **Angry** | 0.83 | 0.89 | 0.86 | Easiest — high energy and wide spectral spread make it acoustically distinct from all others |
| **Neutral** | 0.74 | 0.91 | 0.81 | High recall because the model defaults to neutral when uncertain; it is the "resting" baseline |
| **Happy** | 0.85 | 0.66 | 0.74 | High precision but low recall — the model is confident when it says happy, but misses many happy clips |
| **Fear** | 0.64 | 0.84 | 0.73 | Low precision — the model over-predicts fear, confusing it with angry (both have high pitch and energy) |
| **Sad** | 0.77 | 0.50 | 0.61 | Hardest — consistently the lowest recall across every model in the project |

**Why is sad so difficult?**
Sad speech is acoustically quiet, slow, and low-energy — very similar to neutral on all measurable axes.
The difference is subtle: slightly lower pitch, slightly slower rate, slightly less spectral brightness.
These are exactly the kinds of fine-grained differences that require long-range context to detect,
which is why Wav2Vec2 (with 12 transformer layers able to see the full utterance at once) still
only achieves 50% recall on sad — even though it is the best model overall.

**Why does fear have low precision?**
Fear and angry share the same gross acoustic signature: high pitch, fast rate, high energy. What
separates them is the *quality* of that energy — angry speech has a harsh, clipped quality while
fearful speech is more breathy and tense. These timbral differences are subtle and frequently
confused. The model errs on the side of predicting fear when it sees high-energy clips that do
not match the cleaner angry pattern.

**The happy miss rate** (only 66% recall despite 85% precision) reflects a different problem:
happy clips in acted speech vary enormously. Some actors perform "happy" as calm and warm
(acoustically close to neutral); others perform it as excited and energetic (acoustically close
to fear or angry). The model captures the excited subset well but misses the subdued subset.

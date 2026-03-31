# Data Augmentation Explained

---

## For a 10-Year-Old: The Photo Analogy

Imagine you're teaching a computer to recognize cats. You only have 10 photos of cats. That's not enough — the computer might memorize those exact 10 photos and fail when it sees a new cat.

What do you do? You make *fake* new photos from the ones you already have:
- Flip some photos left/right
- Make some darker
- Rotate some a little

Now you have 30 photos from the same 10 cats — but they all look slightly different. The computer has to learn what a *cat* really looks like, not just memorize those 10 exact images.

**We do the same thing with voices.**

We have recordings of people saying things with different emotions. We apply two simple tricks to each training recording to create extra versions. The emotion is still the same — we just changed some things that don't affect emotion.

---

## Trick 1: Adding Noise (Variant A)

**What we do:** Add a tiny amount of random static to the voice, like a slightly bad phone connection.

**In code:**
```python
noise = random values × 0.5% of the loudest moment in the clip
augmented_voice = original_voice + noise
```

**Why it doesn't hurt:** The noise is very small — about 0.5% of the loudest point. You wouldn't notice it with human ears. The emotion (anger, sadness, etc.) is completely unchanged.

**What it teaches the model:** "Don't depend on the voice being perfectly clean. Real recordings always have *some* background noise."

---

## Trick 2: Pitch Shift (Variant B)

**What we do:** Shift the voice slightly higher (+1 semitone) or lower (-1 semitone) in pitch. One semitone is the smallest interval in music — like pressing one key to the right on a piano. It's barely noticeable.

**In code:**
```python
augmented_voice = librosa.effects.pitch_shift(original_voice, sr=22050, n_steps=+1)
```

**Why it doesn't hurt:** The words, rhythm, and emotional expression are all preserved. Only the absolute pitch changes slightly — like a person speaking slightly higher or lower than their usual voice.

**What it teaches the model:** "Don't recognize 'angry' only at one specific pitch. Angry sounds angry whether the person is a high-pitched speaker or a low-pitched speaker."

---

## The Result: 3× Training Data

For every 1 original training clip, we create 2 new variants:
- Original
- Noisy version
- Pitch-shifted version

This triples the training set from ~4,900 clips to ~14,700 clips — all with accurate emotion labels.

**Important:** We ONLY augment the training split. The validation and test splits keep the original recordings only. This ensures we're testing on real data, not fake variations.

---

## Trick 3: SpecAugment (Online, Happens During Training)

This happens *while* training, not before. During each training batch, we randomly "cover up" parts of the mel spectrogram.

**Time masking:** Pick a random slice of time frames and set them all to zero.

```
Before:  ████████████████████████████  (full spectrogram)
After:   █████████░░░░░░░░████████████  (3 seconds blacked out in the middle)
```

**Frequency masking:** Pick a random range of mel frequency bins and set them to zero.

```
Before:  ████  (full frequency range)
         ████
         ████
After:   ████
         ░░░░  (these frequencies hidden)
         ████
```

**Why this helps:** The model is forced to recognize emotions *even when part of the sound is missing*. It can't "cheat" by relying on one specific frequency band or time segment. It has to understand the emotion from the whole picture.

---

## How Do You Know Augmentation Is Working?

Compare your training accuracy vs your validation accuracy:

| Situation | What it means |
|-----------|--------------|
| Train=90%, Val=60% | **Overfitting** — model memorized training data. Need more augmentation or regularization. |
| Train=80%, Val=78% | **Good** — model generalizes well. Augmentation is working. |
| Train=65%, Val=64% | **Underfitting** — model isn't learning enough. Need more epochs or a better model. |

The goal is for training and validation accuracy to be **close to each other** (within ~5–10%). If they're very far apart, the model is memorizing instead of learning.

You can also look at the training curves plot (`outputs/model*/training_curves.png`). If the validation loss stops improving while training loss keeps dropping, that's overfitting.

---

## What We Do NOT Augment

- **Time stretching** — intentionally skipped. Stretching audio changes the temporal rhythm, which is an important emotion cue. Angry speech is fast; sad speech is slow. Stretching would corrupt this signal.
- **Volume normalization** — skipped. RMS energy (loudness) is a meaningful emotion feature. Normalizing would hide that angry voices are typically louder.
- **Adding reverb/echo** — skipped. This would change the spectral texture in unpredictable ways that don't correspond to real recording conditions in our datasets.

The two augmentations we *do* use (noise + pitch shift) are the safest and most well-established choices for speech emotion recognition.

---

## Summary

| Augmentation | When | Effect on data | Emotion preserved? |
|-------------|------|---------------|-------------------|
| Gaussian noise | Offline (before training) | +1 variant per clip | Yes |
| Pitch shift ±1 semitone | Offline (before training) | +1 variant per clip | Yes |
| Time masking (SpecAugment) | Online (during training) | Random per batch | Yes |
| Frequency masking (SpecAugment) | Online (during training) | Random per batch | Yes |

Net result: **3× training data** (offline) + **random masking** (online) = much better generalization.

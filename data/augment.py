# =============================================================================
# augment.py
#
# Offline data augmentation — TRAIN SPLIT ONLY.
#
# For each original train sample, generates 2 augmented variants:
#   Variant A: Gaussian noise addition
#   Variant B: Pitch shift (±1 semitone)
#
# This triples the training set size (original + 2 variants = 3×).
# Val and test splits are NEVER touched.
#
# CRITICAL RULES:
#   - Only reads rows where split == "train" AND is_augmented == False
#   - Augmented rows are appended to manifest.csv with split="train", is_augmented=True
#   - Safe to re-run: skips files that already have all 6 .npy outputs
#
# Run: python Scripts/data/augment.py
# =============================================================================

import os
import sys
import time

import numpy as np
import librosa
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.extract_features import process_file, get_feature_paths, all_features_exist


# ---------------------------------------------------------------------------
# AUGMENTATION FUNCTIONS
# They operate on y_orig (the original unpadded audio).
# Padding happens inside process_file (same as regular extraction).
# ---------------------------------------------------------------------------

def augment_noise(y_orig: np.ndarray) -> np.ndarray:
    """
    Add Gaussian noise scaled to AUG_NOISE_FACTOR × max absolute amplitude.

    This simulates background noise (e.g., a slightly bad microphone or
    ambient room noise) while keeping the emotion clearly audible.
    """
    amplitude = np.abs(y_orig).max()
    if amplitude == 0:
        return y_orig  # silence — nothing to do
    noise = np.random.normal(0, config.AUG_NOISE_FACTOR * amplitude, len(y_orig))
    return (y_orig + noise).astype(np.float32)


def augment_pitch(y_orig: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Shift the pitch by n_steps semitones.

    n_steps=+1 makes the voice slightly higher (like inhaling helium — very slightly).
    n_steps=-1 makes the voice slightly lower.
    The emotion content is preserved; only the pitch shifts.
    """
    return librosa.effects.pitch_shift(y_orig, sr=config.SAMPLE_RATE, n_steps=float(n_steps))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def augment_train_split():
    """
    Read the current manifest, augment all original train rows, and
    append the new rows back to manifest.csv.
    """
    if not os.path.isfile(config.MANIFEST_PATH):
        raise FileNotFoundError(
            f"Manifest not found: {config.MANIFEST_PATH}\n"
            "Run data/build_manifest.py first."
        )

    df = pd.read_csv(config.MANIFEST_PATH)
    print(f"Loaded manifest: {len(df)} rows")
    print(f"  train: {(df.split=='train').sum()}, "
          f"val: {(df.split=='val').sum()}, "
          f"test: {(df.split=='test').sum()}")

    # Only augment original train rows (not already-augmented rows)
    orig_train = df[(df["split"] == "train") & (df["is_augmented"] == False)].copy()
    print(f"\nOriginal train rows to augment: {len(orig_train)}")

    new_rows    = []
    processed   = 0
    cached      = 0
    errors      = 0
    t_start     = time.time()

    for i, (_, row) in enumerate(orig_train.iterrows()):
        wav_path = row["wav_path"]

        # Load original audio once for both variants
        try:
            y_orig, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        except Exception as e:
            errors += 1
            print(f"  ERROR loading {wav_path}: {e}")
            continue

        # --- Determine pitch step for this sample ---
        # Alternate between +1 and -1 to get balanced representation
        pitch_step = config.AUG_PITCH_STEPS[i % len(config.AUG_PITCH_STEPS)]

        # --- Build variants ---
        variants = [
            ("noise",        augment_noise(y_orig)),
            (f"pitch{pitch_step:+d}", augment_pitch(y_orig, pitch_step)),
        ]

        for aug_type, y_aug in variants:
            aug_stem = f"{row['stem']}_aug_{aug_type}"
            aug_paths = get_feature_paths(aug_stem, is_aug=True)

            if all_features_exist(aug_paths):
                cached += 1
                # Still need to add the row to new_rows if not in manifest yet
            else:
                try:
                    process_file(
                        wav_path=wav_path,
                        stem=aug_stem,
                        is_aug=True,
                        y_override=y_aug,
                    )
                    processed += 1
                except Exception as e:
                    errors += 1
                    print(f"  ERROR augmenting {wav_path} ({aug_type}): {e}")
                    continue

            # Build the new manifest row (copy original, update paths + metadata)
            new_row = row.to_dict()
            new_row["mel_path"]               = aug_paths["mel"]
            new_row["mfcc_path"]              = aug_paths["mfcc"]
            new_row["chroma_path"]            = aug_paths["chroma"]
            new_row["spectral_contrast_path"] = aug_paths["spectral_contrast"]
            new_row["waveform_path"]          = aug_paths["waveform"]
            new_row["scalar_path"]            = aug_paths["scalar"]
            new_row["stem"]                   = aug_stem
            new_row["split"]                  = "train"      # always train
            new_row["is_augmented"]           = True
            new_rows.append(new_row)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (processed + cached) / max(elapsed, 1)
            print(f"  [{i+1}/{len(orig_train)}] {elapsed:.0f}s | "
                  f"{rate:.1f} files/s | new={processed}, cached={cached}, errors={errors}")

    # ------- Combine and save -------
    if new_rows:
        # Remove any rows that might already be in the manifest from a previous run
        existing_stems = set(df["stem"].values)
        truly_new = [r for r in new_rows if r["stem"] not in existing_stems]

        if truly_new:
            new_df = pd.DataFrame(truly_new, columns=df.columns)
            full_df = pd.concat([df, new_df], ignore_index=True)
            full_df.to_csv(config.MANIFEST_PATH, index=False)
            print(f"\nAdded {len(truly_new)} augmented rows to manifest.")
        else:
            print("\nAll augmented rows already in manifest (nothing to add).")
    else:
        print("\nNo new rows generated.")

    # ------- Final verification -------
    final_df = pd.read_csv(config.MANIFEST_PATH)
    val_aug  = final_df[(final_df["split"] == "val")  & (final_df["is_augmented"] == True)].shape[0]
    test_aug = final_df[(final_df["split"] == "test") & (final_df["is_augmented"] == True)].shape[0]

    if val_aug > 0 or test_aug > 0:
        raise RuntimeError(
            f"DATA LEAKAGE: Found {val_aug} augmented val rows and "
            f"{test_aug} augmented test rows. This should NEVER happen."
        )

    print("\nAugmentation complete!")
    print(f"  Total manifest rows: {len(final_df)}")
    train_total = final_df[final_df.split == "train"]
    print(f"  Train: {len(train_total)} "
          f"({(~train_total.is_augmented).sum()} original + "
          f"{train_total.is_augmented.sum()} augmented)")
    print(f"  Val:   {(final_df.split=='val').sum()} (no augmentation)")
    print(f"  Test:  {(final_df.split=='test').sum()} (no augmentation)")
    print(f"\nAugmentation leakage check: PASSED (0 augmented val/test rows)")
    print(f"Errors: {errors}")

    return final_df


if __name__ == "__main__":
    print("=" * 60)
    print("Offline Augmentation — Capstone3 SER Project")
    print("=" * 60)
    print("Augmenting TRAIN split only (3× expansion: noise + pitch shift)")
    print()
    augment_train_split()

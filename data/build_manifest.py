# =============================================================================
# build_manifest.py
#
# Builds Data/manifest.csv with speaker-independent train/val/test splits.
#
# CRITICAL RULES:
#   - NO actor can appear in more than one split (zero leakage guaranteed)
#   - Splits are 70/15/15 using seed=42
#   - RAVDESS and CREMA-D actors are split INDEPENDENTLY so both datasets
#     are proportionally represented in every split
#   - Script CRASHES loudly if any actor overlap is detected
#   - Feature files must already exist (run extract_features.py first)
#
# Run: python Scripts/data/build_manifest.py
# =============================================================================

import os
import sys
import glob
import random
import math

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.parse_labels import parse_file
from data.extract_features import get_feature_paths, make_stem


# ---------------------------------------------------------------------------
# SPEAKER-INDEPENDENT SPLIT
# ---------------------------------------------------------------------------

def assign_actor_splits(actor_ids: list, seed: int, train_ratio: float, val_ratio: float):
    """
    Shuffle a list of actor IDs and split into train/val/test.
    Returns a dict: {actor_id: "train" | "val" | "test"}
    """
    actors = sorted(actor_ids)          # sort first for reproducibility
    rng = random.Random(seed)
    rng.shuffle(actors)

    n = len(actors)
    n_train = math.floor(n * train_ratio)
    n_val   = math.floor(n * val_ratio)
    # n_test = n - n_train - n_val (derived, never hardcoded)

    train_actors = actors[:n_train]
    val_actors   = actors[n_train : n_train + n_val]
    test_actors  = actors[n_train + n_val :]

    split_map = {}
    for a in train_actors: split_map[a] = "train"
    for a in val_actors:   split_map[a] = "val"
    for a in test_actors:  split_map[a] = "test"

    return split_map, train_actors, val_actors, test_actors


def verify_no_leakage(train_actors, val_actors, test_actors, dataset_name: str):
    """
    Assert that no actor appears in more than one split.
    Crashes with a clear error message if leakage is detected.
    """
    ta = set(train_actors)
    va = set(val_actors)
    te = set(test_actors)

    tv = ta & va
    tt = ta & te
    vt = va & te

    if tv or tt or vt:
        msg = (
            f"\n{'='*60}\n"
            f"ACTOR LEAKAGE DETECTED in {dataset_name}!\n"
            f"  train ∩ val  = {tv}\n"
            f"  train ∩ test = {tt}\n"
            f"  val   ∩ test = {vt}\n"
            f"{'='*60}\n"
            f"This would inflate accuracy. Aborting manifest build."
        )
        raise RuntimeError(msg)
    print(f"  [{dataset_name}] No actor leakage — split verified.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_manifest():
    """
    Collect all valid audio files, verify features exist, assign splits,
    and write manifest.csv.
    """
    # ------- 1. Collect all valid audio records -------
    print("Collecting audio files...")

    wav_files = []
    ravdess_pattern = os.path.join(config.RAVDESS_DIR, "**", "*.wav")
    wav_files += glob.glob(ravdess_pattern, recursive=True)
    cremad_pattern = os.path.join(config.CREMAD_DIR, "*.wav")
    wav_files += glob.glob(cremad_pattern)

    records = []
    missing_features = []
    skipped = 0

    for wav_path in sorted(wav_files):
        result = parse_file(wav_path)
        if result is None:
            skipped += 1
            continue

        actor_id, label, emotion_name, dataset = result
        filename = os.path.basename(wav_path)
        stem = make_stem(dataset, actor_id, filename)
        paths = get_feature_paths(stem)

        # Verify all 6 feature files exist
        missing = [k for k, v in paths.items() if not os.path.isfile(v)]
        if missing:
            missing_features.append((wav_path, missing))
            continue

        records.append({
            "wav_path":             wav_path,
            "mel_path":             paths["mel"],
            "mfcc_path":            paths["mfcc"],
            "chroma_path":          paths["chroma"],
            "spectral_contrast_path": paths["spectral_contrast"],
            "waveform_path":        paths["waveform"],
            "scalar_path":          paths["scalar"],
            "emotion":              emotion_name,
            "label":                label,
            "actor_id":             actor_id,
            "dataset":              dataset,
            "stem":                 stem,
            "is_augmented":         False,
        })

    if missing_features:
        print(f"\nERROR: {len(missing_features)} files have missing feature .npy files.")
        print("Please run extract_features.py first.")
        for wav_path, missing in missing_features[:10]:
            print(f"  {os.path.basename(wav_path)}: missing {missing}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features)-10} more.")
        raise RuntimeError("Run data/extract_features.py before building the manifest.")

    print(f"  Total valid records: {len(records)}")
    print(f"  Skipped (wrong emotion class): {skipped}")

    # ------- 2. Separate actor IDs by dataset -------
    ravdess_actors = sorted(set(r["actor_id"] for r in records if r["dataset"] == "RAVDESS"))
    cremad_actors  = sorted(set(r["actor_id"] for r in records if r["dataset"] == "CREMA-D"))

    print(f"\nActor counts: RAVDESS={len(ravdess_actors)}, CREMA-D={len(cremad_actors)}")

    # ------- 3. Split actors independently per dataset -------
    ravdess_split, r_train, r_val, r_test = assign_actor_splits(
        ravdess_actors, config.SPLIT_SEED, config.TRAIN_RATIO, config.VAL_RATIO
    )
    cremad_split, c_train, c_val, c_test = assign_actor_splits(
        cremad_actors, config.SPLIT_SEED, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print(f"  RAVDESS split: train={len(r_train)}, val={len(r_val)}, test={len(r_test)} actors")
    print(f"  CREMA-D split: train={len(c_train)}, val={len(c_val)}, test={len(c_test)} actors")

    # ------- 4. Verify no leakage -------
    verify_no_leakage(r_train, r_val, r_test, "RAVDESS")
    verify_no_leakage(c_train, c_val, c_test, "CREMA-D")

    # ------- 5. Assign split to each record -------
    combined_split = {**ravdess_split, **cremad_split}
    for rec in records:
        rec["split"] = combined_split[rec["actor_id"]]

    # ------- 6. Build DataFrame -------
    columns = [
        "wav_path", "mel_path", "mfcc_path", "chroma_path",
        "spectral_contrast_path", "waveform_path", "scalar_path",
        "emotion", "label", "actor_id", "dataset", "stem", "split", "is_augmented",
    ]
    df = pd.DataFrame(records, columns=columns)

    # ------- 7. Print distribution -------
    print("\nClass distribution per split:")
    print("-" * 60)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        counts = sub.groupby("label")["emotion"].first()
        row = f"  {split:5s} ({len(sub):5d} total): "
        for lbl in sorted(config.EMOTIONS.keys()):
            n = (sub["label"] == lbl).sum()
            row += f"{config.EMOTIONS[lbl]}={n} "
        print(row)
    print()

    # ------- 8. Cross-dataset leakage check on final DataFrame -------
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        actors_a = set(df[df["split"] == split_a]["actor_id"])
        actors_b = set(df[df["split"] == split_b]["actor_id"])
        overlap = actors_a & actors_b
        assert len(overlap) == 0, (
            f"ACTOR LEAKAGE in final DataFrame: {split_a} ∩ {split_b} = {overlap}"
        )
    print("Final DataFrame leakage check: PASSED")

    # ------- 9. Save -------
    df.to_csv(config.MANIFEST_PATH, index=False)
    print(f"\nManifest saved: {config.MANIFEST_PATH}")
    print(f"Rows: {len(df)} (train={len(df[df.split=='train'])}, "
          f"val={len(df[df.split=='val'])}, test={len(df[df.split=='test'])})")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Manifest Builder — Capstone3 SER Project")
    print("=" * 60)
    df = build_manifest()
    print("\nSample rows:")
    print(df[["emotion", "label", "actor_id", "dataset", "split"]].head(10).to_string())

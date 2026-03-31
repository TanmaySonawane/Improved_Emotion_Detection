# =============================================================================
# extract_features.py
#
# Extracts 6 feature types from every valid RAVDESS and CREMA-D audio file
# and saves them as .npy files to Data/Features/.
#
# CRITICAL RULES:
#   - Skip file if all 6 .npy outputs already exist (idempotent, safe to re-run)
#   - MFCC MUST be per-row normalized (fixes the 200× scale bug from v1/v2)
#   - Scalar features MUST be computed on y_orig (BEFORE padding)
#   - Center-pad to exactly config.N_SAMPLES samples
#
# Run: python Scripts/data/extract_features.py
# =============================================================================

import os
import sys
import glob
import time
import traceback

import numpy as np
import librosa

# Add Scripts root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.parse_labels import parse_file


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def center_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Center-pad or center-crop a 1-D audio array to exactly target_len samples.

    Padding: adds silence symmetrically on both sides.
    Cropping: removes samples symmetrically from both ends.
    """
    n = len(y)
    if n == target_len:
        return y
    elif n < target_len:
        pad_total = target_len - n
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant")
    else:
        crop_total = n - target_len
        crop_left  = crop_total // 2
        return y[crop_left : crop_left + target_len]


def compute_mel(y_padded: np.ndarray) -> np.ndarray:
    """
    Mel spectrogram in dB.
    Output shape: (config.N_MELS, config.N_FRAMES) = (128, 173)
    """
    mel = librosa.feature.melspectrogram(
        y=y_padded,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def compute_mfcc_normalized(y_padded: np.ndarray) -> np.ndarray:
    """
    MFCC + delta + delta2, then PER-ROW NORMALIZED.

    Output shape: (config.N_MFCC, config.N_FRAMES, 3) = (40, 173, 3)

    Why per-row normalization?
      Raw MFCC values span -600 to +300. Lower coefficients have much larger
      magnitude than higher ones. This 200× scale difference causes gradient
      instability during training. Normalizing each row (each coefficient)
      independently to mean=0, std=1 puts all 40 coefficients on equal footing.
    """
    mfcc_raw    = librosa.feature.mfcc(
        y=y_padded,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mfcc_delta  = librosa.feature.delta(mfcc_raw, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc_raw, order=2)

    # Stack → (N_MFCC, N_FRAMES, 3)
    mfcc_stack = np.stack([mfcc_raw, mfcc_delta, mfcc_delta2], axis=-1)

    # Per-row normalization: for each channel c and each coefficient row i,
    # subtract the row mean and divide by the row std.
    for c in range(3):
        for i in range(config.N_MFCC):
            row = mfcc_stack[i, :, c]
            mfcc_stack[i, :, c] = (row - row.mean()) / (row.std() + 1e-8)

    return mfcc_stack.astype(np.float32)


def compute_chroma(y_padded: np.ndarray) -> np.ndarray:
    """
    Chroma STFT — captures which musical notes (pitch classes) are active.
    Output shape: (config.N_CHROMA, config.N_FRAMES) = (12, 173)
    """
    chroma = librosa.feature.chroma_stft(
        y=y_padded,
        sr=config.SAMPLE_RATE,
        n_chroma=config.N_CHROMA,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    return chroma.astype(np.float32)


def compute_spectral_contrast(y_padded: np.ndarray) -> np.ndarray:
    """
    Spectral contrast — measures peaks vs valleys across frequency bands.
    Output shape: (N_SPECTRAL_CONTRAST_ROWS, N_FRAMES) = (7, 173)
    """
    sc = librosa.feature.spectral_contrast(
        y=y_padded,
        sr=config.SAMPLE_RATE,
        n_bands=config.N_SPECTRAL_CONTRAST_BANDS,
        hop_length=config.HOP_LENGTH,
    )
    return sc.astype(np.float32)


def compute_scalar(y_orig: np.ndarray) -> np.ndarray:
    """
    Compute a fixed-length scalar feature vector from the ORIGINAL (unpadded) signal.

    Why unpadded? Padding adds silence which would corrupt statistics like RMS
    and ZCR (they'd be dragged toward zero by the silent regions).

    Output shape: (config.SCALAR_DIM,) = (260,)

    Components (in order):
        ZCR mean+std (2), RMS mean+std (2), spectral centroid mean+std (2),
        spectral rolloff mean+std (2), spectral bandwidth mean+std (2),
        spectral flatness mean+std (2), pitch mean+std (2),
        MFCC means[40] + stds[40], chroma means[12] + stds[12],
        spectral contrast means[7] + stds[7], mel means[128]
    """
    sr = config.SAMPLE_RATE
    hop = config.HOP_LENGTH

    def _stats(arr):
        """Return [mean, std] of a 1-D or squeezed array."""
        a = arr.flatten()
        return np.array([a.mean(), a.std()], dtype=np.float32)

    # --- Frame-level features ---
    zcr     = librosa.feature.zero_crossing_rate(y_orig, hop_length=hop)
    rms     = librosa.feature.rms(y=y_orig, hop_length=hop)
    centroid= librosa.feature.spectral_centroid(y=y_orig, sr=sr, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y_orig, sr=sr, hop_length=hop)
    bandwidth=librosa.feature.spectral_bandwidth(y=y_orig, sr=sr, hop_length=hop)
    flatness= librosa.feature.spectral_flatness(y=y_orig, hop_length=hop)

    # --- Pitch (fundamental frequency) ---
    # librosa.yin returns shape (N_frames,); replace 0s with NaN before stats
    pitch = librosa.yin(y_orig, fmin=50, fmax=500, sr=sr, hop_length=hop)
    pitch_clean = pitch.copy().astype(np.float32)
    pitch_clean[pitch_clean == 0] = np.nan
    pitch_mean = float(np.nanmean(pitch_clean)) if not np.all(np.isnan(pitch_clean)) else 0.0
    pitch_std  = float(np.nanstd(pitch_clean))  if not np.all(np.isnan(pitch_clean)) else 0.0
    pitch_stats = np.array([pitch_mean, pitch_std], dtype=np.float32)

    # --- MFCC summary (on original, NOT per-row normalized for scalar vector) ---
    mfcc_orig = librosa.feature.mfcc(
        y=y_orig, sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=hop,
    )
    mfcc_means = mfcc_orig.mean(axis=1).astype(np.float32)  # shape (N_MFCC,)
    mfcc_stds  = mfcc_orig.std(axis=1).astype(np.float32)

    # --- Chroma summary ---
    chroma_orig = librosa.feature.chroma_stft(
        y=y_orig, sr=sr,
        n_chroma=config.N_CHROMA,
        n_fft=config.N_FFT,
        hop_length=hop,
    )
    chroma_means = chroma_orig.mean(axis=1).astype(np.float32)
    chroma_stds  = chroma_orig.std(axis=1).astype(np.float32)

    # --- Spectral contrast summary ---
    sc_orig = librosa.feature.spectral_contrast(
        y=y_orig, sr=sr,
        n_bands=config.N_SPECTRAL_CONTRAST_BANDS,
        hop_length=hop,
    )
    sc_means = sc_orig.mean(axis=1).astype(np.float32)
    sc_stds  = sc_orig.std(axis=1).astype(np.float32)

    # --- Mel summary (mean per frequency band, no std) ---
    mel_orig = librosa.feature.melspectrogram(
        y=y_orig, sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=hop,
    )
    mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)
    mel_means = mel_db_orig.mean(axis=1).astype(np.float32)  # shape (N_MELS,)

    # --- Concatenate everything ---
    scalar = np.concatenate([
        _stats(zcr),        # 2
        _stats(rms),        # 2
        _stats(centroid),   # 2
        _stats(rolloff),    # 2
        _stats(bandwidth),  # 2
        _stats(flatness),   # 2
        pitch_stats,        # 2
        mfcc_means,         # N_MFCC = 40
        mfcc_stds,          # N_MFCC = 40
        chroma_means,       # N_CHROMA = 12
        chroma_stds,        # N_CHROMA = 12
        sc_means,           # N_SPECTRAL_CONTRAST_ROWS = 7
        sc_stds,            # N_SPECTRAL_CONTRAST_ROWS = 7
        mel_means,          # N_MELS = 128
    ])
    # Total = 2+2+2+2+2+2+2 + 40+40 + 12+12 + 7+7 + 128 = 14+80+24+14+128 = 260
    return scalar.astype(np.float32)


# ---------------------------------------------------------------------------
# PER-FILE PROCESSING
# ---------------------------------------------------------------------------

def get_feature_paths(stem: str, is_aug: bool = False) -> dict:
    """
    Return a dict of {feature_type: save_path} for a given file stem.
    is_aug=True uses AUG_DIR instead of FEATURES_DIR.
    """
    base_dir = config.AUG_DIR if is_aug else config.FEATURES_DIR
    return {
        "mel":               os.path.join(base_dir, "mel",               f"{stem}_mel.npy"),
        "mfcc":              os.path.join(base_dir, "mfcc",              f"{stem}_mfcc.npy"),
        "chroma":            os.path.join(base_dir, "chroma",            f"{stem}_chroma.npy"),
        "spectral_contrast": os.path.join(base_dir, "spectral_contrast", f"{stem}_spectral_contrast.npy"),
        "waveform":          os.path.join(base_dir, "waveform",          f"{stem}_waveform.npy"),
        "scalar":            os.path.join(base_dir, "scalar",            f"{stem}_scalar.npy"),
    }


def all_features_exist(paths: dict) -> bool:
    """Return True only if every feature .npy file already exists on disk."""
    return all(os.path.isfile(p) for p in paths.values())


def process_file(wav_path: str, stem: str, is_aug: bool = False,
                 y_override: np.ndarray = None) -> dict:
    """
    Extract all 6 features for a single audio file and save to disk.

    Args:
        wav_path:    Path to the source .wav file (used if y_override is None).
        stem:        Unique identifier string used for output filenames.
        is_aug:      If True, save to AUG_DIR instead of FEATURES_DIR.
        y_override:  If provided, use this pre-loaded/augmented audio instead of
                     loading from wav_path. Must be at the original (unpadded) SR.

    Returns:
        dict of {feature_type: save_path}
    """
    paths = get_feature_paths(stem, is_aug)

    # Skip-if-exists check: if all 6 already exist, do nothing
    if all_features_exist(paths):
        return paths

    # Load audio at 22050 Hz (or use override)
    if y_override is not None:
        y_orig = y_override
    else:
        y_orig, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)

    # Center-pad to N_SAMPLES
    y_padded = center_pad(y_orig, config.N_SAMPLES)

    # --- Compute all 6 features ---
    mel     = compute_mel(y_padded)
    mfcc    = compute_mfcc_normalized(y_padded)
    chroma  = compute_chroma(y_padded)
    sc      = compute_spectral_contrast(y_padded)
    waveform= y_padded.astype(np.float32)
    scalar  = compute_scalar(y_orig)          # <-- ORIGINAL signal, not padded

    # --- Save ---
    np.save(paths["mel"],               mel)
    np.save(paths["mfcc"],              mfcc)
    np.save(paths["chroma"],            chroma)
    np.save(paths["spectral_contrast"], sc)
    np.save(paths["waveform"],          waveform)
    np.save(paths["scalar"],            scalar)

    return paths


def make_stem(dataset: str, actor_id: str, filename: str) -> str:
    """
    Create a unique filename stem for .npy files.
    Example: "ravdess_ravdess_07_03-01-05-01-01-01-07"
    We include dataset prefix to ensure no collisions between datasets.
    """
    base = os.path.splitext(filename)[0]
    return f"{dataset.lower().replace('-', '')}_{actor_id}_{base}"


# ---------------------------------------------------------------------------
# MAIN — walk both datasets and extract features
# ---------------------------------------------------------------------------

def extract_all():
    """
    Walk RAVDESS and CREMA-D directories, parse labels, and extract features
    for every valid audio file.
    """
    config.create_output_dirs()

    wav_files = []

    # Collect RAVDESS files (nested in Actor_XX subdirectories)
    ravdess_pattern = os.path.join(config.RAVDESS_DIR, "**", "*.wav")
    wav_files += glob.glob(ravdess_pattern, recursive=True)

    # Collect CREMA-D files (flat directory)
    cremad_pattern = os.path.join(config.CREMAD_DIR, "*.wav")
    wav_files += glob.glob(cremad_pattern)

    print(f"Found {len(wav_files)} total .wav files.")

    skipped   = 0
    processed = 0
    cached    = 0
    errors    = 0
    t_start   = time.time()

    for i, wav_path in enumerate(sorted(wav_files)):
        # Parse label — returns None if this file should be skipped
        result = parse_file(wav_path)
        if result is None:
            skipped += 1
            continue

        actor_id, label, emotion_name, dataset = result
        filename = os.path.basename(wav_path)
        stem = make_stem(dataset, actor_id, filename)

        # Check if already extracted
        paths = get_feature_paths(stem)
        if all_features_exist(paths):
            cached += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(wav_files)}] {elapsed:.0f}s elapsed — "
                      f"{processed} new, {cached} cached, {skipped} skipped, {errors} errors")
            continue

        try:
            process_file(wav_path, stem)
            processed += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR on {wav_path}: {e}")
            traceback.print_exc()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (processed + cached) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(wav_files)}] {elapsed:.0f}s elapsed | "
                  f"{rate:.1f} files/s | "
                  f"{processed} new, {cached} cached, {skipped} skipped, {errors} errors")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  New:     {processed}")
    print(f"  Cached:  {cached}")
    print(f"  Skipped: {skipped} (wrong emotion class)")
    print(f"  Errors:  {errors}")

    if errors > 0:
        print("\nWARNING: Some files had errors. Check output above.")

    return processed, cached, skipped, errors


# ---------------------------------------------------------------------------
# SHAPE VALIDATION (run after extraction to verify everything looks right)
# ---------------------------------------------------------------------------

def validate_shapes(n_samples: int = 10):
    """
    Load n_samples random .npy files from each feature type and verify shapes.
    Call after extract_all() to catch any silent bugs early.
    """
    import random

    expected = {
        "mel":               (config.N_MELS,                   config.N_FRAMES),
        "mfcc":              (config.N_MFCC,                   config.N_FRAMES, 3),
        "chroma":            (config.N_CHROMA,                  config.N_FRAMES),
        "spectral_contrast": (config.N_SPECTRAL_CONTRAST_ROWS,  config.N_FRAMES),
        "waveform":          (config.N_SAMPLES,),
        "scalar":            (config.SCALAR_DIM,),
    }

    print("\nValidating feature shapes...")
    all_ok = True
    for feat_type, exp_shape in expected.items():
        feat_dir = os.path.join(config.FEATURES_DIR, feat_type)
        files = glob.glob(os.path.join(feat_dir, "*.npy"))
        if not files:
            print(f"  WARNING: No .npy files found in {feat_dir}")
            all_ok = False
            continue
        sample = random.sample(files, min(n_samples, len(files)))
        for fpath in sample:
            arr = np.load(fpath)
            if arr.shape != exp_shape:
                print(f"  SHAPE MISMATCH: {os.path.basename(fpath)}")
                print(f"    Expected: {exp_shape}")
                print(f"    Got:      {arr.shape}")
                all_ok = False

    # MFCC normalization check
    mfcc_dir = os.path.join(config.FEATURES_DIR, "mfcc")
    mfcc_files = glob.glob(os.path.join(mfcc_dir, "*.npy"))
    if mfcc_files:
        check = random.sample(mfcc_files, min(5, len(mfcc_files)))
        for fpath in check:
            arr = np.load(fpath)
            max_val = np.abs(arr).max()
            if max_val > 10.0:
                print(f"  MFCC NORMALIZATION FAILED: {os.path.basename(fpath)} "
                      f"max_abs={max_val:.1f} (expected ≤ 10)")
                all_ok = False

    if all_ok:
        print("  All shape checks passed!")
    else:
        print("  Some checks FAILED — review errors above.")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Extraction — Capstone3 SER Project")
    print("=" * 60)
    print(f"RAVDESS dir: {config.RAVDESS_DIR}")
    print(f"CREMA-D dir: {config.CREMAD_DIR}")
    print(f"Output dir:  {config.FEATURES_DIR}")
    print()

    # Verify GPU info (informational only — extraction runs on CPU)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: not available (extraction runs on CPU, this is fine)")
    except ImportError:
        print("PyTorch not installed — extraction runs on CPU")

    print()
    processed, cached, skipped, errors = extract_all()

    if errors == 0:
        validate_shapes()

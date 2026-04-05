# =============================================================================
# run_pipeline.py — Master Pipeline Runner
#
# Runs the full SER pipeline in the correct order.
# You can run specific steps or the entire sequence.
#
# Usage:
#   python Scripts/run_pipeline.py --all            # run everything
#   python Scripts/run_pipeline.py --step extract   # just feature extraction
#   python Scripts/run_pipeline.py --step manifest  # just build manifest
#   python Scripts/run_pipeline.py --step augment   # just augmentation
#   python Scripts/run_pipeline.py --step verify    # verify data integrity
#   python Scripts/run_pipeline.py --step train1    # traditional ML
#   python Scripts/run_pipeline.py --step train2    # CNN+BiLSTM mel
#   python Scripts/run_pipeline.py --step train3    # CNN+BiLSTM MFCC
#   python Scripts/run_pipeline.py --step train4    # EfficientNet-B0
#   python Scripts/run_pipeline.py --step train5    # ResNet-18
#   python Scripts/run_pipeline.py --step evaluate  # print comparison table
#   python Scripts/run_pipeline.py --from extract   # run from a specific step onward
# =============================================================================

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))
import config


# ---------------------------------------------------------------------------
# STEP 0: Environment Verification
# ---------------------------------------------------------------------------

def step_env():
    """Check Python environment — GPU, NumPy version, key packages."""
    print("\n" + "=" * 60)
    print("STEP 0 — Environment Check")
    print("=" * 60)

    import platform
    print(f"Platform:   {platform.system()} {platform.release()}")
    print(f"Python:     {sys.version.split()[0]}")

    # NumPy version check (Numba requires ≤ 2.3)
    try:
        import numpy as np
        v = tuple(int(x) for x in np.__version__.split(".")[:2])
        status = "OK" if v <= (2, 3) else "WARNING — Numba requires NumPy ≤ 2.3"
        print(f"NumPy:      {np.__version__}  [{status}]")
        if v > (2, 3):
            print("  Fix: pip install \"numpy<2.4\"")
    except ImportError:
        print("NumPy:      NOT INSTALLED")

    # PyTorch + CUDA
    try:
        import torch
        print(f"PyTorch:    {torch.__version__}")
        if torch.cuda.is_available():
            name    = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU:        {name} ({vram_gb:.1f} GB VRAM)  [CUDA AVAILABLE]")
        else:
            print("GPU:        NOT AVAILABLE — training will use CPU (slow)")
            print("  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("PyTorch:    NOT INSTALLED")

    # librosa
    try:
        import librosa
        print(f"librosa:    {librosa.__version__}")
    except ImportError:
        print("librosa:    NOT INSTALLED  →  pip install librosa")

    # torchaudio (optional but recommended for ResNet-18 waveform branch)
    try:
        import torchaudio
        print(f"torchaudio: {torchaudio.__version__}")
    except ImportError:
        print("torchaudio: NOT INSTALLED  →  pip install torchaudio  (optional)")

    # torchvision
    try:
        import torchvision
        print(f"torchvision:{torchvision.__version__}")
    except ImportError:
        print("torchvision:NOT INSTALLED  →  pip install torchvision")

    # scikit-learn
    try:
        import sklearn
        print(f"sklearn:    {sklearn.__version__}")
    except ImportError:
        print("sklearn:    NOT INSTALLED  →  pip install scikit-learn")

    # seaborn / matplotlib
    try:
        import seaborn, matplotlib
        print(f"seaborn:    {seaborn.__version__}  matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("seaborn/matplotlib: NOT INSTALLED  →  pip install seaborn matplotlib")

    # gradio (for deployment — optional until that step)
    try:
        import gradio
        print(f"gradio:     {gradio.__version__}")
    except ImportError:
        print("gradio:     not installed yet  (needed for deployment step only)")

    # Dataset directories
    print(f"\nData root:  {config.DATA_ROOT}")
    for name, path in [("RAVDESS", config.RAVDESS_DIR), ("CREMA-D", config.CREMAD_DIR)]:
        exists = os.path.isdir(path)
        status = "EXISTS" if exists else "MISSING"
        print(f"  {name:8s}: {path}  [{status}]")

    manifest_status = "EXISTS" if os.path.isfile(config.MANIFEST_PATH) else "not yet"
    print(f"  manifest: {config.MANIFEST_PATH}  [{manifest_status}]")


# ---------------------------------------------------------------------------
# STEP 1: Feature Extraction
# ---------------------------------------------------------------------------

def step_extract():
    print("\n" + "=" * 60)
    print("STEP 1 — Feature Extraction")
    print("=" * 60)
    print("Extracting mel, MFCC, chroma, spectral contrast, waveform, scalar...")
    print("(Safe to re-run — already-extracted files are skipped)\n")
    from data.extract_features import extract_all, validate_shapes
    t = time.time()
    processed, cached, skipped, errors = extract_all()
    print(f"\nExtraction complete in {time.time()-t:.1f}s")
    if errors == 0:
        validate_shapes()
    else:
        print(f"WARNING: {errors} extraction errors — check output above before continuing.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# STEP 2: Build Manifest
# ---------------------------------------------------------------------------

def step_manifest():
    print("\n" + "=" * 60)
    print("STEP 2 — Build Manifest")
    print("=" * 60)
    from data.build_manifest import build_manifest
    build_manifest()


# ---------------------------------------------------------------------------
# STEP 3: Augmentation
# ---------------------------------------------------------------------------

def step_augment():
    print("\n" + "=" * 60)
    print("STEP 3 — Offline Augmentation (train split only)")
    print("=" * 60)
    from data.augment import augment_train_split
    augment_train_split()


# ---------------------------------------------------------------------------
# STEP: Data Integrity Verification
# ---------------------------------------------------------------------------

def step_verify():
    """Run all data integrity checks without training anything."""
    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    import pandas as pd
    import numpy as np

    if not os.path.isfile(config.MANIFEST_PATH):
        print("ERROR: manifest.csv not found. Run steps 1-2 first.")
        return

    m = pd.read_csv(config.MANIFEST_PATH)
    print(f"Manifest rows: {len(m)}")

    # 1. No actor leakage
    train_a = set(m[m.split == "train"]["actor_id"])
    val_a   = set(m[m.split == "val"]["actor_id"])
    test_a  = set(m[m.split == "test"]["actor_id"])

    leakage = (train_a & val_a) | (train_a & test_a) | (val_a & test_a)
    if leakage:
        print(f"FAIL — Actor leakage detected: {leakage}")
    else:
        print("PASS — No actor leakage")

    # 2. No augmented data in val/test
    aug_bad = m[(m.split != "train") & (m.is_augmented == True)]
    if len(aug_bad) > 0:
        print(f"FAIL — {len(aug_bad)} augmented rows in val/test splits!")
    else:
        print("PASS — No augmented data in val/test")

    # 3. MFCC normalization check on 5 random samples
    sample = m.sample(min(5, len(m)), random_state=42)
    max_vals = []
    for _, row in sample.iterrows():
        arr = np.load(row["mfcc_path"])
        max_vals.append(np.abs(arr).max())
    avg_max = sum(max_vals) / len(max_vals)
    if avg_max > 10.0:
        print(f"FAIL — MFCC normalization issue: avg max_abs={avg_max:.1f} (should be ≤10)")
    else:
        print(f"PASS — MFCC normalization OK (avg max_abs={avg_max:.2f})")

    # 4. Feature shapes on 1 random sample
    row = m.iloc[0]
    shape_checks = [
        ("mel",               np.load(row["mel_path"]).shape,               (config.N_MELS, config.N_FRAMES)),
        ("mfcc",              np.load(row["mfcc_path"]).shape,              (config.N_MFCC, config.N_FRAMES, 3)),
        ("chroma",            np.load(row["chroma_path"]).shape,            (config.N_CHROMA, config.N_FRAMES)),
        ("spectral_contrast", np.load(row["spectral_contrast_path"]).shape, (config.N_SPECTRAL_CONTRAST_ROWS, config.N_FRAMES)),
        ("waveform",          np.load(row["waveform_path"]).shape,          (config.N_SAMPLES,)),
        ("scalar",            np.load(row["scalar_path"]).shape,            (config.SCALAR_DIM,)),
    ]
    all_shapes_ok = True
    for name, got, expected in shape_checks:
        if got != expected:
            print(f"FAIL — {name} shape: got {got}, expected {expected}")
            all_shapes_ok = False
    if all_shapes_ok:
        print(f"PASS — All feature shapes correct")

    # 5. Class distribution
    print("\nClass distribution:")
    for split in ["train", "val", "test"]:
        sub = m[m.split == split]
        counts = {config.EMOTIONS[lbl]: (sub["label"] == lbl).sum()
                  for lbl in sorted(config.EMOTIONS.keys())}
        print(f"  {split:5s}: " + "  ".join(f"{k}={v}" for k, v in counts.items())
              + f"  TOTAL={len(sub)}")


# ---------------------------------------------------------------------------
# TRAINING STEPS
# ---------------------------------------------------------------------------

def step_train1():
    print("\n" + "=" * 60)
    print("STEP 4 — Model 1: Traditional ML (SVM + Random Forest)")
    print("=" * 60)
    from models.traditional_ml import train_and_evaluate
    config.create_output_dirs()
    train_and_evaluate()


def step_train2():
    print("\n" + "=" * 60)
    print("STEP — Model 2: CNN+BiLSTM+Attention (Mel)")
    print("=" * 60)
    import training.train_cnn_bilstm as t
    t.main("mel")


def step_train3():
    print("\n" + "=" * 60)
    print("STEP — Model 3: CNN+BiLSTM+Attention (MFCC)")
    print("=" * 60)
    import training.train_cnn_bilstm as t
    t.main("mfcc")


def step_train6():
    print("\n" + "=" * 60)
    print("STEP — Model 6: MFCC + Chroma Fusion")
    print("=" * 60)
    import training.train_fusion as t
    t.main()


def step_train4():
    print("\n" + "=" * 60)
    print("STEP 5 — Model 4: EfficientNet-B0")
    print("=" * 60)
    import training.train_efficientnet as t
    t.main()


def step_train5():
    print("\n" + "=" * 60)
    print("STEP 6 — Model 5: ResNet-18 Dual-Input")
    print("=" * 60)
    import training.train_resnet18 as t
    t.main()


# ---------------------------------------------------------------------------
# STEP 11: Evaluate All
# ---------------------------------------------------------------------------

def step_ensemble():
    print("\n" + "=" * 60)
    print("STEP 10 — Ensemble (combines all completed models)")
    print("=" * 60)
    import training.train_ensemble as t
    t.main()


def step_evaluate():
    print("\n" + "=" * 60)
    print("STEP 11 — Model Comparison")
    print("=" * 60)
    from evaluation.evaluate import print_summary_table
    print_summary_table()


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

STEPS = {
    "env":      (step_env,      "Environment check"),
    "extract":  (step_extract,  "Feature extraction"),
    "manifest": (step_manifest, "Build manifest.csv"),
    "augment":  (step_augment,  "Offline augmentation"),
    "verify":   (step_verify,   "Data integrity verification"),
    "train1":   (step_train1,   "Train Model 1: Traditional ML"),
    "train2":   (step_train2,   "Train Model 2: CNN+BiLSTM+Attention (Mel)"),
    "train3":   (step_train3,   "Train Model 3: CNN+BiLSTM+Attention (MFCC)"),
    "train4":   (step_train4,   "Train Model 4: EfficientNet-B0"),
    "train5":   (step_train5,   "Train Model 5: ResNet-18"),
    "train6":   (step_train6,   "Train Model 6: MFCC+Chroma Fusion"),
    "ensemble": (step_ensemble, "Ensemble (combine completed models)"),
    "evaluate": (step_evaluate, "Model comparison table"),
}

FULL_ORDER = ["env", "extract", "manifest", "augment", "verify",
              "train1", "train2", "train3", "train4", "train5", "train6",
              "ensemble", "evaluate"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER Pipeline Runner")
    parser.add_argument("--all",  action="store_true",
                        help="Run the full pipeline from start to finish")
    parser.add_argument("--step", type=str, default=None,
                        choices=list(STEPS.keys()),
                        help="Run a single step")
    parser.add_argument("--from", dest="from_step", type=str, default=None,
                        choices=list(STEPS.keys()),
                        help="Run from this step to the end")
    args = parser.parse_args()

    config.create_output_dirs()

    if args.step:
        fn, desc = STEPS[args.step]
        print(f"\nRunning: {desc}")
        fn()

    elif args.from_step:
        start_idx = FULL_ORDER.index(args.from_step)
        steps_to_run = FULL_ORDER[start_idx:]
        print(f"Running from '{args.from_step}' to end: {steps_to_run}")
        for step_name in steps_to_run:
            fn, desc = STEPS[step_name]
            t0 = time.time()
            fn()
            print(f"  [{step_name}] done in {time.time()-t0:.1f}s")

    elif args.all:
        print("Running full pipeline...")
        for step_name in FULL_ORDER:
            fn, desc = STEPS[step_name]
            t0 = time.time()
            fn()
            print(f"\n[{step_name}] completed in {time.time()-t0:.1f}s")

    else:
        parser.print_help()
        print("\nAvailable steps:")
        for name, (_, desc) in STEPS.items():
            print(f"  {name:12s}  {desc}")

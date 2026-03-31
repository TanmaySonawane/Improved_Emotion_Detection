# =============================================================================
# config.py — Single Source of Truth for the SER Capstone Project
#
# RULE: Every number, path, and hyperparameter lives here.
#       Nothing is hardcoded anywhere else.
#       If you need to change something, change it HERE ONLY.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Using raw strings (r"...") to handle backslashes on Windows safely.

DATA_ROOT    = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone3\Data"
SCRIPTS_ROOT = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone3\Scripts"

RAVDESS_DIR  = os.path.join(DATA_ROOT, "RAVDESS")
CREMAD_DIR   = os.path.join(DATA_ROOT, "CREMA-D")

FEATURES_DIR = os.path.join(DATA_ROOT, "Features")
AUG_DIR      = os.path.join(DATA_ROOT, "Augmented")
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.csv")

OUTPUTS_DIR  = os.path.join(SCRIPTS_ROOT, "outputs")

# Feature subdirectory names (used to build full paths)
FEATURE_TYPES = ["mel", "mfcc", "chroma", "spectral_contrast", "waveform", "scalar"]

# ---------------------------------------------------------------------------
# AUDIO PARAMETERS
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050           # Hz — all audio loaded/resampled to this rate
DURATION    = 4.0             # seconds — all clips padded/cropped to this length
N_SAMPLES   = int(SAMPLE_RATE * DURATION)  # = 88200 samples (derived, not hardcoded)

# Spectrogram parameters
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128             # mel spectrogram frequency bins
N_MFCC      = 40              # MFCC coefficients
N_CHROMA    = 12              # chroma notes (one per semitone in an octave)
N_SPECTRAL_CONTRAST_BANDS = 6 # spectral_contrast uses n_bands, output is n_bands+1 = 7 rows
N_SPECTRAL_CONTRAST_ROWS  = N_SPECTRAL_CONTRAST_BANDS + 1  # = 7

# Number of time frames in a padded spectrogram
# Formula: floor(N_SAMPLES / HOP_LENGTH) + 1 = floor(88200/512)+1 = 172+1 = 173
import math
N_FRAMES = math.floor(N_SAMPLES / HOP_LENGTH) + 1  # = 173 (derived, not hardcoded)

# ---------------------------------------------------------------------------
# EMOTION LABELS  — NEVER hardcode labels elsewhere in the codebase.
# ---------------------------------------------------------------------------
# The EMOTIONS dict is the single definition of label → name.
# Everything else derives from it.

EMOTIONS = {
    0: "angry",
    1: "fear",
    2: "happy",
    3: "neutral",
    4: "sad",
}

NUM_CLASSES = len(EMOTIONS)  # = 5 (derived)

# RAVDESS filename emotion codes → integer label
# File format: [Modality]-[VocalChannel]-[EmotionCode]-[Intensity]-[Statement]-[Rep]-[Actor].wav
# Codes to KEEP: 01=neutral, 03=happy, 04=sad, 05=angry, 06=fearful
# Codes to SKIP (return None in parse_labels): 02=calm, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    "01": 3,  # neutral
    "03": 2,  # happy
    "04": 4,  # sad
    "05": 0,  # angry
    "06": 1,  # fear
}

# CREMA-D filename emotion codes → integer label
# File format: [ActorID]_[Sentence]_[EmotionCode]_[Intensity].wav
# Codes to KEEP: ANG, FEA, HAP, NEU, SAD
# Codes to SKIP: DIS=disgust
CREMAD_EMOTION_MAP = {
    "ANG": 0,  # angry
    "FEA": 1,  # fear
    "HAP": 2,  # happy
    "NEU": 3,  # neutral
    "SAD": 4,  # sad
}

# ---------------------------------------------------------------------------
# DATASET SPLITS
# ---------------------------------------------------------------------------
SPLIT_SEED  = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO is derived: 1.0 - TRAIN_RATIO - VAL_RATIO = 0.15
# Never hardcode 0.15 for test — always compute it.

# ---------------------------------------------------------------------------
# DATA AUGMENTATION (offline, train split only)
# ---------------------------------------------------------------------------
AUG_NOISE_FACTOR = 0.005      # Gaussian noise amplitude relative to max abs value
AUG_PITCH_STEPS  = [-1, 1]   # Semitones for pitch shift (alternating per sample)

# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
BATCH_SIZE          = 32      # Keep ≤ 32 for RTX 3050 4GB VRAM with AMP
NUM_WORKERS         = 0       # CRITICAL: Windows PyTorch requires 0 — NEVER change
LEARNING_RATE       = 1e-3
LABEL_SMOOTHING     = 0.1
EARLY_STOP_PATIENCE = 10      # Epochs with no val improvement before stopping
MAX_EPOCHS          = 100

# EfficientNet two-phase training
EFFICIENTNET_WARMUP_EPOCHS  = 10    # Phase 1: frozen backbone
EFFICIENTNET_FINETUNE_EPOCHS = 40   # Phase 2: top layers unfrozen
EFFICIENTNET_FINETUNE_LR    = 1e-4  # Lower LR for pretrained weights

# ---------------------------------------------------------------------------
# RESNET-18 WAVEFORM BRANCH
# ---------------------------------------------------------------------------
RESAMPLE_HZ        = 8000
# Downsampled waveform length: int(N_SAMPLES * RESAMPLE_HZ / SAMPLE_RATE) = 32000
WAVEFORM_DS_SAMPLES = int(N_SAMPLES * RESAMPLE_HZ / SAMPLE_RATE)  # = 32000 (derived)

# ---------------------------------------------------------------------------
# SCALAR FEATURE VECTOR DIMENSION
# ---------------------------------------------------------------------------
# Computed in extract_features.py — documented here for reference.
# ZCR(2) + RMS(2) + centroid(2) + rolloff(2) + bandwidth(2) + flatness(2) +
# pitch(2) + MFCC_means(40) + MFCC_stds(40) + chroma_means(12) + chroma_stds(12) +
# spectral_contrast_means(7) + spectral_contrast_stds(7) + mel_means(128) = 262
SCALAR_DIM = (
    2 + 2 + 2 + 2 + 2 + 2 + 2     # 7 scalar features × 2 stats = 14
    + N_MFCC * 2                   # 40 means + 40 stds = 80
    + N_CHROMA * 2                 # 12 means + 12 stds = 24
    + N_SPECTRAL_CONTRAST_ROWS * 2 # 7 means + 7 stds = 14
    + N_MELS                       # 128 mel bin means (no std to stay compact)
)
# = 14 + 80 + 24 + 14 + 128 = 260
# Note: if SCALAR_DIM differs from 262 as you change the above, that is fine —
# the models read config.SCALAR_DIM dynamically. Never hardcode 262 in model files.

# ---------------------------------------------------------------------------
# ENSEMBLE WEIGHTS  (higher weight = more trusted model)
# Mel-based models weighted higher per feature importance analysis.
# Keys match the output directory names under OUTPUTS_DIR.
#
# The ensemble script auto-detects which models have finished training
# (checkpoint exists) and normalizes weights to sum to 1.0 automatically.
# Add or remove models here as training completes.
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {
    "model1_traditional":     0.05,   # SVM/RF — low weight, weaker model
    "model2_cnn_bilstm_mel":  0.20,   # pending
    "model3_cnn_bilstm_mfcc": 0.10,   # pending
    "model4_efficientnet":    0.30,
    "model5_resnet18":        0.30,
    "model6_fusion":          0.05,   # pending
}

# ---------------------------------------------------------------------------
# DEPLOYMENT
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5    # Reject prediction if max softmax < this value

# ---------------------------------------------------------------------------
# UTILITY: create output subdirectories automatically on import
# ---------------------------------------------------------------------------
_MODEL_OUTPUT_DIRS = [
    "model1_traditional",
    "model2_cnn_bilstm_mel",
    "model3_cnn_bilstm_mfcc",
    "model4_efficientnet",
    "model5_resnet18",
    "model6_fusion",
    "model7_ensemble",
]

def create_output_dirs():
    """Call once at the start of any training script to ensure output dirs exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    for name in _MODEL_OUTPUT_DIRS:
        os.makedirs(os.path.join(OUTPUTS_DIR, name), exist_ok=True)
    for feat_type in FEATURE_TYPES:
        os.makedirs(os.path.join(FEATURES_DIR, feat_type), exist_ok=True)
        os.makedirs(os.path.join(AUG_DIR, feat_type), exist_ok=True)

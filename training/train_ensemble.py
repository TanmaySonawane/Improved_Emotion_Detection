# =============================================================================
# train_ensemble.py — Run Model 7: Weighted Ensemble
#
# No training happens here — this script:
#   1. Auto-detects which models have saved checkpoints in outputs/
#   2. Runs each model on the test set to collect probabilities
#   3. Computes a weighted average of probabilities
#   4. Evaluates and saves results to outputs/model7_ensemble/
#
# Run AFTER at least one of: train_traditional, train_efficientnet, train_resnet18
#
# Run: python Scripts/training/train_ensemble.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.ensemble import EnsemblePredictor
from training.train_utils import get_device
from evaluation.evaluate import plot_training_curves

OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, "model7_ensemble")


def main():
    print("=" * 60)
    print("Model 7 — Weighted Ensemble")
    print("=" * 60)
    print("No training required. Combining saved model checkpoints.")
    config.create_output_dirs()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()

    ensemble = EnsemblePredictor()
    ensemble.build(device=device)
    acc, macro_f1 = ensemble.save_results(output_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Ensemble complete.")
    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

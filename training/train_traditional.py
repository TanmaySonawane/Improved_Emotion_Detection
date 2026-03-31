# =============================================================================
# train_traditional.py — Train Model 1: SVM + Random Forest
#
# Run: python Scripts/training/train_traditional.py
# =============================================================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.traditional_ml import train_and_evaluate

if __name__ == "__main__":
    print("=" * 60)
    print("Model 1 — Traditional ML (SVM + Random Forest)")
    print("=" * 60)
    config.create_output_dirs()
    output_dir = os.path.join(config.OUTPUTS_DIR, "model1_traditional")
    results = train_and_evaluate(output_dir=output_dir)
    print("\nDone.")
    print(f"  SVM  test accuracy: {results['svm']['test_acc']:.4f}")
    print(f"  RF   test accuracy: {results['rf']['test_acc']:.4f}")

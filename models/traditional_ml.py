# =============================================================================
# traditional_ml.py — Model 1: SVM + Random Forest
#
# Input: scalar feature vector of shape (SCALAR_DIM,) per sample
# Target accuracy: 50–55%
#
# Design notes:
#   - Features are standardized with StandardScaler (fit on TRAIN only)
#   - class_weight="balanced" handles the neutral class imbalance
#   - The scaler is saved so deployment can use identical preprocessing
# =============================================================================

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_scalar_features(split: str, manifest_path: str = None):
    """
    Load all scalar .npy files for a given split from the manifest.

    Returns:
        X: np.ndarray of shape (N_samples, SCALAR_DIM)
        y: np.ndarray of shape (N_samples,) — integer labels
    """
    if manifest_path is None:
        manifest_path = config.MANIFEST_PATH

    df = pd.read_csv(manifest_path)
    sub = df[df["split"] == split].reset_index(drop=True)

    X_list, y_list = [], []
    for _, row in sub.iterrows():
        scalar = np.load(row["scalar_path"])
        X_list.append(scalar)
        y_list.append(int(row["label"]))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_and_evaluate(output_dir: str = None):
    """
    Train SVM and Random Forest on scalar features, evaluate on val and test sets.
    Saves the best model + scaler to output_dir.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUTS_DIR, "model1_traditional")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading scalar features from manifest...")
    X_train, y_train = load_scalar_features("train")
    X_val,   y_val   = load_scalar_features("val")
    X_test,  y_test  = load_scalar_features("test")

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  SCALAR_DIM = {X_train.shape[1]} (expected {config.SCALAR_DIM})")

    # --- Standardize: fit on TRAIN, apply to val/test ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Save scaler for deployment
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")

    results = {}

    # ============================================================
    # SVM (Support Vector Machine)
    # ============================================================
    print("\nTraining SVM (this may take a few minutes)...")
    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,    # needed for confidence scores in deployment
        random_state=config.SPLIT_SEED,
    )
    svm.fit(X_train_s, y_train)

    svm_val_acc  = accuracy_score(y_val,  svm.predict(X_val_s))
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test_s))
    print(f"  SVM  — Val Acc: {svm_val_acc:.4f} | Test Acc: {svm_test_acc:.4f}")

    svm_report = classification_report(
        y_test, svm.predict(X_test_s),
        target_names=[config.EMOTIONS[i] for i in sorted(config.EMOTIONS)],
        digits=4,
    )
    print("  SVM Test Report:")
    print(svm_report)

    svm_path = os.path.join(output_dir, "svm_model.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)

    results["svm"] = {"val_acc": svm_val_acc, "test_acc": svm_test_acc,
                      "report": svm_report, "model_path": svm_path}

    # ============================================================
    # Random Forest
    # ============================================================
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.SPLIT_SEED,
    )
    rf.fit(X_train_s, y_train)

    rf_val_acc  = accuracy_score(y_val,  rf.predict(X_val_s))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test_s))
    print(f"  RF   — Val Acc: {rf_val_acc:.4f} | Test Acc: {rf_test_acc:.4f}")

    rf_report = classification_report(
        y_test, rf.predict(X_test_s),
        target_names=[config.EMOTIONS[i] for i in sorted(config.EMOTIONS)],
        digits=4,
    )
    print("  RF Test Report:")
    print(rf_report)

    rf_path = os.path.join(output_dir, "rf_model.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)

    results["rf"] = {"val_acc": rf_val_acc, "test_acc": rf_test_acc,
                     "report": rf_report, "model_path": rf_path}

    # ============================================================
    # Save summary
    # ============================================================
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model 1 — Traditional ML Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"SVM  Val Acc: {svm_val_acc:.4f} | Test Acc: {svm_test_acc:.4f}\n")
        f.write(f"RF   Val Acc: {rf_val_acc:.4f}  | Test Acc: {rf_test_acc:.4f}\n\n")
        f.write("SVM Test Classification Report:\n")
        f.write(svm_report + "\n")
        f.write("RF Test Classification Report:\n")
        f.write(rf_report + "\n")

    print(f"\nResults saved to: {output_dir}")
    return results

"""
Stage 3 Runner: Model Training, Threshold Selection, and Validation Performance.
Trains:
1. Isolation Forest (Benign Train)
2. One-Class SVM (Subsampled Benign Train, clipped to [-10, 10])
3. Symmetric Keras Autoencoder (PyTorch backend, EarlyStopping on Val Benign Loss, clipped)
4. Random Forest Baseline (Val Train split, class_weight='balanced')
5. XGBoost Baseline (Val Train split, scale_pos_weight)

Outputs:
- Serialized models in models/
- Thresholds in models/autoencoder_thresholds.json
- Training times in models/training_times.json
- Reusable anomaly scores in data/processed/val_scores.parquet and test_scores.parquet
- Validation-set performance table (Precision, Recall, F1, ROC-AUC)
"""

import sys
import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    set_seed,
    ensure_dirs,
    setup_logger,
)
from src.features import load_selected_features
from src.models import (
    clip_features,
    train_isolation_forest,
    score_isolation_forest,
    train_one_class_svm,
    score_one_class_svm,
    train_autoencoder,
    compute_reconstruction_error,
    calibrate_autoencoder_thresholds,
    train_random_forest,
    train_xgboost,
    evaluate_binary_predictions,
    CLIP_MIN,
    CLIP_MAX,
)

logger = setup_logger("Stage3-Runner")

def main():
    overall_start = time.time()
    ensure_dirs()
    set_seed(RANDOM_SEED)

    print("=" * 85)
    print("STAGE 3: ANOMALY DETECTION & SUPERVISED BASELINE MODEL TRAINING")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 85)

    # 1. Load data splits
    logger.info("Loading processed dataset splits...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"))
    feature_cols = load_selected_features()

    print(f"Features: {len(feature_cols)} loaded from selected_features.json")
    print(f"Train Shape (Benign only) : {train_df.shape}")
    print(f"Val Shape (Mixed)         : {val_df.shape}")
    print(f"Test Shape (Mixed)        : {test_df.shape}")

    X_train_benign = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    y_val_bin = val_df["Label_Binary"].values
    y_val_multi = val_df["Label"].values

    # Validation Benign only for Autoencoder early stopping and threshold calibration
    val_benign_mask = (val_df["Label_Binary"] == 0)
    X_val_benign = val_df.loc[val_benign_mask, feature_cols].values

    X_test = test_df[feature_cols].values
    y_test_bin = test_df["Label_Binary"].values
    y_test_multi = test_df["Label"].values

    # 2. Supervised split: 80% train / 20% tune from val.parquet
    logger.info("Splitting val.parquet into 80% val_train and 20% val_tune for supervised baselines...")
    val_train_df, val_tune_df = train_test_split(
        val_df,
        test_size=0.20,
        stratify=val_df["Label_Binary"],
        random_state=RANDOM_SEED,
    )
    X_val_train = val_train_df[feature_cols].values
    y_val_train = val_train_df["Label_Binary"].values
    X_val_tune = val_tune_df[feature_cols].values
    y_val_tune = val_tune_df["Label_Binary"].values
    logger.info("val_train: %d rows | val_tune: %d rows", len(val_train_df), len(val_tune_df))

    training_times = {}
    val_metrics_report = {}

    # Initialize scores DataFrames
    val_scores_df = pd.DataFrame({
        "Label": y_val_multi,
        "Label_Binary": y_val_bin,
    })
    test_scores_df = pd.DataFrame({
        "Label": y_test_multi,
        "Label_Binary": y_test_bin,
    })

    # ==========================================================================
    # Model 1: Isolation Forest
    # ==========================================================================
    print("\n" + "-" * 85)
    print("1. TRAINING ISOLATION FOREST")
    print("-" * 85)
    if_model, if_time = train_isolation_forest(X_train_benign, random_state=RANDOM_SEED)
    training_times["isolation_forest"] = round(if_time, 2)
    joblib.dump(if_model, os.path.join(MODELS_DIR, "isolation_forest.joblib"))

    val_scores_if = score_isolation_forest(if_model, X_val)
    test_scores_if = score_isolation_forest(if_model, X_test)
    val_scores_df["score_isolation_forest"] = val_scores_if
    test_scores_df["score_isolation_forest"] = test_scores_if

    if_val_metrics = evaluate_binary_predictions(y_val_bin, val_scores_if)
    val_metrics_report["Isolation Forest"] = if_val_metrics
    print(f"  [Trained in {if_time:.2f}s] Val F1: {if_val_metrics['f1']:.4f} | ROC-AUC: {if_val_metrics['roc_auc']:.4f} | Recall: {if_val_metrics['recall']:.4f} | Precision: {if_val_metrics['precision']:.4f}")

    # ==========================================================================
    # Model 2: One-Class SVM
    # ==========================================================================
    print("\n" + "-" * 85)
    print("2. TRAINING ONE-CLASS SVM (Subsampled Benign, Feature Clipped [-10, 10])")
    print("-" * 85)
    ocsvm_model, ocsvm_time = train_one_class_svm(
        X_train_benign, subsample_size=25000, random_state=RANDOM_SEED, nu=0.05, kernel="rbf"
    )
    training_times["one_class_svm"] = round(ocsvm_time, 2)
    joblib.dump(ocsvm_model, os.path.join(MODELS_DIR, "one_class_svm.joblib"))

    val_scores_ocsvm = score_one_class_svm(ocsvm_model, X_val)
    test_scores_ocsvm = score_one_class_svm(ocsvm_model, X_test)
    val_scores_df["score_one_class_svm"] = val_scores_ocsvm
    test_scores_df["score_one_class_svm"] = test_scores_ocsvm

    ocsvm_val_metrics = evaluate_binary_predictions(y_val_bin, val_scores_ocsvm)
    val_metrics_report["One-Class SVM"] = ocsvm_val_metrics
    print(f"  [Trained in {ocsvm_time:.2f}s] Val F1: {ocsvm_val_metrics['f1']:.4f} | ROC-AUC: {ocsvm_val_metrics['roc_auc']:.4f} | Recall: {ocsvm_val_metrics['recall']:.4f} | Precision: {ocsvm_val_metrics['precision']:.4f}")

    # ==========================================================================
    # Model 3: Symmetric Keras Autoencoder
    # ==========================================================================
    print("\n" + "-" * 85)
    print("3. TRAINING SYMMETRIC KERAS AUTOENCODER (EarlyStopping on Val Benign Loss)")
    print("-" * 85)
    ae_model, ae_time = train_autoencoder(
        X_train_benign=X_train_benign,
        X_val_benign=X_val_benign,
        epochs=30,
        batch_size=512,
        random_state=RANDOM_SEED,
    )
    training_times["autoencoder"] = round(ae_time, 2)
    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    ae_model.save(ae_path)
    logger.info("Saved Autoencoder model to %s", ae_path)

    # Threshold calibration
    ae_thresholds = calibrate_autoencoder_thresholds(ae_model, X_val_benign)
    thresh_path = os.path.join(MODELS_DIR, "autoencoder_thresholds.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(ae_thresholds, f, indent=4)
    logger.info("Saved Autoencoder thresholds to %s", thresh_path)

    val_scores_ae = compute_reconstruction_error(ae_model, X_val)
    test_scores_ae = compute_reconstruction_error(ae_model, X_test)
    val_scores_df["score_autoencoder"] = val_scores_ae
    test_scores_df["score_autoencoder"] = test_scores_ae

    # Evaluate with all 3 thresholds (p95, p98, p99)
    ae_p95_metrics = evaluate_binary_predictions(y_val_bin, val_scores_ae, threshold=ae_thresholds["p95"])
    ae_p98_metrics = evaluate_binary_predictions(y_val_bin, val_scores_ae, threshold=ae_thresholds["p98"])
    ae_p99_metrics = evaluate_binary_predictions(y_val_bin, val_scores_ae, threshold=ae_thresholds["p99"])

    val_metrics_report["Autoencoder (p95 threshold)"] = ae_p95_metrics
    val_metrics_report["Autoencoder (p98 threshold)"] = ae_p98_metrics
    val_metrics_report["Autoencoder (p99 threshold)"] = ae_p99_metrics

    print(f"  [Trained in {ae_time:.2f}s] Autoencoder ROC-AUC: {ae_p95_metrics['roc_auc']:.4f}")
    print(f"    - p95 (thresh={ae_thresholds['p95']:.4f}): F1={ae_p95_metrics['f1']:.4f} | Recall={ae_p95_metrics['recall']:.4f} | Precision={ae_p95_metrics['precision']:.4f}")
    print(f"    - p98 (thresh={ae_thresholds['p98']:.4f}): F1={ae_p98_metrics['f1']:.4f} | Recall={ae_p98_metrics['recall']:.4f} | Precision={ae_p98_metrics['precision']:.4f}")
    print(f"    - p99 (thresh={ae_thresholds['p99']:.4f}): F1={ae_p99_metrics['f1']:.4f} | Recall={ae_p99_metrics['recall']:.4f} | Precision={ae_p99_metrics['precision']:.4f}")

    # ==========================================================================
    # Model 4: Random Forest Baseline
    # ==========================================================================
    print("\n" + "-" * 85)
    print("4. TRAINING RANDOM FOREST BASELINE (on val_train, class_weight='balanced')")
    print("-" * 85)
    rf_model, rf_time = train_random_forest(X_val_train, y_val_train, random_state=RANDOM_SEED)
    training_times["random_forest"] = round(rf_time, 2)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))

    val_scores_rf = rf_model.predict_proba(X_val)[:, 1]
    test_scores_rf = rf_model.predict_proba(X_test)[:, 1]
    val_scores_df["score_random_forest"] = val_scores_rf
    test_scores_df["score_random_forest"] = test_scores_rf

    # Evaluate RF on val_tune (independent holdout of val)
    val_tune_scores_rf = rf_model.predict_proba(X_val_tune)[:, 1]
    rf_val_metrics = evaluate_binary_predictions(y_val_tune, val_tune_scores_rf, threshold=0.5)
    val_metrics_report["Random Forest (val holdout)"] = rf_val_metrics
    print(f"  [Trained in {rf_time:.2f}s] Val Tune F1: {rf_val_metrics['f1']:.4f} | ROC-AUC: {rf_val_metrics['roc_auc']:.4f} | Recall: {rf_val_metrics['recall']:.4f} | Precision: {rf_val_metrics['precision']:.4f}")

    # ==========================================================================
    # Model 5: XGBoost Baseline
    # ==========================================================================
    print("\n" + "-" * 85)
    print("5. TRAINING XGBOOST BASELINE (on val_train, scale_pos_weight)")
    print("-" * 85)
    xgb_model, xgb_time = train_xgboost(X_val_train, y_val_train, random_state=RANDOM_SEED)
    training_times["xgboost"] = round(xgb_time, 2)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.joblib"))

    val_scores_xgb = xgb_model.predict_proba(X_val)[:, 1]
    test_scores_xgb = xgb_model.predict_proba(X_test)[:, 1]
    val_scores_df["score_xgboost"] = val_scores_xgb
    test_scores_df["score_xgboost"] = test_scores_xgb

    # Evaluate XGB on val_tune (independent holdout of val)
    val_tune_scores_xgb = xgb_model.predict_proba(X_val_tune)[:, 1]
    xgb_val_metrics = evaluate_binary_predictions(y_val_tune, val_tune_scores_xgb, threshold=0.5)
    val_metrics_report["XGBoost (val holdout)"] = xgb_val_metrics
    print(f"  [Trained in {xgb_time:.2f}s] Val Tune F1: {xgb_val_metrics['f1']:.4f} | ROC-AUC: {xgb_val_metrics['roc_auc']:.4f} | Recall: {xgb_val_metrics['recall']:.4f} | Precision: {xgb_val_metrics['precision']:.4f}")

    # ==========================================================================
    # 6. Save Scores, Metadata, and Configurations
    # ==========================================================================
    val_scores_path = os.path.join(PROCESSED_DATA_DIR, "val_scores.parquet")
    test_scores_path = os.path.join(PROCESSED_DATA_DIR, "test_scores.parquet")
    val_scores_df.to_parquet(val_scores_path, index=False)
    test_scores_df.to_parquet(test_scores_path, index=False)
    logger.info("Saved val scores to %s and test scores to %s", val_scores_path, test_scores_path)

    # Save training times
    times_path = os.path.join(MODELS_DIR, "training_times.json")
    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(training_times, f, indent=4)

    # Save scaler config metadata with clipping bounds
    scaler_config = {
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "scaler_path": os.path.join(MODELS_DIR, "scaler.joblib"),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
    }
    with open(os.path.join(MODELS_DIR, "scaler_config.json"), "w", encoding="utf-8") as f:
        json.dump(scaler_config, f, indent=4)

    # Save validation metrics summary
    val_metrics_path = os.path.join(MODELS_DIR, "val_metrics.json")
    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics_report, f, indent=4)

    # ==========================================================================
    # 7. Print Comparative Validation Performance Table
    # ==========================================================================
    print("\n" + "=" * 85)
    print("VALIDATION SET PERFORMANCE COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Model Name':<32} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'ROC-AUC':>10} | {'Time (s)':>9}")
    print("-" * 85)
    for model_name, metrics in val_metrics_report.items():
        # Match training time
        time_key = "isolation_forest" if "Isolation" in model_name else \
                   "one_class_svm" if "SVM" in model_name else \
                   "autoencoder" if "Autoencoder" in model_name else \
                   "random_forest" if "Random" in model_name else "xgboost"
        t_sec = training_times.get(time_key, 0.0)
        print(f"{model_name:<32} | {metrics['precision']:>10.4f} | {metrics['recall']:>10.4f} | {metrics['f1']:>10.4f} | {metrics['roc_auc']:>10.4f} | {t_sec:>9.2f}")
    print("=" * 85)

    elapsed_all = time.time() - overall_start
    print(f"\nSTAGE 3 COMPLETED SUCCESSFULLY in {elapsed_all:.2f} seconds.")


if __name__ == "__main__":
    main()

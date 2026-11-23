"""
Model definitions, training, scoring, and persistence module for Anomaly NIDS.
Includes:
- Feature clipping helper ([-10, 10])
- Isolation Forest (Benign Train)
- One-Class SVM (Subsampled Benign Train)
- Symmetric Keras Autoencoder (PyTorch backend, EarlyStopping on Benign Val Loss)
- Supervised Baselines: Random Forest & XGBoost with class weighting
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier
import keras

from src.utils import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    setup_logger,
    set_seed,
)
from src.features import load_selected_features

logger = setup_logger("Models")

CLIP_MIN = -10.0
CLIP_MAX = 10.0

def clip_features(X: np.ndarray, min_val: float = CLIP_MIN, max_val: float = CLIP_MAX) -> np.ndarray:
    """Clip scaled feature values to [min_val, max_val] to limit outlier influence."""
    return np.clip(X, min_val, max_val)


# ==============================================================================
# 1. Isolation Forest
# ==============================================================================
def train_isolation_forest(
    X_train: np.ndarray,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 100,
    max_samples: Any = "auto",
    contamination: Any = "auto"
) -> Tuple[IsolationForest, float]:
    """Train Isolation Forest on Benign Train split."""
    logger.info("Training Isolation Forest on %d Benign instances...", len(X_train))
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    elapsed = time.time() - t0
    logger.info("Isolation Forest trained in %.2f seconds.", elapsed)
    return model, elapsed


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Compute anomaly score. Higher score = more anomalous."""
    # score_samples returns negative anomaly score (lower is more abnormal), negate it
    return -model.score_samples(X)


# ==============================================================================
# 2. One-Class SVM
# ==============================================================================
def train_one_class_svm(
    X_train: np.ndarray,
    subsample_size: int = 25000,
    random_state: int = RANDOM_SEED,
    nu: float = 0.05,
    kernel: str = "rbf"
) -> Tuple[OneClassSVM, float]:
    """
    Train One-Class SVM with RBF kernel on a subsample of Benign Train data
    with features clipped to [-10, 10].
    """
    set_seed(random_state)
    n_samples = min(len(X_train), subsample_size)
    indices = np.random.choice(len(X_train), size=n_samples, replace=False)
    X_sub = X_train[indices]

    # Clip features
    X_sub_clipped = clip_features(X_sub)

    logger.info("Training One-Class SVM on %d subsampled Benign instances...", n_samples)
    t0 = time.time()
    model = OneClassSVM(
        kernel=kernel,
        gamma="scale",
        nu=nu,
    )
    model.fit(X_sub_clipped)
    elapsed = time.time() - t0
    logger.info("One-Class SVM trained in %.2f seconds.", elapsed)
    return model, elapsed


def score_one_class_svm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """Compute anomaly score from One-Class SVM with clipped features."""
    X_clipped = clip_features(X)
    # decision_function gives signed distance to hyperplane (negative = anomaly)
    return -model.decision_function(X_clipped)


# ==============================================================================
# 3. Symmetric Keras Autoencoder
# ==============================================================================
def build_symmetric_autoencoder(input_dim: int) -> keras.Model:
    """
    Construct symmetric Keras autoencoder:
    input (d) -> 64 -> 32 -> 16 -> 8 (latent) -> 16 -> 32 -> 64 -> output (d)
    with BatchNormalization, LeakyReLU, and Dropout(0.1).
    """
    inputs = keras.Input(shape=(input_dim,), name="flow_input")

    # Encoder
    x = keras.layers.Dense(64, name="enc_dense_64")(inputs)
    x = keras.layers.BatchNormalization(name="enc_bn_64")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="enc_lrelu_64")(x)
    x = keras.layers.Dropout(0.1, name="enc_drop_64")(x)

    x = keras.layers.Dense(32, name="enc_dense_32")(x)
    x = keras.layers.BatchNormalization(name="enc_bn_32")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="enc_lrelu_32")(x)
    x = keras.layers.Dropout(0.1, name="enc_drop_32")(x)

    x = keras.layers.Dense(16, name="enc_dense_16")(x)
    x = keras.layers.BatchNormalization(name="enc_bn_16")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="enc_lrelu_16")(x)
    x = keras.layers.Dropout(0.1, name="enc_drop_16")(x)

    # Bottleneck latent representation
    latent = keras.layers.Dense(8, name="latent_bottleneck")(x)
    latent = keras.layers.BatchNormalization(name="latent_bn")(latent)
    latent = keras.layers.LeakyReLU(negative_slope=0.2, name="latent_lrelu")(latent)

    # Decoder (Symmetric)
    x = keras.layers.Dense(16, name="dec_dense_16")(latent)
    x = keras.layers.BatchNormalization(name="dec_bn_16")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="dec_lrelu_16")(x)
    x = keras.layers.Dropout(0.1, name="dec_drop_16")(x)

    x = keras.layers.Dense(32, name="dec_dense_32")(x)
    x = keras.layers.BatchNormalization(name="dec_bn_32")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="dec_lrelu_32")(x)
    x = keras.layers.Dropout(0.1, name="dec_drop_32")(x)

    x = keras.layers.Dense(64, name="dec_dense_64")(x)
    x = keras.layers.BatchNormalization(name="dec_bn_64")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.2, name="dec_lrelu_64")(x)
    x = keras.layers.Dropout(0.1, name="dec_drop_64")(x)

    outputs = keras.layers.Dense(input_dim, name="reconstruction_output")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="Symmetric_Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return autoencoder


def train_autoencoder(
    X_train_benign: np.ndarray,
    X_val_benign: np.ndarray,
    epochs: int = 35,
    batch_size: int = 512,
    random_state: int = RANDOM_SEED
) -> Tuple[keras.Model, float]:
    """
    Train symmetric autoencoder on clipped Benign Train set with early stopping
    monitored on validation Benign loss.
    """
    set_seed(random_state)
    input_dim = X_train_benign.shape[1]
    model = build_symmetric_autoencoder(input_dim)

    # Clip features to [-10, 10]
    X_train_clipped = clip_features(X_train_benign)
    X_val_benign_clipped = clip_features(X_val_benign)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )

    logger.info("Training Symmetric Autoencoder on %d Benign instances (Val Benign: %d)...",
                len(X_train_clipped), len(X_val_benign_clipped))
    t0 = time.time()
    model.fit(
        X_train_clipped,
        X_train_clipped,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_benign_clipped, X_val_benign_clipped),
        callbacks=[early_stop],
        verbose=1,
        shuffle=True,
    )
    elapsed = time.time() - t0
    logger.info("Autoencoder trained in %.2f seconds.", elapsed)
    return model, elapsed


def compute_reconstruction_error(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample Mean Squared Error (MSE) reconstruction error."""
    X_clipped = clip_features(X)
    reconstructions = model.predict(X_clipped, batch_size=1024, verbose=0)
    mse = np.mean(np.square(X_clipped - reconstructions), axis=1)
    return mse


def calibrate_autoencoder_thresholds(
    model: keras.Model,
    X_val_benign: np.ndarray
) -> Dict[str, float]:
    """Calculate 95th, 98th, and 99th percentiles of validation Benign reconstruction errors."""
    benign_errors = compute_reconstruction_error(model, X_val_benign)
    thresholds = {
        "p95": float(np.percentile(benign_errors, 95)),
        "p98": float(np.percentile(benign_errors, 98)),
        "p99": float(np.percentile(benign_errors, 99)),
    }
    logger.info(f"Calibrated Autoencoder Thresholds: p95={thresholds['p95']:.5f}, p98={thresholds['p98']:.5f}, p99={thresholds['p99']:.5f}")
    return thresholds


# ==============================================================================
# 4. Supervised Baselines (Random Forest & XGBoost)
# ==============================================================================
def train_random_forest(
    X_val_train: np.ndarray,
    y_val_train: np.ndarray,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 100
) -> Tuple[RandomForestClassifier, float]:
    """Train Random Forest baseline with balanced class weights."""
    logger.info("Training Random Forest on %d instances (balanced class weights)...", len(X_val_train))
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_val_train, y_val_train)
    elapsed = time.time() - t0
    logger.info("Random Forest trained in %.2f seconds.", elapsed)
    return rf, elapsed


def train_xgboost(
    X_val_train: np.ndarray,
    y_val_train: np.ndarray,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 100
) -> Tuple[XGBClassifier, float]:
    """Train XGBoost baseline with scale_pos_weight for imbalance compensation."""
    n_neg = int(np.sum(y_val_train == 0))
    n_pos = int(np.sum(y_val_train == 1))
    scale_pos_weight = float(n_neg / max(1, n_pos))
    logger.info("Training XGBoost on %d instances (scale_pos_weight = %.2f)...", len(X_val_train), scale_pos_weight)

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_val_train, y_val_train)
    elapsed = time.time() - t0
    logger.info("XGBoost trained in %.2f seconds.", elapsed)
    return xgb, elapsed


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1, ROC-AUC, and PR-AUC.
    If threshold is None, searches for the threshold that maximizes F1.
    """
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    if threshold is None:
        # Search for threshold maximizing F1 across candidate percentiles
        percentiles = np.linspace(80, 99.5, 40)
        best_f1 = -1.0
        best_thresh = float(np.median(y_scores))
        for p in percentiles:
            cand_thresh = np.percentile(y_scores, p)
            preds = (y_scores >= cand_thresh).astype(int)
            f = f1_score(y_true, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = float(cand_thresh)
        threshold = best_thresh

    y_pred = (y_scores >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

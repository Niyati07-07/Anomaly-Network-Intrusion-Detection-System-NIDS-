"""
Preprocessing and Dataset Splitting Module for Anomaly NIDS.
Handles:
- Missing and infinite value cleaning
- Constant and collinear feature elimination
- Binary and multi-class label generation
- Semi-supervised train/validation/test split
- Strict leakage-free scaling (fitted solely on Benign Train split)
- Parquet split serialization
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler

from src.utils import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    setup_logger,
    ensure_dirs,
    set_seed,
)
from src.features import (
    find_constant_features,
    find_collinear_features,
    save_selected_features,
)

logger = setup_logger("Preprocessing")

def clean_missing_and_inf(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Replace +Inf and -Inf values with NaN, then drop all rows with NaN.
    Returns cleaned DataFrame and number of dropped rows.
    """
    initial_rows = len(df)
    logger.info(f"Checking for infinite and missing values across {initial_rows:,} rows...")

    # Replace inf with NaN
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna().reset_index(drop=True)

    dropped = initial_rows - len(df_clean)
    logger.info(f"Dropped {dropped:,} rows containing NaN / Inf. Remaining clean rows: {len(df_clean):,}")
    return df_clean, dropped


def create_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    Add binary target 'Label_Binary' (0 = BENIGN, 1 = ATTACK) while preserving
    the multi-class 'Label' column.
    """
    df = df.copy()
    df["Label_Binary"] = (df[label_col] != "BENIGN").astype(int)
    logger.info(f"Created 'Label_Binary': {df['Label_Binary'].value_counts().to_dict()}")
    return df


def select_features(
    df: pd.DataFrame,
    label_cols: List[str] = ["Label", "Label_Binary"],
    corr_threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Identify and remove constant features (8 constant columns) and
    highly collinear features (|r| > 0.95).
    """
    all_numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in label_cols]
    n_initial = len(all_numeric)
    logger.info(f"Initial numeric features count: {n_initial}")

    # 1. Drop constant features
    constant_features = find_constant_features(df, all_numeric)
    after_const = [c for c in all_numeric if c not in constant_features]
    logger.info(f"Features after dropping {len(constant_features)} constant features: {len(after_const)}")

    # 2. Drop collinear features (|r| > corr_threshold)
    to_drop_corr, selected_features, _ = find_collinear_features(
        df, after_const, threshold=corr_threshold
    )
    logger.info(f"Features after dropping {len(to_drop_corr)} collinear features: {len(selected_features)}")

    # Retain selected features + label columns
    keep_cols = selected_features + [c for c in label_cols if c in df.columns]
    df_selected = df[keep_cols].copy()

    # Save selected feature list to models/
    save_selected_features(selected_features)

    feature_stats = {
        "n_initial_features": n_initial,
        "constant_features_dropped": constant_features,
        "n_constant_dropped": len(constant_features),
        "collinear_features_dropped": to_drop_corr,
        "n_collinear_dropped": len(to_drop_corr),
        "selected_features": selected_features,
        "n_final_features": len(selected_features),
    }

    return df_selected, selected_features, feature_stats


def semi_supervised_split(
    df: pd.DataFrame,
    label_col: str = "Label",
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset in a semi-supervised configuration:
    - Train: 60% of BENIGN traffic only (0% attacks).
    - Validation: 20% of BENIGN + 50% of each attack class.
    - Test: 20% of BENIGN + remaining 50% of each attack class.
    """
    logger.info("Executing semi-supervised stratified split (seed = %d)...", random_state)

    benign_df = df[df[label_col] == "BENIGN"].copy()
    attack_df = df[df[label_col] != "BENIGN"].copy()

    n_benign = len(benign_df)
    n_attack = len(attack_df)
    logger.info(f"Total Benign rows: {n_benign:,} | Total Attack rows: {n_attack:,}")

    # 1. Benign split (60% Train, 20% Val, 20% Test)
    train_benign = benign_df.sample(frac=0.60, random_state=random_state)
    rem_benign = benign_df.drop(train_benign.index)

    val_benign = rem_benign.sample(frac=0.50, random_state=random_state)
    test_benign = rem_benign.drop(val_benign.index)

    # 2. Attack split (0% Train, 50% Val, 50% Test per attack class)
    val_attack_parts = []
    test_attack_parts = []

    for attack_name, group in attack_df.groupby(label_col):
        n_grp = len(group)
        # Allocate 50% to validation, ensure at least 1 row if >= 2 rows
        n_val = int(np.ceil(n_grp * 0.50))
        n_val = min(n_val, n_grp)

        val_grp = group.sample(n=n_val, random_state=random_state)
        test_grp = group.drop(val_grp.index)

        val_attack_parts.append(val_grp)
        test_attack_parts.append(test_grp)

    val_attack = pd.concat(val_attack_parts, axis=0)
    test_attack = pd.concat(test_attack_parts, axis=0)

    # Combine benign and attack subsets
    train_df = train_benign.reset_index(drop=True)
    val_df = pd.concat([val_benign, val_attack], axis=0).reset_index(drop=True)
    test_df = pd.concat([test_benign, test_attack], axis=0).reset_index(drop=True)

    # Shuffle val and test sets to prevent ordering bias
    val_df = val_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    logger.info(f"Train split shape: {train_df.shape} (100% Benign)")
    logger.info(f"Validation split shape: {val_df.shape} (Benign: {len(val_benign):,}, Attack: {len(val_attack):,})")
    logger.info(f"Test split shape: {test_df.shape} (Benign: {len(test_benign):,}, Attack: {len(test_attack):,})")

    return train_df, val_df, test_df


def fit_and_scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_save_path: str = os.path.join(MODELS_DIR, "scaler.joblib")
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler solely on the Benign Train set to strictly prevent data leakage.
    Transform Train, Validation, and Test feature sets.
    """
    logger.info("Fitting StandardScaler SOLELY on Benign Train split (strict leakage prevention)...")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"Saved fitted scaler to {scaler_save_path}")

    # Transform features while preserving labels
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = PROCESSED_DATA_DIR
) -> Dict[str, str]:
    """
    Serialize the processed splits to Parquet format in data/processed/.
    """
    ensure_dirs()
    paths = {
        "train": os.path.join(output_dir, "train.parquet"),
        "val": os.path.join(output_dir, "val.parquet"),
        "test": os.path.join(output_dir, "test.parquet"),
    }

    for name, p in paths.items():
        data = {"train": train_df, "val": val_df, "test": test_df}[name]
        data.to_parquet(p, index=False)
        logger.info(f"Saved {name} split to {p} ({len(data):,} rows, {data.shape[1]} cols)")

    return paths


def run_stage2_pipeline(
    input_parquet: str = os.path.join(PROCESSED_DATA_DIR, "cicids2017_sampled_20pct.parquet"),
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Execute end-to-end Stage 2 preprocessing pipeline:
    1. Clean Inf and NaN
    2. Drop constant (8) and collinear features (|r| > 0.95)
    3. Generate binary and multi-class labels
    4. Perform semi-supervised 60/20/20 split (attacks 50/50 in val/test)
    5. Fit scaler on Benign train only and serialize
    6. Save splits to Parquet
    """
    ensure_dirs()
    set_seed(random_state)

    logger.info(f"Reading input dataset from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    n_raw_rows = len(df)

    # 1. Clean missing and infinite values
    df_clean, dropped_nan = clean_missing_and_inf(df)

    # 2. Add binary labels
    df_labeled = create_labels(df_clean, label_col="Label")

    # 3. Feature selection (constant + collinear > 0.95)
    df_selected, selected_features, feature_stats = select_features(
        df_labeled, label_cols=["Label", "Label_Binary"], corr_threshold=0.95
    )

    # 4. Semi-supervised split
    train_raw, val_raw, test_raw = semi_supervised_split(
        df_selected, label_col="Label", random_state=random_state
    )

    # 5. Fit scaler on Benign Train split ONLY and transform
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    train_scaled, val_scaled, test_scaled, _ = fit_and_scale_splits(
        train_raw, val_raw, test_raw, feature_cols=selected_features, scaler_save_path=scaler_path
    )

    # 6. Save splits
    split_paths = save_splits(train_scaled, val_scaled, test_scaled)

    # 7. Collect per-class counts for validation and test
    val_class_counts = val_scaled["Label"].value_counts().to_dict()
    test_class_counts = test_scaled["Label"].value_counts().to_dict()

    summary = {
        "raw_rows": n_raw_rows,
        "clean_rows": len(df_selected),
        "dropped_nan_inf_rows": dropped_nan,
        "feature_selection": feature_stats,
        "shapes": {
            "train": train_scaled.shape,
            "val": val_scaled.shape,
            "test": test_scaled.shape,
        },
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "scaler_path": scaler_path,
        "split_paths": split_paths,
    }

    # Save summary metadata
    summary_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved preprocessing summary to {summary_path}")

    return summary

"""
Feature engineering and feature selection module for Anomaly NIDS.
Handles detection and removal of constant (zero-variance) features and
highly collinear features (|r| > 0.95).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from src.utils import setup_logger, MODELS_DIR

logger = setup_logger("Features")

# The 8 constant features with zero variance detected across CICIDS-2017
KNOWN_CONSTANT_FEATURES = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
]

def find_constant_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Identify features with zero variance or nunique <= 1.
    """
    constant_cols = []
    for col in feature_cols:
        col_s = df[col].dropna()
        if col_s.nunique() <= 1 or col_s.std() == 0:
            constant_cols.append(col)

    # Union with known constant features present in df
    all_const = sorted(list(set(constant_cols).union(set(c for c in KNOWN_CONSTANT_FEATURES if c in feature_cols))))
    logger.info(f"Identified {len(all_const)} constant / zero-variance features.")
    return all_const


def find_collinear_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Identify highly collinear features with Pearson absolute correlation > threshold.
    Returns:
        - to_drop: list of features to remove
        - remaining: list of retained features
        - corr_matrix: absolute correlation matrix
    """
    logger.info(f"Calculating correlation matrix for {len(feature_cols)} features (threshold = {threshold})...")
    X = df[feature_cols]
    corr_matrix = X.corr().abs()

    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    remaining = [col for col in feature_cols if col not in to_drop]

    logger.info(f"Identified {len(to_drop)} collinear features (|r| > {threshold}). Retaining {len(remaining)} features.")
    return to_drop, remaining, corr_matrix


def save_selected_features(
    feature_cols: List[str],
    filepath: str = os.path.join(MODELS_DIR, "selected_features.json")
) -> str:
    """Save the ordered list of selected feature names for downstream reproducibility."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)
    logger.info(f"Saved {len(feature_cols)} selected feature names to {filepath}")
    return filepath


def load_selected_features(
    filepath: str = os.path.join(MODELS_DIR, "selected_features.json")
) -> List[str]:
    """Load the ordered list of selected feature names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Selected features file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        features = json.load(f)
    return features

"""
Data loading and ingestion module for CICIDS-2017 dataset.
Handles column name cleaning, character encoding anomalies, label normalization,
and stratified sampling.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from src.utils import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_FRAC,
    RANDOM_SEED,
    setup_logger,
    ensure_dirs,
)

logger = setup_logger("DataLoader")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names:
    - Strip leading and trailing whitespace
    - Remove duplicate columns (e.g. 'Fwd Header Length.1' which duplicates 'Fwd Header Length')
    """
    df = df.copy()
    # Strip whitespace
    df.columns = df.columns.str.strip()

    # Handle duplicated columns (e.g. 'Fwd Header Length.1' from pandas renaming identical headers)
    # Check if duplicate column names exist in index
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Explicitly check for 'Fwd Header Length.1' created by pandas auto-deduplication
    if "Fwd Header Length.1" in df.columns and "Fwd Header Length" in df.columns:
        logger.debug("Dropping redundant 'Fwd Header Length.1' column.")
        df = df.drop(columns=["Fwd Header Length.1"])

    return df


def clean_labels(labels: pd.Series) -> pd.Series:
    """
    Clean and standardize attack label strings:
    - Strip leading and trailing spaces
    - Replace corrupted encoding artifacts (e.g. '', 'ï¿½') with standard hyphens
    """
    cleaned = labels.astype(str).str.strip()

    def _normalize_name(name: str) -> str:
        # Standardize web attack labels corrupted by encoding issues in the raw dataset
        if "Web Attack" in name:
            # Replace replacement character, question mark, or non-ascii between words with hyphen
            name = re.sub(r"Web Attack\s*[^a-zA-Z0-9\s]\s*", "Web Attack - ", name)
            # Remove repeated spaces
            name = re.sub(r"\s+", " ", name).strip()
            return name
        return name

    cleaned = cleaned.apply(_normalize_name)
    return cleaned


# Explicit list of large volume classes to sample with SAMPLE_FRAC
LARGE_CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "DDoS"]
RARE_CLASS_THRESHOLD = 5000  # Any class with fewer rows in full dataset is kept at 100%

def stratified_sample_dataframe(
    df: pd.DataFrame,
    label_col: str = "Label",
    sample_frac: float = 0.2,
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Perform selective stratified sampling:
    - Every class with fewer than 5,000 rows (or not in LARGE_CLASSES) is retained at 100%.
    - Only large classes (BENIGN, DoS Hulk, PortScan, DDoS) are sampled using sample_frac.
    """
    if sample_frac >= 1.0 or sample_frac <= 0.0:
        return df

    sampled_groups: List[pd.DataFrame] = []
    for label_val, group in df.groupby(label_col):
        n_group = len(group)
        # If class is rare (< 5000 rows) or not one of the designated large classes, keep 100%
        if n_group < RARE_CLASS_THRESHOLD and label_val not in LARGE_CLASSES:
            logger.info(f"Retaining 100% of rare class '{label_val}': {n_group:,} rows.")
            sampled_groups.append(group)
        elif label_val in LARGE_CLASSES:
            n_samples = max(1, int(np.ceil(n_group * sample_frac)))
            n_samples = min(n_group, n_samples)
            sampled_group = group.sample(n=n_samples, random_state=random_state)
            logger.info(f"Sampled {n_samples:,}/{n_group:,} ({sample_frac*100:.0f}%) of large class '{label_val}'.")
            sampled_groups.append(sampled_group)
        else:
            # Medium classes >= 5000 that are not in LARGE_CLASSES
            # To be safe and preserve rare evaluation signals, retain 100%
            logger.info(f"Retaining 100% of class '{label_val}': {n_group:,} rows.")
            sampled_groups.append(group)

    result = pd.concat(sampled_groups, axis=0, ignore_index=True)
    return result


def load_single_raw_file(
    file_path: str,
    random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Load a single CICIDS-2017 CSV file with robust encoding fallback.
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Loading {file_name}...")

    encodings = ["utf-8", "latin-1", "cp1252", "ISO-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            logger.debug(f"Successfully loaded {file_name} with encoding {enc}.")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.debug(f"Failed loading {file_name} with {enc}: {e}")
            continue

    if df is None:
        raise ValueError(f"Unable to read file {file_path} with standard encodings.")

    # Clean columns and labels
    df = clean_column_names(df)

    if "Label" not in df.columns:
        raise KeyError(f"'Label' column not found in {file_name}. Columns: {df.columns.tolist()[:5]}...")

    df["Label"] = clean_labels(df["Label"])
    logger.info(f"Loaded {file_name}: {len(df):,} rows.")
    return df


def load_all_raw_data(
    raw_dir: str = RAW_DATA_DIR,
    sample_frac: float = SAMPLE_FRAC,
    random_state: int = RANDOM_SEED,
    save_processed: bool = True
) -> pd.DataFrame:
    """
    Load all 8 CICIDS-2017 CSV files, drop duplicate rows first across the entire dataset,
    then apply selective sampling (100% of rare/medium classes, SAMPLE_FRAC of large classes).
    Optionally save the combined dataset to data/processed/.
    """
    ensure_dirs()
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_dir}")

    logger.info(f"Found {len(csv_files)} dataset files in {raw_dir}.")
    logger.info(f"Sampling configuration: Only large classes {LARGE_CLASSES} sampled at {sample_frac*100:.1f}%.")
    logger.info("All classes < 5,000 rows retained at 100% after deduplication.")

    dfs: List[pd.DataFrame] = []
    for f in csv_files:
        df_part = load_single_raw_file(f, random_state=random_state)
        dfs.append(df_part)

    combined_full = pd.concat(dfs, axis=0, ignore_index=True)
    raw_total_rows = len(combined_full)
    logger.info(f"Total raw rows across all 8 files: {raw_total_rows:,}")

    # 1. Drop duplicates FIRST before sampling so rare classes are not wasted on duplicates
    logger.info("Deduplicating full dataset before sampling...")
    combined_dedup = combined_full.drop_duplicates()
    dedup_rows = len(combined_dedup)
    dropped_dups = raw_total_rows - dedup_rows
    logger.info(f"Dropped {dropped_dups:,} duplicate rows. Remaining unique rows: {dedup_rows:,}")

    # 2. Apply selective sampling
    sampled_df = stratified_sample_dataframe(
        combined_dedup,
        label_col="Label",
        sample_frac=sample_frac,
        random_state=random_state
    )
    logger.info(f"Sampled dataset shape: {sampled_df.shape} ({len(sampled_df):,} rows, {sampled_df.shape[1]} columns)")

    if save_processed:
        pct_tag = int(sample_frac * 100) if sample_frac < 1.0 else 100
        output_parquet = os.path.join(PROCESSED_DATA_DIR, f"cicids2017_sampled_{pct_tag}pct.parquet")
        try:
            sampled_df.to_parquet(output_parquet, index=False)
            logger.info(f"Saved updated sampled dataset to Parquet: {output_parquet}")
        except Exception as e:
            logger.warning(f"Parquet save failed ({e}), saving to CSV...")
            sampled_df.to_csv(output_parquet.replace(".parquet", ".csv.gz"), index=False, compression="gzip")

    return sampled_df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive structural inspection of the dataset.
    """
    logger.info("Computing dataset inspection summary...")
    n_rows, n_cols = df.shape
    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Label distributions
    label_counts = df["Label"].value_counts()
    label_pcts = df["Label"].value_counts(normalize=True) * 100

    binary_label = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    binary_counts = binary_label.value_counts()
    binary_pcts = binary_label.value_counts(normalize=True) * 100

    # Inf and NaN checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    nan_counts = {}
    for col in numeric_cols:
        col_series = df[col]
        inf_c = int(np.isinf(col_series).sum())
        nan_c = int(col_series.isna().sum())
        if inf_c > 0:
            inf_counts[col] = inf_c
        if nan_c > 0:
            nan_counts[col] = nan_c

    # Duplicates
    dup_rows = int(df.duplicated().sum())

    # Constant columns (std == 0)
    constant_cols = []
    for col in numeric_cols:
        # replace inf with nan temporarily for std calculation
        clean_s = df[col].replace([np.inf, -np.inf], np.nan)
        if clean_s.std() == 0 or clean_s.nunique() <= 1:
            constant_cols.append(col)

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(mem_usage_mb, 2),
        "binary_distribution": {
            "counts": binary_counts.to_dict(),
            "percentages": {k: round(v, 2) for k, v in binary_pcts.items()}
        },
        "multiclass_distribution": {
            "counts": label_counts.to_dict(),
            "percentages": {k: round(v, 4) for k, v in label_pcts.items()}
        },
        "infinite_values": inf_counts,
        "nan_values": nan_counts,
        "duplicate_rows": dup_rows,
        "constant_columns": constant_cols,
    }

    return summary

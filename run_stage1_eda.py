"""
Stage 1 Runner: Data Loading, Cleaning, Stratified Sampling, and EDA.
Demonstrates and validates:
- Space-trimmed column names
- Dropped duplicate 'Fwd Header Length.1'
- Encoding normalization (Web Attack labels)
- Stratified 20% sample preserving rare attacks (Heartbleed, Infiltration)
- Inf and NaN detection
- Constant column identification
- Publication-quality EDA plots in reports/figures/
"""

import sys
import os
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import (
    RANDOM_SEED,
    SAMPLE_FRAC,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    set_seed,
    ensure_dirs,
    setup_logger,
)
from src.data_loader import load_all_raw_data
from src.eda import run_eda

logger = setup_logger("Stage1-Runner")

def main():
    start_time = time.time()
    ensure_dirs()
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("STAGE 1: CICIDS-2017 DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"Random Seed: {RANDOM_SEED} | Stratified Sample Fraction: {SAMPLE_FRAC * 100:.1f}%")
    print("=" * 80)

    # 1. Check if processed parquet already exists to allow fast reuse or load fresh
    pct_tag = int(SAMPLE_FRAC * 100) if SAMPLE_FRAC < 1.0 else 100
    parquet_path = os.path.join(PROCESSED_DATA_DIR, f"cicids2017_sampled_{pct_tag}pct.parquet")

    logger.info("Loading and sampling all 8 CICIDS-2017 raw CSV files...")
    df = load_all_raw_data(sample_frac=SAMPLE_FRAC, random_state=RANDOM_SEED, save_processed=True)

    print("\n" + "-" * 80)
    print("1. DATASET DIMENSIONS & MEMORY")
    print("-" * 80)
    print(f"Total Rows Loaded (Sampled): {len(df):,}")
    print(f"Total Columns: {df.shape[1]}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory Usage: {mem_mb:.2f} MB")

    # 2. Run EDA
    print("\n" + "-" * 80)
    print("2. RUNNING EXPLORATORY DATA ANALYSIS & GENERATING FIGURES")
    print("-" * 80)
    eda_results = run_eda(df)
    summary = eda_results["summary"]

    # 3. Print Key Findings
    print("\n" + "-" * 80)
    print("3. BINARY DISTRIBUTION (BENIGN vs ATTACK)")
    print("-" * 80)
    for cls_name, count in summary["binary_distribution"]["counts"].items():
        pct = summary["binary_distribution"]["percentages"][cls_name]
        print(f"  {cls_name:<10}: {count:>8,} ({pct:6.2f}%)")

    print("\n" + "-" * 80)
    print("4. MULTI-CLASS DISTRIBUTION (PRESERVED ACROSS ALL 15 CLASSES)")
    print("-" * 80)
    for cls_name, count in summary["multiclass_distribution"]["counts"].items():
        pct = summary["multiclass_distribution"]["percentages"][cls_name]
        print(f"  {cls_name:<30}: {count:>8,} ({pct:8.4f}%)")

    print("\n" + "-" * 80)
    print("5. DATA INTEGRITY & DATASET ISSUES DETECTED")
    print("-" * 80)
    print(f"Duplicate Rows Count : {summary['duplicate_rows']:,}")
    print(f"Infinite Values (+/-Inf):")
    for col, count in summary["infinite_values"].items():
        print(f"  - {col}: {count:,} occurrences")

    print(f"Missing (NaN) Values:")
    for col, count in summary["nan_values"].items():
        print(f"  - {col}: {count:,} occurrences")

    print(f"Zero-Variance / Constant Columns ({len(summary['constant_columns'])} detected):")
    for col in sorted(summary["constant_columns"]):
        print(f"  - {col}")

    print("\n" + "-" * 80)
    print("6. GENERATED FIGURES IN reports/figures/")
    print("-" * 80)
    for fig_name, fig_path in eda_results["figures"].items():
        if fig_path and os.path.exists(fig_path):
            file_sz_kb = os.path.getsize(fig_path) / 1024
            print(f"  [OK] {fig_name:<25}: {fig_path} ({file_sz_kb:.1f} KB)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Stage 2 Runner: Preprocessing, Feature Selection, Semi-Supervised Splitting, and Scaling.
Demonstrates:
- NaN and Inf cleaning
- Constant (8) and Collinear (|r| > 0.95) feature elimination
- Binary (0=benign, 1=attack) and multi-class target retention
- Semi-supervised split (Train=60% benign, Val=20% benign + 50% attacks, Test=20% benign + 50% attacks)
- Leakage-free StandardScaler fitted solely on Train set
- Parquet split persistence
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    set_seed,
    ensure_dirs,
    setup_logger,
)
from src.preprocessing import run_stage2_pipeline

logger = setup_logger("Stage2-Runner")

def main():
    start_time = time.time()
    ensure_dirs()
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("STAGE 2: PREPROCESSING, FEATURE SELECTION, SPLITTING & LEAKAGE-FREE SCALING")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 80)

    parquet_input = os.path.join(PROCESSED_DATA_DIR, "cicids2017_sampled_20pct.parquet")
    summary = run_stage2_pipeline(input_parquet=parquet_input, random_state=RANDOM_SEED)

    feat_stats = summary["feature_selection"]
    shapes = summary["shapes"]
    val_counts = summary["val_class_counts"]
    test_counts = summary["test_class_counts"]

    print("\n" + "-" * 80)
    print("1. FEATURE SELECTION SUMMARY")
    print("-" * 80)
    print(f"Initial Numeric Features        : {feat_stats['n_initial_features']}")
    print(f"Constant Features Dropped ({feat_stats['n_constant_dropped']})   : {feat_stats['constant_features_dropped']}")
    print(f"Features after Constant Drop    : {feat_stats['n_initial_features'] - feat_stats['n_constant_dropped']}")
    print(f"Collinear Features Dropped (|r| > 0.95, {feat_stats['n_collinear_dropped']}):")
    for f in feat_stats["collinear_features_dropped"]:
        print(f"  - {f}")
    print(f"Final Selected Features Count   : {feat_stats['n_final_features']}")

    print("\n" + "-" * 80)
    print("2. DATASET SPLIT SHAPES (SEMI-SUPERVISED SETUP)")
    print("-" * 80)
    print(f"Train Set Shape       : {shapes['train']}  (Features: {shapes['train'][1] - 2} + Label + Label_Binary)")
    print(f"  -> Traffic Type     : 100% BENIGN only (Strict Semi-Supervised training)")
    print(f"Validation Set Shape  : {shapes['val']}  (Features: {shapes['val'][1] - 2} + Label + Label_Binary)")
    print(f"Test Set Shape        : {shapes['test']}  (Features: {shapes['test'][1] - 2} + Label + Label_Binary)")

    print("\n" + "-" * 80)
    print("3. PER-CLASS COUNTS IN VALIDATION AND TEST SPLITS")
    print("-" * 80)
    all_classes = sorted(list(set(list(val_counts.keys()) + list(test_counts.keys()))))
    print(f"{'Class Name':<30} | {'Validation Count':>16} | {'Test Count':>12} | {'Total':>10}")
    print("-" * 76)
    val_total = 0
    test_total = 0
    for cls in all_classes:
        vc = val_counts.get(cls, 0)
        tc = test_counts.get(cls, 0)
        val_total += vc
        test_total += tc
        print(f"{cls:<30} | {vc:>16,} | {tc:>12,} | {vc + tc:>10,}")
    print("-" * 76)
    print(f"{'TOTAL ROWS':<30} | {val_total:>16,} | {test_total:>12,} | {val_total + test_total:>10,}")

    print("\n" + "-" * 80)
    print("4. SERIALIZED ARTIFACTS VERIFICATION")
    print("-" * 80)
    print(f"Fitted Scaler    : {summary['scaler_path']} (Exists: {os.path.exists(summary['scaler_path'])})")
    for split_name, path in summary["split_paths"].items():
        sz_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{split_name.capitalize():<16} : {path} ({sz_mb:.2f} MB)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"STAGE 2 COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()

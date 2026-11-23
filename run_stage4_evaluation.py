"""
Stage 4 Runner: Comprehensive Test Evaluation & Comparative Analysis.
Executes:
1. Strict Test Evaluation across all models with fixed validation thresholds
2. Confusion Matrices (06_confusion_matrices.png)
3. Multi-Model ROC & PR Curves (07_multi_model_roc_curves.png, 08_multi_model_pr_curves.png)
4. Per-Attack Detection Rate Table & Heatmap (reports/per_attack_detection.csv, 09_per_attack_detection_heatmap.png)
5. Autoencoder Threshold Sensitivity Analysis (10_autoencoder_threshold_analysis.png)
6. Leave-One-Attack-Out Zero-Day Experiment (11_leave_one_out_comparison.png)
7. Final Comparison Table (reports/final_comparison.csv & .md)
8. Comprehensive Diagnostic Analysis Report (reports/analysis.md)
"""

import sys
import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    set_seed,
    ensure_dirs,
    setup_logger,
)
from src.features import load_selected_features
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrices,
    plot_multi_model_roc_pr_curves,
    compute_per_attack_detection,
    plot_per_attack_heatmap,
    plot_autoencoder_threshold_analysis,
)

logger = setup_logger("Stage4-Runner")

def main():
    start_time = time.time()
    ensure_dirs()
    set_seed(RANDOM_SEED)

    print("=" * 95)
    print("STAGE 4: COMPREHENSIVE TEST EVALUATION & BENCHMARKING (test.parquet ONLY)")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 95)

    # 1. Load precomputed test scores & ground truth
    test_scores_path = os.path.join(PROCESSED_DATA_DIR, "test_scores.parquet")
    logger.info("Loading test scores from %s...", test_scores_path)
    test_df = pd.read_parquet(test_scores_path)
    y_test_bin = test_df["Label_Binary"].values
    y_test_multi = test_df["Label"]

    # Load thresholds from validation stage
    with open(os.path.join(MODELS_DIR, "val_metrics.json"), "r") as f:
        val_metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "autoencoder_thresholds.json"), "r") as f:
        ae_thresholds = json.load(f)
    with open(os.path.join(MODELS_DIR, "training_times.json"), "r") as f:
        training_times = json.load(f)

    # Calculate Autoencoder val-F1-optimal threshold on validation scores
    val_scores_path = os.path.join(PROCESSED_DATA_DIR, "val_scores.parquet")
    val_scores_df = pd.read_parquet(val_scores_path)
    val_y_bin = val_scores_df["Label_Binary"].values
    val_ae_scores = val_scores_df["score_autoencoder"].values

    best_ae_f1 = -1.0
    ae_val_f1_th = float(np.percentile(val_ae_scores, 95))
    for p in np.linspace(50, 99.5, 200):
        cand_th = np.percentile(val_ae_scores, p)
        f = f1_score(val_y_bin, (val_ae_scores >= cand_th).astype(int), zero_division=0)
        if f > best_ae_f1:
            best_ae_f1 = f
            ae_val_f1_th = float(cand_th)

    logger.info(f"Autoencoder Val-F1-Optimal Threshold: {ae_val_f1_th:.5f} (Val F1: {best_ae_f1:.4f})")

    # Define models, score columns, and their fixed validation thresholds
    model_configs = {
        "Isolation Forest": {
            "scores": test_df["score_isolation_forest"].values,
            "threshold": val_metrics["Isolation Forest"]["threshold"],
            "time_key": "isolation_forest",
            "is_anomaly": True,
        },
        "One-Class SVM": {
            "scores": test_df["score_one_class_svm"].values,
            "threshold": val_metrics["One-Class SVM"]["threshold"],
            "time_key": "one_class_svm",
            "is_anomaly": True,
        },
        "Autoencoder (p95)": {
            "scores": test_df["score_autoencoder"].values,
            "threshold": ae_thresholds["p95"],
            "time_key": "autoencoder",
            "is_anomaly": True,
        },
        "Autoencoder (p98)": {
            "scores": test_df["score_autoencoder"].values,
            "threshold": ae_thresholds["p98"],
            "time_key": "autoencoder",
            "is_anomaly": True,
        },
        "Autoencoder (p99)": {
            "scores": test_df["score_autoencoder"].values,
            "threshold": ae_thresholds["p99"],
            "time_key": "autoencoder",
            "is_anomaly": True,
        },
        "Autoencoder (val-F1-optimal)": {
            "scores": test_df["score_autoencoder"].values,
            "threshold": ae_val_f1_th,
            "time_key": "autoencoder",
            "is_anomaly": True,
        },
        "Random Forest": {
            "scores": test_df["score_random_forest"].values,
            "threshold": 0.5,
            "time_key": "random_forest",
            "is_anomaly": False,
        },
        "XGBoost": {
            "scores": test_df["score_xgboost"].values,
            "threshold": 0.5,
            "time_key": "xgboost",
            "is_anomaly": False,
        },
    }

    # ==========================================================================
    # 1. Compute Test Metrics
    # ==========================================================================
    print("\n" + "-" * 95)
    print("1. COMPUTING TEST SET PERFORMANCE METRICS")
    print("-" * 95)

    test_results = []
    binary_preds = {}

    for m_name, cfg in model_configs.items():
        th = cfg["threshold"]
        scores = cfg["scores"]
        metrics = compute_metrics(y_test_bin, scores, threshold=th)
        binary_preds[m_name] = (scores >= th).astype(int)

        t_train = training_times.get(cfg["time_key"], 0.0)
        res_row = {
            "Model": m_name,
            "Accuracy": round(metrics["accuracy"] * 100, 2),
            "Precision": round(metrics["precision"] * 100, 2),
            "Recall": round(metrics["recall"] * 100, 2),
            "F1-Score": round(metrics["f1"] * 100, 2),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "PR-AUC": round(metrics["pr_auc"], 4),
            "FPR (%)": round(metrics["fpr"] * 100, 2),
            "Threshold": round(th, 5),
            "Train Time (s)": t_train,
        }
        test_results.append(res_row)

    df_comparison = pd.DataFrame(test_results)
    csv_comp_path = os.path.join(REPORTS_DIR, "final_comparison.csv")
    df_comparison.to_csv(csv_comp_path, index=False)
    logger.info("Saved final comparison table to %s", csv_comp_path)

    # Save markdown version
    md_comp_path = os.path.join(REPORTS_DIR, "final_comparison.md")
    with open(md_comp_path, "w", encoding="utf-8") as f:
        f.write("# CICIDS-2017: Final Test Set Model Benchmark\n\n")
        f.write(df_comparison.to_markdown(index=False))
        f.write("\n")
    logger.info("Saved final comparison markdown to %s", md_comp_path)

    # ==========================================================================
    # 2. Confusion Matrices & Curves
    # ==========================================================================
    print("\n" + "-" * 95)
    print("2. GENERATING CONFUSION MATRICES & MULTI-MODEL CURVES")
    print("-" * 95)
    cm_path = plot_confusion_matrices(binary_preds, y_test_bin)
    print(f"  [OK] Confusion matrices saved: {cm_path}")

    # Multi-model ROC and PR curves (use p95 for autoencoder)
    curve_scores = {
        "Isolation Forest": model_configs["Isolation Forest"]["scores"],
        "One-Class SVM": model_configs["One-Class SVM"]["scores"],
        "Autoencoder": model_configs["Autoencoder (p95)"]["scores"],
        "Random Forest": model_configs["Random Forest"]["scores"],
        "XGBoost": model_configs["XGBoost"]["scores"],
    }
    roc_p, pr_p = plot_multi_model_roc_pr_curves(curve_scores, y_test_bin)
    print(f"  [OK] Multi-model ROC curves saved: {roc_p}")
    print(f"  [OK] Multi-model PR curves saved: {pr_p}")

    # ==========================================================================
    # 3. Per-Attack Detection Rate
    # ==========================================================================
    print("\n" + "-" * 95)
    print("3. COMPUTING PER-ATTACK DETECTION RATES ACROSS ALL 14 ATTACK CLASSES")
    print("-" * 95)
    df_per_attack = compute_per_attack_detection(y_test_multi, binary_preds, unstable_threshold=20)
    per_attack_csv = os.path.join(REPORTS_DIR, "per_attack_detection.csv")
    df_per_attack.to_csv(per_attack_csv, index=False)
    print(f"  [OK] Saved per-attack table to {per_attack_csv}")

    per_attack_heatmap_p = plot_per_attack_heatmap(df_per_attack)
    print(f"  [OK] Saved per-attack heatmap to {per_attack_heatmap_p}")

    # ==========================================================================
    # 4. Autoencoder Threshold Sensitivity Analysis
    # ==========================================================================
    print("\n" + "-" * 95)
    print("4. AUTOENCODER THRESHOLD SENSITIVITY ANALYSIS (Precision, Recall, FPR)")
    print("-" * 95)
    ae_errors = model_configs["Autoencoder (p95)"]["scores"]
    ae_thresh_fig = plot_autoencoder_threshold_analysis(y_test_bin, ae_errors, ae_thresholds)
    print(f"  [OK] Saved Autoencoder threshold analysis to {ae_thresh_fig}")

    # ==========================================================================
    # 5. Leave-One-Attack-Out Zero-Day Experiment
    # ==========================================================================
    print("\n" + "-" * 95)
    print("5. LEAVE-ONE-ATTACK-OUT ZERO-DAY GENERALIZATION EXPERIMENT")
    print("-" * 95)
    feature_cols = load_selected_features()
    val_parquet = os.path.join(PROCESSED_DATA_DIR, "val.parquet")
    test_parquet = os.path.join(PROCESSED_DATA_DIR, "test.parquet")
    raw_val_df = pd.read_parquet(val_parquet)
    raw_test_df = pd.read_parquet(test_parquet)

    target_classes = ["Bot", "Web Attack - XSS", "Infiltration", "DoS Hulk"]
    loo_records = []

    for target_attack in target_classes:
        logger.info(f"Running Leave-One-Out for held-out attack '{target_attack}'...")
        # Remove target attack from val_train
        train_pool = raw_val_df[raw_val_df["Label"] != target_attack]
        X_tr = train_pool[feature_cols].values
        y_tr = train_pool["Label_Binary"].values

        # Train Random Forest without knowledge of target attack
        rf_loo = RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_loo.fit(X_tr, y_tr)

        # Test on the specific target attack in test set
        target_mask = (raw_test_df["Label"] == target_attack)
        n_target = int(target_mask.sum())
        X_target = raw_test_df.loc[target_mask, feature_cols].values

        # Supervised RF Recall on Zero-Day Attack
        preds_rf = rf_loo.predict(X_target)
        recall_rf = (np.sum(preds_rf == 1) / n_target) * 100.0

        # Autoencoder (p95) Recall on the same Attack
        ae_scores_target = ae_errors[target_mask]
        recall_ae = (np.sum(ae_scores_target >= ae_thresholds["p95"]) / n_target) * 100.0

        diff = recall_ae - recall_rf
        loo_records.append({
            "Held-Out Attack (Zero-Day)": target_attack,
            "Test Instances": n_target,
            "Supervised RF Recall (%)": round(recall_rf, 2),
            "Unsupervised AE Recall (%)": round(recall_ae, 2),
            "AE Advantage (Delta %)": round(diff, 2),
        })

    df_loo = pd.DataFrame(loo_records)
    loo_csv_path = os.path.join(REPORTS_DIR, "leave_one_out_experiment.csv")
    df_loo.to_csv(loo_csv_path, index=False)
    print(f"  [OK] Saved Leave-One-Out results to {loo_csv_path}")

    # Plot Leave-One-Out Comparison
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_pos = np.arange(len(target_classes))
    bar_width = 0.35

    ax.bar(x_pos - bar_width/2, df_loo["Supervised RF Recall (%)"], width=bar_width,
           label="Supervised Random Forest (Trained without Attack)", color="#e74c3c", edgecolor="black")
    ax.bar(x_pos + bar_width/2, df_loo["Unsupervised AE Recall (%)"], width=bar_width,
           label="Unsupervised Autoencoder (Trained only on Benign)", color="#27ae60", edgecolor="black")

    ax.set_ylabel("Detection Recall (%)")
    ax.set_title("Zero-Day Detection: Leave-One-Attack-Out Experiment\nSupervised Model (No Prior Attack Signatures) vs. Autoencoder", pad=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_classes)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for i in range(len(target_classes)):
        rf_v = df_loo.loc[i, "Supervised RF Recall (%)"]
        ae_v = df_loo.loc[i, "Unsupervised AE Recall (%)"]
        ax.text(i - bar_width/2, rf_v + 1.5, f"{rf_v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + bar_width/2, ae_v + 1.5, f"{ae_v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    loo_fig_path = os.path.join(FIGURES_DIR, "11_leave_one_out_comparison.png")
    fig.savefig(loo_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved Leave-One-Out figure to {loo_fig_path}")

    # ==========================================================================
    # 5b. Leave-One-FAMILY-Out Experiment
    # ==========================================================================
    print("\n" + "-" * 95)
    print("5b. LEAVE-ONE-FAMILY-OUT GENERALIZATION EXPERIMENT")
    print("-" * 95)
    families = {
        "Web Attacks": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
        "DoS Family": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"],
        "Patator (Brute Force)": ["FTP-Patator", "SSH-Patator"],
    }

    fam_records = []
    for fam_name, fam_classes in families.items():
        logger.info(f"Running Leave-Family-Out for family '{fam_name}' {fam_classes}...")
        # Train RF excluding the entire family
        train_pool = raw_val_df[~raw_val_df["Label"].isin(fam_classes)]
        X_tr = train_pool[feature_cols].values
        y_tr = train_pool["Label_Binary"].values

        rf_fam = RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_fam.fit(X_tr, y_tr)

        # Test on the entire held-out family in test set
        fam_mask = raw_test_df["Label"].isin(fam_classes)
        n_fam = int(fam_mask.sum())
        X_fam = raw_test_df.loc[fam_mask, feature_cols].values

        preds_rf = rf_fam.predict(X_fam)
        recall_rf = (np.sum(preds_rf == 1) / n_fam) * 100.0

        ae_scores_fam = ae_errors[fam_mask]
        recall_ae = (np.sum(ae_scores_fam >= ae_thresholds["p95"]) / n_fam) * 100.0

        fam_records.append({
            "Held-Out Attack Family": fam_name,
            "Included Classes": ", ".join(fam_classes),
            "Test Instances": n_fam,
            "Supervised RF Recall (%)": round(recall_rf, 2),
            "Unsupervised AE (p95) Recall (%)": round(recall_ae, 2),
            "Delta (AE - RF)": round(recall_ae - recall_rf, 2),
        })

    df_fam = pd.DataFrame(fam_records)
    fam_csv_path = os.path.join(REPORTS_DIR, "leave_family_out_experiment.csv")
    df_fam.to_csv(fam_csv_path, index=False)
    print(f"  [OK] Saved Leave-Family-Out results to {fam_csv_path}")

    # Plot Leave-Family-Out
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_pos = np.arange(len(families))
    bar_width = 0.35

    ax.bar(x_pos - bar_width/2, df_fam["Supervised RF Recall (%)"], width=bar_width,
           label="Supervised RF (Trained without Entire Family)", color="#e74c3c", edgecolor="black")
    ax.bar(x_pos + bar_width/2, df_fam["Unsupervised AE (p95) Recall (%)"], width=bar_width,
           label="Unsupervised Autoencoder (Trained on Benign only)", color="#27ae60", edgecolor="black")

    ax.set_ylabel("Detection Recall (%)")
    ax.set_title("Zero-Day Family Detection: Leave-One-FAMILY-Out Benchmark\nEvaluating Entirely Novel Attack Families", pad=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{name}\n(n={cnt:,})" for name, cnt in zip(df_fam["Held-Out Attack Family"], df_fam["Test Instances"])])
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for i in range(len(families)):
        rf_v = df_fam.loc[i, "Supervised RF Recall (%)"]
        ae_v = df_fam.loc[i, "Unsupervised AE (p95) Recall (%)"]
        ax.text(i - bar_width/2, rf_v + 1.5, f"{rf_v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + bar_width/2, ae_v + 1.5, f"{ae_v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fam_fig_path = os.path.join(FIGURES_DIR, "12_leave_family_out_comparison.png")
    fig.savefig(fam_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved Leave-Family-Out figure to {fam_fig_path}")

    # ==========================================================================
    # 6. Comprehensive Written Analysis (reports/analysis.md)
    # ==========================================================================
    analysis_text = """# Comprehensive Analysis: Anomaly-Based NIDS on CICIDS-2017

## 1. Overview of Experimental Findings
In this project, semi-supervised anomaly detection models (Isolation Forest, One-Class SVM, and a Symmetric Keras Autoencoder) trained strictly on Benign network traffic were benchmarked against supervised classifiers (Random Forest and XGBoost) across 140,048 test flows containing 14 distinct attack classes.

## 2. Thresholding Methodology
- **Autoencoder Percentiles (p95, p98, p99)**: Derived strictly from the empirical reconstruction error distribution of **validation Benign traffic only**. This represents true zero-knowledge anomaly thresholding with zero exposure to attack labels.
- **Validation-F1 Optimal Thresholds**: The Isolation Forest (0.48646), One-Class SVM (13.94477), and Autoencoder val-F1 (0.03041) thresholds were tuned on validation anomaly scores to maximize the validation F1-score against labeled attack samples. Under this identical evaluation protocol, the Autoencoder achieves an F1-score of **73.41%** (Recall: 82.65%, Precision: 66.03%) on the test set, substantially outperforming Isolation Forest (51.55%) and One-Class SVM (64.63%).

## 3. Attacks Well-Detected by Anomaly Models
Unsupervised models—particularly the Symmetric Autoencoder—excelled at detecting volumetric and connection-disrupting intrusions:
- **DoS Hulk (89.2%), DoS Slowhttptest (94.1%), DoS GoldenEye (73.2%), and DDoS (62.9%)**: These attacks alter fundamental transport-layer statistics (extreme packet transmission rates, distorted forward/backward byte ratios, and anomalous inter-arrival times). These violate the reconstruction manifold learned by the Autoencoder, generating high reconstruction errors ($MSE \gg p_{99}$).

## 4. Attacks Missed by Anomaly Models and Architectural Explanations
- **PortScan (AE p95 Recall: 5.7%)**: PortScan in CICIDS-2017 consists of single-packet or two-packet probe flows with 0 payload bytes and sub-millisecond durations. In benign network traffic, millions of flows are also lightweight, short-lived connections (e.g., DNS queries, quick SYN checks). Without temporal aggregation or session-window correlation across multiple ports, an individual PortScan flow viewed in isolation lies squarely within the high-density cluster of normal short benign flows.
- **Web Attacks (XSS: 3.4%, Brute Force: 4.6%, SQL Injection: 0.0%)**: Web attacks are application-layer exploits embedded inside syntactically valid HTTP GET and POST requests. At the flow/transport level (flow duration, byte counts, TCP flags), an SQL injection flow closely resembles normal web browsing. Detecting payload-level web attacks requires deep packet inspection (DPI) rather than statistical transport flow metrics.
- **Botnets (2.7%) & Patator Brute Force (<0.3%)**: Stealthy botnet command-and-control and single authentication attempts operate at low volume to avoid statistical detection, mimicking regular client-server exchanges.

## 5. Zero-Day Generalization: Leave-One-Out and Leave-Family-Out Experiments
The leave-one-out experiments offer a nuanced, mixed picture of supervised versus unsupervised generalization:
- **Stealthy Privilege Escalation (Infiltration)**: When Infiltration was removed from Random Forest training, its recall on test Infiltration dropped to **0.00%**. The Autoencoder detected **83.33%** purely from benign divergence. However, this must be interpreted with caution, as Infiltration has only 18 test instances and is statistically unstable.
- **Sibling Class Leakage (Web Attacks)**: When Web Attack - XSS was held out individually, Random Forest still achieved 95.4% recall. This occurred because Web Attack - Brute Force remained in training, allowing the tree to generalize across sibling classes sharing similar HTTP server ports and payload lengths.
- **Leave-One-FAMILY-Out Findings**: When the entire **Web Attack family** was held out together, Random Forest recall collapsed to **0.00%**. Similarly, when the **Patator family** (FTP and SSH) was held out, Random Forest recall dropped to **0.00%**. On the **DoS family**, Random Forest retained 54.6% recall due to volumetric overlap with other traffic, while the Autoencoder maintained **87.2%** recall across the entire DoS family without having been trained on any attack signatures.

## 6. Limitations and Evaluation Caveats
1. **Downsampling and Base Rate Fallacy**: In the raw CICIDS-2017 dataset, attacks constitute ~15-20% of traffic. Selective benign downsampling elevates the test-set attack rate to ~40.2%. Consequently, Precision and PR-AUC reported in this benchmark are optimistic compared to real-world deployments where attacks represent <1% of traffic. The **False Positive Rate (FPR)** and **ROC-AUC** provide more invariant measures of operational performance.
2. **Small-Sample Instability on Rare Classes**: Certain attack classes in CICIDS-2017 contain very few unique instances after deduplication. In our test set, Heartbleed has only 5 instances, Web Attack - Sql Injection has 10, and Infiltration has 18. Detection rates on these classes carry high variance and should be viewed as exploratory rather than statistically conclusive.
"""
    analysis_path = os.path.join(REPORTS_DIR, "analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    logger.info("Saved written analysis to %s", analysis_path)

    # ==========================================================================
    # 7. Print Terminal Output
    # ==========================================================================
    print("\n" + "=" * 95)
    print("FINAL TEST SET PERFORMANCE COMPARISON (test.parquet, N=140,048)")
    print("=" * 95)
    print(df_comparison.to_string(index=False))

    print("\n" + "=" * 95)
    print("PER-ATTACK CLASS DETECTION RECALL (%) ON TEST SET")
    print("=" * 95)
    print(df_per_attack.to_string(index=False))

    print("\n" + "=" * 95)
    print("LEAVE-ONE-ATTACK-OUT (ZERO-DAY) EXPERIMENT RESULTS")
    print("=" * 95)
    print(df_loo.to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 95)
    print(f"STAGE 4 COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print("=" * 95)


if __name__ == "__main__":
    main()

"""
Evaluation, metrics computation, and visualization module for Anomaly NIDS.
Handles:
- Comprehensive test evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR)
- Multi-model ROC and Precision-Recall curve generation
- Multi-model Confusion Matrix visualization
- Per-attack class detection rate table and heatmap
- Autoencoder threshold sensitivity analysis
- Leave-one-attack-out zero-day generalization experiment
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.utils import (
    FIGURES_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    setup_logger,
    ensure_dirs,
)

logger = setup_logger("Evaluation")

# Plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
})

def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and FPR on test set
    using a fixed pre-determined threshold.
    """
    y_pred = (y_scores >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    # Confusion Matrix for FPR
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }


def plot_confusion_matrices(
    model_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: str = os.path.join(FIGURES_DIR, "06_confusion_matrices.png")
) -> str:
    """Plot confusion matrices for all models side-by-side in a multi-panel figure."""
    n_models = len(model_preds)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, (m_name, y_pred) in enumerate(model_preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Annotations with counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
            linewidths=0.8,
            linecolor="gray",
        )
        axes[idx].set_title(m_name, fontweight="bold", fontsize=11)
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    # Hide extra unused subplots
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Test Set Confusion Matrices Across All Models", fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices to %s", output_path)
    return output_path


def plot_multi_model_roc_pr_curves(
    model_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str = FIGURES_DIR
) -> Tuple[str, str]:
    """Plot multi-model ROC and Precision-Recall curves on test set."""
    # 1. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    # 2. PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    colors = {
        "Isolation Forest": "#e67e22",
        "One-Class SVM": "#8e44ad",
        "Autoencoder": "#2980b9",
        "Random Forest": "#27ae60",
        "XGBoost": "#c0392b",
    }

    for name, scores in model_scores.items():
        color = colors.get(name, None)
        # ROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=1.8, color=color)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", linewidth=1.8, color=color)

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.50)")
    ax_roc.set_title("Test Set: Multi-Model ROC Curves", pad=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)
    ax_roc.grid(True, linestyle="--", alpha=0.6)
    fig_roc.tight_layout()
    roc_path = os.path.join(output_dir, "07_multi_model_roc_curves.png")
    fig_roc.savefig(roc_path, bbox_inches="tight")
    plt.close(fig_roc)

    # PR formatting
    baseline_pr = np.mean(y_true)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], "k--", alpha=0.6, label=f"No Skill ({baseline_pr:.2f})")
    ax_pr.set_title("Test Set: Multi-Model Precision-Recall Curves", pad=12, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])
    ax_pr.legend(loc="lower left", frameon=True, fontsize=9)
    ax_pr.grid(True, linestyle="--", alpha=0.6)
    fig_pr.tight_layout()
    pr_path = os.path.join(output_dir, "08_multi_model_pr_curves.png")
    fig_pr.savefig(pr_path, bbox_inches="tight")
    plt.close(fig_pr)

    logger.info("Saved ROC curve to %s and PR curve to %s", roc_path, pr_path)
    return roc_path, pr_path


def compute_per_attack_detection(
    y_true_multi: pd.Series,
    model_preds: Dict[str, np.ndarray],
    unstable_threshold: int = 20
) -> pd.DataFrame:
    """
    Compute detection rate (recall) for each attack class across all models.
    Marks classes with fewer than unstable_threshold samples as unstable.
    """
    attack_classes = sorted([c for c in y_true_multi.unique() if c != "BENIGN"])
    records = []

    for cls in attack_classes:
        mask = (y_true_multi == cls)
        count = int(mask.sum())
        is_unstable = count < unstable_threshold

        row = {
            "Attack Class": cls,
            "Test Count": count,
            "Status": "Unstable (<20)" if is_unstable else "Stable",
        }

        for m_name, preds in model_preds.items():
            cls_preds = preds[mask]
            detected = int(np.sum(cls_preds == 1))
            recall = float(detected / count) if count > 0 else 0.0
            row[m_name] = round(recall * 100.0, 2)

        records.append(row)

    df_per_attack = pd.DataFrame(records)
    return df_per_attack


def plot_per_attack_heatmap(
    df_per_attack: pd.DataFrame,
    output_path: str = os.path.join(FIGURES_DIR, "09_per_attack_detection_heatmap.png")
) -> str:
    """Render heatmap of per-attack detection rates (recall %) for all models."""
    plot_df = df_per_attack.copy()

    # Create formatted index with sample counts and instability flag
    labels = [
        f"{row['Attack Class']} (n={row['Test Count']})" + (" *" if row["Status"] == "Unstable (<20)" else "")
        for _, row in plot_df.iterrows()
    ]

    metric_cols = [c for c in plot_df.columns if c not in ["Attack Class", "Test Count", "Status"]]
    data_mat = plot_df[metric_cols].values

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data_mat,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        xticklabels=metric_cols,
        yticklabels=labels,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Detection Rate (Recall %)", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Per-Attack Class Detection Rate (%) on Test Set\n(* indicates rare class with < 20 test instances)", pad=15, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-attack detection heatmap to %s", output_path)
    return output_path


def plot_autoencoder_threshold_analysis(
    y_true: np.ndarray,
    reconstruction_errors: np.ndarray,
    thresholds: Dict[str, float],
    output_path: str = os.path.join(FIGURES_DIR, "10_autoencoder_threshold_analysis.png")
) -> str:
    """Plot Precision, Recall, and FPR across varying reconstruction error thresholds."""
    # Evaluate across a wide percentile grid of errors
    p_grid = np.linspace(80, 99.8, 100)
    thresh_grid = [np.percentile(reconstruction_errors, p) for p in p_grid]

    precisions = []
    recalls = []
    fprs = []

    total_benign = int(np.sum(y_true == 0))
    total_attack = int(np.sum(y_true == 1))

    for th in thresh_grid:
        preds = (reconstruction_errors >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, total_attack)
        fpr = fp / max(1, total_benign)

        precisions.append(prec)
        recalls.append(rec)
        fprs.append(fpr)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresh_grid, precisions, label="Precision", color="#2980b9", linewidth=2.0)
    ax.plot(thresh_grid, recalls, label="Recall (TPR)", color="#27ae60", linewidth=2.0)
    ax.plot(thresh_grid, fprs, label="False Positive Rate (FPR)", color="#e74c3c", linewidth=2.0, linestyle="--")

    # Mark p95, p98, p99
    palette = {"p95": "#f39c12", "p98": "#d35400", "p99": "#c0392b"}
    for p_name in ["p95", "p98", "p99"]:
        th_val = thresholds[p_name]
        ax.axvline(
            th_val,
            color=palette[p_name],
            linestyle=":",
            linewidth=1.8,
            label=f"{p_name.upper()} Threshold ({th_val:.4f})",
        )

    ax.set_title("Autoencoder Threshold Sensitivity: Precision, Recall & FPR", pad=12, fontweight="bold")
    ax.set_xlabel("Reconstruction Error Threshold (MSE)")
    ax.set_ylabel("Score / Rate")
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc="center right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved autoencoder threshold sensitivity analysis to %s", output_path)
    return output_path

"""
Exploratory Data Analysis (EDA) module for CICIDS-2017.
Generates comprehensive distribution plots, missing/infinite value reports,
correlation heatmaps, and dataset diagnostics saved to reports/figures/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from src.utils import (
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    setup_logger,
    ensure_dirs,
    set_seed,
)
from src.data_loader import inspect_dataset

logger = setup_logger("EDA")

# Set consistent plotting styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
})

def plot_binary_distribution(df: pd.DataFrame, output_dir: str = FIGURES_DIR) -> str:
    """Plot distribution of Benign vs Attack instances."""
    binary_labels = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    counts = binary_labels.value_counts()
    pcts = binary_labels.value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max(counts.values) * 0.01),
            f"{int(height):,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("CICIDS-2017: Binary Traffic Distribution (BENIGN vs. ATTACK)", pad=15)
    ax.set_ylabel("Number of Flow Instances")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "01_class_distribution_binary.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved binary distribution plot to {out_path}")
    return out_path


def plot_multiclass_distribution(df: pd.DataFrame, output_dir: str = FIGURES_DIR) -> str:
    """Plot detailed breakdown of all traffic classes on linear and log scale."""
    counts = df["Label"].value_counts()
    labels = counts.index
    values = counts.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette: distinct color for Benign, gradient for attacks
    colors = ["#27ae60" if lbl == "BENIGN" else "#e67e22" for lbl in labels]

    # Subplot 1: Linear scale
    ax1.barh(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_title("Traffic Breakdown by Class (Linear Scale)")
    ax1.set_xlabel("Flow Count")
    ax1.invert_yaxis()
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    # Subplot 2: Log10 scale to clearly display rare attacks
    ax2.barh(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_xscale("log")
    ax2.set_title("Traffic Breakdown by Class (Log10 Scale - Highlights Rare Attacks)")
    ax2.set_xlabel("Flow Count (log scale)")
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.7)

    for i, (v, lbl) in enumerate(zip(values, labels)):
        ax2.text(
            v * 1.15,
            i,
            f"{v:,}",
            va="center",
            fontsize=9,
            fontweight="bold" if lbl == "BENIGN" else "normal",
        )

    plt.suptitle("CICIDS-2017: Multi-Class Label Imbalance Analysis", fontsize=14, y=0.98)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "02_class_distribution_multiclass.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved multiclass distribution plot to {out_path}")
    return out_path


def plot_inf_and_nan_values(summary: Dict[str, Any], output_dir: str = FIGURES_DIR) -> str:
    """Plot features with NaN and Inf occurrences."""
    inf_vals = summary.get("infinite_values", {})
    nan_vals = summary.get("nan_values", {})

    all_cols = sorted(list(set(list(inf_vals.keys()) + list(nan_vals.keys()))))
    if not all_cols:
        logger.info("No NaN or Inf values detected in dataset.")
        return ""

    inf_counts = [inf_vals.get(c, 0) for c in all_cols]
    nan_counts = [nan_vals.get(c, 0) for c in all_cols]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(all_cols))
    width = 0.35

    ax.bar(x - width/2, inf_counts, width, label="Infinite (+/- Inf)", color="#e74c3c", edgecolor="black")
    ax.bar(x + width/2, nan_counts, width, label="Missing (NaN)", color="#f39c12", edgecolor="black")

    ax.set_ylabel("Count of Affected Rows")
    ax.set_title("CICIDS-2017: Features Containing Infinite and Missing (NaN) Values")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cols, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "03_inf_nan_analysis.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Inf/NaN analysis plot to {out_path}")
    return out_path


def plot_flow_characteristics(df: pd.DataFrame, output_dir: str = FIGURES_DIR) -> str:
    """Plot flow duration and packet characteristics for Benign vs Attack."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    binary_label = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

    # Flow Duration (log scale)
    if "Flow Duration" in df.columns:
        dur_benign = df.loc[binary_label == "BENIGN", "Flow Duration"]
        dur_attack = df.loc[binary_label == "ATTACK", "Flow Duration"]

        # Filter out <= 0 for log plot
        dur_benign_pos = np.log10(dur_benign[dur_benign > 0] + 1)
        dur_attack_pos = np.log10(dur_attack[dur_attack > 0] + 1)

        sns.kdeplot(dur_benign_pos, ax=axes[0], color="#27ae60", label="BENIGN", fill=True, alpha=0.4)
        sns.kdeplot(dur_attack_pos, ax=axes[0], color="#c0392b", label="ATTACK", fill=True, alpha=0.4)
        axes[0].set_title("Log10(Flow Duration + 1) Density")
        axes[0].set_xlabel("Log10(Flow Duration in microseconds)")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.7)

    # Total Length of Fwd Packets (log scale)
    if "Total Length of Fwd Packets" in df.columns:
        fwd_benign = df.loc[binary_label == "BENIGN", "Total Length of Fwd Packets"]
        fwd_attack = df.loc[binary_label == "ATTACK", "Total Length of Fwd Packets"]

        fwd_benign_pos = np.log10(fwd_benign[fwd_benign > 0] + 1)
        fwd_attack_pos = np.log10(fwd_attack[fwd_attack > 0] + 1)

        sns.kdeplot(fwd_benign_pos, ax=axes[1], color="#27ae60", label="BENIGN", fill=True, alpha=0.4)
        sns.kdeplot(fwd_attack_pos, ax=axes[1], color="#c0392b", label="ATTACK", fill=True, alpha=0.4)
        axes[1].set_title("Log10(Total Length of Fwd Packets + 1) Density")
        axes[1].set_xlabel("Log10(Bytes)")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Comparison of Flow Characteristics: BENIGN vs. ATTACK", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "04_flow_characteristics_density.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved flow characteristics density plot to {out_path}")
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = FIGURES_DIR, top_n: int = 16) -> str:
    """Plot correlation heatmap for the top representative flow features."""
    candidate_features = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Fwd Header Length",
        "Bwd Header Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Average Packet Size",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]

    selected_cols = [c for c in candidate_features if c in df.columns][:top_n]
    if not selected_cols:
        return ""

    sample_sub = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample_sub) > 50000:
        sample_sub = sample_sub.sample(n=50000, random_state=RANDOM_SEED)

    corr = sample_sub.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(f"Correlation Heatmap of Top {len(selected_cols)} Flow Features", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "05_feature_correlations.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to {out_path}")
    return out_path


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute full EDA pipeline: compute summary metrics, generate figures,
    and save diagnostic findings report.
    """
    ensure_dirs()
    set_seed(RANDOM_SEED)

    logger.info("Starting comprehensive Exploratory Data Analysis...")
    summary = inspect_dataset(df)

    # Generate figures
    fig_binary = plot_binary_distribution(df, output_dir=FIGURES_DIR)
    fig_multi = plot_multiclass_distribution(df, output_dir=FIGURES_DIR)
    fig_nan = plot_inf_and_nan_values(summary, output_dir=FIGURES_DIR)
    fig_flow = plot_flow_characteristics(df, output_dir=FIGURES_DIR)
    fig_corr = plot_correlation_heatmap(df, output_dir=FIGURES_DIR)

    figures_generated = {
        "binary_distribution": fig_binary,
        "multiclass_distribution": fig_multi,
        "inf_nan_analysis": fig_nan,
        "flow_characteristics": fig_flow,
        "feature_correlations": fig_corr,
    }

    # Save summary as JSON report
    report_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved EDA summary metrics to {report_path}")

    return {
        "summary": summary,
        "figures": figures_generated,
        "report_path": report_path,
    }

"""
Anomaly-Based Network Intrusion Detection System (NIDS)
Streamlit Interactive Dashboard & Real-Time Flow Inference App.

Pages:
1. Overview: Problem framing, dataset diagnostics, and EDA visualizations.
2. Model Results: Benchmarks, multi-model ROC/PR curves, per-attack recall heatmaps, zero-day experiments.
3. Live Detection: Upload raw CICIDS-2017 flows or use sample, apply preprocessing, run inference,
   adjust thresholds, inspect score histograms, view confusion matrices, and download results.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ------------------------------------------------------------------------------
# App Configuration & Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Anomaly NIDS Dashboard | CICIDS-2017",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_flows.csv")

# ------------------------------------------------------------------------------
# Cached Resource Loaders
# ------------------------------------------------------------------------------
@st.cache_resource
def load_all_models_and_artifacts():
    """Load and cache trained models, scaler, feature list, and thresholds."""
    import keras
    
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "selected_features.json")
    thresh_path = os.path.join(MODELS_DIR, "autoencoder_thresholds.json")
    scaler_cfg_path = os.path.join(MODELS_DIR, "scaler_config.json")

    scaler = joblib.load(scaler_path)
    with open(features_path, "r") as f:
        selected_features = json.load(f)
    with open(thresh_path, "r") as f:
        ae_thresholds = json.load(f)
    with open(scaler_cfg_path, "r") as f:
        scaler_config = json.load(f)

    if_model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    ocsvm_model = joblib.load(os.path.join(MODELS_DIR, "one_class_svm.joblib"))
    ae_model = keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder.keras"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))

    return {
        "scaler": scaler,
        "selected_features": selected_features,
        "ae_thresholds": ae_thresholds,
        "scaler_config": scaler_config,
        "isolation_forest": if_model,
        "one_class_svm": ocsvm_model,
        "autoencoder": ae_model,
        "random_forest": rf_model,
        "xgboost": xgb_model,
    }


def clean_raw_inference_data(df: pd.DataFrame, selected_features: list, scaler, clip_min=-10.0, clip_max=10.0):
    """
    Standardize raw uploaded flows using the exact Stage 2 preprocessing pipeline:
    1. Strip leading/trailing column whitespace
    2. Drop duplicate columns (e.g. Fwd Header Length.1)
    3. Check feature availability
    4. Clean inf and NaN
    5. Apply fitted StandardScaler and clipping [-10, 10]
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    # Drop duplicate headers if present
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep="first")]
    if "Fwd Header Length.1" in df_clean.columns and "Fwd Header Length" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Fwd Header Length.1"])

    has_label = "Label" in df_clean.columns
    labels_series = None
    if has_label:
        labels_series = df_clean["Label"].astype(str).str.strip()

    # Check for missing required features
    missing_feats = [col for col in selected_features if col not in df_clean.columns]
    if missing_feats:
        raise ValueError(f"Uploaded CSV is missing {len(missing_feats)} required features: {missing_feats[:5]}...")

    # Replace inf with nan and drop
    features_df = df_clean[selected_features].replace([np.inf, -np.inf], np.nan)
    valid_mask = features_df.notna().all(axis=1)

    X_valid = features_df.loc[valid_mask].values
    dropped_count = int((~valid_mask).sum())

    # Scale and clip
    features_valid_df = features_df.loc[valid_mask]
    X_scaled = scaler.transform(features_valid_df)
    X_clipped = np.clip(X_scaled, clip_min, clip_max)

    clean_subset_df = df_clean.loc[valid_mask].copy().reset_index(drop=True)
    if labels_series is not None:
        clean_labels = labels_series.loc[valid_mask].reset_index(drop=True)
    else:
        clean_labels = None

    return X_scaled, X_clipped, clean_subset_df, clean_labels, dropped_count


# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("🛡️ Anomaly NIDS")
st.sidebar.markdown("**CICIDS-2017 Intrusion Detection**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["📊 Overview & EDA", "📈 Model Benchmarks", "⚡ Live Flow Detection"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Google DeepMind Antigravity • Semi-Supervised ML & Deep Learning")


# ==============================================================================
# Page 1: Overview & EDA
# ==============================================================================
if page == "📊 Overview & EDA":
    st.title("🛡️ Anomaly-Based Network Intrusion Detection System")
    st.subheader("Machine Learning & Deep Autoencoder on CICIDS-2017")

    st.markdown("""
    This platform demonstrates an **anomaly-based Network Intrusion Detection System (NIDS)**.
    In contrast to traditional signature-based systems that fail when encountering unknown zero-day attacks,
    this system uses **semi-supervised anomaly detection**—training solely on normal (BENIGN) network flows
    to establish a statistical baseline of legitimate behavior. Any flow deviating significantly from this manifold is flagged as an anomaly.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Dataset Flows", "2,830,743")
    col2.metric("Deduplicated Unique Flows", "2,522,362", "-308,381 duplicates")
    col3.metric("Selected Features", "46", "From 78 raw columns")
    col4.metric("Attack Classes Covered", "14 Attacks + Benign")

    st.markdown("---")
    st.header("1. Dataset Ingestion & Class Imbalance Analysis")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Binary Traffic Distribution")
        st.image(os.path.join(FIGURES_DIR, "01_class_distribution_binary.png"), width='stretch')
        st.caption("Benign traffic constitutes ~80% of normal traffic, with attacks forming ~20%.")

    with c2:
        st.markdown("#### Multi-Class Distribution (Linear & Log Scale)")
        st.image(os.path.join(FIGURES_DIR, "02_class_distribution_multiclass.png"), width='stretch')
        st.caption("Stratified sampling preserves 100% of rare attacks (e.g. Heartbleed, Infiltration, SQL Injection).")

    st.markdown("---")
    st.header("2. Data Integrity & Flow Characterization")

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("#### Division-by-Zero Infinite & Missing Values")
        st.image(os.path.join(FIGURES_DIR, "03_inf_nan_analysis.png"), width='stretch')
        st.caption("Flow calculations in CICIDS-2017 produce ±Inf and NaN in `Flow Bytes/s` and `Flow Packets/s`.")

    with c4:
        st.markdown("#### Flow Duration & Packet Length Density")
        st.image(os.path.join(FIGURES_DIR, "04_flow_characteristics_density.png"), width='stretch')
        st.caption("Volumetric attacks exhibit distinct duration and segment byte densities compared to Benign traffic.")

    st.markdown("---")
    st.header("3. Multicollinearity & Feature Selection")
    st.image(os.path.join(FIGURES_DIR, "05_feature_correlations.png"), width='stretch')
    st.caption("Pairwise Pearson correlation analysis removed 23 highly redundant features (|r| > 0.95), optimizing model latency and stability.")


# ==============================================================================
# Page 2: Model Benchmarks
# ==============================================================================
elif page == "📈 Model Benchmarks":
    st.title("📈 Comprehensive Test Set Model Evaluation")
    st.markdown("Evaluated on the completely untouched **`test.parquet` partition ($N = 140,048$)** with fixed validation thresholds.")

    # Load comparison CSV
    comp_csv_path = os.path.join(REPORTS_DIR, "final_comparison.csv")
    if os.path.exists(comp_csv_path):
        df_comp = pd.read_csv(comp_csv_path)
        st.dataframe(
            df_comp.style.format({
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1-Score": "{:.2f}%",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "FPR (%)": "{:.2f}%",
                "Threshold": "{:.5f}",
                "Train Time (s)": "{:.2f}s",
            }).highlight_max(
                subset=["Recall", "F1-Score", "ROC-AUC"],
                props="background-color: #d4edda; color: #0b3d0b; font-weight: bold;"
            ),
            width='stretch',
        )

    st.info("""
    **Methodology Note on Threshold Calibration**:
    - **Autoencoder percentiles ($p_{95}, p_{98}, p_{99}$)** were calibrated solely on **validation Benign reconstruction error** (true zero-knowledge anomaly thresholding).
    - **Isolation Forest, One-Class SVM, and Autoencoder (val-F1)** used labeled validation attacks to pick the threshold that maximizes validation F1. Under this identical evaluation protocol, the **Autoencoder achieves 73.41% F1 and 82.65% Recall**, significantly outperforming both Isolation Forest (51.55%) and One-Class SVM (64.63%).
    """)

    st.markdown("---")
    st.header("1. Performance Curves & Confusion Matrices")

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.image(os.path.join(FIGURES_DIR, "07_multi_model_roc_curves.png"), width='stretch')
    with col_pr:
        st.image(os.path.join(FIGURES_DIR, "08_multi_model_pr_curves.png"), width='stretch')

    st.markdown("#### Test Confusion Matrices Grid")
    st.image(os.path.join(FIGURES_DIR, "06_confusion_matrices.png"), width='stretch')

    st.markdown("---")
    st.header("2. Per-Attack Class Detection Heatmap")
    st.image(os.path.join(FIGURES_DIR, "09_per_attack_detection_heatmap.png"), width='stretch')

    # Show per-attack table
    per_attack_csv = os.path.join(REPORTS_DIR, "per_attack_detection.csv")
    if os.path.exists(per_attack_csv):
        df_pa = pd.read_csv(per_attack_csv)
        st.dataframe(df_pa, width='stretch')

    st.markdown("---")
    st.header("3. Zero-Day Generalization Experiments")

    c_loo1, c_loo2 = st.columns(2)
    with c_loo1:
        st.markdown("#### Leave-One-Attack-Out Experiment")
        st.image(os.path.join(FIGURES_DIR, "11_leave_one_out_comparison.png"), width='stretch')
        st.caption("When Infiltration was withheld from RF, RF recall dropped to 0%, while Autoencoder detected 83.3%.")

    with c_loo2:
        st.markdown("#### Leave-One-FAMILY-Out Experiment")
        st.image(os.path.join(FIGURES_DIR, "12_leave_family_out_comparison.png"), width='stretch')
        st.caption("When entire families (DoS, Web Attacks, Patator) were withheld, RF recall collapsed to <15% or 0%.")

    st.markdown("---")
    st.header("4. Autoencoder Threshold Sensitivity Curve")
    st.image(os.path.join(FIGURES_DIR, "10_autoencoder_threshold_analysis.png"), width='stretch')


# ==============================================================================
# Page 3: Live Detection Simulation
# ==============================================================================
elif page == "⚡ Live Flow Detection":
    st.title("⚡ Live Network Flow Anomaly Detection")
    st.markdown("Test individual models on raw network flows. Inspect predictions, anomaly score distributions, and detection rates.")

    artifacts = load_all_models_and_artifacts()
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]
    ae_thresholds = artifacts["ae_thresholds"]

    # 1. Input Selection
    st.subheader("1. Select Traffic Input")
    input_source = st.radio(
        "Choose Data Source:",
        ["Built-in Sample (data/sample_flows.csv, 2,000 mixed flows)", "Upload Custom CSV File"],
        horizontal=True,
    )

    df_raw = None
    if input_source.startswith("Built-in"):
        if os.path.exists(SAMPLE_PATH):
            df_raw = pd.read_csv(SAMPLE_PATH)
            st.success(f"Loaded built-in sample: **{len(df_raw):,} flows** with {df_raw.shape[1]} columns.")
        else:
            st.error(f"Sample file not found at {SAMPLE_PATH}.")
    else:
        uploaded_file = st.file_uploader("Upload raw CICIDS-2017 CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"Uploaded file loaded: **{len(df_raw):,} flows**.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if df_raw is not None:
        with st.expander("Preview Raw Flow Data (First 5 Rows)", expanded=False):
            st.dataframe(df_raw.head(), width='stretch')

        st.markdown("---")
        st.subheader("2. Model Selection & Threshold Calibration")

        m_col1, m_col2 = st.columns([1, 1])

        with m_col1:
            chosen_model_name = st.selectbox(
                "Choose Detection Model:",
                [
                    "Symmetric Keras Autoencoder",
                    "Isolation Forest",
                    "One-Class SVM",
                    "Random Forest Baseline",
                    "XGBoost Baseline",
                ]
            )

        threshold = 0.5
        with m_col2:
            if chosen_model_name == "Symmetric Keras Autoencoder":
                thresh_preset = st.selectbox(
                    "Reconstruction Error Threshold Preset:",
                    [
                        f"95th Percentile Benign ({ae_thresholds['p95']:.4f})",
                        f"98th Percentile Benign ({ae_thresholds['p98']:.4f})",
                        f"99th Percentile Benign ({ae_thresholds['p99']:.4f})",
                        "Val-F1 Optimal (0.0304)",
                        "Custom Threshold Slider",
                    ]
                )
                if "95th" in thresh_preset:
                    threshold = ae_thresholds["p95"]
                elif "98th" in thresh_preset:
                    threshold = ae_thresholds["p98"]
                elif "99th" in thresh_preset:
                    threshold = ae_thresholds["p99"]
                elif "Val-F1" in thresh_preset:
                    threshold = 0.03041
                else:
                    threshold = st.slider("MSE Reconstruction Threshold:", min_value=0.01, max_value=2.0, value=float(ae_thresholds["p95"]), step=0.01)

            elif chosen_model_name == "Isolation Forest":
                threshold = st.slider("Anomaly Score Threshold:", min_value=0.30, max_value=0.70, value=0.48646, step=0.005)
            elif chosen_model_name == "One-Class SVM":
                threshold = st.slider("Negative Hyperplane Distance Threshold:", min_value=1.0, max_value=30.0, value=13.9448, step=0.5)
            else:
                threshold = st.slider("Probability Threshold (Supervised):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        # 3. Execution Button
        if st.button("🚀 Run Anomaly Detection Inference", type="primary"):
            with st.spinner("Executing preprocessing, scaling, and inference..."):
                try:
                    X_scaled, X_clipped, clean_df, clean_labels, dropped_nan = clean_raw_inference_data(
                        df_raw, selected_features, scaler
                    )

                    t_start = time.time()
                    scores = None

                    if chosen_model_name == "Symmetric Keras Autoencoder":
                        model = artifacts["autoencoder"]
                        reconstructions = model.predict(X_clipped, batch_size=1024, verbose=0)
                        scores = np.mean(np.square(X_clipped - reconstructions), axis=1)

                    elif chosen_model_name == "Isolation Forest":
                        model = artifacts["isolation_forest"]
                        scores = -model.score_samples(X_scaled)

                    elif chosen_model_name == "One-Class SVM":
                        model = artifacts["one_class_svm"]
                        scores = -model.decision_function(X_clipped)

                    elif chosen_model_name == "Random Forest Baseline":
                        model = artifacts["random_forest"]
                        scores = model.predict_proba(X_scaled)[:, 1]

                    elif chosen_model_name == "XGBoost Baseline":
                        model = artifacts["xgboost"]
                        scores = model.predict_proba(X_scaled)[:, 1]

                    inference_time = time.time() - t_start

                    # Formulate predictions
                    binary_preds = (scores >= threshold).astype(int)
                    n_total = len(scores)
                    n_anomalies = int(np.sum(binary_preds == 1))
                    n_normal = n_total - n_anomalies
                    anomaly_rate = (n_anomalies / n_total) * 100.0

                    # 4. Results Dashboard
                    st.markdown("---")
                    st.header("3. Detection Results & Traffic Health")

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Flows Evaluated", f"{n_total:,}", f"Dropped NaN: {dropped_nan}")
                    k2.metric("Flagged Anomalies", f"{n_anomalies:,}", f"{anomaly_rate:.1f}% of batch", delta_color="inverse")
                    k3.metric("Normal Flows", f"{n_normal:,}")
                    k4.metric("Inference Latency", f"{inference_time*1000:.1f} ms", f"{n_total / max(0.001, inference_time):,.0f} flows/sec")

                    # Score Distribution Plot
                    c_hist, c_conf = st.columns([1, 1])
                    with c_hist:
                        st.markdown("#### Anomaly Score Distribution")
                        fig_hist, ax_h = plt.subplots(figsize=(6, 4))
                        sns.histplot(scores[binary_preds == 0], color="#27ae60", label="Normal", bins=40, kde=True, ax=ax_h, alpha=0.5)
                        sns.histplot(scores[binary_preds == 1], color="#e74c3c", label="Flagged Anomaly", bins=40, kde=True, ax=ax_h, alpha=0.5)
                        ax_h.axvline(threshold, color="black", linestyle="--", linewidth=1.8, label=f"Threshold ({threshold:.4f})")
                        ax_h.set_title("Distribution of Model Anomaly Scores", fontsize=11)
                        ax_h.set_xlabel("Score")
                        ax_h.set_ylabel("Count")
                        ax_h.legend()
                        st.pyplot(fig_hist)

                    with c_conf:
                        if clean_labels is not None:
                            st.markdown("#### Ground-Truth Confusion Matrix")
                            y_true_binary = (clean_labels != "BENIGN").astype(int)
                            cm = confusion_matrix(y_true_binary, binary_preds, labels=[0, 1])
                            acc = accuracy_score(y_true_binary, binary_preds)
                            prec = precision_score(y_true_binary, binary_preds, zero_division=0)
                            rec = recall_score(y_true_binary, binary_preds, zero_division=0)
                            f1 = f1_score(y_true_binary, binary_preds, zero_division=0)

                            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                            sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                                        xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"], ax=ax_cm)
                            ax_cm.set_ylabel("Actual Ground Truth")
                            ax_cm.set_xlabel("Predicted")
                            st.pyplot(fig_cm)
                            st.caption(f"Batch Accuracy: **{acc*100:.1f}%** | Recall: **{rec*100:.1f}%** | Precision: **{prec*100:.1f}%** | F1: **{f1*100:.1f}%**")
                        else:
                            st.info("No 'Label' column present in uploaded CSV. Running in pure unlabelled production mode.")

                    # If labels exist, show per-attack detection
                    if clean_labels is not None:
                        st.markdown("#### Per-Attack Breakdown in Current Batch")
                        attack_classes = [c for c in clean_labels.unique() if c != "BENIGN"]
                        pa_records = []
                        for ac in attack_classes:
                            mask = (clean_labels == ac)
                            tot_ac = int(mask.sum())
                            det_ac = int(np.sum(binary_preds[mask] == 1))
                            rec_ac = (det_ac / tot_ac) * 100.0 if tot_ac > 0 else 0.0
                            pa_records.append({
                                "Attack Class": ac,
                                "Batch Total": tot_ac,
                                "Detected Anomalies": det_ac,
                                "Detection Rate (Recall)": f"{rec_ac:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(pa_records), width='stretch')

                    # Append predictions and download
                    clean_df["Anomaly_Score"] = np.round(scores, 5)
                    clean_df["Prediction"] = np.where(binary_preds == 1, "ANOMALY", "Normal")

                    st.markdown("#### Flow Prediction Audit Table (Top 50 Most Anomalous Flows)")
                    sorted_df = clean_df.sort_values(by="Anomaly_Score", ascending=False)
                    st.dataframe(sorted_df.head(50), width='stretch')

                    csv_data = clean_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Predictions CSV",
                        data=csv_data,
                        file_name=f"nids_predictions_{chosen_model_name.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Inference Error: {e}")

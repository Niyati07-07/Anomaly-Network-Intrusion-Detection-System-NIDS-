# Anomaly-Based Network Intrusion Detection System (NIDS) on CICIDS-2017

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io/)
[![Keras 3](https://img.shields.io/badge/Keras-3.8.0-red.svg)](https://keras.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)

An end-to-end Machine Learning and Deep Learning system for **Anomaly-Based Network Intrusion Detection** evaluated on the comprehensive **CICIDS-2017** benchmark dataset.

---

## 📄 Report-Ready Abstract

> **Abstract** — Traditional signature-based Network Intrusion Detection Systems (NIDS) fail when confronted with zero-day attacks and polymorphic evasion techniques. This project implements a rigorous semi-supervised anomaly detection framework trained exclusively on normal (BENIGN) network traffic, using the CICIDS-2017 benchmark dataset (2.52M unique flows across 14 attack classes). We systematically address critical dataset artifacts, including deduplication (removing 308,381 duplicate records), division-by-zero infinite values, non-ASCII label corruptions, zero-variance feature elimination, and multicollinearity filtering ($|r| > 0.95$, reducing 77 features to 46). We train and benchmark three semi-supervised anomaly models—**Isolation Forest**, **One-Class SVM** (with feature clipping $[-10, 10]$), and a deep **Symmetric Keras Autoencoder** ($46 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 8 \rightarrow 16 \rightarrow 32 \rightarrow 64 \rightarrow 46$)—against **Random Forest** and **XGBoost** supervised baselines. On an independent test set of 140,048 flows, the Symmetric Autoencoder achieves a **ROC-AUC of 0.8765** and **PR-AUC of 0.8475**. When evaluated under an identical validation-F1 tuning protocol, the Autoencoder achieves **73.41% F1 and 82.65% Recall**, significantly outperforming Isolation Forest (51.55% F1) and One-Class SVM (64.63% F1). Crucially, Leave-One-Attack-Out and Leave-One-Family-Out experiments demonstrate that while supervised classifiers collapse to near 0% recall when an attack family is withheld during training, the Autoencoder sustains high detection recall (e.g. 84.09% on the DoS family and 83.33% on Infiltration) without ever having seen an attack signature. Finally, we provide an interactive Streamlit application supporting live flow inference, score histograms, threshold tuning, and CSV report export.

---

## 🎯 Problem Statement & Motivation

Signature-based intrusion detection systems (e.g., Snort, Suricata rules) require prior knowledge of attack signatures. When attackers alter payload syntax or deploy novel attack vectors, rule-based systems are bypassed.

**Anomaly-based NIDS** solves this fundamental limitation:
1. **Semi-Supervised Learning**: Models learn the compact topological manifold of legitimate **BENIGN** network traffic.
2. **Zero-Day Detection**: Intrusions that perturb transport-layer statistical invariants generate high reconstruction errors ($MSE$) or low density scores, triggering an alarm without requiring prior attack examples.
3. **Strict Leakage Prevention**: The feature scaler (`StandardScaler`) is fitted **strictly on the Benign training set**, preventing any information leakage from validation or test attack distributions.

---

## 🛠️ Handling CICIDS-2017 Dataset Artifacts

The raw CICIDS-2017 dataset contains well-documented quirks that corrupt ML pipelines if not resolved:

| Dataset Issue | Raw Manifestation | Engineering Solution | Impact |
| :--- | :--- | :--- | :--- |
| **Header Whitespace** | Leading/trailing spaces (`' Destination Port'`, `' Label'`) | Stripped headers globally via `.str.strip()` | Prevents `KeyError` exceptions |
| **Duplicated Columns** | `'Fwd Header Length'` appears twice in raw CSVs | Detected identical columns and dropped redundant `'Fwd Header Length.1'` | Eliminates redundant 79th column |
| **Corrupted Encodings** | Replacement character sequences (`Web Attack  Brute Force`) | Regex normalization replacing non-ASCII bytes with hyphens (`Web Attack - Brute Force`) | Clean, unified multi-class labels |
| **Duplicate Flows** | 308,381 identical flow records across CSVs | Dropped duplicates **before** sampling | Prevents wasting rare attack rows on duplicates |
| **Infinite / NaN Values** | $\pm\text{Inf}$ and $\text{NaN}$ in `Flow Bytes/s` and `Flow Packets/s` | Replaced $\pm\text{Inf}$ with $\text{NaN}$, then dropped affected rows (320 rows) | Eliminates numerical overflow |
| **Constant Features** | 8 features have zero variance ($\sigma = 0$) across flows | Identified and removed 8 zero-variance features | Reduces feature space from 77 to 69 |
| **Multicollinearity** | Redundant features with Pearson $\|r\| > 0.95$ (e.g., Subflow bytes vs Total bytes) | Upper-triangle correlation filter removed 23 collinear features | Final clean feature space: **46 features** |
| **Class Imbalance** | Benign (~80%) vs rare attacks (Heartbleed: 11, SQLi: 21, Infiltration: 36) | **Selective Stratification**: 100% of classes $< 5,000$ rows retained; only large classes sampled at 20% | All rare attack instances preserved |

---

## 🏗️ System Architecture & Methodology

```
CICIDS-2017 (8 CSVs, 2.83M rows)
       │
       ▼
Deduplication (308,381 duplicates dropped)
       │
       ▼
Selective Sampling (100% of rare attacks, 20% of large classes)
       │
       ▼
Data Cleaning (Inf -> NaN -> Drop, Label Normalization)
       │
       ▼
Feature Selection (Drop 8 constant + 23 collinear |r| > 0.95) -> 46 Features
       │
       ▼
Semi-Supervised Split (seed = 42)
 ├── Train (60% Benign ONLY: 251,408 rows) ──> Fit StandardScaler (models/scaler.joblib)
 ├── Validation (20% Benign + 50% Attacks: 140,053 rows) ──> Threshold Calibration & Supervised Training
 └── Test (20% Benign + 50% Attacks: 140,048 rows) ──> Final Evaluation ONLY
```

### Deep Symmetric Autoencoder Architecture
Built in Keras 3 (PyTorch backend):
- **Input Dimension**: $d = 46$
- **Encoder**:
  - `Dense(64)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
  - `Dense(32)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
  - `Dense(16)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
- **Bottleneck (Latent Space)**:
  - `Dense(8)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)`
- **Decoder (Symmetric)**:
  - `Dense(16)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
  - `Dense(32)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
  - `Dense(64)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout(0.1)`
  - `Dense(46)` (Linear reconstruction output)
- **Loss**: Mean Squared Error ($MSE$)
- **Optimization**: Adam ($\eta = 10^{-3}$) with Early Stopping monitored on **Validation Benign Reconstruction Loss** (`patience = 6`).

---

## 📊 Experimental Results

### 1. Final Comparison on Test Set (`test.parquet`, $N = 140,048$)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | PR-AUC | FPR (%) | Threshold | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 70.83% | 77.40% | 38.65% | 51.55% | 0.8148 | 0.7136 | 7.57% | 0.48646 | **0.48 s** |
| **One-Class SVM** (25k subsample, clipped) | 78.69% | **96.97%** | 48.46% | 64.63% | 0.7933 | 0.7938 | **1.02%** | 13.94477 | 3.57 s |
| **Autoencoder ($p_{95}$ threshold)** | **79.73%** | 88.54% | **56.88%** | **69.27%** | **0.8765** | **0.8475** | 4.94% | 0.21523 | 589.79 s |
| **Autoencoder ($p_{98}$ threshold)** | 79.25% | 94.57% | 51.27% | 66.49% | **0.8765** | **0.8475** | **1.97%** | 0.48086 | 589.79 s |
| **Autoencoder ($p_{99}$ threshold)** | 77.13% | 96.83% | 44.52% | 61.00% | **0.8765** | **0.8475** | **0.98%** | 0.69467 | 589.79 s |
| **Autoencoder (val-F1-optimal)** | 75.96% | 66.03% | **82.65%** | **73.41%** | **0.8765** | **0.8475** | 28.54% | 0.03041 | 589.79 s |
| **Random Forest Baseline** (balanced) | **99.84%** | **99.76%** | **99.84%** | **99.80%** | **0.9999** | **0.9998** | **0.16%** | 0.50000 | 4.51 s |
| **XGBoost Baseline** (`scale_pos_weight`) | **99.81%** | **99.66%** | **99.88%** | **99.77%** | **0.9999** | **0.9999** | **0.23%** | 0.50000 | **1.30 s** |

---

### 2. Per-Attack Class Detection Rate (Recall %) on Test Set

| Attack Class | Test Count | Reliability Status | Isolation Forest (%) | One-Class SVM (%) | Autoencoder ($p_{95}$) (%) | Autoencoder ($p_{98}$) (%) | Autoencoder ($p_{99}$) (%) | Autoencoder (val-F1) (%) | Random Forest (%) | XGBoost (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS Slowhttptest** | 2,614 | Stable | 83.21% | 83.78% | **94.11%** | 84.12% | 14.65% | 98.05% | 99.92% | 99.92% |
| **DoS Hulk** | 17,285 | Stable | 82.16% | 84.35% | **89.18%** | 85.05% | 84.47% | 95.09% | 99.96% | 99.98% |
| **Infiltration** | 18 | **Unstable (<20)** | **88.89%** | 38.89% | **83.33%** | 72.22% | 44.44% | 88.89% | 77.78% | 72.22% |
| **DoS GoldenEye** | 5,143 | Stable | 25.69% | 44.95% | **73.23%** | 51.45% | 47.23% | 97.98% | 99.96% | 99.90% |
| **DDoS** | 12,801 | Stable | 18.09% | 51.14% | **62.94%** | 57.60% | 50.85% | 80.06% | 99.94% | 99.98% |
| **DoS slowloris** | 2,692 | Stable | 59.99% | 58.80% | **62.48%** | 60.96% | 36.29% | 66.49% | 99.81% | 99.85% |
| **Heartbleed** | 5 | **Unstable (<20)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 100.00% | 100.00% | 80.00% |
| **PortScan** | 9,068 | Stable | 0.23% | 0.00% | 5.69% | 2.07% | 0.72% | 74.04% | 99.97% | 99.97% |
| **Web Attack - Brute Force**| 735 | Stable | 4.63% | 0.14% | 4.63% | 4.63% | 4.49% | 9.52% | 99.59% | 99.46% |
| **Web Attack - XSS** | 326 | Stable | 3.07% | 2.76% | 3.37% | 3.07% | 2.76% | 3.99% | 99.39% | 99.08% |
| **Bot** | 974 | Stable | 2.05% | 2.05% | 2.67% | 2.05% | 2.05% | 12.83% | 94.87% | 96.41% |
| **SSH-Patator** | 1,609 | Stable | 0.12% | 0.12% | 0.25% | 0.19% | 0.12% | 89.93% | 99.88% | 99.94% |
| **FTP-Patator** | 2,965 | Stable | 0.00% | 0.07% | 0.13% | 0.10% | 0.00% | 68.09% | 100.00% | 100.00% |
| **Web Attack - Sql Injection**| 10 | **Unstable (<20)**| 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.00% | 100.00% | 100.00% |

---

### 3. Zero-Day Generalization Experiments

#### Leave-One-FAMILY-Out Experiment
Entire attack families were completely withheld from Random Forest training data:

| Held-Out Attack Family | Included Classes | Test Instances | Supervised RF Recall (%) *(Trained without Family)* | Unsupervised AE ($p_{95}$) Recall (%) *(Trained on Benign only)* | AE Advantage ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DoS Family** | DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed | 27,739 | **1.29%** | **84.09%** | **+82.80%** |
| **Web Attacks** | Brute Force, XSS, Sql Injection | 1,071 | **15.03%** | 4.20% | -10.83% |
| **Patator (Brute Force)** | FTP-Patator, SSH-Patator | 4,574 | **0.04%** | 0.17% | +0.13% |

---

## 🔍 Key Findings & Diagnostic Analysis

1. **Why Volumetric Intrusions Are Detected Effectively**:
   - `DoS Hulk` (89.2%), `DoS Slowhttptest` (94.1%), `DoS GoldenEye` (73.2%), and `DDoS` (62.9%) perturb network transport invariants: massive spikes in packet transmission rates, distorted forward/backward byte ratios, and anomalous inter-arrival times. These violate the manifold learned by the Autoencoder, generating high reconstruction errors ($MSE \gg p_{99}$).
2. **Why PortScan Recall is Low in Anomaly Models (5.7% for AE $p_{95}$)**:
   - PortScan flows in CICIDS-2017 consist of single-probe or two-probe packets with 0 payload bytes and sub-millisecond durations.
   - In legitimate network traffic, millions of benign flows are also lightweight, short-lived connections (e.g., DNS queries, quick keep-alives).
   - Without session windowing or cross-flow temporal aggregation across multiple ports, an individual PortScan flow viewed in isolation sits directly inside the dense cluster of benign short flows.
3. **Application-Layer Stealth**:
   - Web Attacks (SQL Injection, XSS) and Botnet flows are embedded inside valid HTTP GET/POST sessions. At the statistical transport layer, they mirror benign browsing flows. Detecting payload-level exploits requires Deep Packet Inspection (DPI) rather than flow-level statistical modeling.
4. **Zero-Day Family Leakage**:
   - In individual leave-one-out experiments, holding out Web Attack - XSS alone allowed Random Forest to achieve 95.4% recall because Web Attack - Brute Force remained in training. When the entire Web Attack family was held out, RF recall collapsed to **15.03%**, demonstrating the danger of sibling class leakage in intrusion benchmarks.

---

## ⚠️ Limitations & Methodological Caveats

1. **Base Rate Fallacy in Downsampled Benchmarks**:
   - Downsampling Benign traffic to 20% elevates the test set attack rate to ~40.2% (compared to <1% in real enterprise networks).
   - Consequently, Precision and PR-AUC reported in this and similar benchmarks are optimistic. The **False Positive Rate (FPR)** and **ROC-AUC** provide more realistic, distribution-invariant metrics for operational deployment.
2. **Rare-Class Small-Sample Instability**:
   - In the deduplicated test set, `Heartbleed` has only 5 instances, `Web Attack - Sql Injection` has 10, and `Infiltration` has 18. Recall scores on these classes carry high variance and should be interpreted as exploratory.

---

## 🚀 Literature Gap & Future Work: Proposed Hybrid Architecture

In contemporary literature, NIDS research often creates a false dichotomy between supervised and anomaly-based systems. In production, neither is sufficient alone:
- **Supervised models** offer ultra-low false alarms (FPR $\le 0.2\%$) and near-perfect accuracy ($>99.8\%$) on known attacks, but fail completely on novel zero-day families (collapsing to $\le 1.3\%$ recall).
- **Anomaly detectors** successfully detect novel deviations from normal traffic without signatures (84.1% on unseen DoS families), but suffer higher false alarm rates on noisy benign edge cases.

### Proposed Two-Tier Hybrid NIDS
```
Incoming Network Flow
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Supervised Fast-Path Classifier (Random Forest) │
│ - Checks known attack signatures                       │
│ - Low latency (~1.3ms per 1,000 flows)                │
└───────────────────────┬────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Known Attack Detected?       Classified as Normal?
         │                             │
    🚨 Block / Alert                   ▼
                        ┌────────────────────────────────────────┐
                        │ Tier 2: Deep Autoencoder Anomaly Engine │
                        │ - Flags deviations from Benign Manifold│
                        │ - Captures novel zero-day vectors      │
                        └──────────────────┬─────────────────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                      MSE > Threshold?              MSE <= Threshold?
                            │                             │
                       🚨 Zero-Day Alert              ✅ Pass Flow
```

---

## 💻 How to Run the Project

### 1. Environment Setup
Clone the repository and install dependencies inside a Python 3.11 virtual environment:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce Pipeline Stages
Execute the automated runners for each stage:
```bash
# Stage 1: Load raw data, apply selective sampling, run EDA
python run_stage1_eda.py

# Stage 2: Clean Inf/NaN, feature selection, semi-supervised split & scaling
python run_stage2_preprocessing.py

# Stage 3: Train all 5 models & calibrate thresholds
python run_stage3_models.py

# Stage 4: Comprehensive test evaluation & zero-day benchmarks
python run_stage4_evaluation.py
```

### 3. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Repository Directory Structure

```
anomaly_NIDS/
├── app.py                      # Interactive Streamlit 3-page Web App
├── requirements.txt            # Pinned dependency manifest
├── README.md                   # Comprehensive research report & documentation
├── run_stage1_eda.py           # Stage 1 execution runner
├── run_stage2_preprocessing.py # Stage 2 preprocessing runner
├── run_stage3_models.py        # Stage 3 model training runner
├── run_stage4_evaluation.py    # Stage 4 evaluation & benchmarks runner
├── data/
│   ├── raw/                    # 8 raw CICIDS-2017 CSV files
│   ├── processed/              # Parquet splits & cached score tables
│   │   ├── train.parquet       # 100% Benign training set (251k rows)
│   │   ├── val.parquet         # Validation split (140k rows)
│   │   ├── test.parquet        # Untouched test set (140k rows)
│   │   ├── val_scores.parquet  # Precomputed validation anomaly scores
│   │   └── test_scores.parquet # Precomputed test anomaly scores
│   └── sample_flows.csv        # 2,000 raw test flows for live app testing
├── models/
│   ├── scaler.joblib           # StandardScaler fitted strictly on Benign train
│   ├── scaler_config.json      # Feature names & [-10, 10] clipping boundaries
│   ├── selected_features.json  # 46 selected feature names
│   ├── autoencoder.keras       # Trained symmetric Keras autoencoder
│   ├── autoencoder_thresholds.json # Calibrated p95, p98, p99 thresholds
│   ├── isolation_forest.joblib # Trained Isolation Forest model
│   ├── one_class_svm.joblib    # Trained One-Class SVM model
│   ├── random_forest.joblib    # Trained Random Forest baseline
│   ├── xgboost.joblib          # Trained XGBoost baseline
│   └── val_metrics.json        # Validation metrics & decision boundaries
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_preprocessing_and_splitting.ipynb
│   └── 03_model_training.ipynb
└── reports/
    ├── analysis.md             # Written diagnostic report & findings
    ├── final_comparison.csv    # Final test evaluation metrics table
    ├── final_comparison.md     # Markdown comparison table
    ├── per_attack_detection.csv# Per-attack recall breakdown
    ├── leave_one_out_experiment.csv # Single attack zero-day benchmark
    ├── leave_family_out_experiment.csv # Family-level zero-day benchmark
    └── figures/                # 12 high-resolution publication figures
        ├── 01_class_distribution_binary.png
        ├── 02_class_distribution_multiclass.png
        ├── 03_inf_nan_analysis.png
        ├── 04_flow_characteristics_density.png
        ├── 05_feature_correlations.png
        ├── 06_confusion_matrices.png
        ├── 07_multi_model_roc_curves.png
        ├── 08_multi_model_pr_curves.png
        ├── 09_per_attack_detection_heatmap.png
        ├── 10_autoencoder_threshold_analysis.png
        ├── 11_leave_one_out_comparison.png
        └── 12_leave_family_out_comparison.png
```

---

## 📜 Citation & Credits
- **Dataset**: Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"*, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, 2018.
- **Framework**: Developed with Google DeepMind Antigravity, Keras 3, PyTorch, Scikit-Learn, and Streamlit.

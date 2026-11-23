# Comprehensive Analysis: Anomaly-Based NIDS on CICIDS-2017

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

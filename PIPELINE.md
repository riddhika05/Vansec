# VANET Intrusion Detection Pipeline (Leakage-Free v2)

This document explains the end-to-end machine learning pipeline used to train the Intrusion Detection System (IDS) for Vehicular Ad-Hoc Networks (VANET). The pipeline focuses on building a highly realistic, robust model by carefully preventing data leakage and simulation artifacts from artificially inflating performance.

## 1. Dataset Overview
The model uses the `v2x_dataset_Main_run.csv` dataset, which contains **466,454 rows** representing network traffic and physical metrics for connected vehicles. 
There are four possible labels:
- **Normal** (Benign traffic)
- **Sybil** (Attacker pretending to be multiple vehicles)
- **DDoS** (Distributed Denial of Service)
- **Blackhole** (Attacker dropping all received packets)

## 2. Preprocessing & Leakage Removal (Crucial)
In synthetic or simulated datasets, certain features mathematically guarantee 100% accuracy because they are artifacts of the simulation rather than real-world observables. This pipeline aggressively removes 38 such features across 6 categories:

1. **Identifiers / Pre-computed labels:** `event_type`, `is_fake`, `fake_id`, `sybil_score`, `ddos_score`, `blackhole_score`
2. **Attacker-side counters (Unobservable by a real IDS):** `my_attack_packets`, `my_fake_ids_sent`, `my_ddos_packets`, `my_dropped`, `attack_rate`, `ddos_rate_actual`
3. **Derived Identity Ratios:** `fake_ratio`, `ddos_ratio`, `blackhole_ratio`, `drop_ratio`, `norm_attack_rate`
4. **Zero/Near-Zero Variance:** `avg_pkt_size_sent`, `norm_pkt_size`
5. **Global/Network-Wide Metrics:** `global_beacons_sent`, `global_beacons_rcvd`, `global_attack_pkts`, `global_dropped`, `network_load` *(A single car's IDS cannot know the global state of the network)*
6. **Simulation Artifacts:** Features that the simulator sets to deterministic values for attackers (e.g., all `blackhole` networking metrics being exactly `0`, or `sybil` beacon rates being exactly `60.0`). 

By removing these, the model is forced to learn the actual *behavioral patterns* of the attacks.

## 3. Feature Encoding and Imputation
- Categorical target labels are converted to numerical classes using `LabelEncoder`.
- Any missing values in the features are filled with the median of their respective columns.
- Any remaining object-type columns are encoded using `LabelEncoder`.

## 4. Feature Selection (Mutual Information)
To reduce dimensionality and improve training efficiency, we use **Mutual Information (MI)** to score how much information each feature contributes to predicting the label.
- A 50,000-row sample is used to compute MI scores quickly.
- We select the top features that have an MI score strictly greater than `0.01` (typically resulting in ~9 core features like `speed_mps`, `distance_moved`, etc.).

## 5. Group-Based Splitting
**This is the most critical evaluation step.**
The dataset only contains 5 unique attacker cars per attack type. If we use a standard random train/test split, data from the *same* attacker car will appear in both the training set and the test set. The model would just memorize the specific driving path or location (`pos_x`, `pos_y`) of those 5 cars.

To prevent this temporal/spatial leakage, the pipeline uses **`GroupShuffleSplit`** grouped by `car_id`. 
- **80% of cars** are placed in the training set.
- **20% of cars** are placed in the test set.
- A car is *never* split across both sets. 
This provides a realistic measurement of how the model will perform on a brand-new car it has never seen before.

## 6. Model Training & Class Imbalance Handling
Because the dataset is heavily imbalanced (Normal traffic vastly outnumbers attack traffic), we must account for this during training. 

**Note on SMOTE:** 
Synthetic Minority Over-sampling Technique (SMOTE) was *removed* from this pipeline. SMOTE generates fake data points by interpolating between minority class samples. However, in our dataset, doing this after a group-split corrupts the strict boundary between cars, and doing it before splitting causes data leakage.

Instead of SMOTE, we use **Algorithm-Level Balancing**:
1. **Random Forest:** Trained with `class_weight='balanced'`. The algorithm automatically penalizes mistakes on minority classes more heavily.
2. **XGBoost:** Trained using explicit `sample_weight` arrays calculated based on the inverse frequency of each class in the training set.

Both models are evaluated on Accuracy, Weighted F1 Score, and Macro F1 Score.

## 7. Artifact Generation
At the end of the pipeline, the system outputs:
- Visualizations (`feature_importance.png`, `label_distribution.png`, `model_comparison.png`).
- Confusion matrices for each model.
- The best performing model weights (`best_model.joblib`), along with the `scaler.joblib`, `label_encoder.joblib`, and `feature_list.joblib` for future inference deployments.

# VANET Intrusion Detection System - Pipeline Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Problem Statement](#3-problem-statement)
4. [Data Leakage Analysis](#4-data-leakage-analysis)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Feature Selection](#6-feature-selection)
7. [Train/Test Split Strategy](#7-traintest-split-strategy)
8. [Model Training](#8-model-training)
9. [Results & Interpretation](#9-results--interpretation)
10. [File Structure](#10-file-structure)

---

## 1. Project Overview

This project builds a **machine learning-based Intrusion Detection System (IDS)** for
**Vehicular Ad-hoc Networks (VANETs)**. VANETs enable vehicle-to-vehicle (V2V) and
vehicle-to-infrastructure (V2X) communication, but are vulnerable to cyberattacks.
The IDS classifies network traffic into four categories:

| Label       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| **normal**  | Legitimate beacon/communication traffic                                     |
| **sybil**   | Attacker creates multiple fake vehicle identities to manipulate the network |
| **ddos**    | Distributed Denial of Service - flooding the network with packets           |
| **blackhole** | Attacker drops packets it should forward, disrupting routing              |

**Models used:** Random Forest and XGBoost (ensemble tree-based classifiers).

**Key constraint:** SMOTE (Synthetic Minority Over-sampling) is intentionally NOT used.
Class imbalance is handled via `class_weight='balanced'` (RF) and manual `sample_weight` (XGBoost).

---

## 2. Dataset Description

**File:** `v2x_dataset_Main_run.csv`

| Property         | Value          |
|------------------|----------------|
| Rows             | 466,454        |
| Original columns | 49             |
| Source           | SUMO/OMNeT++ simulation |
| Missing values   | 0              |

### Class Distribution

| Class     | Count   | Percentage |
|-----------|---------|------------|
| normal    | 280,478 | 60.1%      |
| sybil     | 148,950 | 31.9%      |
| ddos      | 24,677  | 5.3%       |
| blackhole | 12,349  | 2.6%       |

The dataset is **imbalanced** - normal traffic dominates at 60%, while blackhole is only 2.6%.
This imbalance is handled through class weighting rather than synthetic oversampling.

### Column Categories in the Raw Dataset

The 49 original columns fall into these categories:

| Category                  | Example Columns                                   | Count |
|---------------------------|---------------------------------------------------|-------|
| Identifiers               | `car_id`, `event_type`, `fake_id`                 | 3     |
| Pre-computed labels/scores | `sybil_score`, `ddos_score`, `blackhole_score`    | 4     |
| Attacker-side counters    | `my_attack_packets`, `my_ddos_packets`, `my_dropped` | 6  |
| Derived attack ratios     | `fake_ratio`, `ddos_ratio`, `drop_ratio`          | 5     |
| Global network metrics    | `global_beacons_sent`, `network_load`             | 5     |
| Simulation artifacts      | `beacon_regularity`, `throughput_bps`             | 14    |
| Vehicle dynamics          | `speed_mps`, `pos_x`, `pos_y`, `avg_speed`       | 7     |
| Packet info               | `pkt_size_bytes`                                  | 1     |
| Temporal                  | `timestamp`                                       | 1     |
| Redundant derived         | `norm_speed`, `norm_pkt_size`                     | 2     |
| Target                    | `label`                                           | 1     |

---

## 3. Problem Statement

### The Original Issue: 100% Accuracy

The original notebook achieved **100% accuracy** on all models. This is a classic sign of
**data leakage** - the model has access to information that directly reveals the answer,
rather than learning genuine traffic patterns.

### Root Causes Identified

**Cause 1: Leaky Features (addressed in the original notebook)**
Columns like `sybil_score`, `ddos_ratio`, `my_attack_packets` directly encode the attack
type. For example, `sybil_score > 0` perfectly predicts the sybil class. The original
notebook dropped 38 such columns, bringing accuracy from 100% to ~82%.

**Cause 2: Vehicle Identity Leakage (NEW - fixed in this version)**
The original `train_test_split` used a plain random split. Since each `car_id` generates
thousands of rows, the same car appeared in both train and test sets. The model memorized
per-vehicle behavioral fingerprints rather than learning generalizable attack patterns.

**Cause 3: Temporal Leakage via `timestamp` (NEW - fixed in this version)**
The `timestamp` column is a monotonic simulation clock (0.2s to 397.5s). Different attack
types are injected at different simulation phases, so the model could learn
"if timestamp > 200, it's probably sybil" - a pattern that would never generalize to
real-world deployment.

**Cause 4: Redundant Feature `norm_speed` (NEW - fixed in this version)**
`norm_speed = speed_mps / max_speed` is a perfect linear transformation. Having both
features inflates their combined importance in mutual information scoring without adding
any actual information.

**Cause 5: Missing Class Stratification in Group Split (NEW - fixed in this version)**
A plain `GroupShuffleSplit` doesn't stratify by label. With only ~133 cars total and some
attack types concentrated in very few cars, an entire class (DDoS) ended up with 0 test
samples, making evaluation meaningless for that class.

---

## 4. Data Leakage Analysis

### What is Data Leakage?

Data leakage occurs when information from outside the training dataset is used to create
the model. In the context of IDS, this means:

- **Feature leakage:** Using columns that contain the answer (e.g., `sybil_score`)
- **Identity leakage:** Same entity (car) in train and test, allowing memorization
- **Temporal leakage:** Using time-based patterns specific to the simulation run

### Columns Dropped (40 total)

```
Category 1 - Identifiers / Pre-computed Labels:
  event_type, is_fake, fake_id, sybil_score, ddos_score, blackhole_score

Category 2 - Attacker-side Counters:
  my_attack_packets, my_fake_ids_sent, my_ddos_packets, my_dropped,
  attack_rate, ddos_rate_actual

Category 3 - Derived Ratios (encode attack type directly):
  fake_ratio, ddos_ratio, blackhole_ratio, drop_ratio, norm_attack_rate

Category 4 - Zero/Near-zero Variance:
  avg_pkt_size_sent, norm_pkt_size

Category 5 - Global/Network-wide Metrics:
  global_beacons_sent, global_beacons_rcvd, global_attack_pkts,
  global_dropped, network_load

Category 6 - Simulation Artifacts:
  my_beacons_sent, my_beacons_received, beacon_send_rate, beacon_recv_rate,
  total_bytes_sent, total_bytes_rcvd, avg_pkt_size_rcvd, throughput_bps,
  send_recv_ratio, active_duration, beacon_regularity, norm_send_rate,
  norm_recv_rate, norm_bytes_sent

Category 7 - Temporal Leakage:
  timestamp

Category 8 - Redundant:
  norm_speed
```

### Why Each Category Leaks

| Category | Why it leaks | Real-world analogy |
|----------|-------------|-------------------|
| Pre-computed labels | `sybil_score` is computed FROM the label - it IS the answer | Giving a student the answer key during an exam |
| Attacker counters | `my_ddos_packets` is only non-zero for DDoS attackers - a deployed IDS can't see this | A security camera that only records criminals |
| Derived ratios | `fake_ratio = fake_ids / total` directly encodes attack intensity | Labeling suspicious packages before screening |
| Global metrics | `network_load` requires omniscient knowledge of all traffic | Expecting a single guard to see all entrances simultaneously |
| Simulation artifacts | `beacon_regularity` has deterministic values per attack type in the simulation | The simulation "tells" the model the answer via fixed parameters |
| Timestamp | Attacks happen at fixed simulation phases (e.g., sybil starts at t=60) | Knowing the robbery schedule |
| Redundant | `norm_speed = speed / max_speed` carries identical information to `speed_mps` | Counting the same vote twice |

---

## 5. Preprocessing Pipeline

### Step-by-Step Flow

```
Raw CSV (466,454 x 49)
        |
        v
  Drop 40 leaky columns
        |
        v
  Remaining: 9 columns (7 features + car_id + label)
        |
        v
  Encode labels: normal=2, sybil=3, ddos=1, blackhole=0
        |
        v
  Separate: X (features), y (labels), groups (car_id)
        |
        v
  Handle missing values (median imputation - none found)
        |
        v
  Mutual Information feature selection
        |
        v
  Stratified Group Split (80/20 by car_id)
        |
        v
  StandardScaler (fit on train only, transform both)
        |
        v
  Model Training
```

### Final Feature Set (7 features)

| Feature           | MI Score | Description                                      |
|-------------------|----------|--------------------------------------------------|
| `avg_speed`       | 0.287    | Average speed of the vehicle over the session     |
| `distance_moved`  | 0.219    | Total distance traveled by the vehicle            |
| `speed_mps`       | 0.157    | Instantaneous speed in meters per second          |
| `position_changes`| 0.138    | Number of times the vehicle changed position      |
| `pkt_size_bytes`  | 0.121    | Size of the transmitted packet in bytes           |
| `pos_y`           | 0.079    | Y-coordinate of the vehicle on the road network   |
| `pos_x`           | 0.076    | X-coordinate of the vehicle on the road network   |

### Why These Features Make Sense for IDS

- **Speed/movement features** (`avg_speed`, `speed_mps`, `distance_moved`, `position_changes`):
  Sybil attackers create fake vehicle identities that may have unrealistic movement
  patterns. Blackhole attackers may exhibit different mobility (e.g., stationary).
  
- **Packet size** (`pkt_size_bytes`): DDoS attacks often use crafted packets of specific
  sizes. Normal beacons have a fixed size (92 bytes), while attack packets may differ.

- **Position** (`pos_x`, `pos_y`): While these have lower MI scores, different vehicle
  positions on the road network correlate with different traffic patterns and attack
  exposure. NOTE: These could be simulation-specific if attackers are always initialized
  in certain zones.

---

## 6. Feature Selection

### Method: Mutual Information

Mutual Information (MI) measures how much knowing the value of a feature reduces
uncertainty about the class label. Unlike correlation, MI captures non-linear
relationships.

```
MI(X, Y) = sum over x,y of p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

- MI = 0: Feature is independent of the label (useless)
- MI > 0: Feature carries some information about the label
- Higher MI = more informative feature

### Selection Threshold

Features with MI > 0.01 are kept. All 7 remaining features passed this threshold.
If fewer than 5 features had passed, the top 15 would be kept as a fallback.

### Computation

MI is computed on a 50,000-row subsample (for speed) with `random_state=42` for
reproducibility.

---

## 7. Train/Test Split Strategy

### The Problem with Random Splits

In this dataset, each `car_id` generates ~3,500 rows on average. A random 80/20 split
would scatter rows from the same car into both train and test:

```
Car #5:  [row1, row2, row3, ..., row3500]
          ^^^^^^^^^^^^                       -> train
                        ^^^^^^^^^^^^^^^^^^^^  -> test
```

The model would learn: "Car #5 always behaves like THIS" rather than learning general
attack patterns. This is **vehicle identity leakage**.

### The Problem with Plain GroupShuffleSplit

`GroupShuffleSplit` fixes the identity leak by keeping all rows from a car in one split.
But it doesn't consider labels:

```
With 133 total cars:
  - 5 blackhole cars
  - 8 DDoS cars
  - 70 normal cars
  - 50 sybil cars

Plain GroupShuffleSplit might put ALL 8 DDoS cars in training -> 0 DDoS test samples!
```

This is exactly what happened in our first run (DDoS support = 0 in test).

### Our Solution: Stratified Group Split

We split cars **within each label group** independently:

```
For each attack type:
  1. Get all car_ids with that label
  2. Shuffle them
  3. Put ~20% into test, ~80% into train (at least 1 car per group)
  4. Combine all train cars, combine all test cars
```

### Results of the Split

| Metric              | Value   |
|---------------------|---------|
| Train samples       | 375,261 |
| Test samples        | 91,193  |
| Cars in train       | 107     |
| Cars in test        | 26      |
| Car overlap         | 0       |

Per-class counts after split:

| Class     | Train   | Test   |
|-----------|---------|--------|
| blackhole | 9,854   | 2,495  |
| ddos      | 19,816  | 4,861  |
| normal    | 225,963 | 54,515 |
| sybil     | 119,628 | 29,322 |

All classes are represented in both train and test.

---

## 8. Model Training

### 8a. Random Forest

| Hyperparameter    | Value      | Rationale                                        |
|-------------------|------------|--------------------------------------------------|
| n_estimators      | 200        | Sufficient ensemble size for stable predictions  |
| max_depth         | 25         | Deep enough for complex patterns, not overfitting|
| min_samples_split | 5          | Prevents splits on tiny groups                   |
| min_samples_leaf  | 2          | Ensures leaf nodes have sufficient samples       |
| class_weight      | "balanced" | Automatically adjusts weights inversely proportional to class frequency |
| random_state      | 42         | Reproducibility                                  |
| n_jobs            | -1         | Use all CPU cores                                |

**How class_weight='balanced' works:**

```
weight_i = total_samples / (n_classes * count_of_class_i)

Example for blackhole (12,349 out of 466,454):
  weight = 466,454 / (4 * 12,349) = 9.44

Example for normal (280,478 out of 466,454):
  weight = 466,454 / (4 * 280,478) = 0.42
```

This means blackhole samples count ~22x more than normal samples during training,
compensating for the class imbalance without generating synthetic data.

### 8b. XGBoost

| Hyperparameter    | Value          | Rationale                                    |
|-------------------|----------------|----------------------------------------------|
| n_estimators      | 300            | More boosting rounds for gradual learning    |
| max_depth         | 10             | Shallower trees (boosting builds depth iteratively) |
| learning_rate     | 0.1            | Standard step size                           |
| subsample         | 0.8            | Row subsampling to reduce overfitting        |
| colsample_bytree  | 0.8            | Feature subsampling to reduce overfitting    |
| objective         | multi:softmax  | Multi-class classification                   |
| eval_metric       | mlogloss       | Multi-class log loss                         |
| sample_weight     | (manual)       | Same balanced weighting as RF, applied per-sample |

### Feature Scaling

`StandardScaler` is applied: `X_scaled = (X - mean) / std`

- **Fitted on training data only** to prevent test data information from leaking into
  the scaler statistics
- Applied to both train and test data using the same transform
- Important for algorithms sensitive to feature magnitude (though tree-based models
  are generally scale-invariant, it's good practice for pipeline consistency)

---

## 9. Results & Interpretation

### Overall Performance

| Model         | Accuracy | F1 (Weighted) | F1 (Macro) | Training Time |
|---------------|----------|---------------|------------|---------------|
| Random Forest | 82.6%    | 82.7%         | 71.0%      | ~50s          |
| XGBoost       | 78.9%    | 79.9%         | 69.1%      | ~25s          |

### Per-Class Performance (Random Forest)

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| blackhole | 1.00      | 1.00   | 1.00     | 2,495   |
| ddos      | 0.11      | 0.11   | 0.11     | 4,861   |
| normal    | 0.84      | 0.88   | 0.86     | 54,515  |
| sybil     | 0.92      | 0.83   | 0.87     | 29,322  |

### Interpretation

**Blackhole (F1 = 1.00):** Perfectly detected. Blackhole attackers drop packets, which
creates a very distinct behavioral pattern in speed/movement features - they likely
remain stationary or exhibit very different mobility compared to normal vehicles.

**Normal (F1 = 0.86):** Well classified. Most normal traffic is correctly identified,
with some confusion with sybil attacks (false positives).

**Sybil (F1 = 0.87):** Good detection. Sybil attacks create fake identities with
potentially unrealistic movement patterns (e.g., abnormal speeds, impossible position
changes), which the movement-based features capture well.

**DDoS (F1 = 0.11):** Poor detection. This is the weakest class because:
1. Only 5.3% of the dataset (small class)
2. DDoS attackers flood packets but their vehicle dynamics (speed, position) may look
   identical to normal vehicles
3. The features that would best identify DDoS (packet rates, send counts) were correctly
   removed as simulation artifacts
4. **Improving DDoS detection would require feature engineering** - e.g., computing
   inter-arrival time statistics, packet burst patterns, or sliding-window packet counts
   from raw packet logs

### Why 82% is Realistic

| Version | Accuracy | What changed |
|---------|----------|-------------|
| Original (leaky) | 100% | Leaky columns gave away the answer |
| After dropping leaky cols | 82% | Leaky features removed, but broken split |
| After fixing split (no stratification) | 72% | GroupShuffleSplit but DDoS missing from test |
| **Final (stratified group split)** | **82.6%** | Proper stratified group split, all classes tested |

The ~82% accuracy with a proper leakage-free pipeline is realistic for this synthetic
dataset using only vehicle dynamics features.

---

## 10. File Structure

```
Intrusion Detection/
  |
  |-- v2x_dataset_Main_run.csv          # Raw dataset (466K rows)
  |-- Intrusion_Detection.ipynb         # Original notebook (has syntax errors)
  |-- Intrusion_Detection_Clean.ipynb   # Fixed clean notebook (run this)
  |-- train_ids_model.py                # Standalone training script
  |-- generate_notebook.py              # Script that generated the clean notebook
  |-- PIPELINE_DOCUMENTATION.md         # This file
  |
  |-- model_outputs/
       |-- best_model.joblib            # Saved Random Forest model
       |-- scaler.joblib                # Fitted StandardScaler
       |-- label_encoder.joblib         # Fitted LabelEncoder
       |-- feature_list.joblib          # List of selected feature names
       |-- label_distribution.png       # Bar chart of class distribution
       |-- mi_scores.png               # Mutual Information bar chart
       |-- rf_confusion_matrix.png      # Random Forest confusion matrix
       |-- xgb_confusion_matrix.png     # XGBoost confusion matrix
       |-- model_comparison.png         # Side-by-side metric comparison
```

### How to Run

**Option 1: Notebook**
```
jupyter notebook Intrusion_Detection_Clean.ipynb
# Run all cells
```

**Option 2: Script**
```
python train_ids_model.py
```

### How to Use the Saved Model for Inference

```python
import joblib
import pandas as pd

model    = joblib.load('model_outputs/best_model.joblib')
scaler   = joblib.load('model_outputs/scaler.joblib')
encoder  = joblib.load('model_outputs/label_encoder.joblib')
features = joblib.load('model_outputs/feature_list.joblib')

# new_data is a DataFrame with the required feature columns
new_data_scaled = scaler.transform(new_data[features])
predictions     = encoder.inverse_transform(model.predict(new_data_scaled))
```

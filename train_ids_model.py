"""
VANET Intrusion Detection System - Model Training Pipeline (Leakage-Free v2)
=============================================================================
Dataset : v2x_dataset_Main_run.csv  (466,454 rows x 50 columns)
Labels  : normal | sybil | ddos | blackhole
Approach: Random Forest + XGBoost
          - Aggressive leakage column removal
          - Group-based train/test split by car_id (prevents temporal leakage)
          - NO SMOTE (uses class_weight='balanced')
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed - will skip XGBoost model.")

# ======================================================================
# 1. LOAD DATA
# ======================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "v2x_dataset_Main_run.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  VANET IDS - Training Pipeline v2 (Leakage-Free, No SMOTE)")
print("=" * 70)

t0 = time.time()
print(f"\n[1/8] Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}  |  Loaded in {time.time()-t0:.1f}s")

# ======================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ======================================================================
print("\n[2/8] Exploratory Data Analysis ...")

label_counts = df["label"].value_counts()
print(f"\n      Label distribution:\n{label_counts.to_string()}")

# Car-level analysis
car_label = df.groupby("car_id")["label"].first()
print(f"\n      Unique cars per label:")
print(f"{df.groupby('label')['car_id'].nunique().to_string()}")
print(f"      Total unique car_ids: {df['car_id'].nunique()}")
print(f"\n      WARNING: Only 5 cars per attack type!")
print(f"      -> Random split leaks car identity into test set")
print(f"      -> Using GROUP-BASED split by car_id instead")

# Save label distribution chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
label_counts.plot.bar(ax=ax, color=colors, edgecolor="black")
ax.set_title("Label Distribution in V2X Dataset", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Attack Type")
for i, v in enumerate(label_counts):
    ax.text(i, v + 2000, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=150)
plt.close(fig)
print("      [OK] Saved label_distribution.png")

# ======================================================================
# 3. PREPROCESSING - AGGRESSIVE LEAKAGE REMOVAL
# ======================================================================
print("\n[3/8] Preprocessing (aggressive leakage removal) ...")

# CATEGORY 1: Obvious identifiers and pre-computed labels
# CATEGORY 2: Attacker-side counters (not observable by IDS in deployment)
# CATEGORY 3: Derived ratios that directly encode attack type
# CATEGORY 4: Zero/near-zero variance columns
# CATEGORY 5: Global/Network-wide metrics (unrealistic for a single-node IDS)
# CATEGORY 6: Simulation artifacts (e.g. blackhole=0, sybil=60)
LEAK_COLS = [
    # Category 1: Identifiers / pre-computed labels
    "event_type", "is_fake", "fake_id",
    "sybil_score", "ddos_score", "blackhole_score",
    # Category 2: Attacker-side counters
    "my_attack_packets", "my_fake_ids_sent", "my_ddos_packets", "my_dropped",
    "attack_rate", "ddos_rate_actual",
    # Category 3: Derived ratios directly encoding attack identity
    "fake_ratio", "ddos_ratio", "blackhole_ratio", "drop_ratio",
    "norm_attack_rate",
    # Category 4: Zero/near-zero variance
    "avg_pkt_size_sent", "norm_pkt_size",
    # Category 5: Global/Network-wide metrics
    "global_beacons_sent", "global_beacons_rcvd", 
    "global_attack_pkts", "global_dropped", "network_load",
    # Category 6: Simulation artifacts (deterministic values for attacks)
    "my_beacons_sent", "my_beacons_received", 
    "beacon_send_rate", "beacon_recv_rate", 
    "total_bytes_sent", "total_bytes_rcvd", 
    "avg_pkt_size_rcvd", "throughput_bps", 
    "send_recv_ratio", "active_duration", 
    "beacon_regularity", "norm_send_rate", 
    "norm_recv_rate", "norm_bytes_sent",
]

# Keep car_id for group splitting but don't use it as a feature
dropped = [c for c in LEAK_COLS if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"      Dropped {len(dropped)} leaky/non-predictive columns:")
for c in dropped:
    print(f"        - {c}")

# --- Encode target ---
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])
label_names = list(le.classes_)
print(f"\n      Label encoding: {dict(zip(label_names, le.transform(label_names)))}")

# --- Feature / Target split (exclude car_id and label from features) ---
FEATURE_COLS = [c for c in df.columns if c not in ("label", "label_enc", "car_id")]
X = df[FEATURE_COLS].copy()
y = df["label_enc"].copy()
groups = df["car_id"].copy()  # for group-based splitting

# --- Handle missing values ---
if X.isnull().sum().sum() > 0:
    X.fillna(X.median(), inplace=True)
    print("      Filled missing values with column medians.")

# --- Handle remaining object columns ---
obj_cols = X.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print(f"      Encoding remaining categorical columns: {obj_cols}")
    for col in obj_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"\n      Final feature matrix: {X.shape}")

# ======================================================================
# 4. FEATURE IMPORTANCE (Mutual Information)
# ======================================================================
print("\n[4/8] Feature importance (Mutual Information) ...")

sample_idx = np.random.RandomState(42).choice(len(X), size=min(50_000, len(X)), replace=False)
mi_scores = mutual_info_classif(X.iloc[sample_idx], y.iloc[sample_idx], random_state=42)
mi_df = pd.DataFrame({"feature": FEATURE_COLS, "mi_score": mi_scores}).sort_values(
    "mi_score", ascending=False
)
print(mi_df.to_string(index=False))

# Select top features (MI > 0.01)
TOP_FEATURES = mi_df.loc[mi_df["mi_score"] > 0.01, "feature"].tolist()
if len(TOP_FEATURES) < 5:
    TOP_FEATURES = mi_df.head(10)["feature"].tolist()
print(f"\n      Selected {len(TOP_FEATURES)} features with MI > 0.01")

# Save feature importance chart
fig, ax = plt.subplots(figsize=(10, 8))
plot_df = mi_df.head(20)
ax.barh(plot_df["feature"][::-1], plot_df["mi_score"][::-1], color="#3498db", edgecolor="black")
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Importance - Mutual Information (Leakage-Free)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)
print("      [OK] Saved feature_importance.png")

X = X[TOP_FEATURES]

# ======================================================================
# 5. GROUP-BASED TRAIN/TEST SPLIT (by car_id)
# ======================================================================
print("\n[5/8] Group-based Train/Test split by car_id (80/20) ...")
print("      This ensures NO car appears in both train and test sets.")
print("      Critical because there are only 5 attack cars per type.")

# GroupShuffleSplit: splits by groups (car_id) not individual rows
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Show which car_ids went where
train_cars = groups.iloc[train_idx].unique()
test_cars = groups.iloc[test_idx].unique()
print(f"\n      Train cars ({len(train_cars)}): {sorted(train_cars)[:10]}{'...' if len(train_cars)>10 else ''}")
print(f"      Test  cars ({len(test_cars)}): {sorted(test_cars)[:10]}{'...' if len(test_cars)>10 else ''}")
print(f"      Overlap: {len(set(train_cars) & set(test_cars))} cars (should be 0)")

print(f"\n      Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"      Train label distribution:")
train_labels = pd.Series(y_train).map(dict(enumerate(label_names)))
print(f"{train_labels.value_counts().to_string()}")
print(f"\n      Test label distribution:")
test_labels = pd.Series(y_test.values).map(dict(enumerate(label_names)))
print(f"{test_labels.value_counts().to_string()}")

# Check if all classes are represented in both splits
train_classes = set(y_train.unique())
test_classes = set(y_test.unique())
if train_classes != test_classes:
    print(f"\n      WARNING: Not all classes in both splits!")
    print(f"      Train classes: {train_classes}")
    print(f"      Test  classes: {test_classes}")
    missing = test_classes - train_classes
    if missing:
        print(f"      Missing from train: {missing} - results may be unreliable for these classes")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ======================================================================
# 6. MODEL TRAINING & EVALUATION
# ======================================================================
print("\n[6/8] Training models (class_weight='balanced', no SMOTE) ...")

results = {}


def evaluate_model(name, model, Xtr, ytr, Xte, yte):
    """Train, predict, and report metrics."""
    t_start = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t_start

    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    f1_w = f1_score(yte, y_pred, average="weighted")
    f1_mac = f1_score(yte, y_pred, average="macro")

    print(f"\n  -- {name} --")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1_w:.4f}")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print(f"  Training time : {train_time:.1f}s")
    
    # Only include classes present in test set
    present_labels = sorted(yte.unique())
    present_names = [label_names[i] for i in present_labels]
    print(f"\n{classification_report(yte, y_pred, target_names=present_names, labels=present_labels)}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(yte, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=present_names)
    disp.plot(ax=ax, cmap="Blues", values_format=",")
    ax.set_title(f"{name} - Confusion Matrix (Group Split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {fname}")

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_mac,
        "train_time": train_time,
        "y_pred": y_pred,
    }
    return model


# -- 6a. Random Forest --
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
evaluate_model("Random Forest", rf, X_train_sc, y_train, X_test_sc, y_test)

# -- 6b. XGBoost --
if HAS_XGB:
    class_counts = np.bincount(y_train)
    total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = np.array([total / (n_classes * class_counts[yi]) for yi in y_train])

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(label_names),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    
    print("\n  -- XGBoost (with sample weights) --")
    t_start = time.time()
    xgb_model.fit(X_train_sc, y_train, sample_weight=sample_weights)
    train_time = time.time() - t_start

    y_pred = xgb_model.predict(X_test_sc)
    present_labels = sorted(y_test.unique())
    present_names = [label_names[i] for i in present_labels]
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_mac = f1_score(y_test, y_pred, average="macro")

    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1_w:.4f}")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print(f"  Training time : {train_time:.1f}s")
    print(f"\n{classification_report(y_test, y_pred, target_names=present_names, labels=present_labels)}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=present_names)
    disp.plot(ax=ax, cmap="Blues", values_format=",")
    ax.set_title("XGBoost - Confusion Matrix (Group Split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_xgboost.png"), dpi=150)
    plt.close(fig)
    print("  [OK] Saved confusion_matrix_xgboost.png")

    results["XGBoost"] = {
        "model": xgb_model,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_mac,
        "train_time": train_time,
        "y_pred": y_pred,
    }

# ======================================================================
# 7. CROSS-VALIDATION (Group-based)
# ======================================================================
print("\n[7/8] Note on Cross-Validation ...")
print("      With only 5 cars per attack type, proper group-based CV")
print("      is limited. The group split above is the honest evaluation.")

# ======================================================================
# 8. SAVE BEST MODEL + ARTIFACTS
# ======================================================================
print("\n[8/8] Saving artifacts ...")

best_name = max(results, key=lambda k: results[k]["f1_weighted"])
best_model = results[best_name]["model"]

joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
joblib.dump(TOP_FEATURES, os.path.join(OUTPUT_DIR, "feature_list.joblib"))

print(f"      [OK] Best model: {best_name} (F1w={results[best_name]['f1_weighted']:.4f})")
print(f"      [OK] Saved to {OUTPUT_DIR}/")

# -- Comparison bar chart --
if len(results) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    model_names = list(results.keys())
    metrics = ["accuracy", "f1_weighted", "f1_macro"]
    titles = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals, color=["#2ecc71", "#3498db"], edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(max(0, min(vals) - 0.15), 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print("      [OK] Saved model_comparison.png")

# -- Final summary --
elapsed = time.time() - t0
print("\n" + "=" * 70)
print("  TRAINING COMPLETE (Leakage-Free v2)")
print(f"  Total time     : {elapsed:.1f}s")
print(f"  Best model     : {best_name}")
print(f"  Accuracy       : {results[best_name]['accuracy']:.4f}")
print(f"  F1 (weighted)  : {results[best_name]['f1_weighted']:.4f}")
print(f"  F1 (macro)     : {results[best_name]['f1_macro']:.4f}")
print(f"  Features used  : {len(TOP_FEATURES)}")
print(f"  SMOTE          : NOT USED (class_weight='balanced')")
print(f"  Leaked cols    : {len(dropped)} removed")
print(f"  Split method   : GroupShuffleSplit by car_id")
print(f"  Output dir     : {OUTPUT_DIR}")
print("=" * 70)

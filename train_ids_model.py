"""
VANET Intrusion Detection System - Clean Training Script
=========================================================
Dataset  : v2x_dataset_Main_run.csv (466,454 rows x 50 cols)
Labels   : normal | sybil | DDoS | blackhole
Models   : Random Forest + XGBoost (NO SMOTE)

Issues fixed vs. the original notebook:
  1. 6 syntax errors (missing parentheses) repaired
  2. GroupShuffleSplit now actually used (stratified by class)
  3. timestamp dropped (temporal / simulation leakage)
  4. norm_speed dropped (redundant linear transform of speed_mps)
  5. Class imbalance via class_weight / sample_weight (no SMOTE)
"""

import os, time, warnings
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, ConfusionMatrixDisplay)
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("All imports loaded successfully.")

# --- 2. Load Dataset ---
t0 = time.time()
df = pd.read_csv("v2x_dataset_Main_run.csv")
print(f"Shape: {df.shape}  |  Loaded in {time.time()-t0:.1f}s")
print(df.head())

# --- 3. EDA ---
label_counts = df["label"].value_counts()
print("\nLabel Distribution:")
print(label_counts)
print(f"\nTotal samples: {len(df):,}")

fig, ax = plt.subplots(figsize=(8,4))
colors = ["#2ecc71","#e74c3c","#f39c12","#9b59b6"]
label_counts.plot.bar(ax=ax, color=colors, edgecolor="black")
ax.set_title("Label Distribution in V2X Dataset", fontsize=14, fontweight="bold")
ax.set_ylabel("Count"); ax.set_xlabel("Attack Type")
for i,v in enumerate(label_counts):
    ax.text(i, v+2000, f"{v:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"))
plt.close()

missing = df.isnull().sum()
print(f"\nTotal missing values: {missing.sum()}")
if missing.sum() == 0:
    print("No missing values!")
print(df.describe())

# --- 4. Drop Leaky / Non-Predictive Columns ---
LEAK_COLS = [
    "event_type","is_fake","fake_id",
    "sybil_score","ddos_score","blackhole_score",
    "my_attack_packets","my_fake_ids_sent","my_ddos_packets","my_dropped",
    "attack_rate","ddos_rate_actual",
    "fake_ratio","ddos_ratio","blackhole_ratio","drop_ratio","norm_attack_rate",
    "avg_pkt_size_sent","norm_pkt_size",
    "global_beacons_sent","global_beacons_rcvd",
    "global_attack_pkts","global_dropped","network_load",
    "my_beacons_sent","my_beacons_received",
    "beacon_send_rate","beacon_recv_rate",
    "total_bytes_sent","total_bytes_rcvd",
    "avg_pkt_size_rcvd","throughput_bps",
    "send_recv_ratio","active_duration",
    "beacon_regularity","norm_send_rate",
    "norm_recv_rate","norm_bytes_sent",
    "timestamp",   # temporal / simulation leakage
    "norm_speed",  # redundant linear transform of speed_mps
]
dropped = [c for c in LEAK_COLS if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"\nDropped {len(dropped)} leaky / ID / redundant columns")
print(f"Remaining columns: {df.shape[1]}")
print("Feature columns:", [c for c in df.columns if c not in ("label","car_id")])

# --- 5. Encode Labels ---
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])
label_names = list(le.classes_)
print(f"\nLabel encoding: {dict(zip(label_names, le.transform(label_names)))}")

# --- 6. Feature / Target Split ---
FEATURE_COLS = [c for c in df.columns if c not in ("label","label_enc","car_id")]
X = df[FEATURE_COLS].copy()
y = df["label_enc"].copy()
groups = df["car_id"].copy()
if X.isnull().sum().sum() > 0:
    X.fillna(X.median(), inplace=True)
    print("Filled missing values with column medians.")
obj_cols = X.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print(f"Encoding categorical columns: {obj_cols}")
    for col in obj_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
print(f"\nFinal feature matrix: {X.shape}")
print("Features:", FEATURE_COLS)

# --- 7. Feature Selection (Mutual Information) ---
sample_idx = np.random.RandomState(42).choice(len(X), size=min(50000,len(X)), replace=False)
mi_scores = mutual_info_classif(X.iloc[sample_idx], y.iloc[sample_idx], random_state=42)
mi_df = pd.DataFrame({"feature": FEATURE_COLS, "mi_score": mi_scores}).sort_values("mi_score", ascending=False)
print("\nMutual Information scores:")
print(mi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8,5))
top = mi_df.head(15)
ax.barh(top["feature"][::-1], top["mi_score"][::-1], color="#3498db", edgecolor="black")
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top Features - Mutual Information", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mi_scores.png"))
plt.close()

TOP_FEATURES = mi_df.loc[mi_df["mi_score"]>0.01, "feature"].tolist()
if len(TOP_FEATURES) < 5:
    TOP_FEATURES = mi_df.head(15)["feature"].tolist()
print(f"\nSelected {len(TOP_FEATURES)} features: {TOP_FEATURES}")
X = X[TOP_FEATURES]

# --- 8. Train/Test Split - STRATIFIED GROUP split by car_id ---
# Why stratified-group?
#   1. Plain random split: same car in train+test -> vehicle identity leak
#   2. Plain GroupShuffleSplit: doesn't stratify by label -> entire class
#      (e.g. DDoS) can have 0 test samples
# Solution: split cars within each label group ~80/20
rng = np.random.RandomState(42)
car_labels = df.groupby("car_id")["label_enc"].agg(lambda s: s.mode()[0])
train_cars_list, test_cars_list = [], []
for lv in sorted(car_labels.unique()):
    cars = car_labels[car_labels == lv].index.values.copy()
    rng.shuffle(cars)
    n_test = max(1, int(len(cars)*0.20))
    test_cars_list.extend(cars[:n_test])
    train_cars_list.extend(cars[n_test:])
train_car_set = set(train_cars_list)
test_car_set = set(test_cars_list)
train_idx = np.where(groups.isin(train_car_set))[0]
test_idx = np.where(groups.isin(test_car_set))[0]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Unique cars in train: {groups.iloc[train_idx].nunique()}")
print(f"Unique cars in test : {groups.iloc[test_idx].nunique()}")
overlap = set(groups.iloc[train_idx].unique()) & set(groups.iloc[test_idx].unique())
print(f"Car overlap: {len(overlap)}  (should be 0)")
assert len(overlap) == 0, "LEAKAGE: same car in train AND test!"

print("\nPer-class sample counts:")
for cn in label_names:
    ci = le.transform([cn])[0]
    print(f"  {cn:12s}  train={int((y_train==ci).sum()):7,}  test={int((y_test==ci).sum()):7,}")
assert all((y_test==c).sum()>0 for c in range(len(label_names))), "Class with 0 test samples!"

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- 9. Model Training & Evaluation ---
results = {}

def evaluate_model(name, model, Xtr, ytr, Xte, yte):
    t_start = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t_start
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    f1_w = f1_score(yte, y_pred, average="weighted")
    f1_mac = f1_score(yte, y_pred, average="macro")
    results[name] = {"model":model, "accuracy":acc, "f1_weighted":f1_w,
                     "f1_macro":f1_mac, "train_time":train_time, "y_pred":y_pred}
    return acc, f1_w, f1_mac, train_time, y_pred

# 9a. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_split=5,
                            min_samples_leaf=2, class_weight="balanced",
                            random_state=42, n_jobs=-1)
acc,f1_w,f1_mac,t,y_pred_rf = evaluate_model("Random Forest",rf,X_train_sc,y_train,X_test_sc,y_test)
print("\n-- Random Forest --")
print(f"Accuracy: {acc:.4f}  F1w: {f1_w:.4f}  F1m: {f1_mac:.4f}  Time: {t:.1f}s")
print(classification_report(y_test, y_pred_rf, target_names=label_names))

fig,ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_rf), display_labels=label_names).plot(ax=ax,cmap="Blues",values_format=",")
ax.set_title("Random Forest - Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"rf_confusion_matrix.png")); plt.close()

# 9b. XGBoost
class_counts = np.bincount(y_train)
sample_weights = np.array([len(y_train)/(len(class_counts)*class_counts[yi]) for yi in y_train])
xgb = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, objective="multi:softmax", num_class=len(label_names),
                    eval_metric="mlogloss", use_label_encoder=False, random_state=42, n_jobs=-1)
t_start = time.time()
xgb.fit(X_train_sc, y_train, sample_weight=sample_weights)
t = time.time()-t_start
y_pred_xgb = xgb.predict(X_test_sc)
acc = accuracy_score(y_test,y_pred_xgb)
f1_w = f1_score(y_test,y_pred_xgb,average="weighted")
f1_mac = f1_score(y_test,y_pred_xgb,average="macro")
results["XGBoost"] = {"model":xgb,"accuracy":acc,"f1_weighted":f1_w,"f1_macro":f1_mac,"train_time":t,"y_pred":y_pred_xgb}
print("\n-- XGBoost --")
print(f"Accuracy: {acc:.4f}  F1w: {f1_w:.4f}  F1m: {f1_mac:.4f}  Time: {t:.1f}s")
print(classification_report(y_test, y_pred_xgb, target_names=label_names))

fig,ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_xgb), display_labels=label_names).plot(ax=ax,cmap="Greens",values_format=",")
ax.set_title("XGBoost - Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"xgb_confusion_matrix.png")); plt.close()

# --- 10. Comparison ---
comp = pd.DataFrame({n:{"Acc":r["accuracy"],"F1w":r["f1_weighted"],"F1m":r["f1_macro"],"Time":round(r["train_time"],1)} for n,r in results.items()}).T
print("\nModel Comparison:")
print(comp.to_string())

fig,axes = plt.subplots(1,3,figsize=(14,4))
for ax,m,title in zip(axes,["accuracy","f1_weighted","f1_macro"],["Accuracy","F1 Weighted","F1 Macro"]):
    vals = [results[n][m] for n in results]
    bars = ax.bar(list(results.keys()), vals, color=["#2ecc71","#3498db"], edgecolor="black")
    ax.set_title(title, fontweight="bold"); ax.set_ylim(0.5,1.05)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"model_comparison.png")); plt.close()

# --- 11. Save ---
best_name = max(results, key=lambda k: results[k]["f1_weighted"])
best_model = results[best_name]["model"]
joblib.dump(best_model, os.path.join(OUTPUT_DIR,"best_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR,"scaler.joblib"))
joblib.dump(le, os.path.join(OUTPUT_DIR,"label_encoder.joblib"))
joblib.dump(TOP_FEATURES, os.path.join(OUTPUT_DIR,"feature_list.joblib"))
print(f"\nBest model: {best_name}")
print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"F1w: {results[best_name]['f1_weighted']:.4f}  F1m: {results[best_name]['f1_macro']:.4f}")
print(f"Features: {len(TOP_FEATURES)}")
print(f"Saved to: {OUTPUT_DIR}/")

# --- 12. Inference Demo ---
lm = joblib.load(os.path.join(OUTPUT_DIR,"best_model.joblib"))
ls = joblib.load(os.path.join(OUTPUT_DIR,"scaler.joblib"))
ll = joblib.load(os.path.join(OUTPUT_DIR,"label_encoder.joblib"))
lf = joblib.load(os.path.join(OUTPUT_DIR,"feature_list.joblib"))
sample = X_test.sample(5, random_state=123)
preds = ll.inverse_transform(lm.predict(ls.transform(sample[lf])))
actuals = ll.inverse_transform(y_test.loc[sample.index].values)
print("\nInference Demo:")
print(pd.DataFrame({"Predicted":preds,"Actual":actuals,"Correct":preds==actuals}, index=sample.index).to_string())

"""Generate the Intrusion_Detection_Clean.ipynb notebook from the training script logic."""
import json

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type":"markdown","metadata":{},"source": source.split("\n") if isinstance(source,str) else source}
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": source.split("\n") if isinstance(source,str) else source}

def lines(*args):
    return [l+"\n" for l in args[:-1]] + [args[-1]]

nb = {
 "nbformat":4,"nbformat_minor":5,
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python","version":"3.13.0"}},
 "cells":[]
}
c = nb["cells"]

# Title
c.append(cell("# VANET Intrusion Detection System\n\n**Dataset:** v2x_dataset_Main_run.csv (466K rows x 49 cols)  \n**Labels:** normal | sybil | DDoS | blackhole  \n**Models:** Random Forest + XGBoost (no SMOTE)  \n\n**Fixes applied:**\n1. All syntax errors repaired\n2. Stratified group split by car_id (prevents vehicle identity leakage)\n3. `timestamp` dropped (temporal/simulation leakage)\n4. `norm_speed` dropped (redundant linear transform of speed_mps)\n5. Class imbalance handled via class_weight / sample_weight","markdown"))

# Imports
c.append(cell(lines(
    "import os, time, warnings",
    "import joblib, numpy as np, pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from sklearn.model_selection import GroupShuffleSplit",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.metrics import (classification_report, confusion_matrix,",
    "                             accuracy_score, f1_score, ConfusionMatrixDisplay)",
    "from sklearn.feature_selection import mutual_info_classif",
    "from xgboost import XGBClassifier",
    "",
    "warnings.filterwarnings('ignore')",
    "%matplotlib inline",
    "plt.rcParams['figure.dpi'] = 120",
    "OUTPUT_DIR = 'model_outputs'",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)",
    "print('All imports loaded.')"
)))

# Load
c.append(cell("## 1. Load Dataset","markdown"))
c.append(cell(lines(
    "t0 = time.time()",
    "df = pd.read_csv('v2x_dataset_Main_run.csv')",
    "print(f'Shape: {df.shape}  |  Loaded in {time.time()-t0:.1f}s')",
    "df.head()"
)))

# EDA
c.append(cell("## 2. Exploratory Data Analysis","markdown"))
c.append(cell(lines(
    "label_counts = df['label'].value_counts()",
    "print('Label Distribution:')",
    "print(label_counts)",
    "print(f'\\nTotal samples: {len(df):,}')"
)))

c.append(cell(lines(
    "fig, ax = plt.subplots(figsize=(8,4))",
    "colors = ['#2ecc71','#e74c3c','#f39c12','#9b59b6']",
    "label_counts.plot.bar(ax=ax, color=colors, edgecolor='black')",
    "ax.set_title('Label Distribution in V2X Dataset', fontsize=14, fontweight='bold')",
    "ax.set_ylabel('Count'); ax.set_xlabel('Attack Type')",
    "for i,v in enumerate(label_counts):",
    "    ax.text(i, v+2000, f'{v:,}', ha='center', fontsize=9)",
    "plt.tight_layout()",
    "plt.show()"
)))

c.append(cell(lines(
    "missing = df.isnull().sum()",
    "print(f'Total missing values: {missing.sum()}')",
    "if missing.sum() == 0:",
    "    print('No missing values!')",
    "df.describe()"
)))

# Drop columns
c.append(cell("## 3. Drop Leaky / Non-Predictive Columns","markdown"))
c.append(cell(lines(
    "LEAK_COLS = [",
    "    # Identifiers / pre-computed labels",
    "    'event_type','is_fake','fake_id',",
    "    'sybil_score','ddos_score','blackhole_score',",
    "    # Attacker-side counters",
    "    'my_attack_packets','my_fake_ids_sent','my_ddos_packets','my_dropped',",
    "    'attack_rate','ddos_rate_actual',",
    "    # Derived ratios directly encoding attack identity",
    "    'fake_ratio','ddos_ratio','blackhole_ratio','drop_ratio','norm_attack_rate',",
    "    # Zero/near-zero variance",
    "    'avg_pkt_size_sent','norm_pkt_size',",
    "    # Global/Network-wide metrics",
    "    'global_beacons_sent','global_beacons_rcvd',",
    "    'global_attack_pkts','global_dropped','network_load',",
    "    # Simulation artifacts",
    "    'my_beacons_sent','my_beacons_received',",
    "    'beacon_send_rate','beacon_recv_rate',",
    "    'total_bytes_sent','total_bytes_rcvd',",
    "    'avg_pkt_size_rcvd','throughput_bps',",
    "    'send_recv_ratio','active_duration',",
    "    'beacon_regularity','norm_send_rate',",
    "    'norm_recv_rate','norm_bytes_sent',",
    "    # Temporal leakage - simulation timestamp",
    "    'timestamp',",
    "    # Redundant (linear transform of speed_mps)",
    "    'norm_speed',",
    "]",
    "dropped = [c for c in LEAK_COLS if c in df.columns]",
    "df.drop(columns=dropped, inplace=True)",
    "print(f'Dropped {len(dropped)} leaky / ID / redundant columns')",
    "print(f'Remaining columns: {df.shape[1]}')",
    "print('Feature columns:', [c for c in df.columns if c not in ('label','car_id')])"
)))

# Encode
c.append(cell("## 4. Encode Labels","markdown"))
c.append(cell(lines(
    "le = LabelEncoder()",
    "df['label_enc'] = le.fit_transform(df['label'])",
    "label_names = list(le.classes_)",
    "print(f'Label encoding: {dict(zip(label_names, le.transform(label_names)))}')"
)))

# Feature split
c.append(cell("## 5. Feature / Target Split","markdown"))
c.append(cell(lines(
    "FEATURE_COLS = [c for c in df.columns if c not in ('label','label_enc','car_id')]",
    "X = df[FEATURE_COLS].copy()",
    "y = df['label_enc'].copy()",
    "groups = df['car_id'].copy()",
    "if X.isnull().sum().sum() > 0:",
    "    X.fillna(X.median(), inplace=True)",
    "    print('Filled missing values with column medians.')",
    "obj_cols = X.select_dtypes(include='object').columns.tolist()",
    "if obj_cols:",
    "    print(f'Encoding categorical columns: {obj_cols}')",
    "    for col in obj_cols:",
    "        X[col] = LabelEncoder().fit_transform(X[col].astype(str))",
    "print(f'Final feature matrix: {X.shape}')",
    "print('Features:', FEATURE_COLS)"
)))

# MI
c.append(cell("## 6. Feature Selection (Mutual Information)","markdown"))
c.append(cell(lines(
    "sample_idx = np.random.RandomState(42).choice(len(X), size=min(50000,len(X)), replace=False)",
    "mi_scores = mutual_info_classif(X.iloc[sample_idx], y.iloc[sample_idx], random_state=42)",
    "mi_df = pd.DataFrame({'feature': FEATURE_COLS, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)",
    "print('Mutual Information scores:')",
    "mi_df"
)))

c.append(cell(lines(
    "fig, ax = plt.subplots(figsize=(8,5))",
    "top = mi_df.head(15)",
    "ax.barh(top['feature'][::-1], top['mi_score'][::-1], color='#3498db', edgecolor='black')",
    "ax.set_xlabel('Mutual Information Score')",
    "ax.set_title('Top Features - Mutual Information', fontsize=14, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

c.append(cell(lines(
    "TOP_FEATURES = mi_df.loc[mi_df['mi_score']>0.01, 'feature'].tolist()",
    "if len(TOP_FEATURES) < 5:",
    "    TOP_FEATURES = mi_df.head(15)['feature'].tolist()",
    "print(f'Selected {len(TOP_FEATURES)} features: {TOP_FEATURES}')",
    "X = X[TOP_FEATURES]"
)))

# Split
c.append(cell("## 7. Train/Test Split - Stratified Group Split by car_id\n\n**Why stratified-group?**\n1. Plain random split: same car in train+test -> vehicle identity leak\n2. Plain GroupShuffleSplit: doesn't stratify by label -> entire class (e.g. DDoS) can have 0 test samples\n\n**Solution:** split cars within each label group ~80/20","markdown"))
c.append(cell(lines(
    "rng = np.random.RandomState(42)",
    "car_labels = df.groupby('car_id')['label_enc'].agg(lambda s: s.mode()[0])",
    "train_cars_list, test_cars_list = [], []",
    "for lv in sorted(car_labels.unique()):",
    "    cars = car_labels[car_labels == lv].index.values.copy()",
    "    rng.shuffle(cars)",
    "    n_test = max(1, int(len(cars)*0.20))",
    "    test_cars_list.extend(cars[:n_test])",
    "    train_cars_list.extend(cars[n_test:])",
    "train_car_set = set(train_cars_list)",
    "test_car_set = set(test_cars_list)",
    "train_idx = np.where(groups.isin(train_car_set))[0]",
    "test_idx = np.where(groups.isin(test_car_set))[0]",
    "",
    "X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]",
    "y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]",
    "print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')",
    "print(f'Unique cars in train: {groups.iloc[train_idx].nunique()}')",
    "print(f'Unique cars in test : {groups.iloc[test_idx].nunique()}')",
    "overlap = set(groups.iloc[train_idx].unique()) & set(groups.iloc[test_idx].unique())",
    "print(f'Car overlap: {len(overlap)}  (should be 0)')",
    "assert len(overlap) == 0, 'LEAKAGE: same car in train AND test!'"
)))

c.append(cell(lines(
    "print('Per-class sample counts:')",
    "for cn in label_names:",
    "    ci = le.transform([cn])[0]",
    "    print(f'  {cn:12s}  train={int((y_train==ci).sum()):7,}  test={int((y_test==ci).sum()):7,}')",
    "assert all((y_test==c).sum()>0 for c in range(len(label_names))), 'Class with 0 test samples!'"
)))

c.append(cell(lines(
    "scaler = StandardScaler()",
    "X_train_sc = scaler.fit_transform(X_train)",
    "X_test_sc = scaler.transform(X_test)"
)))

# RF
c.append(cell("## 8. Random Forest","markdown"))
c.append(cell(lines(
    "rf = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_split=5,",
    "                            min_samples_leaf=2, class_weight='balanced',",
    "                            random_state=42, n_jobs=-1)",
    "t_start = time.time()",
    "rf.fit(X_train_sc, y_train)",
    "rf_time = time.time() - t_start",
    "y_pred_rf = rf.predict(X_test_sc)",
    "rf_acc = accuracy_score(y_test, y_pred_rf)",
    "rf_f1w = f1_score(y_test, y_pred_rf, average='weighted')",
    "rf_f1m = f1_score(y_test, y_pred_rf, average='macro')",
    "print(f'Accuracy: {rf_acc:.4f}  F1w: {rf_f1w:.4f}  F1m: {rf_f1m:.4f}  Time: {rf_time:.1f}s')",
    "print(classification_report(y_test, y_pred_rf, target_names=label_names))"
)))

c.append(cell(lines(
    "fig, ax = plt.subplots(figsize=(7,6))",
    "ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),",
    "                       display_labels=label_names).plot(ax=ax, cmap='Blues', values_format=',')",
    "ax.set_title('Random Forest - Confusion Matrix', fontsize=13, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# XGB
c.append(cell("## 9. XGBoost","markdown"))
c.append(cell(lines(
    "class_counts = np.bincount(y_train)",
    "sample_weights = np.array([len(y_train)/(len(class_counts)*class_counts[yi]) for yi in y_train])",
    "xgb = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8,",
    "                    colsample_bytree=0.8, objective='multi:softmax', num_class=len(label_names),",
    "                    eval_metric='mlogloss', use_label_encoder=False, random_state=42, n_jobs=-1)",
    "t_start = time.time()",
    "xgb.fit(X_train_sc, y_train, sample_weight=sample_weights)",
    "xgb_time = time.time() - t_start",
    "y_pred_xgb = xgb.predict(X_test_sc)",
    "xgb_acc = accuracy_score(y_test, y_pred_xgb)",
    "xgb_f1w = f1_score(y_test, y_pred_xgb, average='weighted')",
    "xgb_f1m = f1_score(y_test, y_pred_xgb, average='macro')",
    "print(f'Accuracy: {xgb_acc:.4f}  F1w: {xgb_f1w:.4f}  F1m: {xgb_f1m:.4f}  Time: {xgb_time:.1f}s')",
    "print(classification_report(y_test, y_pred_xgb, target_names=label_names))"
)))

c.append(cell(lines(
    "fig, ax = plt.subplots(figsize=(7,6))",
    "ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_xgb),",
    "                       display_labels=label_names).plot(ax=ax, cmap='Greens', values_format=',')",
    "ax.set_title('XGBoost - Confusion Matrix', fontsize=13, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# Comparison
c.append(cell("## 10. Model Comparison","markdown"))
c.append(cell(lines(
    "results = {",
    "    'Random Forest': {'accuracy':rf_acc,'f1_weighted':rf_f1w,'f1_macro':rf_f1m,'train_time':rf_time,'model':rf,'y_pred':y_pred_rf},",
    "    'XGBoost': {'accuracy':xgb_acc,'f1_weighted':xgb_f1w,'f1_macro':xgb_f1m,'train_time':xgb_time,'model':xgb,'y_pred':y_pred_xgb},",
    "}",
    "comp = pd.DataFrame({n:{'Accuracy':r['accuracy'],'F1 Weighted':r['f1_weighted'],'F1 Macro':r['f1_macro'],'Time (s)':round(r['train_time'],1)} for n,r in results.items()}).T",
    "comp"
)))

c.append(cell(lines(
    "fig, axes = plt.subplots(1,3,figsize=(14,4))",
    "for ax,m,title in zip(axes,['accuracy','f1_weighted','f1_macro'],['Accuracy','F1 Weighted','F1 Macro']):",
    "    vals = [results[n][m] for n in results]",
    "    bars = ax.bar(list(results.keys()), vals, color=['#2ecc71','#3498db'], edgecolor='black')",
    "    ax.set_title(title, fontweight='bold'); ax.set_ylim(0.5,1.05)",
    "    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.4f}', ha='center', fontsize=10)",
    "plt.tight_layout()",
    "plt.show()"
)))

# Save
c.append(cell("## 11. Save Best Model","markdown"))
c.append(cell(lines(
    "best_name = max(results, key=lambda k: results[k]['f1_weighted'])",
    "best_model = results[best_name]['model']",
    "joblib.dump(best_model, os.path.join(OUTPUT_DIR,'best_model.joblib'))",
    "joblib.dump(scaler, os.path.join(OUTPUT_DIR,'scaler.joblib'))",
    "joblib.dump(le, os.path.join(OUTPUT_DIR,'label_encoder.joblib'))",
    "joblib.dump(TOP_FEATURES, os.path.join(OUTPUT_DIR,'feature_list.joblib'))",
    "print(f'Best model: {best_name}')",
    "print(f'Accuracy: {results[best_name][\"accuracy\"]:.4f}')",
    "print(f'F1w: {results[best_name][\"f1_weighted\"]:.4f}  F1m: {results[best_name][\"f1_macro\"]:.4f}')",
    "print(f'Features: {len(TOP_FEATURES)}')",
    "print(f'Saved to: {OUTPUT_DIR}/')"
)))

# Inference
c.append(cell("## 12. Inference Demo","markdown"))
c.append(cell(lines(
    "lm = joblib.load(os.path.join(OUTPUT_DIR,'best_model.joblib'))",
    "ls = joblib.load(os.path.join(OUTPUT_DIR,'scaler.joblib'))",
    "ll = joblib.load(os.path.join(OUTPUT_DIR,'label_encoder.joblib'))",
    "lf = joblib.load(os.path.join(OUTPUT_DIR,'feature_list.joblib'))",
    "sample = X_test.sample(5, random_state=123)",
    "preds = ll.inverse_transform(lm.predict(ls.transform(sample[lf])))",
    "actuals = ll.inverse_transform(y_test.loc[sample.index].values)",
    "pd.DataFrame({'Predicted':preds,'Actual':actuals,'Correct':preds==actuals}, index=sample.index)"
)))

with open("Intrusion_Detection_Clean.ipynb","w",encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook created: Intrusion_Detection_Clean.ipynb")

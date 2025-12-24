"""
Aggressive Feature Selection + Ensemble
Approach: Use ONLY the most robust features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AGGRESSIVE OPTIMIZATION STRATEGY")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc, on='object_id', how='inner')
test_full = test_meta.merge(test_lc, on='object_id', how='inner')

# Metadata features
for df in [train_full, test_full]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2

# Get ALL features
metadata_feats = ['Z', 'EBV', 'Z_EBV_ratio', 'Z_squared']
lc_feats = [col for col in train_lc.columns if col != 'object_id']
all_feats = metadata_feats + lc_feats

X_all = train_full[all_feats].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y = train_full['target']
X_test_all = test_full[all_feats].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

# Find most important features
print("\n[2] Finding most important features...")
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])

temp_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    scale_pos_weight=scale_pos
)
temp_model.fit(X_train, y_train)

feat_imp = pd.DataFrame({
    'feature': all_feats,
    'importance': temp_model.feature_importances_
}).sort_values('importance', ascending=False)

# Select ONLY top 12 features (very aggressive!)
top_n = 12
selected_feats = feat_imp.head(top_n)['feature'].tolist()

print(f"\n✓ Selected {top_n} features:")
for i, f in enumerate(selected_feats, 1):
    print(f"  {i}. {f}: {feat_imp[feat_imp['feature']==f]['importance'].values[0]:.4f}")

# Prepare selected data
X_sel = X_all[selected_feats]
X_test_sel = X_test_all[selected_feats]

X_train_sel, X_val_sel, y_train, y_val = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# MODEL 1: XGBoost with heavy regularization
# ==========================================
print("\n[3] Model 1: XGBoost (heavy regularization)...")

xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.02,  # Very slow
    max_depth=3,  # Very shallow
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=1.0,  # Strong regularization
    min_child_weight=10,
    reg_alpha=0.5,
    reg_lambda=2.0
)

xgb_model.fit(X_train_sel, y_train)
xgb_val_proba = xgb_model.predict_proba(X_val_sel)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)
print(f"  Validation AUC: {xgb_val_auc:.4f}")

# Train on full data
xgb_final = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.02,
    max_depth=3,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=1.0,
    min_child_weight=10,
    reg_alpha=0.5,
    reg_lambda=2.0
)
xgb_final.fit(X_sel, y)
xgb_test_proba = xgb_final.predict_proba(X_test_sel)[:, 1]

# ==========================================
# MODEL 2: Logistic Regression (simple!)
# ==========================================
print("\n[4] Model 2: Logistic Regression (simpler model)...")

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=0.1  # Strong regularization
)

lr_model.fit(X_train_sel, y_train)
lr_val_proba = lr_model.predict_proba(X_val_sel)[:, 1]
lr_val_auc = roc_auc_score(y_val, lr_val_proba)
print(f"  Validation AUC: {lr_val_auc:.4f}")

# Train on full
lr_final = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    C=0.1
)
lr_final.fit(X_sel, y)
lr_test_proba = lr_final.predict_proba(X_test_sel)[:, 1]

# ==========================================
# MODEL 3: Random Forest (moderate complexity)
# ==========================================
print("\n[5] Model 3: Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_sel, y_train)
rf_val_proba = rf_model.predict_proba(X_val_sel)[:, 1]
rf_val_auc = roc_auc_score(y_val, rf_val_proba)
print(f"  Validation AUC: {rf_val_auc:.4f}")

# Train on full
rf_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    random_state=42,
    class_weight='balanced'
)
rf_final.fit(X_sel, y)
rf_test_proba = rf_final.predict_proba(X_test_sel)[:, 1]

# ==========================================
# ENSEMBLE: Average predictions
# ==========================================
print("\n[6] Creating ensemble...")

# Weighted ensemble (based on validation performance)
weights = np.array([xgb_val_auc, lr_val_auc, rf_val_auc])
weights = weights / weights.sum()

print(f"\nEnsemble weights:")
print(f"  XGBoost: {weights[0]:.3f}")
print(f"  LogReg:  {weights[1]:.3f}")
print(f"  RF:      {weights[2]:.3f}")

ensemble_test_proba = (
    weights[0] * xgb_test_proba +
    weights[1] * lr_test_proba +
    weights[2] * rf_test_proba
)

# Also try simple average
simple_avg_proba = (xgb_test_proba + lr_test_proba + rf_test_proba) / 3

# ==========================================
# CREATE SUBMISSIONS
# ==========================================
print("\n[7] Creating submissions...")

# Submission 1: Ensemble (weighted)
ensemble_binary = (ensemble_test_proba >= 0.6).astype(int)
sub_ensemble = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': ensemble_binary
})
sub_ensemble.to_csv('submission_ensemble_weighted.csv', index=False)
print(f"✓ submission_ensemble_weighted.csv")
print(f"  TDEs: {ensemble_binary.sum()} ({ensemble_binary.mean()*100:.2f}%)")

# Submission 2: Simple average
simple_binary = (simple_avg_proba >= 0.6).astype(int)
sub_simple = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': simple_binary
})
sub_simple.to_csv('submission_ensemble_simple.csv', index=False)
print(f"✓ submission_ensemble_simple.csv")
print(f"  TDEs: {simple_binary.sum()} ({simple_binary.mean()*100:.2f}%)")

# Submission 3: XGBoost alone (best single model usually)
xgb_binary = (xgb_test_proba >= 0.6).astype(int)
sub_xgb = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': xgb_binary
})
sub_xgb.to_csv('submission_xgb_minimal.csv', index=False)
print(f"✓ submission_xgb_minimal.csv")
print(f"  TDEs: {xgb_binary.sum()} ({xgb_binary.mean()*100:.2f}%)")

# Submission 4: Logistic Regression (simplest)
lr_binary = (lr_test_proba >= 0.5).astype(int)
sub_lr = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': lr_binary
})
sub_lr.to_csv('submission_logreg_simple.csv', index=False)
print(f"✓ submission_logreg_simple.csv")
print(f"  TDEs: {lr_binary.sum()} ({lr_binary.mean()*100:.2f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
KEY CHANGES:
1. Aggressive feature selection: {len(all_feats)} → {top_n} features
2. Heavy regularization on all models
3. Ensemble of 3 different model types
4. Simpler models (lower complexity)

RECOMMENDATIONS (try in order):
1. submission_ensemble_weighted.csv (most robust)
2. submission_ensemble_simple.csv (simple average)
3. submission_xgb_minimal.csv (if you trust XGBoost)
4. submission_logreg_simple.csv (simplest baseline)

All use binary predictions, threshold tuned for each.
""")

print("="*70)

"""
ROBUST OVERSAMPLING ENSEMBLE + GP FEATURES (PHASE 9)

Features:
1. Physics-Corrected features (Light Curve + Advanced Temporal)
2. Gaussian Process features (Length Scale, Amplitude, Noise) - NEW!

Method:
- Load all feature sets
- Merge on object_id
- Robust Oversampling (5x) on Training data ONLY
- Stratified 5-Fold CV
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ROBUST ENSEMBLE + GAUSSIAN PROCESS FEATURES")
print("="*70)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n[1] Loading Feature Sets...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

# Existing Corrected Features
train_lc = pd.read_csv('lightcurve_features_train_corrected.csv')
test_lc = pd.read_csv('lightcurve_features_test_corrected.csv')
train_adv = pd.read_csv('advanced_temporal_features_train_corrected.csv')
test_adv = pd.read_csv('advanced_temporal_features_test_corrected.csv')

# NEW: GP Features
try:
    train_gp = pd.read_csv('gp_features_train_corrected.csv')
    test_gp = pd.read_csv('gp_features_test_corrected.csv')
    print("  Loaded GP Features!")
except FileNotFoundError:
    print("  ERROR: GP features not found. Make sure extract_gp_features.py finished.")
    exit(1)

def prepare_features(meta, lc, adv, gp):
    # Merge metadata with features on object_id
    df = meta.merge(lc, on='object_id', how='left')
    df = df.merge(adv, on='object_id', how='left')
    df = df.merge(gp, on='object_id', how='left')
    
    # Feature Engineering
    X = df[['Z', 'EBV']].copy()
    X['Z_EBV_ratio'] = X['Z'] / (X['EBV'] + 1e-5)
    X['Z_squared'] = X['Z'] ** 2
    
    # Add LC features
    lc_cols = [c for c in lc.columns if c != 'object_id']
    for col in lc_cols:
        X[f'lc_{col}'] = df[col].values
        
    # Add Advanced features
    adv_cols = [c for c in adv.columns if c != 'object_id']
    for col in adv_cols:
        X[f'adv_{col}'] = df[col].values

    # Add GP features
    gp_cols = [c for c in gp.columns if c != 'object_id']
    for col in gp_cols:
        X[f'gp_{col}'] = df[col].values
        
    return X.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

X_train_raw = prepare_features(train_meta, train_lc, train_adv, train_gp)
X_test_raw = prepare_features(test_meta, test_lc, test_adv, test_gp)
y_train_raw = train_meta['target'].values

print(f"Train Matrix: {X_train_raw.shape}")
print(f"Test Matrix:  {X_test_raw.shape}")

# ==========================================
# 2. FEATURE SELECTION
# ==========================================
print("\n[2] Feature Selection...")
sel = xgb.XGBClassifier(n_estimators=100, random_state=42)
sel.fit(X_train_raw, y_train_raw)
imp = sel.feature_importances_

# Check top features to see if GP is there
feat_imp = pd.DataFrame({'feature': X_train_raw.columns, 'importance': imp})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print("  Top 10 Features:")
print(feat_imp.head(10))

# Select top 60
top_cols = feat_imp['feature'].values[:60]
X_train_sel = X_train_raw[top_cols]
X_test_sel = X_test_raw[top_cols]

print(f"  Selected {len(top_cols)} features.")

# ==========================================
# 3. ROBUST TRAINING
# ==========================================
print("\n[3] Training GP-Enhanced Ensemble...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_sel))
test_preds = np.zeros(len(X_test_sel))

val_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_sel, y_train_raw), 1):
    X_tr, y_tr = X_train_sel.iloc[tr_idx], y_train_raw[tr_idx]
    X_val, y_val = X_train_sel.iloc[val_idx], y_train_raw[val_idx]
    
    # Robust Oversampling used in Phase 5
    pos_mask = y_tr == 1
    X_pos = X_tr[pos_mask]
    
    # 5x Augmentation
    aug_list = [X_tr]
    y_aug_list = [y_tr]
    
    for _ in range(5):
        noise = np.random.normal(0, 0.05, X_pos.shape) * X_pos.std().values
        X_new = X_pos + pd.DataFrame(noise, columns=X_pos.columns, index=X_pos.index)
        aug_list.append(X_new)
        y_aug_list.append(y_tr[pos_mask])
        
    X_tr_final = pd.concat(aug_list)
    y_tr_final = np.concatenate(y_aug_list)
    
    # Train
    clf1 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.7, random_state=42)
    clf1.fit(X_tr_final, y_tr_final)
    
    clf2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    clf2.fit(X_tr_final, y_tr_final)
    
    # Predict Val
    p1 = clf1.predict_proba(X_val)[:, 1]
    p2 = clf2.predict_proba(X_val)[:, 1]
    prob_fold = (p1 + p2) / 2
    oof_preds[val_idx] = prob_fold
    
    # Predict Test
    tp1 = clf1.predict_proba(X_test_sel)[:, 1]
    tp2 = clf2.predict_proba(X_test_sel)[:, 1]
    test_preds += (tp1 + tp2) / 2
    
    # Fold Score
    best_f1_fold = 0
    for t in np.arange(0.1, 0.9, 0.05):
        s = f1_score(y_val, (prob_fold >= t).astype(int))
        if s > best_f1_fold: best_f1_fold = s
    val_scores.append(best_f1_fold)
    
    print(f"  Fold {fold}: F1={best_f1_fold:.4f}")

test_preds /= 5
mean_cv = np.mean(val_scores)

# Final Optimization
best_f1 = 0
best_thr = 0.5
for thr in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_train_raw, (oof_preds >= thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\n{'='*70}")
print(f"GP-ENHANCED CV F1: {best_f1:.4f}")
print(f"Threshold: {best_thr:.3f}")

# Submission
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

binary = (test_preds >= best_thr).astype(int)
sub = pd.DataFrame({'object_id': test_ids, 'prediction': binary})
sub.to_csv('submission_gp_enhanced.csv', index=False)

print(f"\n✓ Saved submission_gp_enhanced.csv (TDEs: {binary.sum()})")

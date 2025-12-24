"""
ROBUST OVERSAMPLING ENSEMBLE (GENERATIVE PHYSICS AUGMENTED)

Problem: TDEs are only 4.8% of training data (148 samples).
Solution:
1. Load REAL data (Physics Corrected).
2. Load SYNTHETIC data (Generated from TDE Physics distributions).
3. Strategy: 
   - CV Split: Stratified 5-Fold on REAL data only (Strict Validation).
   - Training: Real Training Fold + ALL Synthetic TDEs.
   - Result: Model sees 148 Real + 500 Synthetic = ~650 Positives per fold.
4. Train Random Forest + XGBoost with strict regularization.
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
print("ROBUST ENSEMBLE + GENERATIVE PHYSICS AUGMENTATION")
print("="*70)

# ==========================================
# 1. LOAD DATA (REAL + SYNTHETIC)
# ==========================================
print("\n[1] Loading REAL (Corrected) data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

train_lc = pd.read_csv('lightcurve_features_train_corrected.csv')
test_lc = pd.read_csv('lightcurve_features_test_corrected.csv')
train_adv = pd.read_csv('advanced_temporal_features_train_corrected.csv')
test_adv = pd.read_csv('advanced_temporal_features_test_corrected.csv')

print("\n[2] Loading SYNTHETIC data...")
syn_meta = pd.read_csv('synthetic_log.csv') # Generated in Step 4
syn_lc = pd.read_csv('lightcurve_features_synthetic.csv') # Generated in Step 5
syn_adv = pd.read_csv('advanced_temporal_features_synthetic.csv') # Generated in Step 5

def prepare_features(meta, lc, adv):
    # Merge metadata with features on object_id to ensure alignment
    df = meta.merge(lc, on='object_id', how='left')
    df = df.merge(adv, on='object_id', how='left')
    
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
        
    return X.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

# Prepare REAL datasets
X_real_raw = prepare_features(train_meta, train_lc, train_adv)
y_real_raw = train_meta['target'].values
X_test_raw = prepare_features(test_meta, test_lc, test_adv)

# Prepare SYNTHETIC dataset
X_syn_raw = prepare_features(syn_meta, syn_lc, syn_adv)
y_syn_raw = syn_meta['target'].values

# Ensure synthetic has same columns as real (order matters)
# Some columns might assume different orders or missing if not careful
# We select common columns
common_cols = [c for c in X_real_raw.columns if c in X_syn_raw.columns]
X_real_raw = X_real_raw[common_cols]
X_syn_raw = X_syn_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

print(f"Real Train: {X_real_raw.shape} - Positives: {sum(y_real_raw)}")
print(f"Synthetic:  {X_syn_raw.shape} - Positives: {sum(y_syn_raw)}")

# ==========================================
# 2. FEATURE SELECTION (Using Real Data + Syn)
# ==========================================
print("\n[3] Feature Selection...")
# Train selection model on ALL data
X_all = pd.concat([X_real_raw, X_syn_raw])
y_all = np.concatenate([y_real_raw, y_syn_raw])

sel = xgb.XGBClassifier(n_estimators=100, random_state=42)
sel.fit(X_all, y_all)
imp = sel.feature_importances_
top_idx = np.argsort(imp)[-60:] # Increase to 60 features
top_cols = X_real_raw.columns[top_idx]

X_real = X_real_raw[top_cols]
X_syn = X_syn_raw[top_cols]
X_test = X_test_raw[top_cols]

print(f"Selected {len(top_cols)} features.")

# ==========================================
# 3. ROBUST TRAINING (CV on REAL, Train on REAL+SYN)
# ==========================================
print("\n[4] Training Generative Ensemble...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_real)) # OOF for Real data only
test_preds = np.zeros(len(X_test))

val_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_real, y_real_raw), 1):
    # Split Real Data
    X_tr_real, y_tr_real = X_real.iloc[tr_idx], y_real_raw[tr_idx]
    X_val_real, y_val_real = X_real.iloc[val_idx], y_real_raw[val_idx]
    
    # Validation is strictly REAL (No leakage)
    
    # Training is REAL + SYNTHETIC
    # We also apply a small amount of noise oversampling to REAL positives 
    # to bridge the "real vs synthetic" gap, but mostly rely on Synthetic.
    
    pos_mask_real = y_tr_real == 1
    X_pos_real = X_tr_real[pos_mask_real]
    y_pos_real = y_tr_real[pos_mask_real]
    
    # 2x Oversampling of Real Positives
    noise = np.random.normal(0, 0.05, X_pos_real.shape) * X_pos_real.std().values
    X_pos_aug = X_pos_real + pd.DataFrame(noise, columns=X_pos_real.columns, index=X_pos_real.index)
    
    # Combine: Real Train + Real Augmented + Synthetic
    X_train_fold = pd.concat([X_tr_real, X_pos_aug, X_syn], ignore_index=True)
    y_train_fold = np.concatenate([y_tr_real, y_pos_real, y_syn_raw])
    
    # Train Models
    # 1. XGB
    clf1 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.7, random_state=42)
    clf1.fit(X_train_fold, y_train_fold)
    
    # 2. Random Forest
    clf2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    clf2.fit(X_train_fold, y_train_fold)
    
    # Predict Val (Real)
    p1 = clf1.predict_proba(X_val_real)[:, 1]
    p2 = clf2.predict_proba(X_val_real)[:, 1]
    prob_fold = (p1 + p2) / 2
    oof_preds[val_idx] = prob_fold
    
    # Score
    f1_fold = 0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        s = f1_score(y_val_real, (prob_fold >= t).astype(int))
        if s > f1_fold: 
            f1_fold = s
            best_t = t
            
    val_scores.append(f1_fold)
    print(f"  Fold {fold}: F1={f1_fold:.4f} (Thr={best_t:.2f})")
    
    # Predict Test
    tp1 = clf1.predict_proba(X_test)[:, 1]
    tp2 = clf2.predict_proba(X_test)[:, 1]
    test_preds += (tp1 + tp2) / 2
    
test_preds /= 5

# Overall Optimization
best_f1 = 0
best_thr = 0.5
for thr in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_real_raw, (oof_preds >= thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\n{'='*70}")
print(f"GENERATIVE PHYSICS CV F1: {best_f1:.4f}")
print(f"Threshold: {best_thr:.3f}")

# Submission
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

binary = (test_preds >= best_thr).astype(int)
sub = pd.DataFrame({'object_id': test_ids, 'prediction': binary})
sub.to_csv('submission_generative_physics.csv', index=False)

print(f"\n✓ Saved submission_generative_physics.csv (TDEs: {binary.sum()})")

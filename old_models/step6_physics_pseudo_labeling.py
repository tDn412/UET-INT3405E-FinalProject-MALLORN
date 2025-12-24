"""
STEP 6: PHYSICS-BASED PSEUDO-LABELING
Goal: Use the best Physics-Corrected Model (CV 0.49) to label confident Test samples,
then retrain to refine the decision boundary.

Process:
1. Load `submission_robust_oversample_physics.csv` (The 0.49 model predictions).
2. Select Test objects predicted as TDE (Target=1).
3. Add these to the Training set with Target=1.
4. Retrain the Robust Ensemble on (Real Train + Pseudo Test).
5. Generate Final Submission.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHYSICS-BASED PSEUDO-LABELING")
print("="*70)

# 1. Load Data
print("\n[1] Loading Data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

# Use CORRECTED features
train_lc = pd.read_csv('lightcurve_features_train_corrected.csv')
test_lc = pd.read_csv('lightcurve_features_test_corrected.csv')
train_adv = pd.read_csv('advanced_temporal_features_train_corrected.csv')
test_adv = pd.read_csv('advanced_temporal_features_test_corrected.csv')

# Load Predictions from Best Model
try:
    sub = pd.read_csv('submission_robust_oversample_physics.csv')
    pseudo_tde_ids = sub[sub['prediction'] == 1]['object_id'].values
    print(f"  Found {len(pseudo_tde_ids)} High-Confidence TDE candidates in Test set.")
except FileNotFoundError:
    print("  ERROR: submission_robust_oversample_physics.csv not found!")
    exit(1)

def prepare_features(meta, lc, adv):
    # Merge metadata with features on object_id
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

X_train_raw = prepare_features(train_meta, train_lc, train_adv)
X_test_raw = prepare_features(test_meta, test_lc, test_adv)
y_train_raw = train_meta['target'].values

# Common columns
cols = X_train_raw.columns
X_test_raw = X_test_raw[cols]

# 2. Construct Pseudo-Label Dataset
print("\n[2] Constructing Pseudo-Label Training Set...")

# Identify pseudo samples in Test Features
pseudo_mask = test_meta['object_id'].isin(pseudo_tde_ids)
X_pseudo = X_test_raw[pseudo_mask]
y_pseudo = np.ones(len(X_pseudo)) # Label as 1

print(f"  Adding {len(X_pseudo)} pseudo-positives to training data.")

# Combine Real + Pseudo
X_combined = pd.concat([X_train_raw, X_pseudo], ignore_index=True)
y_combined = np.concatenate([y_train_raw, y_pseudo])

print(f"  New Training Size: {len(X_combined)} (Positives: {sum(y_combined)})")

# 3. Feature Selection
print("\n[3] Feature Selection...")
sel = xgb.XGBClassifier(n_estimators=100, random_state=42)
sel.fit(X_combined, y_combined) # Fit on combined to learn from pseudo labels
imp = sel.feature_importances_
top_idx = np.argsort(imp)[-50:] 
top_cols = X_combined.columns[top_idx]

X_tr_final = X_combined[top_cols]
X_te_final = X_test_raw[top_cols]

print(f"  Selected {len(top_cols)} features.")

# 4. Train Final Ensemble (No CV needed, we trust the pseudo strategy)
# Or we can do CV to check stability? 
# Let's do 5-Fold CV on the REAL data part to see if pseudo-labeling improves validation
print("\n[4] Training with Pseudo-Labels...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_preds = np.zeros(len(X_te_final))

# We split REAL data for validation
real_idx = np.arange(len(X_train_raw))
X_real_sel = X_train_raw[top_cols]

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_real_sel, y_train_raw), 1):
    # Train = Real Train Fold + ALL Pseudo
    X_tr_fold = X_real_sel.iloc[tr_idx]
    y_tr_fold = y_train_raw[tr_idx]
    
    # Validation = Real Val Fold (Strict)
    X_val_fold = X_real_sel.iloc[val_idx]
    y_val_fold = y_train_raw[val_idx]
    
    # Combine with Pseudo
    X_train_aug = pd.concat([X_tr_fold, X_pseudo[top_cols]], ignore_index=True)
    y_train_aug = np.concatenate([y_tr_fold, y_pseudo])
    
    # Oversample Real Positives (2x) to fix imbalance within the real part
    pos_mask = y_tr_fold == 1
    X_pos = X_tr_fold[pos_mask]
    noise = np.random.normal(0, 0.05, X_pos.shape) * X_pos.std().values
    X_pos_noise = X_pos + pd.DataFrame(noise, columns=X_pos.columns, index=X_pos.index)
    
    X_train_aug = pd.concat([X_train_aug, X_pos_noise], ignore_index=True)
    y_train_aug = np.concatenate([y_train_aug, np.ones(len(X_pos_noise))])
    
    # Train
    clf1 = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.7, random_state=42)
    clf1.fit(X_train_aug, y_train_aug)
    
    clf2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    clf2.fit(X_train_aug, y_train_aug)
    
    # Validate
    p1 = clf1.predict_proba(X_val_fold)[:, 1]
    p2 = clf2.predict_proba(X_val_fold)[:, 1]
    p_val = (p1 + p2) / 2
    
    # Optimize Threshold
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.05):
        s = f1_score(y_val_fold, (p_val >= t).astype(int))
        if s > best_f1: best_f1 = s
            
    val_scores.append(best_f1)
    
    # Predict Test
    tp1 = clf1.predict_proba(X_te_final)[:, 1]
    tp2 = clf2.predict_proba(X_te_final)[:, 1]
    test_preds += (tp1 + tp2) / 2
    
    print(f"  Fold {fold}: F1={best_f1:.4f}")

test_preds /= 5
mean_cv = np.mean(val_scores)

print(f"\n{'='*70}")
print(f"PSEUDO-LABEL CV F1: {mean_cv:.4f}") # Compare to 0.49

# Final Threshold logic: Use the one that worked best on CV on average? ~0.35 typical
sub_thr = 0.35 
binary = (test_preds >= sub_thr).astype(int)

# Submission
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

sub = pd.DataFrame({'object_id': test_ids, 'prediction': binary})
sub.to_csv('submission_pseudo_physics.csv', index=False)

print(f"\n✓ Saved submission_pseudo_physics.csv (TDEs: {binary.sum()})")

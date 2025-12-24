"""
STEP 11: FINAL ENSEMBLE (PHYSICS + GP + SSL + TEMPLATE/CHI2)
Goal: The Ultimate Classifier combining all "Paradigm Shift" features.
Inputs:
1. Physics Features (Extinction Corrected)
2. Gaussian Process Features (Length Scale)
3. SSL Autoencoder Embeddings (Shape)
4. Template Chi-Square Features (Goodness of Fit)
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
print("FINAL ENSEMBLE: PHYSICS + GP + SSL + CHI-SQUARE")
print("="*70)

# 1. Load Data
print("\n[1] Loading All Feature Sets...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

# Load Features
lc_train = pd.read_csv('lightcurve_features_train_corrected.csv')
lc_test = pd.read_csv('lightcurve_features_test_corrected.csv')

gp_train = pd.read_csv('gp_features_train_corrected.csv')
gp_test = pd.read_csv('gp_features_test_corrected.csv')

emb_train = pd.read_csv('embeddings_autoencoder.csv') # Includes Test objects (train/test split handled by ID)
# Embeddings file has one row per object_id in train+test combined usually, 
# or we need to ensure it splits correctly.
# The script `step7` saved all valid embeddings in one file usually, or split?
# Let's check the previous script. Ah, it saved `embeddings_autoencoder.csv` with 10k rows.
# So we load it and merge.

chi_train = pd.read_csv('template_features_train.csv')
chi_test = pd.read_csv('template_features_test.csv')

def prepare_data(meta, lc, gp, emb, chi):
    # Merge
    df = meta.merge(lc, on='object_id', how='left')
    df = df.merge(gp, on='object_id', how='left')
    df = df.merge(emb, on='object_id', how='left')
    df = df.merge(chi, on='object_id', how='left')
    
    # Simple engineering
    X = df[['Z', 'EBV']].copy()
    
    # Add Feature Groups
    for prefix, data in [('phys', lc), ('gp', gp), ('emb', emb), ('chi', chi)]:
        cols = [c for c in data.columns if c != 'object_id']
        for c in cols:
            X[f'{prefix}_{c}'] = df[c].values
            
    # Key Ratios
    # Chi2 Ratio
    if 'chi_chisq_tde_g' in X.columns and 'chi_chisq_sn_g' in X.columns:
        X['ratio_chi2_g'] = X['chi_chisq_tde_g'] / (X['chi_chisq_sn_g'] + 1e-5)
    
    return X.fillna(0).replace([np.inf, -np.inf], [0, 0])

X_train = prepare_data(train_meta, lc_train, gp_train, emb_train, chi_train)
y_train = train_meta['target'].values
X_test = prepare_data(test_meta, lc_test, gp_test, emb_train, chi_test)

# Align
cols = X_train.columns
X_test = X_test[cols]

print(f"  Feature Matrix: {X_train.shape}")
print(f"  (Includes Physics, GP, Embeddings, Chi-Square)")

# 2. Training
print("\n[2] Training Ultimate Ensemble...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
val_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    # 3x Oversample
    pos_mask = y_tr == 1
    X_pos = X_tr[pos_mask]
    
    aug_list = [X_tr]
    y_aug_list = [y_tr]
    for _ in range(3):
        noise = np.random.normal(0, 0.05, X_pos.shape)
        aug_list.append(X_pos + noise)
        y_aug_list.append(y_tr[pos_mask])
        
    X_aug = pd.concat(aug_list)
    y_aug = np.concatenate(y_aug_list)
    
    # Model
    clf = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.02, 
                            subsample=0.7, colsample_bytree=0.5, random_state=42)
    clf.fit(X_aug, y_aug)
    
    p_val = clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = p_val
    test_preds += clf.predict_proba(X_test)[:, 1] / 5
    
    # Score
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.05):
        s = f1_score(y_val, (p_val >= t).astype(int))
        if s > best_f1: best_f1 = s
    val_scores.append(best_f1)
    print(f"  Fold {fold}: F1={best_f1:.4f}")

f1_mean = np.mean(val_scores)
print(f"\n{'='*70}")
print(f"ULTIMATE ENSEMBLE CV F1: {f1_mean:.4f}")

# Threshold
best_thr = 0.5
best_f1 = 0
for thr in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_train, (oof_preds >= thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print(f"Optimal Threshold: {best_thr:.3f}")

# Submit
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)
    
sub = pd.DataFrame({'object_id': test_ids, 'prediction': (test_preds >= best_thr).astype(int)})
sub.to_csv('submission_ultimate.csv', index=False)
print("✓ Saved submission_ultimate.csv")

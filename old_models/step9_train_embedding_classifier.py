"""
STEP 9: TRAIN CLASSIFIER ON SSL EMBEDDINGS
Goal: Validate if the Self-Supervised Embeddings (from Autoencoder) improve detection.
Method:
1. Load `embeddings_autoencoder.csv`.
2. Load `lightcurve_features_train_corrected.csv` (Physics features).
3. Combine them: X = [Embeddings (64) + Physics (10)].
4. Train the Robust Ensemble.
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
print("TRAINING ON SSL EMBEDDINGS (PARADIGM SHIFT)")
print("="*70)

# 1. Load Data
print("\n[1] Loading Data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

# Load Embeddings
try:
    emb_df = pd.read_csv('embeddings_autoencoder.csv')
    print(f"  Loaded {len(emb_df)} embeddings.")
except:
    print("  ERROR: Embeddings not found. Wait for step7 to finish.")
    exit(1)

# Load Physics Features (Optional, but recommended to Combine)
lc_train = pd.read_csv('lightcurve_features_train_corrected.csv')
lc_test = pd.read_csv('lightcurve_features_test_corrected.csv')
gp_train = pd.read_csv('gp_features_train_corrected.csv')
gp_test = pd.read_csv('gp_features_test_corrected.csv')

def prepare_data(meta, emb, lc, gp):
    # Merge
    df = meta.merge(emb, on='object_id', how='left')
    df = df.merge(lc, on='object_id', how='left')
    df = df.merge(gp, on='object_id', how='left')
    
    # Feature Engineering (Just Z)
    X = df[['Z', 'EBV']].copy()
    
    # Add Embeddings
    emb_cols = [c for c in emb.columns if c != 'object_id']
    for c in emb_cols:
        X[c] = df[c].values
        
    # Add Physics
    lc_cols = [c for c in lc.columns if c != 'object_id']
    for c in lc_cols:
        X[f'phys_{c}'] = df[c].values

    # Add GP
    gp_cols = [c for c in gp.columns if c != 'object_id']
    for c in gp_cols:
        X[f'gp_{c}'] = df[c].values
        
    return X.fillna(0).replace([np.inf, -np.inf], [0, 0])

X_train = prepare_data(train_meta, emb_df, lc_train, gp_train)
y_train = train_meta['target'].values
X_test = prepare_data(test_meta, emb_df, lc_test, gp_test)

# Align columns
cols = X_train.columns
X_test = X_test[cols]

print(f"  Train Matrix: {X_train.shape}")
print("  (Includes 64 SSL Embeddings + Physics Features)")

# 2. Training
print("\n[2] Training Ensemble...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
val_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    # Oversample TDEs
    pos_mask = y_tr == 1
    X_pos = X_tr[pos_mask]
    
    # 3x Oversampling
    aug_list = [X_tr]
    y_aug_list = [y_tr]
    for _ in range(3): 
        # Add noise to Embeddings?
        noise = np.random.normal(0, 0.05, X_pos.shape)
        X_new = X_pos + noise
        aug_list.append(X_new)
        y_aug_list.append(y_tr[pos_mask])
        
    X_tr_aug = pd.concat(aug_list)
    y_tr_aug = np.concatenate(y_aug_list)
    
    # XGBoost
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
    clf.fit(X_tr_aug, y_tr_aug)
    
    # Predict
    p_val = clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = p_val
    
    # Test
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
print(f"SSL EMBEDDING ENSEMBLE CV F1: {f1_mean:.4f}")

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
sub.to_csv('submission_ssl_autoencoder.csv', index=False)
print("✓ Saved submission_ssl_autoencoder.csv")

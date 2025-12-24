"""
HYBRID STACKING V2 (Corrected Leakage)

Strategy:
1. Load Original + Physics Features (X_manual) [~90 features]
2. Load OOF Embeddings (X_dl) [192+256 = 448 features]
3. Combine: X_all = [X_manual, X_dl]
4. Feature Selection: Keep Top 100 features to avoid dimensionality curse
5. Stacking Ensemble:
   - Base: XGBoost, RF, ExtraTrees
   - Meta: LogisticRegression

This fixes the F1=0.96 overfitting by using clean OOF features.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYBRID STACKING V2 (PHASE 2)")
print("="*70)

# 1. Load Manual Features
print("\n[1] Loading Manual Features...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')
train_adv = pd.read_csv('advanced_temporal_features_train.csv')
test_adv = pd.read_csv('advanced_temporal_features_test.csv')

X_train_man = train_meta[['Z', 'EBV']].copy()
X_test_man = test_meta[['Z', 'EBV']].copy()

# Add interactions
for df in [X_train_man, X_test_man]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2

# Add LC stats
for col in train_lc.columns:
    if col != 'object_id':
        X_train_man[f'lc_{col}'] = train_lc[col].values
        X_test_man[f'lc_{col}'] = test_lc[col].values

# Add Physics features
for col in train_adv.columns:
    X_train_man[f'adv_{col}'] = train_adv[col].values
    X_test_man[f'adv_{col}'] = test_adv[col].values

X_train_man = X_train_man.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
X_test_man = X_test_man.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"Manual Features: {X_train_man.shape[1]}")

# 2. Load Deep Learning OOF Embeddings
print("\n[2] Loading DL OOF Embeddings...")
lstm_oof = np.load('lstm_embeddings_oof.npy')
cnn_oof = np.load('cnn_embeddings_oof.npy')
lstm_test = np.load('lstm_embeddings_test.npy')
cnn_test = np.load('cnn_embeddings_test.npy')

X_train_dl = np.hstack([lstm_oof, cnn_oof])
X_test_dl = np.hstack([lstm_test, cnn_test])

print(f"DL Features: {X_train_dl.shape[1]}")

# 3. Combine
X_train_all = np.hstack([X_train_man.values, X_train_dl])
X_test_all = np.hstack([X_test_man.values, X_test_dl])
y_train = train_meta['target'].values

feature_names = list(X_train_man.columns) + \
                [f'dl_{i}' for i in range(X_train_dl.shape[1])]

print(f"Total Features: {X_train_all.shape[1]}")

# 4. Feature Selection (Top 100)
print("\n[3] Selecting Top 100 Features...")
sel_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
sel_model.fit(X_train_all, y_train)

importances = sel_model.feature_importances_
indices = np.argsort(importances)[-100:] # Keep top 100

X_train_sel = X_train_all[:, indices]
X_test_sel = X_test_all[:, indices]

selected_names = [feature_names[i] for i in indices]
print("Top 10 selected features:")
print(selected_names[-10:][::-1])

# 5. Stacking Ensemble
print("\n[4] Training Stacking Ensemble...")

base_models = [
    ('xgb1', xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=42)),
    ('xgb2', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.6, random_state=123)), # Deeper
    ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1))
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train_sel), len(base_models)))
test_preds = np.zeros((len(X_test_sel), len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"  Training {name}...")
    # CV for OOF
    for tr_idx, val_idx in skf.split(X_train_sel, y_train):
        X_tr, y_tr = X_train_sel[tr_idx], y_train[tr_idx]
        X_val = X_train_sel[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
    
    # Train on full for Test
    model.fit(X_train_sel, y_train)
    test_preds[:, i] = model.predict_proba(X_test_sel)[:, 1]

# Meta Learner
print("  Training Meta-Learner (LogisticRegression)...")
meta = LogisticRegression(C=0.1, random_state=42)
meta.fit(oof_preds, y_train)

# Predictions
meta_train_prob = meta.predict_proba(oof_preds)[:, 1]
meta_test_prob = meta.predict_proba(test_preds)[:, 1]

# Optimize Threshold
best_f1 = 0
best_thr = 0.5
for thr in np.arange(0.05, 0.5, 0.005):
    f1 = f1_score(y_train, (meta_train_prob >= thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\n{'='*70}")
print(f"HYBRID V2 RESULTS (CV F1: {best_f1:.4f})")
print(f"Threshold: {best_thr:.3f}")
print(f"{'='*70}")

# Save Submission
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

sub = pd.DataFrame({'object_id': test_ids, 'prediction': (meta_test_prob >= best_thr).astype(int)})
sub.to_csv('submission_hybrid_v2.csv', index=False)
print(f"✓ Saved submission_hybrid_v2.csv (TDEs: {sub['prediction'].sum()})")

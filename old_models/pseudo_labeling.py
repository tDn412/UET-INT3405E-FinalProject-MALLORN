"""
PSEUDO-LABELING (Phase 4)

Strategy:
1. Load trained Stacking Model (best so far)
2. Predict on Test Set
3. Select High-Confidence samples (Prob > 0.9 or < 0.1)
4. Add to Training Set (Pseudo-labels)
5. Retrain Stacking Model on Expanded Dataset

Goal: Squeeze extra performance from unlabeled test data.
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
print("PSEUDO-LABELING RETRAINING (PHASE 4)")
print("="*70)

# 1. Load Data
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')
train_adv = pd.read_csv('advanced_temporal_features_train.csv')
test_adv = pd.read_csv('advanced_temporal_features_test.csv')

def load_features(meta, lc, adv):
    X = meta[['Z', 'EBV']].copy()
    X['Z_EBV_ratio'] = X['Z'] / (X['EBV'] + 1e-5)
    X['Z_squared'] = X['Z'] ** 2
    
    for col in lc.columns:
        if col != 'object_id':
            X[f'lc_{col}'] = lc[col].values
            
    for col in adv.columns:
        X[f'adv_{col}'] = adv[col].values
        
    return X.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

X_train = load_features(train_meta, train_lc, train_adv)
X_test = load_features(test_meta, test_lc, test_adv)
y_train = train_meta['target'].values

print(f"Original Train: {X_train.shape}")
print(f"Test Set: {X_test.shape}")

# 2. Load Model & Predict
print("\n[1] Loading saved model & predicting...")
with open('stacking_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)

# Reconstruct stacking pipeline
# Note: saved_model contains trained base models and meta learner
# But for pseudo-labeling we need to PREDICT first.
# The base models in 'stacking_model.pkl' were trained on X_train_selected (top 30 features).
# We need to recreate X_test_selected using the same top features.

top_features = saved_model['top_features']
X_test_sel = X_test[top_features]

meta_learner = saved_model['meta_learner']
base_models = saved_model['base_models']

# Get Base Predictions on Test
n_models = len(base_models)
test_preds_base = np.zeros((len(X_test), n_models))

for i, (name, model) in enumerate(base_models):
    test_preds_base[:, i] = model.predict_proba(X_test_sel)[:, 1]

# Meta Prediction
test_probs = meta_learner.predict_proba(test_preds_base)[:, 1]

# 3. Select Pseudo-Labels
print("\n[2] Selecting Pseudo-Labels...")
CONF_THRESH_HIGH = 0.40 # optimal was 0.065, so 0.40 is very confident
CONF_THRESH_LOW = 0.02

high_conf_mask = (test_probs >= CONF_THRESH_HIGH) | (test_probs <= CONF_THRESH_LOW)
pseudo_X = X_test[high_conf_mask]
pseudo_y = (test_probs[high_conf_mask] >= 0.5).astype(int)

print(f"  > Found {len(pseudo_X)} pseudo-labels ({len(pseudo_X)/len(X_test)*100:.1f}%)")
print(f"  > TDEs: {sum(pseudo_y == 1)}, Non-TDEs: {sum(pseudo_y == 0)}")

# 4. Combine Datasets
X_train_expanded = pd.concat([X_train, pseudo_X], axis=0).reset_index(drop=True)
y_train_expanded = np.concatenate([y_train, pseudo_y])

print(f"Expanded Train: {X_train_expanded.shape}")

# 5. Retrain Stacking on Expanded Data
print("\n[3] Retraining Stacking Ensemble...")

# Select features again (on expanded data)
temp_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
temp_model.fit(X_train_expanded, y_train_expanded)
imp = temp_model.feature_importances_
top_idx = np.argsort(imp)[-30:] # Keep top 30
top_cols = X_train_expanded.columns[top_idx].tolist()

X_tr_sel = X_train_expanded[top_cols]
X_te_sel = X_test[top_cols]

# Retrain Base Models
new_base_models = []
for name, _ in base_models: # Re-instantiate
    if 'xgb' in name:
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    elif 'rf' in name:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    else:
        m = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)
    new_base_models.append((name, m))

# OOF Loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_tr_sel), len(new_base_models)))
test_base_preds = np.zeros((len(X_te_sel), len(new_base_models)))

for i, (name, model) in enumerate(new_base_models):
    print(f"  Training {name}...")
    for tr_idx, val_idx in skf.split(X_tr_sel, y_train_expanded):
        model.fit(X_tr_sel.iloc[tr_idx], y_train_expanded[tr_idx])
        oof_preds[val_idx, i] = model.predict_proba(X_tr_sel.iloc[val_idx])[:, 1]
    
    # Fit full
    model.fit(X_tr_sel, y_train_expanded)
    test_base_preds[:, i] = model.predict_proba(X_te_sel)[:, 1]

# Retrain Meta
print("  Training Meta-Learner...")
new_meta = LogisticRegression(C=0.1)
new_meta.fit(oof_preds, y_train_expanded)

# Predict
test_final_probs = new_meta.predict_proba(test_base_preds)[:, 1]

# Submission
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# Use fairly standard threshold or optimize?
# Hard to optimize without true labels, so stick to what worked or 0.5
best_threshold = 0.5 
binary = (test_final_probs >= best_threshold).astype(int)

sub = pd.DataFrame({'object_id': test_ids, 'prediction': binary})
sub.to_csv('submission_pseudo.csv', index=False)
print(f"\n✓ Saved submission_pseudo.csv (TDEs: {binary.sum()})")

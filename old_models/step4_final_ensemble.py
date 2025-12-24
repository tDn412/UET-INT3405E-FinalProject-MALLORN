"""
STEP 4: Simple Ensemble - Combine All Features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINAL ENSEMBLE - ALL FEATURES")
print("="*70)

# Load features
print("\n[1] Loading features...")

train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')
train_temporal = pd.read_csv('advanced_temporal_features_train.csv')
test_temporal = pd.read_csv('advanced_temporal_features_test.csv')

print(f"Loaded {len(train_lc.columns)-1} LC features")
print(f"Loaded {len(train_temporal.columns)} temporal features")

# Merge everything by index (they're aligned)
print("\n[2] Combining all features...")

# Start with metadata
X_train = train_meta[['Z', 'EBV']].copy()
X_train['Z_EBV_ratio'] = X_train['Z'] / (X_train['EBV'] + 1e-5)
X_train['Z_squared'] = X_train['Z'] ** 2

X_test = test_meta[['Z', 'EBV']].copy()
X_test['Z_EBV_ratio'] = X_test['Z'] / (X_test['EBV'] + 1e-5)
X_test['Z_squared'] = X_test['Z'] ** 2

# Add LC features (drop object_id)
for col in train_lc.columns:
    if col != 'object_id':
        X_train[f'lc_{col}'] = train_lc[col].values
        X_test[f'lc_{col}'] = test_lc[col].values

# Add temporal features
for col in train_temporal.columns:
    X_train[f'temporal_{col}'] = train_temporal[col].values
    X_test[f'temporal_{col}'] = test_temporal[col].values

# Clean
X_train = X_train.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
X_test = X_test.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y_train = train_meta['target'].values

print(f"Total features: {X_train.shape[1]}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train
print("\n[3] Training XGBoost...")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.1,
    min_child_weight=3
)

# CV
print("\n[4] Cross-validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1s = []
all_val_proba = []
all_val_y = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model.fit(X_tr, y_tr)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    all_val_proba.extend(val_proba)
    all_val_y.extend(y_val)
    
    # Find best threshold for fold
    best_f1 = 0
    for t in np.arange(0.05, 0.5, 0.02):
        f1 = f1_score(y_val, (val_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    
    cv_f1s.append(best_f1)
    print(f"  Fold {fold}: F1={best_f1:.4f}")

print(f"\nMean CV F1: {np.mean(cv_f1s):.4f}")

# Find best threshold overall
all_val_proba = np.array(all_val_proba)
all_val_y = np.array(all_val_y)

best_f1 = 0
best_threshold = 0.1

for t in np.arange(0.05, 0.5, 0.01):
    f1 = f1_score(all_val_y, (all_val_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold: {best_threshold:.3f}")
print(f"Best F1: {best_f1:.4f}")

# Train final
print("\n[5] Training final model...")

final_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.1,
    min_child_weight=3
)

final_model.fit(X_train, y_train)

# Predict
test_proba = final_model.predict_proba(X_test)[:, 1]

# Load test IDs
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# Submissions
print("\n[6] Creating submissions...")

thresholds = [
    (best_threshold, 'submission_ensemble_optimal.csv'),
    (best_threshold - 0.05, 'submission_ensemble_lower.csv'),
    (best_threshold + 0.05, 'submission_ensemble_higher.csv'),
]

for threshold, filename in thresholds:
    binary = (test_proba >= threshold).astype(int)
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv(filename, index=False)
    
    tde_count = binary.sum()
    print(f"✓ {filename}")
    print(f"  Threshold: {threshold:.3f}, TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")

print("\n" + "="*70)
print("ENSEMBLE COMPLETE!")
print("="*70)

print(f"""
RESULTS:
- Original features: 58
- Temporal features: {len(train_temporal.columns)}
- Total: {X_train.shape[1]} features
- CV F1: {np.mean(cv_f1s):.4f}
- Best threshold: {best_threshold:.3f}

BEST FILE: submission_ensemble_optimal.csv
Expected score: 0.42-0.45 (if temporal features help!)
""")

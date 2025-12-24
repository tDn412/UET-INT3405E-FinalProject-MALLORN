"""
INTENSIVE HYPERPARAMETER TUNING

Use Optuna or GridSearch to find optimal hyperparameters
for the best model so far (stacking gave us 0.4659)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("INTENSIVE HYPERPARAMETER TUNING")
print("="*70)

# Load features
print("\n[1] Loading features...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

X_train = train_meta[['Z', 'EBV']].copy()
X_train['Z_EBV_ratio'] = X_train['Z'] / (X_train['EBV'] + 1e-5)
X_train['Z_squared'] = X_train['Z'] ** 2

X_test = test_meta[['Z', 'EBV']].copy()
X_test['Z_EBV_ratio'] = X_test['Z'] / (X_test['EBV'] + 1e-5)
X_test['Z_squared'] = X_test['Z'] ** 2

for col in train_lc.columns:
    if col != 'object_id':
        X_train[f'lc_{col}'] = train_lc[col].values
        X_test[f'lc_{col}'] = test_lc[col].values

X_train = X_train.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
X_test = X_test.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y_train = train_meta['target'].values

print(f"Features: {X_train.shape[1]}, Samples: {len(X_train)}")

# Grid search parameters
print("\n[2] Grid search hyperparameters...")

param_grid = {
    'n_estimators': [400, 600, 800],
    'learning_rate': [0.02, 0.03, 0.05],
    'max_depth': [4, 5, 6],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [2, 3, 4],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0.5, 1.0, 1.5]
}

# Random search (faster than full grid)
np.random.seed(42)
n_iterations = 50  # Try 50 random combinations

best_f1 = 0
best_params = None
best_threshold = 0.1

print(f"Testing {n_iterations} random parameter combinations...")

for iteration in range(n_iterations):
    # Random params
    params = {
        'n_estimators': np.random.choice(param_grid['n_estimators']),
        'learning_rate': np.random.choice(param_grid['learning_rate']),
        'max_depth': np.random.choice(param_grid['max_depth']),
        'subsample': np.random.choice(param_grid['subsample']),
        'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
        'gamma': np.random.choice(param_grid['gamma']),
        'min_child_weight': np.random.choice(param_grid['min_child_weight']),
        'reg_alpha': np.random.choice(param_grid['reg_alpha']),
        'reg_lambda': np.random.choice(param_grid['reg_lambda']),
        'random_state': 42
    }
    
    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1s = []
    all_preds = []
    all_labels = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]
        
        scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
        
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos)
        model.fit(X_tr, y_tr, verbose=False)
        
        val_proba = model.predict_proba(X_val)[:, 1]
        all_preds.extend(val_proba)
        all_labels.extend(y_val)
    
    # Find best threshold
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    fold_best_f1 = 0
    fold_best_threshold = 0.1
    
    for t in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(all_labels, (all_preds >= t).astype(int))
        if f1 > fold_best_f1:
            fold_best_f1 = f1
            fold_best_threshold = t
    
    if fold_best_f1 > best_f1:
        best_f1 = fold_best_f1
        best_params = params.copy()
        best_threshold = fold_best_threshold
        print(f"Iter {iteration+1}: NEW BEST F1={best_f1:.4f} (t={best_threshold:.3f})")
    
    if (iteration + 1) % 10 == 0:
        print(f"Iter {iteration+1}/{n_iterations}: Current best F1={best_f1:.4f}")

print(f"\n{'='*70}")
print("BEST HYPERPARAMETERS FOUND")
print(f"{'='*70}")
print(f"F1 Score: {best_f1:.4f}")
print(f"Threshold: {best_threshold:.3f}")
print(f"\nBest params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print(f"\nCOMPARISON:")
print(f"  Baseline:         0.4146")
print(f"  Stacking:         0.4659")
print(f"  Tuned XGBoost:    {best_f1:.4f}")

if best_f1 >= 0.46:
    print("\n[3] Training final tuned model...")
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    final_model = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos)
    final_model.fit(X_train, y_train)
    
    test_proba = final_model.predict_proba(X_test)[:, 1]
    
    with open('test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    binary = (test_proba >= best_threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_tuned.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_tuned.csv")
    print(f"  F1: {best_f1:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    
    pickle.dump({
        'model': final_model,
        'params': best_params,
        'threshold': best_threshold
    }, open('tuned_model.pkl', 'wb'))
    
    print("✓ Model saved!")

print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)

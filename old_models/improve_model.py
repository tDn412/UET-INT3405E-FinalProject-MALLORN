"""
MALLORN TDE - Improved Model Strategy
Diagnose low score and try improvements
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

print("="*70)
print("DIAGNOSIS & IMPROVEMENT")
print("="*70)

# Load features
train_df = pd.read_csv('physics_features_train_full.csv')
test_df = pd.read_csv('physics_features_test_full.csv')

feature_cols = [col for col in train_df.columns 
               if col not in ['object_id', 'target']]

X = train_df[feature_cols]
y = train_df['target']
X_test = test_df[feature_cols]
test_ids = test_df['object_id']

# Fill NaN
for col in feature_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

print(f"\n[1] Current Model Analysis")
print(f"  Features: {len(feature_cols)}")
print(f"  TDE ratio: {y.mean()*100:.2f}%")

# Try different strategies
strategies = {
    'conservative': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01},
    'balanced': {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05},
    'aggressive': {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.1}
}

best_strategy = None
best_f1 = 0

for strategy_name, params in strategies.items():
    print(f"\n[Testing: {strategy_name.upper()}]")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        **params
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_fold, y_train_fold)
        
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Try multiple thresholds
        best_fold_f1 = 0
        for threshold in np.arange(0.01, 0.5, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val_fold, y_pred)
            if f1 > best_fold_f1:
                best_fold_f1 = f1
        
        fold_scores.append(best_fold_f1)
    
    mean_f1 = np.mean(fold_scores)
    print(f"  Mean F1: {mean_f1:.4f} (+/- {np.std(fold_scores):.4f})")
    
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_strategy = strategy_name
        best_params = lgb_params

print(f"\n{'='*70}")
print(f"BEST STRATEGY: {best_strategy.upper()} - F1: {best_f1:.4f}")
print(f"{'='*70}")

# Train with best strategy and try different submission strategies
print(f"\n[2] Generating Multiple Submissions...")

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)
test_proba = final_model.predict_proba(X_test)[:, 1]

# Strategy 1: Very conservative (high precision)
threshold_conservative = 0.3
pred_conservative = (test_proba >= threshold_conservative).astype(int)
pd.DataFrame({
    'object_id': test_ids,
    'prediction': pred_conservative
}).to_csv('submission_conservative.csv', index=False)
print(f"  ✓ submission_conservative.csv (threshold={threshold_conservative}, predicted TDEs: {pred_conservative.sum()})")

# Strategy 2: Moderate
threshold_moderate = 0.15
pred_moderate = (test_proba >= threshold_moderate).astype(int)
pd.DataFrame({
    'object_id': test_ids,
    'prediction': pred_moderate
}).to_csv('submission_moderate.csv', index=False)
print(f"  ✓ submission_moderate.csv (threshold={threshold_moderate}, predicted TDEs: {pred_moderate.sum()})")

# Strategy 3: Aggressive (high recall)
threshold_aggressive = 0.05
pred_aggressive = (test_proba >= threshold_aggressive).astype(int)
pd.DataFrame({
    'object_id': test_ids,
    'prediction': pred_aggressive
}).to_csv('submission_aggressive.csv', index=False)
print(f"  ✓ submission_aggressive.csv (threshold={threshold_aggressive}, predicted TDEs: {pred_aggressive.sum()})")

# Strategy 4: Based on training ratio
n_test = len(test_ids)
n_tde_expected = int(n_test * y.mean())
top_indices = np.argsort(test_proba)[-n_tde_expected:]
pred_ratio_based = np.zeros(len(test_proba), dtype=int)
pred_ratio_based[top_indices] = 1
pd.DataFrame({
    'object_id': test_ids,
    'prediction': pred_ratio_based
}).to_csv('submission_ratio_based.csv', index=False)
print(f"  ✓ submission_ratio_based.csv (top {n_tde_expected} predictions)")

print(f"\n{'='*70}")
print("TRY THESE SUBMISSIONS:")
print("  1. submission_conservative.csv - High precision")
print("  2. submission_moderate.csv - Balanced")
print("  3. submission_aggressive.csv - High recall")
print("  4. submission_ratio_based.csv - Match training ratio")
print(f"{'='*70}")

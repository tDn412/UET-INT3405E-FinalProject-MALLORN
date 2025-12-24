"""
Strategy 1: Submit PROBABILITIES instead of binary predictions
This is the quickest fix to try!
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRATEGY 1: PROBABILITY SUBMISSION")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc_features = pd.read_csv('lightcurve_features_train.csv')
test_lc_features = pd.read_csv('lightcurve_features_test.csv')

# Merge
train_full = train_meta.merge(train_lc_features, on='object_id', how='inner')
test_full = test_meta.merge(test_lc_features, on='object_id', how='inner')

# Feature engineering
metadata_features = ['Z', 'EBV']
for df in [train_full, test_full]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2
    df['EBV_squared'] = df['EBV'] ** 2
    df['Z_EBV_interaction'] = df['Z'] * df['EBV']

engineered_meta = ['Z_EBV_ratio', 'Z_squared', 'EBV_squared', 'Z_EBV_interaction']
lc_feature_cols = [col for col in train_lc_features.columns if col != 'object_id']
all_features = metadata_features + engineered_meta + lc_feature_cols

X = train_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y = train_full['target']
X_test = test_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"Features: {len(all_features)}")
print(f"Training samples: {len(X)}")
print(f"Test samples: {len(X_test)}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with LESS aggressive parameters to reduce overfitting
print("\n[2] Training model with regularization...")

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb_model = xgb.XGBClassifier(
    n_estimators=200,  # Reduced from 300
    learning_rate=0.03,  # Lower learning rate
    max_depth=4,  # Shallower trees
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    subsample=0.7,  # More aggressive subsampling
    colsample_bytree=0.7,  # Sample features
    gamma=0.5,  # Higher regularization
    min_child_weight=5,  # More regularization
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0  # L2 regularization
)

xgb_model.fit(X_train, y_train, verbose=False)

# Evaluate
from sklearn.metrics import roc_auc_score
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# Train on full data
print("\n[3] Training final model on all data...")
final_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=4,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.5,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0
)

final_model.fit(X, y)

# Generate PROBABILITY predictions
print("\n[4] Generating probability predictions...")
test_predictions_proba = final_model.predict_proba(X_test)[:, 1]

print(f"\nProbability statistics:")
print(f"  Min: {test_predictions_proba.min():.4f}")
print(f"  Max: {test_predictions_proba.max():.4f}")
print(f"  Mean: {test_predictions_proba.mean():.4f}")
print(f"  Median: {np.median(test_predictions_proba):.4f}")

# Create submission with PROBABILITIES
submission_proba = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': test_predictions_proba  # Keep as probabilities!
})

submission_proba.to_csv('submission_probability.csv', index=False)

print(f"\n✅ Created submission_probability.csv")
print(f"   Format: Probabilities (0.0 to 1.0)")
print(f"   This might work better if competition evaluates on probabilities!")

# Also create version with different thresholds
print("\n[5] Creating versions with different thresholds...")

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    binary_preds = (test_predictions_proba >= thresh).astype(int)
    tde_count = binary_preds.sum()
    tde_rate = tde_count / len(binary_preds) * 100
    print(f"  Threshold {thresh}: {tde_count} TDEs ({tde_rate:.2f}%)")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("""
1. TRY FIRST: submission_probability.csv
   → If competition uses ROC-AUC, probabilities are better!
   
2. If that doesn't work, competition might use binary
   → Then we need to find optimal threshold
   → Training has ~4.9% TDE, but optimal threshold might differ

3. Note: Reduced overfitting with:
   - Shallower trees (max_depth=4 instead of 6)
   - Lower learning rate (0.03 vs 0.05)
   - More regularization (gamma, min_child_weight, reg_alpha, reg_lambda)
   - More aggressive subsampling
""")

print("\n✅ Ready to submit: submission_probability.csv")
print("="*70)

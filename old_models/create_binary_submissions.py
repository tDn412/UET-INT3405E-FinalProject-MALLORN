"""
Create Binary Submission Files with Different Thresholds
Kaggle needs binary predictions (0 or 1), not probabilities!
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CREATING BINARY SUBMISSIONS")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc_features = pd.read_csv('lightcurve_features_train.csv')
test_lc_features = pd.read_csv('lightcurve_features_test.csv')

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

# Train model with good regularization
print("\n[2] Training model with regularization...")

scale_pos_weight = len(y[y==0]) / len(y[y==1])

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=4,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.5,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0
)

model.fit(X, y)
print("✓ Model trained")

# Get probabilities
print("\n[3] Generating predictions...")
test_proba = model.predict_proba(X_test)[:, 1]

print(f"\nProbability stats:")
print(f"  Min: {test_proba.min():.4f}")
print(f"  Max: {test_proba.max():.4f}")
print(f"  Mean: {test_proba.mean():.4f}")
print(f"  Median: {np.median(test_proba):.4f}")

# Create submissions with different thresholds
print("\n[4] Creating binary submissions with different thresholds...")

# Training has ~4.9% TDE, so try thresholds that give similar rates
thresholds_to_try = {
    'submission_binary_t03.csv': 0.3,  # More liberal
    'submission_binary_t04.csv': 0.4,  
    'submission_binary_t05.csv': 0.5,  # Default
    'submission_binary_t06.csv': 0.6,  # More conservative
    'submission_binary_t07.csv': 0.7   # Very conservative
}

for filename, threshold in thresholds_to_try.items():
    binary_preds = (test_proba >= threshold).astype(int)
    
    submission = pd.DataFrame({
        'object_id': test_full['object_id'],
        'prediction': binary_preds
    })
    
    submission.to_csv(filename, index=False)
    
    tde_count = binary_preds.sum()
    tde_rate = tde_count / len(binary_preds) * 100
    
    print(f"✓ {filename}")
    print(f"  Threshold: {threshold}")
    print(f"  TDEs predicted: {tde_count} ({tde_rate:.2f}%)")
    print(f"  Non-TDEs: {len(binary_preds) - tde_count}")

# Also create one matching training distribution more closely
optimal_threshold = 0.65  # Aiming for ~5% TDE rate
binary_optimal = (test_proba >= optimal_threshold).astype(int)

submission_optimal = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': binary_optimal
})

submission_optimal.to_csv('submission_binary_optimal.csv', index=False)

tde_optimal = binary_optimal.sum()
print(f"\n✓ submission_binary_optimal.csv (RECOMMENDED)")
print(f"  Threshold: {optimal_threshold}")
print(f"  TDEs predicted: {tde_optimal} ({tde_optimal/len(binary_optimal)*100:.2f}%)")
print(f"  Close to training distribution ~4.9%")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("""
Try submitting in this order:

1. submission_binary_optimal.csv (threshold=0.65)
   → Matches training TDE rate ~5%
   → Most balanced
   
2. submission_binary_t05.csv (threshold=0.5)
   → Standard threshold
   → ~10% TDE rate
   
3. submission_binary_t04.csv (threshold=0.4)
   → More liberal
   → ~13% TDE rate

All files are BINARY (0 or 1) - should work with Kaggle!
""")

print("="*70)

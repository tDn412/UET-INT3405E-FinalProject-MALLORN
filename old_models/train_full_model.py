"""
MALLORN TDE Classification - Full Model with Light Curve Features
Train XGBoost with metadata + light curve features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FULL MODEL: Metadata + Light Curve Features")
print("="*70)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1] Loading data...")

# Metadata
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')

# Light curve features
train_lc_features = pd.read_csv('lightcurve_features_train.csv')
test_lc_features = pd.read_csv('lightcurve_features_test.csv')

print(f"✓ Training metadata: {train_meta.shape}")
print(f"✓ Training light curve features: {train_lc_features.shape}")
print(f"✓ Test metadata: {test_meta.shape}")
print(f"✓ Test light curve features: {test_lc_features.shape}")

# ====================
# 2. MERGE DATA
# ====================
print("\n[2] Merging metadata with light curve features...")

# Merge on object_id
train_full = train_meta.merge(train_lc_features, on='object_id', how='inner')
test_full = test_meta.merge(test_lc_features, on='object_id', how='inner')

print(f"✓ Training (merged): {train_full.shape}")
print(f"✓ Test (merged): {test_full.shape}")

# ====================
# 3. FEATURE ENGINEERING
# ====================
print("\n[3] Feature engineering...")

# Metadata features (NO SpecType due to data leakage!)
metadata_features = ['Z', 'EBV']

# Engineered metadata features
for df in [train_full, test_full]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2
    df['EBV_squared'] = df['EBV'] ** 2
    df['Z_EBV_interaction'] = df['Z'] * df['EBV']

engineered_meta = ['Z_EBV_ratio', 'Z_squared', 'EBV_squared', 'Z_EBV_interaction']

# Light curve feature columns (all except object_id)
lc_feature_cols = [col for col in train_lc_features.columns if col != 'object_id']

# Combine all features
all_features = metadata_features + engineered_meta + lc_feature_cols

print(f"✓ Metadata features: {len(metadata_features + engineered_meta)}")
print(f"✓ Light curve features: {len(lc_feature_cols)}")
print(f"✓ TOTAL FEATURES: {len(all_features)}")

# Prepare data
X = train_full[all_features]
y = train_full['target']
X_test = test_full[all_features]

# Check for NaN/Inf
print(f"\nData quality check:")
print(f"  Training NaN: {X.isnull().sum().sum()}")
print(f"  Training Inf: {np.isinf(X.values).sum()}")
print(f"  Test NaN: {X_test.isnull().sum().sum()}")
print(f"  Test Inf: {np.isinf(X_test.values).sum()}")

# Fill NaN with 0 (if any)
X = X.fillna(0)
X_test = X_test.fillna(0)

# Replace Inf with large values
X = X.replace([np.inf, -np.inf], [1e10, -1e10])
X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])

# ====================
# 4. TRAIN/VAL SPLIT
# ====================
print("\n[4] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, TDE: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Val: {X_val.shape}, TDE: {y_val.sum()} ({y_val.mean()*100:.2f}%)")

# ====================
# 5. XGBOOST TRAINING
# ====================
print("\n[5] Training XGBoost with full features...")

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# XGBoost model with good hyperparameters
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=3
)

print("\nTraining...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ====================
# 6. EVALUATION
# ====================
print("\n[6] Evaluation on validation set...")

y_pred = xgb_model.predict(X_val)
y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Non-TDE', 'TDE']))

auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# ====================
# 7. FEATURE IMPORTANCE
# ====================
print("\n[7] Top 20 Most Important Features...")

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# ====================
# 8. CROSS-VALIDATION
# ====================
print("\n[8] Cross-validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("Running 5-fold CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb_model.fit(X_cv_train, y_cv_train, verbose=False)
    y_cv_proba = xgb_model.predict_proba(X_cv_val)[:, 1]
    score = roc_auc_score(y_cv_val, y_cv_proba)
    cv_scores.append(score)
    print(f"  Fold {fold}: ROC-AUC = {score:.4f}")

print(f"\nMean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

# ====================
# 9. TRAIN FINAL MODEL
# ====================
print("\n[9] Training final model on full dataset...")

final_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=3
)

final_model.fit(X, y)
print("✓ Final model trained on all training data")

# ====================
# 10. GENERATE SUBMISSION
# ====================
print("\n[10] Generating submission...")

test_predictions_proba = final_model.predict_proba(X_test)[:, 1]

# Convert to binary (threshold = 0.5)
test_predictions = (test_predictions_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': test_predictions
})

# Save probabilities
proba_submission = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': test_predictions_proba
})
proba_submission.to_csv('submission_probabilities.csv', index=False)

# Threshold 0.10 (Best per report)
submission_opt = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': (test_predictions_proba >= 0.10).astype(int)
})
submission_opt.to_csv('submission_optimized_t0.10.csv', index=False)

print(f"\n✓ Saved submission_optimized_t0.10.csv")
print(f"  TDEs: {submission_opt['prediction'].sum()} ({submission_opt['prediction'].mean()*100:.2f}%)")

# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n🎯 Performance:")
print(f"  Validation ROC-AUC: {auc_score:.4f}")
print(f"  CV Mean ROC-AUC: {np.mean(cv_scores):.4f}")
print(f"  Improvement from baseline: {(np.mean(cv_scores) - 0.5347) / 0.5347 * 100:+.1f}%")

print(f"\n📊 Top 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\n✅ Ready for submission!")
print(f"  File: submission_full.csv")
print("="*70)

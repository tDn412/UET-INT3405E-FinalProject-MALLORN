"""
Strategy 2: Feature Selection - Use Only Top Important Features
Reduce overfitting by using fewer, more important features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRATEGY 2: FEATURE SELECTION")
print("="*70)

# Load data
print("\n[1] Loading and preparing data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc_features = pd.read_csv('lightcurve_features_train.csv')
test_lc_features = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc_features, on='object_id', how='inner')
test_full = test_meta.merge(test_lc_features, on='object_id', how='inner')

#  Feature engineering
metadata_features = ['Z', 'EBV']
for df in [train_full, test_full]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2
    df['EBV_squared'] = df['EBV'] ** 2
    df['Z_EBV_interaction'] = df['Z'] * df['EBV']

engineered_meta = ['Z_EBV_ratio', 'Z_squared', 'EBV_squared', 'Z_EBV_interaction']
lc_feature_cols = [col for col in train_lc_features.columns if col != 'object_id']
all_features = metadata_features + engineered_meta + lc_feature_cols

X_full = train_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y = train_full['target']
X_test_full = test_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"Total features: {len(all_features)}")

# Train model to get feature importances
print("\n[2] Training model to get feature importances...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

temp_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
temp_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': temp_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(feature_importance.head(20).to_string(index=False))

# Select top N features
top_n = 25  # Use only top 25 features
selected_features = feature_importance.head(top_n)['feature'].tolist()

print(f"\n[3] Selected top {top_n} features:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i}. {feat}")
print(f"  ... and {top_n-10} more")

# Train with selected features
print(f"\n[4] Training with {top_n} selected features...")

X_selected = X_full[selected_features]
X_test_selected = X_test_full[selected_features]

X_train_sel, X_val_sel, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

model_selected = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.04,
    max_depth=5,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    min_child_weight=4,
    reg_alpha=0.05,
    reg_lambda=0.5
)

model_selected.fit(X_train_sel, y_train)

# Evaluate
y_val_proba = model_selected.predict_proba(X_val_sel)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# Cross-validation
print("\n[5] Cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
    X_cv_train, X_cv_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model_selected.fit(X_cv_train, y_cv_train)
    y_cv_proba = model_selected.predict_proba(X_cv_val)[:, 1]
    score = roc_auc_score(y_cv_val, y_cv_proba)
    cv_scores.append(score)
    print(f"  Fold {fold}: {score:.4f}")

print(f"\nMean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

# Train final model
print("\n[6] Training final model...")
final_model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.04,
    max_depth=5,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    min_child_weight=4,
    reg_alpha=0.05,
    reg_lambda=0.5
)

final_model.fit(X_selected, y)

# Predict
test_proba = final_model.predict_proba(X_test_selected)[:, 1]

# Save
submission = pd.DataFrame({
    'object_id': test_full['object_id'],
    'prediction': test_proba
})

submission.to_csv('submission_feature_selected.csv', index=False)

print(f"\n✅ Created: submission_feature_selected.csv")
print(f"   Features used: {top_n} (instead of {len(all_features)})")
print(f"   Less overfitting, potentially better generalization!")

print("\n" + "="*70)

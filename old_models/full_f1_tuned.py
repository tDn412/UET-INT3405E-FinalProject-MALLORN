"""
Full Features + F1-Specific Tuning
Best of both worlds: Keep all features, but tune FOR F1
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FULL FEATURES + F1-SPECIFIC HYPERPARAMETERS")
print("="*70)

# Load data (same as submission_full which got 0.4068)
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc, on='object_id', how='inner')
test_full = test_meta.merge(test_lc, on='object_id', how='inner')

# Same features as before
metadata_features = ['Z', 'EBV']
for df in [train_full, test_full]:
    df['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    df['Z_squared'] = df['Z'] ** 2
    df['EBV_squared'] = df['EBV'] ** 2
    df['Z_EBV_interaction'] = df['Z'] * df['EBV']

engineered_meta = ['Z_EBV_ratio', 'Z_squared', 'EBV_squared', 'Z_EBV_interaction']
lc_feature_cols = [col for col in train_lc.columns if col != 'object_id']
all_features = metadata_features + engineered_meta + lc_feature_cols

X = train_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y = train_full['target']
X_test = test_full[all_features].fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"Features: {len(all_features)}")

# F1 scorer
f1_scorer = make_scorer(f1_score)

# Focused grid search on key parameters for F1
print("\n[2] Grid search with F1 metric...")

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.07],
    'scale_pos_weight': [18, 19.6, 22],  # Around true ratio
    'min_child_weight': [2, 3, 4]
}

base_model = xgb.XGBClassifier(
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1
)

grid_search = GridSearchCV(
    base_model,
    param_grid,
    scoring=f1_scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("Running grid search (this may take a few minutes)...")
grid_search.fit(X, y)

print(f"\nBest CV F1 score: {grid_search.best_score_:.4f}")
print(f"Best parameters: {grid_search.best_params_}")

# Find optimal threshold
print("\n[3] Finding optimal threshold...")

best_model = grid_search.best_estimator_
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_val_proba = []
all_val_y = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    best_model.fit(X_tr, y_tr)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    all_val_proba.extend(val_proba)
    all_val_y.extend(y_val)

all_val_proba = np.array(all_val_proba)
all_val_y = np.array(all_val_y)

# Find best threshold
thresholds = np.arange(0.05, 0.5, 0.02)
f1_scores = []

for threshold in thresholds:
    y_pred = (all_val_proba >= threshold).astype(int)
    f1 = f1_score(all_val_y, y_pred)
    f1_scores.append((threshold, f1))

f1_scores.sort(key=lambda x: x[1], reverse=True)
best_threshold = f1_scores[0][0]
best_f1 = f1_scores[0][1]

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1: {best_f1:.4f}")

print("\nTop 5 thresholds:")
for thresh, f1 in f1_scores[:5]:
    print(f"  t={thresh:.3f}: F1={f1:.4f}")

# Train final model
print("\n[4] Training final model...")
final_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42,
                                 subsample=0.8, colsample_bytree=0.8, gamma=0.1)
final_model.fit(X, y)

# Predict
test_proba = final_model.predict_proba(X_test)[:, 1]

# Create submissions
print("\n[5] Creating submissions...")

for i, (thresh, _) in enumerate(f1_scores[:3]):
    binary = (test_proba >= thresh).astype(int)
    sub = pd.DataFrame({
        'object_id': test_full['object_id'],
        'prediction': binary
    })
    
    filename = f'submission_f1_full_v{i+1}.csv'
    sub.to_csv(filename, index=False)
    
    tde_count = binary.sum()
    print(f"✓ {filename}: t={thresh:.3f}, TDEs={tde_count} ({tde_count/len(binary)*100:.2f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
FULL FEATURES (58) + F1 HYPERPARAMETERS

Previous: 
- submission_full: 0.4068 (old params, t=0.5)
- submission_f1_optimal: 0.4146 (old params, t=0.1)

New:
- CV F1: {best_f1:.4f}
- Best threshold: {best_threshold:.3f}
- Tuned FOR F1 metric!

FILES TO TRY:
1. submission_f1_full_v1.csv (best threshold)
2. submission_f1_full_v2.csv (2nd best)
3. submission_f1_full_v3.csv (3rd best)

Expected: Should improve on 0.4146!
""")
print("="*70)

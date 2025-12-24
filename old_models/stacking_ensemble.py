"""
ADVANCED STACKING WITH PROPER REGULARIZATION

Why hybrid failed: Too many features (506), DL embeddings overfit

New strategy:
1. Train diverse base models with DIFFERENT features
2. Use their PROBABILITIES (not embeddings) as meta-features
3. Train meta-learner with heavy regularization
4. Proper out-of-fold predictions to avoid overfitting
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
print("ADVANCED STACKING - PROPER OUT-OF-FOLD")
print("="*70)

# Load features
print("\n[1] Loading features...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

# Build feature matrix (original 58)
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

# Load ADVANCED temporal features (Physics-based)
train_adv = pd.read_csv('advanced_temporal_features_train.csv')
test_adv = pd.read_csv('advanced_temporal_features_test.csv')

for col in train_adv.columns:
    X_train[f'adv_{col}'] = train_adv[col].values
    X_test[f'adv_{col}'] = test_adv[col].values

print(f"Added {len(train_adv.columns)} advanced features")

X_train = X_train.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
X_test = X_test.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
y_train = train_meta['target'].values

print(f"Features: {X_train.shape[1]}, Samples: {len(X_train)}")

# Select top features by importance (reduce overfitting)
print("\n[2] Selecting top features...")

temp_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
temp_model.fit(X_train, y_train)
importance = temp_model.feature_importances_

# Keep top 30 features
top_indices = np.argsort(importance)[-30:]
top_features = X_train.columns[top_indices].tolist()

X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

print(f"Selected {len(top_features)} top features")

# Define diverse base models
print("\n[3] Training base models with out-of-fold predictions...")

base_models = [
    ('xgb1', xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42
    )),
    ('xgb2', xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.7, colsample_bytree=0.8,
        random_state=123
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        random_state=42, n_jobs=-1
    )),
]

# Out-of-fold predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_train_preds = np.zeros((len(X_train_selected), len(base_models)))
test_preds_all = np.zeros((len(X_test_selected), len(base_models)))

for model_idx, (name, model) in enumerate(base_models):
    print(f"\nTraining {name}...")
    
    oof_preds = np.zeros(len(X_train_selected))
    test_preds_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train), 1):
        X_tr = X_train_selected.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train_selected.iloc[val_idx]
        
        # Train
        model.fit(X_tr, y_tr)
        
        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Test predictions
        test_preds_folds.append(model.predict_proba(X_test_selected)[:, 1])
        
        print(f"  Fold {fold} done", end='\r')
    
    # Average test predictions across folds
    oof_train_preds[:, model_idx] = oof_preds
    test_preds_all[:, model_idx] = np.mean(test_preds_folds, axis=0)
    
    print(f"  {name} complete - OOF predictions ready")

# Meta-learner: Logistic Regression with L2 regularization
print("\n[4] Training meta-learner...")

meta_learner = LogisticRegression(
    C=0.1,  # Strong regularization
    penalty='l2',
    max_iter=1000,
    random_state=42
)

meta_learner.fit(oof_train_preds, y_train)

# Meta predictions
meta_train_proba = meta_learner.predict_proba(oof_train_preds)[:, 1]
meta_test_proba = meta_learner.predict_proba(test_preds_all)[:, 1]

# Find best threshold
best_threshold = 0.1
best_f1 = 0
for t in np.arange(0.05, 0.5, 0.005):
    f1 = f1_score(y_train, (meta_train_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n{'='*70}")
print("STACKING RESULTS")
print(f"{'='*70}")
print(f"Meta-learner F1: {best_f1:.4f}")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"\nCOMPARISON:")
print(f"  XGBoost baseline: 0.4146")
print(f"  Stacking:         {best_f1:.4f}")
print(f"  Difference:       {(best_f1 - 0.4146):.4f}")

# Create submission if competitive
if best_f1 >= 0.40:
    print("\n[5] Creating submission...")
    
    with open('test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    binary = (meta_test_proba >= best_threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_stacking.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_stacking.csv")
    print(f"  F1: {best_f1:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    
    # Save models
    pickle.dump({
        'base_models': base_models,
        'meta_learner': meta_learner,
        'top_features': top_features,
        'threshold': best_threshold
    }, open('stacking_model.pkl', 'wb'))
    
    print("✓ Models saved!")

else:
    print(f"\n❌ Stacking not better than baseline")

print("\n" + "="*70)
print("STACKING COMPLETE!")
print("="*70)

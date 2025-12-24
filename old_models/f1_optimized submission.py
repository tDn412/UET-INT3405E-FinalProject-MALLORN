"""
F1-SCORE OPTIMIZED SUBMISSION  
KEY INSIGHT: Competition uses F1, not ROC-AUC!
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("F1-SCORE OPTIMIZATION (NOT ROC-AUC!)")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc, on='object_id', how='inner')
test_full = test_meta.merge(test_lc, on='object_id', how='inner')

# Feature engineering (same as submission_full which got 0.4068)
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
print(f"Training samples: {len(X)}")

# Train XGBoost (same model that got best score 0.4068)
print("\n[2] Training XGBoost...")

scale_pos = (y == 0).sum() / (y == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos,
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=3
)

# Cross-validation to find BEST F1 THRESHOLD
print("\n[3] Finding optimal F1 threshold via CV...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_val_proba = []
all_val_y = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_tr, y_tr, verbose=False)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    all_val_proba.extend(val_proba)
    all_val_y.extend(y_val)

all_val_proba = np.array(all_val_proba)
all_val_y = np.array(all_val_y)

# Try different thresholds and find best F1
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []

print("\nTesting thresholds for F1 score:")
for threshold in thresholds:
    y_pred = (all_val_proba >= threshold).astype(int)
    f1 = f1_score(all_val_y, y_pred)
    precision = precision_score(all_val_y, y_pred, zero_division=0)
    recall = recall_score(all_val_y, y_pred)
    f1_scores.append(f1)
    
    if threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  t={threshold:.2f}: F1={f1:.4f} (P={precision:.4f}, R={recall:.4f})")

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"\n✓ BEST F1 THRESHOLD: {best_threshold:.3f}")
print(f"✓ BEST CV F1 SCORE: {best_f1:.4f}")

# Also check ROC-AUC for comparison
roc_auc = roc_auc_score(all_val_y, all_val_proba)
print(f"  (ROC-AUC: {roc_auc:.4f})")

# Train final model on ALL data
print("\n[4] Training final model on all data...")
model.fit(X, y)

# Predict on test
test_proba = model.predict_proba(X_test)[:, 1]

# Create multiple submissions with different thresholds around the best
print("\n[5] Creating submissions...")

submissions = {
    f'submission_f1_optimal.csv': best_threshold,
    f'submission_f1_lower.csv': max(0.1, best_threshold - 0.1),
    f'submission_f1_higher.csv': min(0.8, best_threshold + 0.1),
}

for filename, threshold in submissions.items():
    binary = (test_proba >= threshold).astype(int)
    sub = pd.DataFrame({
        'object_id': test_full['object_id'],
        'prediction': binary
    })
    sub.to_csv(filename, index=False)
    
    tde_count = binary.sum()
    print(f"✓ {filename}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
KEY LEARNING: Competition evaluates on F1 SCORE, not ROC-AUC!

This is WHY previous submissions failed:
- They optimized for ROC-AUC (probabilities)
- But competition needs F1 (precision/recall balance)

F1-OPTIMIZED APPROACH:
- Used same features as best submission (0.4068)
- But optimized THRESHOLD for F1 score
- Best threshold: {best_threshold:.3f}
- Best CV F1: {best_f1:.4f}

TRY IN ORDER:
1. submission_f1_optimal.csv (threshold={best_threshold:.3f})
2. submission_f1_lower.csv (threshold={max(0.1, best_threshold - 0.1):.3f})
3. submission_f1_higher.csv (threshold={min(0.8, best_threshold + 0.1):.3f})

This should perform MUCH better than previous submissions!
""")
print("="*70)

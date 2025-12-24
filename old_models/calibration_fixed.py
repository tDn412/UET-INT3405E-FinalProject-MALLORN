"""
Fixed Calibration - Use OOF probs for isotonic fitting
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

print("="*80)
print("FIXED PROBABILITY CALIBRATION")
print("="*80)

# Load
phys = pd.read_csv('physics_features_train_full.csv')
log = pd.read_csv('train_log.csv')

if 'target' in phys.columns:
    phys = phys.drop(columns=['target'])

data = phys.merge(log[['object_id', 'target']], on='object_id', how='inner')

feat_cols = [c for c in phys.columns if c not in ['object_id', 'Z', 'EBV', 'Z_err']]
X = data[feat_cols].fillna(data[feat_cols].median())
y = data['target'].values

print(f"Dataset: {len(data)} objects")

# Generate OOF predictions for calibration
print("\nGenerating OOF predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs_raw = np.zeros(len(X))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, random_state=42, verbose=-1
    )
    
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[val], y[val])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    oof_probs_raw[val] = model.predict_proba(X.iloc[val])[:, 1]
    print(f"  Fold {fold+1}/5 done")

# Fit isotonic calibrator on OOF
print("\nFitting isotonic calibrator on OOF predictions...")
iso_calibrator = IsotonicRegression(out_of_bounds='clip')
iso_calibrator.fit(oof_probs_raw, y)

# Check OOF improvement
oof_probs_calibrated = iso_calibrator.predict(oof_probs_raw)

def get_f1(probs, y_true, n=400):
    preds = np.zeros(len(y_true))
    top_idx = np.argsort(probs)[::-1][:n]
    preds[top_idx] = 1
    return f1_score(y_true, preds)

f1_raw = get_f1(oof_probs_raw, y)
f1_calibrated = get_f1(oof_probs_calibrated, y)

print(f"\nOOF F1 Scores (N=400):")
print(f"  Raw:        {f1_raw:.4f}")
print(f"  Calibrated: {f1_calibrated:.4f} ({f1_calibrated-f1_raw:+.4f})")

if f1_calibrated > f1_raw + 0.01:
    print("  ✅ Calibration shows meaningful improvement!")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, random_state=42, verbose=-1
    )
    final_model.fit(X, y)
    
    # Load test
    phys_test = pd.read_csv('physics_features_test_full.csv')
    feat_cols_test = [c for c in phys_test.columns if c not in ['object_id', 'Z', 'EBV', 'Z_err']]
    X_test = phys_test[feat_cols_test].fillna(phys_test[feat_cols_test].median())
    
    # Get raw test probs
    probs_test_raw = final_model.predict_proba(X_test)[:, 1]
    
    # Apply calibration
    probs_test_calibrated = iso_calibrator.predict(probs_test_raw)
    
    # Save submission
    top_idx = np.argsort(probs_test_calibrated)[::-1][:400]
    
    sub = pd.DataFrame({
        'object_id': phys_test['object_id'].values,
        'prediction': 0
    })
    sub.loc[top_idx, 'prediction'] = 1
    
    sub.to_csv('submission_isotonic_calibrated_top400.csv', index=False)
    print(f"\n✅ Saved: submission_isotonic_calibrated_top400.csv")
    print(f"   Expected improvement: ~{(f1_calibrated-f1_raw)*100:.1f}% over baseline")
else:
    print("  ❌ Calibration doesn't improve F1 meaningfully.")
    print("     Stick with current best (N=400, raw probs)")

"""
Simplified Error Analysis - Focus on patterns without Z complications
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

# Load
phys = pd.read_csv('physics_features_train_full.csv')
log = pd.read_csv('train_log.csv')

# Drop duplicate target from phys
if 'target' in phys.columns:
    phys = phys.drop(columns=['target'])

# Merge
data = phys.merge(log, on='object_id', how='inner')

print("Dataset:", data.shape)

# Prepare features (exclude metadata)
feat_cols = [c for c in phys.columns if c not in ['object_id', 'Z', 'EBV', 'Z_err']]
X = data[feat_cols].fillna(data[feat_cols].median())
y = data['target'].values

print(f"Features: {len(feat_cols)}")
print(f"TDEs: {y.sum()}")

# Generate OOF predictions
print("\nGenerating OOF predictions...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(X))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5...", end='')
    
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, random_state=42, verbose=-1
    )
    
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[val], y[val])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    oof_probs[val] = model.predict_proba(X.iloc[val])[:, 1]
    print(" Done")

# Select Top-400
oof_preds = np.zeros(len(X))
top_idx = np.argsort(oof_probs)[::-1][:400]
oof_preds[top_idx] = 1

# Metrics
f1 = f1_score(y, oof_preds)
cm = confusion_matrix(y, oof_preds)

print(f"\nOOF F1: {f1:.4f}")
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# False Positives
fp_mask = (oof_preds == 1) & (y == 0)
fp_data = data[fp_mask].copy()
fp_data['prob'] = oof_probs[fp_mask]

print(f"\n{'='*80}")
print(f"FALSE POSITIVES: {fp_mask.sum()}")
print(f"{'='*80}")

print("\nTop 20 by probability:")
print(fp_data[['object_id', 'prob', 'SpecType']].sort_values('prob', ascending=False).head(20).to_string(index=False))

print(f"\nSpecType distribution:")
for spec, count in fp_data['SpecType'].value_counts().head(10).items():
    pct = count / len(fp_data) * 100
    print(f"  {spec:15s}: {count:3d} ({pct:5.1f}%)")

# False Negatives
fn_mask = (oof_preds == 0) & (y == 1)
fn_data = data[fn_mask].copy()  
fn_data['prob'] = oof_probs[fn_mask]

print(f"\n{'='*80}")
print(f"FALSE NEGATIVES: {fn_mask.sum()} missed TDEs")
print(f"{'='*80}")

print("\nTop 20 missed (highest prob):")
print(fn_data[['object_id', 'prob', 'SpecType']].sort_values('prob', ascending=False).head(20).to_string(index=False))

# Insights
print(f"\n{'='*80}")
print("ACTIONABLE INSIGHTS")
print(f"{'='*80}")

fp_spec_counts = fp_data['SpecType'].value_counts()
if len(fp_spec_counts) > 0:
    top_spec_name = fp_spec_counts.index[0]
    top_spec_count = fp_spec_counts.iloc[0]
    
    print(f"\nTop FP SpecType: {top_spec_name} ({top_spec_count/len(fp_data)*100:.1f}%)")
    
    if top_spec_count / len(fp_data) > 0.3:
        print(f"RECOMMENDATION: Focus on improving TDE vs {top_spec_name} discrimination")



# Save
fp_data[['object_id', 'prob', 'SpecType']].to_csv('false_positives.csv', index=False)
fn_data[['object_id', 'prob', 'SpecType']].to_csv('false_negatives.csv', index=False)

print("\n✅ Saved: false_positives.csv, false_negatives.csv")

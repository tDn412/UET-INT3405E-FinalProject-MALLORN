import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_proba):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

# Load
gp = pd.read_csv('gp_features_train.csv')
phys = pd.read_csv('physics_features_train_full.csv')  
log = pd.read_csv('train_log.csv')

# Merge
data = gp.merge(phys, on='object_id').merge(log[['object_id','target']], on='object_id')

# Features
exclude = ['object_id','target','split','fold','SpecType','Z','Z_err','EBV','English Translation']
feats = [c for c in data.columns if c not in exclude]

X = data[feats].fillna(data[feats].median())
y = data['target'].values

print(f"Dataset: {len(data)} objects, {len(feats)} features")
print(f"  GP: {sum([1 for f in feats if 'gp_' in f])}, Physics: {sum([1 for f in feats if 'gp_' not in f])}")  
print(f"  TDEs: {y.sum()}")

# CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s = []

for fold, (tr, val) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                is_unbalance=True, random_state=42, verbose=-1)
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[val], y[val])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    proba = model.predict_proba(X.iloc[val])[:, 1]
    thresh, f1 = find_optimal_threshold(y[val], proba)
    f1s.append(f1)
    print(f"Fold {fold+1}: F1={f1:.4f} (t={thresh:.2f})")

print(f"\n{'='*60}")
print(f"AVG F1: {np.mean(f1s):.4f}")
print(f"Baseline (Physics): ~0.46")
print(f"Improvement: {np.mean(f1s)-0.46:+.4f} ({(np.mean(f1s)-0.46)/0.46*100:+.1f}%)")

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

print("=" * 80)
print("GP FEATURES VALIDATION (CRITICAL EXPERIMENT)")
print("=" * 80)

# Load
gp = pd.read_csv('gp_features_train.csv')
phys = pd.read_csv('physics_features_train_full.csv')  
log = pd.read_csv('train_log.csv')

# Drop target from physics (it's already there but may be wrong!)
if 'target' in phys.columns:
    phys = phys.drop(columns=['target'])
    print("Dropped duplicate 'target' from physics features")

# Merge
data = gp.merge(phys, on='object_id', how='inner')
data = data.merge(log[['object_id','target']], on='object_id', how='inner')

print(f"\nData: {len(data)} objects")
print(f"Has target: {'target' in data.columns}")
print(f"TDEs: {data['target'].sum()}")

# Features
exclude = ['object_id','target','split','fold','SpecType','Z','Z_err','EBV','English Translation']
feats = [c for c in data.columns if c not in exclude]

X = data[feats].fillna(data[feats].median())
y = data['target'].values

gp_count = sum([1 for f in feats if 'gp_' in f])
phys_count = sum([1 for f in feats if 'gp_' not in f])

print(f"\nFeatures: {len(feats)} total")
print(f"  GP features: {gp_count}")
print(f"  Physics features: {phys_count}")

# Train with CV
print("\n" + "=" * 80)
print("TRAINING LIGHTGBM WITH 5-FOLD CV")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s = []

for fold, (tr, val) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        num_leaves=31,
        is_unbalance=True, 
        random_state=42, 
        verbose=-1
    )
    
    model.fit(
        X.iloc[tr], y[tr], 
        eval_set=[(X.iloc[val], y[val])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    proba = model.predict_proba(X.iloc[val])[:, 1]
    thresh, f1 = find_optimal_threshold(y[val], proba)
    f1s.append(f1)
    
    print(f"Fold {fold+1}: F1={f1:.4f} (threshold={thresh:.3f})")
    
    # Show top features for fold 1
    if fold == 0:
        imp = pd.DataFrame({'feat': feats, 'imp': model.feature_importances_})
        imp = imp.sort_values('imp', ascending=False)
        
        print("\n  Top 15 Features:")
        for i, row in imp.head(15).iterrows():
            ftype = "GP  " if row['feat'].startswith('gp_') else "PHYS"
            print(f"    {ftype} | {row['feat']:30s} | {row['imp']:.0f}")
        print()

print("=" * 80)
print(f"🎯 FINAL RESULT: Avg F1 = {np.mean(f1s):.4f}")
print("=" * 80)
print(f"\nBaseline (Physics only):  ~0.46")
print(f"GP + Physics:              {np.mean(f1s):.4f}")
print(f"Improvement:              {np.mean(f1s)-0.46:+.4f} ({(np.mean(f1s)-0.46)/0.46*100:+.1f}%)")

if np.mean(f1s) > 0.55:
    print("\n🎉 BREAKTHROUGH! GP features ARE the magic for 0.7+!")
elif np.mean(f1s) > 0.50:
    print("\n✅ Significant improvement! GP features help.")
else:
    print("\n⚠️  Marginal improvement. GP alone may not be enough.")

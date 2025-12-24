"""
BREAKTHROUGH: AGN-Penalty Ranking
Error analysis showed 45% of FPs are AGN!
Solution: Rank by P(TDE) / (1 + P(AGN)) instead of raw P(TDE)
"""

import pandas as pd
import numpy as np

print("="*80)
print("IMPLEMENTING AGN-PENALTY RANKING")
print("="*80)

# Load existing multiclass probabilities
probs_df = pd.read_csv('submission_multiclass_refined_probs.csv')

# Need to get full class probabilities, not just TDE prob
# Let's retrain quickly to get all class probs
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load data
phys = pd.read_csv('physics_features_train_full.csv')  
log = pd.read_csv('train_log.csv')

if 'target' in phys.columns:
    phys = phys.drop(columns=['target'])

data = phys.merge(log[['object_id', 'target', 'SpecType']], on='object_id', how='inner')

# Map SpecType to simplified classes
def simplify_spectype(spec):
    if spec == 'TDE':
        return 0  # TDE
    elif 'AGN' in spec:
        return 1  # AGN
    elif 'Ia' in spec:
        return 2  # SN Ia
    else:
        return 3  # Other

data['class'] = data['SpecType'].apply(simplify_spectype)

# Features
feat_cols = [c for c in phys.columns if c not in ['object_id', 'Z', 'EBV', 'Z_err']]
X = data[feat_cols].fillna(data[feat_cols].median())
y_multiclass = data['class'].values

print(f"\nTraining multi-class model to get P(AGN)...")

# Train final model on all data to get test probs
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    objective='multiclass',
    num_class=4,
    random_state=42,
    verbose=-1
)

model.fit(X, y_multiclass)

# Load test data
phys_test = pd.read_csv('physics_features_test_full.csv')
feat_cols_test = [c for c in phys_test.columns if c not in ['object_id', 'Z', 'EBV', 'Z_err']]
X_test = phys_test[feat_cols_test].fillna(phys_test[feat_cols_test].median())

# Get multi-class probabilities
probs_multiclass = model.predict_proba(X_test)

# Extract relevant probs
P_TDE = probs_multiclass[:, 0]
P_AGN = probs_multiclass[:, 1]

print(f"\nComputing ranking scores...")

# STRATEGY 1: AGN Penalty
score_agn_penalty = P_TDE / (1 + P_AGN)

# STRATEGY 2: Pure margin (TDE must dominate)
other_max = np.max(probs_multiclass[:, 1:], axis=1)
score_margin = P_TDE - other_max

# STRATEGY 3: Certainty-weighted
certainty = probs_multiclass.max(axis=1)
score_certainty = P_TDE * certainty

# Create submissions for each strategy
strategies = {
    '400_original': P_TDE,
    '400_agn_penalty': score_agn_penalty,
    '400_margin': score_margin,
    '400_certainty': score_certainty
}

for name, scores in strategies.items():
    # Rank and select Top-400
    sorted_idx = np.argsort(scores)[::-1]
    
    preds = pd.DataFrame({
        'object_id': phys_test['object_id'].values,
        'prediction': 0
    })
    preds.loc[sorted_idx[:400], 'prediction'] = 1
    
    filename = f'submission_refined_{name}.csv'
    preds.to_csv(filename, index=False)
    print(f"  Saved: {filename}")

print(f"\n{"="*80}")
print("GENERATED 4 SUBMISSIONS TO TEST:")
print(f"{"="*80}")
print("1. submission_refined_400_original.csv      - Baseline (raw P(TDE))")
print("2. submission_refined_400_agn_penalty.csv   - P(TDE) / (1 + P(AGN)) ⭐")
print("3. submission_refined_400_margin.csv        - P(TDE) - max(others)")  
print("4. submission_refined_400_certainty.csv     - P(TDE) * certainty")
print("\nRECOMMENDATION: Submit AGN-penalty first (targets our 45% FP issue)")

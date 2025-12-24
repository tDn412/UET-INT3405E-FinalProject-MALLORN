
import pandas as pd
import numpy as np

# Load files
phys_probs = pd.read_csv('submission_physics_tuned_probs.csv')
xgb_probs = pd.read_csv('submission_xgboost_probs.csv')
ens_sub = pd.read_csv('submission_ensemble_final.csv')
phys_sub = pd.read_csv('submission_physics_tuned_final.csv')

# Merge probs
df = phys_probs.merge(xgb_probs, on='object_id', suffixes=('_phys', '_xgb'))

# Correlation
corr = df['prediction_phys'].corr(df['prediction_xgb'])
print(f"Correlation between Physics and XGBoost: {corr:.4f}")

# Threshold check
thresh = 0.154
phys_preds_chk = (df['prediction_phys'] >= thresh).astype(int)
xgb_preds_chk = (df['prediction_xgb'] >= thresh).astype(int)

print(f"Physics Model (Thresh {thresh}) Count: {phys_preds_chk.sum()}")
print(f"XGBoost Model (Thresh {thresh}) Count: {xgb_preds_chk.sum()}")
print(f"Ensemble Final Count: {ens_sub['prediction'].sum()}")
print(f"Physics Tuned Final Count (Discrepancy check): {phys_sub['prediction'].sum()}")

# Check overlap
overlap = (phys_preds_chk & xgb_preds_chk).sum()
print(f"Overlap (Both=1): {overlap}")

"""
MALLORN TDE - Raw Features Model (Reverse Engineering)
Test hypothesis: Extinction correction might be hurting performance.
We reverse the correction to get back 'raw' flux ratios.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

# Fitzpatrick 1999 coefficients used in physics_features.py
R = {
    'u': 4.239, 'g': 3.303, 'r': 2.285,
    'i': 1.698, 'z': 1.263, 'y': 1.086
}

def reverse_correction(df):
    """
    Recover raw flux ratios from corrected ones.
    Ratio_raw = Ratio_corr * 10^(0.4 * EBV * (R_denom - R_num))
    """
    df = df.copy()
    
    # 1. Reverse Flux Ratios
    # u/g
    df['flux_ratio_u_g'] = df['flux_ratio_u_g'] * 10**(0.4 * df['EBV'] * (R['g'] - R['u']))
    # g/r
    df['flux_ratio_g_r'] = df['flux_ratio_g_r'] * 10**(0.4 * df['EBV'] * (R['r'] - R['g']))
    # r/i
    df['flux_ratio_r_i'] = df['flux_ratio_r_i'] * 10**(0.4 * df['EBV'] * (R['i'] - R['r']))
    # i/z
    df['flux_ratio_i_z'] = df['flux_ratio_i_z'] * 10**(0.4 * df['EBV'] * (R['z'] - R['i']))
    
    # 2. Reverse Max Flux (Approximation - assuming Peak is in g-band or dominant band, 
    # but we don't know which filter. Let's just keep corrected max_flux as a proxy for luminosity 
    # or drop it. Let's drop it to be purely 'raw ratio' focused)
    # df.drop(columns=['max_flux_corrected'], inplace=True)
    
    return df

def main():
    print("="*70)
    print("TRAINING ON RECOVERED RAW FEATURES")
    print("="*70)
    
    train = pd.read_csv('physics_features_train_full.csv')
    test = pd.read_csv('physics_features_test_full.csv')
    
    # Reverse corrections
    print("[1] Reversing Extinction Correction...")
    train_raw = reverse_correction(train)
    test_raw = reverse_correction(test)
    
    # Clean (Clip only, no NaN filling yet)
    ratio_cols = [c for c in train_raw.columns if 'flux_ratio' in c]
    for col in ratio_cols:
        train_raw[col] = train_raw[col].clip(-10, 100)
        test_raw[col] = test_raw[col].clip(-10, 100)
        
    features = [c for c in train.columns if c not in ['object_id', 'target', 'split']]
    print(f"Features: {features}")

    X = train_raw[features]
    y = train_raw['target']
    X_test = test_raw[features]
    
    # Metrics
    scale_pos_weight = (y==0).sum() / (y==1).sum()
    
    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1
    }
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        cv_scores.append(auc)
        
        test_preds += model.predict_proba(X_test)[:, 1] / 10
        
    print(f"\nMean AUC (Raw Features): {np.mean(cv_scores):.4f}")
    
    # Submission
    submission = pd.DataFrame({'object_id': test['object_id']})
    # Use global threshold ~0.5 for raw
    submission['prediction'] = (test_preds >= 0.5).astype(int)
    
    submission.to_csv('submission_raw_reversed.csv', index=False)
    print("✓ Saved submission_raw_reversed.csv")

if __name__ == "__main__":
    main()

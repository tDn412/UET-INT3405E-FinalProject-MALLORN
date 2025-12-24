"""
MALLORN TDE - Robust XGBoost Training with Data Cleaning
Refined strategy based on data auditing to fix CV-LB gap.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

def clean_data(df):
    """Clean physics features logically."""
    df = df.copy()
    
    # 1. Fix Rise Time: Cannot be negative. If negative, likely noise or peak error.
    # We'll set negative rise times to 0 or NaN (XGBoost handles NaN)
    df.loc[df['rise_time'] < 0, 'rise_time'] = np.nan
    
    # 2. Fix Fade Time: Cannot be negative
    df.loc[df['fade_time'] < 0, 'fade_time'] = np.nan
    
    # 3. Flux Ratios: Negative ratios are physically weird for TDEs (usually hot/positive).
    # Extreme values due to division by near-zero. Clip them?
    # Let's clip ratios to a reasonable range [-10, 100] to remove outliers
    ratio_cols = [c for c in df.columns if 'flux_ratio' in c]
    for col in ratio_cols:
        df[col] = df[col].clip(-10, 100)

    # 4. Asymmetry Ratio: If rise or fade is NaN, this should be NaN
    # Also clip extreme values
    df['asymmetry_ratio'] = df['asymmetry_ratio'].clip(0, 50)
    
    return df

def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def main():
    print("="*70)
    print("ROBUST XGBOOST TRAINING PIPELINE")
    print("="*70)
    
    # Load Data
    print("[1] Loading & Cleaning Data...")
    train = pd.read_csv('physics_features_train_full.csv')
    test = pd.read_csv('physics_features_test_full.csv')
    
    print(f"  Original Train: {train.shape}")
    train = clean_data(train)
    test = clean_data(test)
    
    # Features
    exclude = ['object_id', 'target', 'split']
    features = [c for c in train.columns if c not in exclude]
    print(f"  Features ({len(features)}): {features}")
    
    X = train[features]
    y = train['target']
    X_test = test[features]
    
    # Calculate scale_pos_weight
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos
    print(f"  Class Balance: {pos} pos, {neg} neg")
    print(f"  Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Robust Scaler (good for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost Parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight * 1.5,
        'tree_method': 'hist',
        'early_stopping_rounds': 100, # Move here
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Stratified K-Fold
    print("\n[2] Training with StratifiedKFold (K=10)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cv_f1s = []
    cv_aucs = []
    thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X_scaled[train_idx], y.iloc[train_idx]
        X_val, y_val = X_scaled[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        
        # Test predictions (average over folds)
        test_preds += model.predict_proba(X_test_scaled)[:, 1] / skf.get_n_splits()
        
        # Metrics
        auc = roc_auc_score(y_val, val_proba)
        thresh, f1 = find_best_threshold(y_val, val_proba)
        
        cv_aucs.append(auc)
        cv_f1s.append(f1)
        thresholds.append(thresh)
        
        print(f"  Fold {fold+1}: AUC={auc:.4f}, F1={f1:.4f} @ {thresh:.2f}")

    print(f"\n[3] Results Summary")
    print(f"  Mean AUC: {np.mean(cv_aucs):.4f}")
    print(f"  Mean F1:  {np.mean(cv_f1s):.4f}")
    print(f"  Mean Threshold: {np.mean(thresholds):.2f}")
    
    # Global Optimal Threshold on OOF
    global_thresh, global_f1 = find_best_threshold(y, oof_preds)
    print(f"  Global Optimal Threshold (OOF): {global_thresh:.2f} -> F1: {global_f1:.4f}")
    
    # Create Submission
    submission = pd.DataFrame({'object_id': test['object_id']})
    submission['prediction'] = (test_preds >= global_thresh).astype(int)
    
    out_file = 'submission_xgboost_robust.csv'
    submission.to_csv(out_file, index=False)
    print(f"\n✓ Saved submission to {out_file}")
    print(f"  Predicted TDEs: {submission['prediction'].sum()} ({submission['prediction'].mean()*100:.2f}%)")

if __name__ == "__main__":
    main()

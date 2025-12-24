"""
Validate Wavelet & Fourier Features
Compare: Physics Features ONLY vs Physics + Frequency Features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1

def main():
    print("="*80)
    print("VALIDATING WAVELET & FOURIER FEATURES")
    print("="*80)
    
    # 1. Load Data
    print("Loading features...")
    freq_train = pd.read_csv('frequency_features_train.csv')
    phys_train = pd.read_csv('physics_features_train_full.csv')
    log_train = pd.read_csv('train_log.csv')
    
    print(f"Frequency features: {freq_train.shape}")
    print(f"Physics features: {phys_train.shape}")
    
    # 2. Merge
    if 'target' in phys_train.columns:
        phys_train = phys_train.drop(columns=['target'])
        
    data = phys_train.merge(freq_train, on='object_id', how='left')  # Left merge to keep all physics objects
    data = data.merge(log_train[['object_id', 'target']], on='object_id', how='inner')
    
    # Fill Nans (some objects might have failed extraction)
    # Frequency features might be NaN if not enough points
    feat_cols = [c for c in data.columns if c not in ['object_id', 'target', 'Z', 'EBV', 'Z_err', 'split', 'fold']]
    
    # Check NaN count
    nan_counts = data[feat_cols].isnull().sum().sum()
    print(f"Total NaNs in features: {nan_counts}")
    
    X = data[feat_cols].fillna(data[feat_cols].median())
    y = data['target'].values
    
    print(f"Final training set: {X.shape}")
    print(f"Feature count: {len(feat_cols)}")
    
    # 3. Train with CV
    print("\nTraining LightGBM with 5-fold CV...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))
    
    fold_f1s = []
    
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X.iloc[tr], y[tr],
                 eval_set=[(X.iloc[val], y[val])],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        val_probs = model.predict_proba(X.iloc[val])[:, 1]
        oof_probs[val] = val_probs
        
        t, f1 = find_optimal_threshold(y[val], val_probs)
        fold_f1s.append(f1)
        print(f"  Fold {fold+1} F1: {f1:.4f} (Thresh={t:.2f})")
        
        if fold == 0:
            # Feature Importance
            imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_})
            imp = imp.sort_values('importance', ascending=False)
            print("\n  Top 20 Features:")
            print(imp.head(20))
            
    avg_f1 = np.mean(fold_f1s)
    print(f"\nAverage CV F1: {avg_f1:.4f}")
    
    # Save OOF for Stacking
    oof_df = pd.DataFrame({'object_id': log_train['object_id'], 'oof_prob': oof_probs, 'target': y})
    oof_df.to_csv('oof_lgbm_wavelet.csv', index=False)
    print("Saved OOF predictions to oof_lgbm_wavelet.csv")
    
    # Global optimization

    best_t, best_global_f1 = find_optimal_threshold(y, oof_probs)
    print(f"Global Optimal Threshold: {best_t:.3f} -> F1: {best_global_f1:.4f}")
    
    print("\n" + "-"*30)
    print("COMPARISON")
    print(f"Baseline (Physics only): ~0.46")
    print(f"New (Physics + Freq):    {avg_f1:.4f}")
    
    if avg_f1 > 0.47:
        print("\n✅ POSITIVE SIGNAL! Features add value.")
    else:
        print("\n❌ NEGATIVE/NEUTRAL. Features may not be helpful.")

if __name__ == "__main__":
    main()

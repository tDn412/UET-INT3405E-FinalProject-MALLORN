"""
SIMPLIFIED VALIDATION: Train model with GP features
Using simpler merge approach
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.01, 0.99, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def main():
    print("=" * 80)
    print("GP FEATURES VALIDATION")
    print("=" * 80)
    
    # Load all data
    print("\n1. Loading data...")
    gp_df = pd.read_csv('gp_features_train.csv')
    phys_df = pd.read_csv('physics_features_train_full.csv')
    log_df = pd.read_csv('train_log.csv')
    
    print(f"   GP: {gp_df.shape}")
    print(f"   Physics: {phys_df.shape}")  
    print(f"   Log: {log_df.shape}")
    
    #  Merge step by step
    print("\n2. Merging...")
    df = gp_df.merge(phys_df[['object_id'] + [c for c in phys_df.columns if c != 'object_id']], 
                      on='object_id', how='inner')
    print(f"   After GP+Physics: {df.shape}")
    
    df = df.merge(log_df[['object_id', 'target']], on='object_id', how='inner')
    print(f"   After adding target: {df.shape}")
    
    # Verify target
    print(f"\n3. Target check:")
    print(f"   Has target column: {'target' in df.columns}")
    print(f"   Target nulls: {df['target'].isnull().sum()}")
    print(f"   TDE count: {df['target'].sum()}")
    
    if 'target' not in df.columns or df['target'].isnull().all():
        print("FATAL: No target!")
        return
    
    # Prepare features
    exclude_cols = ['object_id', 'target', 'split', 'fold', 'SpecType', 'Z', 'Z_err', 'EBV', 'English Translation']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['target']

    print(f"\n4. Features: {len(feature_cols)} total")
    print(f"   GP features: {sum([1 for c in feature_cols if 'gp_' in c])}")
    print(f"   Physics features: {sum([1 for c in feature_cols if 'gp_' not in c])}")
    
    # Train
    print("\n5. Training LightGBM with 5-Fold CV...")
    print("=" * 80)
    
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        val_proba = model.predict_proba(X_val)[:, 1]
        thresh, f1 = find_optimal_threshold(y_val, val_proba)
        fold_f1s.append(f1)
        print(f"   Fold {fold+1}: F1={f1:.4f} (thresh={thresh:.3f})")
    
    avg_f1 = np.mean(fold_f1s)
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT: Average CV F1 = {avg_f1:.4f}")
    print(f"{'=' * 80}")
    print(f"Baseline (Physics only): F1 ~0.46")
    print(f"GP + Physics:           F1 {avg_f1:.4f}")
    print(f"Improvement:            {avg_f1-0.46:+.4f} ({(avg_f1-0.46)/0.46*100:+.1f}%)\n")
    
    if avg_f1 > 0.55:
        print("🎉 BREAKTHROUGH! GP features ARE the magic!")
    elif avg_f1 > 0.50:
        print("✅ Significant improvement!")
    else:
        print("⚠️  Marginal improvement.")

if __name__ == "__main__":
    main()

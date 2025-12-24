"""
CRITICAL VALIDATION: Train model with GP features to test hypothesis
If CV F1 score jumps significantly, GP features ARE the magic for 0.7+
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

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
    print("VALIDATING GP FEATURES (THE HYPOTHESIS TEST)")
    print("=" * 80)
    
    # Load GP features
    print("\nLoading GP features...")
    gp_train = pd.read_csv('gp_features_train.csv')
    print(f"GP features shape: {gp_train.shape}")
    
    # Load physics features
    print("Loading physics features...")
    phys_train = pd.read_csv('physics_features_train_full.csv')
    print(f"Physics features shape: {phys_train.shape}")
    
    # Load labels
    print("Loading labels...")
    log_df = pd.read_csv('train_log.csv')
    
    # Merge features
    print("\nMerging features...")

    
    # Start with GP features as base (has correct object_ids)
    combined = gp_train.copy()
    print(f"Base (GP) shape: {combined.shape}")
    
    # Add physics features
    combined = combined.merge(phys_train, on='object_id', how='left')
    print(f"After adding physics: {combined.shape}")
    
    # Add target
    combined = combined.merge(log_df[['object_id', 'target']], on='object_id', how='left')
    print(f"After adding target: {combined.shape}")
    
    # Check if target is loaded
    if 'target' not in combined.columns:
        print("\nERROR: Target not found. Debugging...")
        print(f"GP object_id sample: {gp_train['object_id'].head(3).tolist()}")
        print(f"Log object_id sample: {log_df['object_id'].head(3).tolist()}")
        return
    
    print(f"Target distribution: {combined['target'].value_counts().to_dict()}")


    
    # Prepare for training
    feature_cols = [c for c in combined.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType']]
    X = combined[feature_cols]
    y = combined['target']
    
    # Fill NaNs
    median_values = X.median(numeric_only=True)
    X = X.fillna(median_values)
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"GP features: {len([c for c in feature_cols if 'gp_' in c])}")
    print(f"Physics features: {len([c for c in feature_cols if 'gp_' not in c])}")
    
    # Train with Cross-Validation
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM WITH GP + PHYSICS FEATURES")
    print("=" * 80)
    
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        
        # Optimize threshold
        thresh, f1 = find_optimal_threshold(y_val, val_proba)
        fold_f1s.append(f1)
        print(f"Fold {fold+1} F1: {f1:.4f} (Thresh: {thresh:.3f})")
        
        # Feature importance for this fold
        if fold == 0:
            feat_imp = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 20 Features (Fold 1):")
            for idx, row in feat_imp.head(20).iterrows():
                feat_type = "GP" if 'gp_' in row['feature'] else "PHYS"
                print(f"  {feat_type:4s} | {row['feature']:30s} | {row['importance']:.1f}")
    
    avg_f1 = np.mean(fold_f1s)
    print(f"\n{'=' * 80}")
    print(f"AVERAGE CV F1 SCORE: {avg_f1:.4f}")
    print(f"{'=' * 80}")
    
    # Global optimal threshold
    global_thresh, global_f1 = find_optimal_threshold(y, oof_preds)
    print(f"Global Optimal Threshold: {global_thresh:.3f} with F1: {global_f1:.4f}")
    
    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print("Physics-only model (previous best): F1 ~0.46")
    print(f"GP + Physics model (this run):   F1 {avg_f1:.4f}")
    
    improvement = avg_f1 - 0.46
    print(f"\nImprovement: {improvement:+.4f} ({improvement/0.46*100:+.1f}%)")
    
    if avg_f1 > 0.55:
        print("\n🎉 BREAKTHROUGH! GP features are THE MAGIC!")
        print("This explains the 0.7+ leaderboard scores.")
    elif avg_f1 > 0.50:
        print("\n✅ Significant improvement! GP features are valuable.")
    else:
        print("\n❌ Marginal improvement. GP features alone may not be the answer.")

if __name__ == "__main__":
    main()

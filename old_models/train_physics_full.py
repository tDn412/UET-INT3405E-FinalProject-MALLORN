"""
MALLORN TDE - Train Final Model on Full Dataset  
LightGBM with physics features from all 20 splits
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_optimal_threshold(y_true, y_proba):
    """Find F1-maximizing threshold."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def main():
    print("="*70)
    print("FINAL MODEL - FULL DATASET (ALL 20 SPLITS)")
    print("="*70)
    
    # Load full features
    print("\n[1] Loading full dataset...")
    train_df = pd.read_csv('physics_features_train_full.csv')
    test_df = pd.read_csv('physics_features_test_full.csv')
    
    print(f"  Train: {len(train_df):,} objects")
    print(f"  Test: {len(test_df):,} objects")
    
    # Features
    feature_cols = [col for col in train_df.columns 
                   if col not in ['object_id', 'target']]
    
    X = train_df[feature_cols]
    y = train_df['target']
    X_test = test_df[feature_cols]
    test_ids = test_df['object_id']
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  TDE count: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")
    
    # Handle missing values
    print("\n[2] Handling missing values...")
    for col in feature_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            print(f"  {col}: filled {X[col].isna().sum()} NaN with {median_val:.3f}")
    
    # LightGBM configuration
    print("\n[3] Training LightGBM with 5-Fold CV...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,  # More estimators for larger dataset
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/5")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Optimize threshold
        threshold, f1_opt = find_optimal_threshold(y_val_fold, y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Metrics
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        recall = recall_score(y_val_fold, y_pred, zero_division=0)
        auc = roc_auc_score(y_val_fold, y_proba)
        
        print(f"    Threshold: {threshold:.3f}")
        print(f"    F1: {f1_opt:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"    ROC-AUC: {auc:.4f}")
        
        cv_results.append({
            'fold': fold,
            'threshold': threshold,
            'f1': f1_opt,
            'precision': precision,
            'recall': recall,
            'auc': auc
        })
    
    # Summary
    cv_df = pd.DataFrame(cv_results)
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    print(cv_df.to_string(index=False))
    print(f"\nMean F1-Score: {cv_df['f1'].mean():.4f} (+/- {cv_df['f1'].std():.4f})")
    print(f"Mean ROC-AUC: {cv_df['auc'].mean():.4f}")
    print(f"Mean Threshold: {cv_df['threshold'].mean():.3f}")
    
    # Train final model
    print("\n[4] Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)
    
    optimal_threshold = cv_df['threshold'].mean()
    print(f"  Using threshold: {optimal_threshold:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'LightGBM Feature Importance\\nFull Dataset ({len(train_df)} objects)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_lgbm_full_importance.png', dpi=150)
    print("\n✓ Saved: physics_lgbm_full_importance.png")
    plt.close()
    
    # Generate predictions
    print("\n[5] Generating test predictions...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Submissions
    submission_proba = pd.DataFrame({
        'object_id': test_ids,
        'prediction': test_proba
    })
    submission_proba.to_csv('submission_physics_full.csv', index=False)
    print("  ✓ Saved: submission_physics_full.csv")
    
    print(f"\n  Prediction stats:")
    print(f"    Mean: {test_proba.mean():.4f}")
    print(f"    Median: {np.median(test_proba):.4f}")
    print(f"    Max: {test_proba.max():.4f}")
    print(f"    Predicted TDEs (>{optimal_threshold:.3f}): {(test_proba > optimal_threshold).sum()}")
    
    # Save summary
    summary = f"""MALLORN TDE Classification - Physics-Informed Full Model
Generated: {pd.Timestamp.now()}

DATASET:
- Training: {len(train_df):,} objects ({y.sum()} TDEs, {y.mean()*100:.2f}%)
- Test: {len(test_df):,} objects
- Features: {len(feature_cols)}

MODEL: LightGBM
- is_unbalance: True
- n_estimators: {lgb_params['n_estimators']}
- max_depth: {lgb_params['max_depth']}
- learning_rate: {lgb_params['learning_rate']}

CROSS-VALIDATION (5-Fold Stratified):
{cv_df.to_string(index=False)}

Mean F1-Score: {cv_df['f1'].mean():.4f} (+/- {cv_df['f1'].std():.4f})
Mean ROC-AUC: {cv_df['auc'].mean():.4f}
Optimal Threshold: {optimal_threshold:.3f}

TOP 5 FEATURES:
{importance_df.head(5).to_string(index=False)}

SUBMISSION: submission_physics_full.csv
"""
    
    with open('physics_full_summary.txt', 'w') as f:
        f.write(summary)
    print("\n✓ Saved: physics_full_summary.txt")
    
    print("\n" + "="*70)
    print("✅ FINAL MODEL COMPLETE")
    print("="*70)
    print(f"\n🎯 Cross-Validation F1-Score: {cv_df['f1'].mean():.4f}")
    print(f"📊 Submit: submission_physics_full.csv")
    print("="*70)


if __name__ == "__main__":
    main()

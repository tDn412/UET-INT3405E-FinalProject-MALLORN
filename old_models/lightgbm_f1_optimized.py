"""
MALLORN TDE Classification - LightGBM with F1 Optimization
Physics-informed model optimized for rare TDE detection

Key strategies:
1. LightGBM with is_unbalance=True for class imbalance
2. Stratified 5-Fold CV to maintain TDE ratio
3. F1-threshold optimization (not default 0.5)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_optimal_threshold(y_true, y_proba):
    """
    Find probability threshold that maximizes F1-Score.
    Default 0.5 is suboptimal for imbalanced data.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1, thresholds, f1_scores


def train_and_evaluate():
    """Train LightGBM with physics features and F1 optimization."""
    
    print("="*70)
    print("LIGHTGBM WITH F1 OPTIMIZATION")
    print("="*70)
    
    # Load physics features
    print("\n[1] Loading features...")
    train_df = pd.read_csv('physics_features_train.csv')
    test_df = pd.read_csv('physics_features_test.csv')
    
    print(f"  Train: {len(train_df)} objects")
    print(f"  Test: {len(test_df)} objects")
    
    # Prepare features
    feature_cols = [col for col in train_df.columns 
                   if col not in ['object_id', 'target']]
    
    X = train_df[feature_cols]
    y = train_df['target']
    X_test = test_df[feature_cols]
    test_ids = test_df['object_id']
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  TDE ratio: {y.mean()*100:.3f}%")
    
    # Handle missing values (fill with median for now)
    print("\n[2] Handling missing values...")
    for col in feature_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # LightGBM configuration optimized for imbalanced data
    print("\n[3] Configuring LightGBM...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,  # Auto-adjust class weights
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    print(f"  Parameters: {lgb_params}")
    
    # Stratified 5-Fold Cross-Validation
    print("\n[4] 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = []
    fold_thresholds = []
    fold_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/5")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train LightGBM
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict probabilities
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Find optimal threshold
        threshold, f1_opt, _, _ = find_optimal_threshold(y_val_fold, y_proba)
        
        # Apply optimal threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Metrics
        precision = precision_score(y_val_fold, y_pred)
        recall = recall_score(y_val_fold, y_pred)
        auc = roc_auc_score(y_val_fold, y_proba)
        
        print(f"    Optimal threshold: {threshold:.3f}")
        print(f"    F1-Score: {f1_opt:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    ROC-AUC: {auc:.4f}")
        
        cv_results.append({
            'fold': fold,
            'threshold': threshold,
            'f1': f1_opt,
            'precision': precision,
            'recall': recall,
            'auc': auc
        })
        
        fold_thresholds.append(threshold)
        fold_f1_scores.append(f1_opt)
    
    # CV Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.to_string(index=False))
    print(f"\nMean F1-Score: {cv_df['f1'].mean():.4f} (+/- {cv_df['f1'].std():.4f})")
    print(f"Mean Threshold: {cv_df['threshold'].mean():.3f}")
    
    # Train final model on full dataset
    print("\n[5] Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)
    
    # Use mean threshold from CV
    optimal_threshold = cv_df['threshold'].mean()
    print(f"  Using threshold: {optimal_threshold:.3f}")
    
    # Feature importance
    print("\n[6] Feature importance...")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('LightGBM Feature Importance (Top 15)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_lgbm_feature_importance.png', dpi=150)
    print("  ✓ Saved: physics_lgbm_feature_importance.png")
    plt.close()
    
    # Generate test predictions
    print("\n[7] Generating test predictions...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= optimal_threshold).astype(int)
    
    # Create submissions with different strategies
    # 1. Probability-based (recommended for Kaggle)
    submission_proba = pd.DataFrame({
        'object_id': test_ids,
        'prediction': test_proba
    })
    submission_proba.to_csv('submission_physics_proba.csv', index=False)
    print("  ✓ Saved: submission_physics_proba.csv (probabilities)")
    
    # 2. Binary with optimal threshold
    submission_binary = pd.DataFrame({
        'object_id': test_ids,
        'prediction': test_pred
    })
    submission_binary.to_csv('submission_physics_binary.csv', index=False)
    print(f"  ✓ Saved: submission_physics_binary.csv (threshold={optimal_threshold:.3f})")
    
    print(f"\n  Predicted TDEs: {test_pred.sum()} / {len(test_pred)} ({test_pred.mean()*100:.2f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print("\nModel Performance:")
    print(f"  Mean CV F1-Score: {cv_df['f1'].mean():.4f}")
    print(f"  Mean CV ROC-AUC: {cv_df['auc'].mean():.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print("\nTop 3 Features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")
    print("\nSubmissions:")
    print("  - submission_physics_proba.csv (RECOMMENDED)")
    print("  - submission_physics_binary.csv")
    print("="*70)


if __name__ == "__main__":
    train_and_evaluate()

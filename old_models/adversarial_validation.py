"""
Adversarial Validation
Goal: distinct Train vs Test distribution?
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def main():
    print("ADVERSARIAL VALIDATION: Train vs Test")
    print("="*60)
    
    # 1. Load Data
    try:
        # Load Refined Features (67 cols)
        train_df = pd.read_csv('features_combined_train.csv')
        test_df = pd.read_csv('features_combined_test.csv')
        
        feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
        
        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        # Fill NaNs
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_cols)}")
        
    except FileNotFoundError:
        print("Features not found.")
        return

    # 2. Label Data
    X_train['adversarial_target'] = 0
    X_test['adversarial_target'] = 1
    
    # 3. Combine
    X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_all = X_all['adversarial_target'].values
    X_all = X_all.drop(columns=['adversarial_target'])
    
    # 4. Train Model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = feature_cols
    feature_importance['importance'] = 0
    
    print("\nTraining LGBM Classifier...")
    for fold, (tr, val) in enumerate(skf.split(X_all, y_all)):
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            metric='auc',
            verbose=-1
        )
        
        model.fit(
            X_all.iloc[tr], y_all[tr],
            eval_set=[(X_all.iloc[val], y_all[val])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict_proba(X_all.iloc[val])[:, 1]
        auc = roc_auc_score(y_all[val], preds)
        auc_scores.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
        
        feature_importance['importance'] += model.feature_importances_ / 5
        
    mean_auc = np.mean(auc_scores)
    print(f"\nMean AUC: {mean_auc:.4f}")
    
    if mean_auc > 0.7:
        print("⚠️  SIGNIFICANT DISTRIBUTION SHIFT DETECTED!")
        print("\nTop 20 Drifting Features:")
        print(feature_importance.sort_values('importance', ascending=False).head(20))
    else:
        print("✅ Distribution mostly similar (No major shift).")

if __name__ == "__main__":
    main()

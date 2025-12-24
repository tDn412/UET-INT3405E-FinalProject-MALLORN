
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def main():
    print("Training Refined Multi-Class Model...")
    
    # 1. Load Data
    train_df = pd.read_csv('features_combined_train.csv')
    test_df = pd.read_csv('features_combined_test.csv')
    log_df = pd.read_csv('train_log.csv')
    
    # 2. Add Target Labels
    train_df = train_df.merge(log_df[['object_id', 'SpecType']], on='object_id', how='left')
    
    # Simplify Classes
    def simplify_class(cls):
        if cls in ['AGN', 'SN Ia', 'SN II', 'TDE']:
            return cls
        return 'Other'
    train_df['simple_class'] = train_df['SpecType'].apply(simplify_class)
    
    le = LabelEncoder()
    y_multi = le.fit_transform(train_df['simple_class'])
    tde_idx = list(le.classes_).index('TDE')
    
    # 3. Prepare
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
    X = train_df[feature_cols]
    
    # Fill NaNs
    median_values = X.median(numeric_only=True)
    X = X.fillna(median_values)
    
    X_test = test_df[feature_cols]
    X_test = X_test.fillna(median_values)
    
    print(f"Training on {X.shape[1]} features.")
    
    # 4. Train
    params = {
        'n_estimators': 2000,
        'learning_rate': 0.02, # Slower learning for robustness
        'num_leaves': 31,
        'max_depth': -1,
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
        'colsample_bytree': 0.7, 
        'subsample': 0.7,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_probs_accum = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multi)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_multi[train_idx], y_multi[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
        )
        
        # Test Preds
        probs = model.predict_proba(X_test)
        test_probs_accum += probs[:, tde_idx]
        print(f"Fold {fold+1} Done.")
        
    avg_test_proba = test_probs_accum / 5
    
    # 5. Top-N Selection (Target 350)
    # Estimate: Train TDE % = 4.8%. Test Size 7137. Expected ~342.
    TARGET = 350
    
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prob': avg_test_proba})
    sub_df = sub_df.sort_values('prob', ascending=False)
    
    sub_df['prediction'] = 0
    sub_df.iloc[:TARGET, sub_df.columns.get_loc('prediction')] = 1
    
    # Save
    sub_df[['object_id', 'prediction']].to_csv('submission_multiclass_refined.csv', index=False)
    sub_df[['object_id', 'prob']].to_csv('submission_multiclass_refined_probs.csv', index=False)
    
    print(f"Saved submission_multiclass_refined.csv with top {TARGET} predictions.")

if __name__ == "__main__":
    main()

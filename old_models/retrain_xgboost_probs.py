
import pandas as pd
import xgboost as xgb
import numpy as np

def main():
    print("Retraining XGBoost to get probabilities...")
    
    # Load features (Statistical features)
    train_df = pd.read_csv('lightcurve_features_train.csv')
    test_df = pd.read_csv('lightcurve_features_test.csv')
    
    # Load labels
    train_log = pd.read_csv('train_log.csv')
    
    # Merge
    train_df = train_df.merge(train_log[['object_id', 'target']], on='object_id', how='left')
    
    # Fill NaNs
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    test_df = test_df.fillna(train_df.median(numeric_only=True))
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'label', 'split', 'English Translation', 'SpecType', 'Z', 'Z_err', 'EBV']]
    X = train_df[feature_cols]
    y = train_df['target']
    X_test = test_df[feature_cols]
    
    # Params from FINAL_SUMMARY.md
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 4,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': 19.6, # Handle imbalance
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    
    # Predict Probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': probs})
    sub_df.to_csv('submission_xgboost_probs.csv', index=False)
    print("Saved submission_xgboost_probs.csv")

if __name__ == "__main__":
    main()

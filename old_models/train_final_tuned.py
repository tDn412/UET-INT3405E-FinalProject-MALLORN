
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def main():
    print("Training FINAL TUNED Model...")
    
    # Load data
    train_df = pd.read_csv('physics_features_train_full.csv')
    test_df = pd.read_csv('physics_features_test_full.csv')
    
    # Handle NaNs
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    test_df = test_df.fillna(train_df.median(numeric_only=True))
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target']]
    X = train_df[feature_cols]
    y = train_df['target']
    X_test = test_df[feature_cols]
    
    # Best Params from Optuna (CV: 0.4597)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'is_unbalance': True,
        'n_estimators': 500,
        'learning_rate': 0.07260882855256458,
        'num_leaves': 79,
        'max_depth': 7,
        'min_child_samples': 42,
        'subsample': 0.5120716984593039,
        'colsample_bytree': 0.6276184340390644,
        'reg_alpha': 0.0007228882736203657,
        'reg_lambda': 1.6373396825642394e-05
    }
    
    # 1. Determine optimal threshold using CV again
    print("Determining optimal threshold with CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = []
    cv_f1s = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold)
        probas = model.predict_proba(X_val_fold)[:, 1]
        
        thresh, f1 = find_optimal_threshold(y_val_fold, probas)
        thresholds.append(thresh)
        cv_f1s.append(f1)
        
    avg_thresh = np.mean(thresholds)
    avg_f1 = np.mean(cv_f1s)
    print(f"Mean CV F1: {avg_f1:.4f}")
    print(f"Optimal Threshold: {avg_thresh:.3f}")
    
    # 2. Train Full Model
    print("Retraining on full dataset...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    
    # 3. Predict Test
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= avg_thresh).astype(int)
    
    # 4. Save
    # Save probabilities for ensemble
    prob_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': test_proba})
    prob_df.to_csv('submission_physics_tuned_probs.csv', index=False)
    print("Saved submission_physics_tuned_probs.csv")

    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': test_preds})
    sub_df.to_csv('submission_physics_tuned_final.csv', index=False)
    print("Saved submission_physics_tuned_final.csv")
    print(f"Predicted TDEs: {test_preds.sum()}")

if __name__ == "__main__":
    main()

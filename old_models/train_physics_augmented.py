
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("Loading data for Augmented Training...")
    # Load features (Physics features) - assuming these are the best ones
    # We need to recreate the full feature set as in train_physics_full.py
    # But since we didn't save the exact comprehensive csv, we might need to rerun extraction or use what we have.
    # Ah, 'physics_features_train_full.csv' might be the one.
    
    try:
        train_df = pd.read_csv('physics_features_train_full.csv')
    except FileNotFoundError:
        print("physics_features_train_full.csv not found. Please run feature extraction.")
        return

    test_df = pd.read_csv('physics_features_test_full.csv')
    
    # Fill NaNs with median
    median_values = train_df.median(numeric_only=True)
    train_df = train_df.fillna(median_values)
    test_df = test_df.fillna(median_values)
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'split', 'fold']]
    X = train_df[feature_cols]
    y = train_df['target']
    X_test = test_df[feature_cols]
    
    print(f"Training Data Shape: {X.shape}")
    print(f"Class Balance: {y.value_counts()}")
    
    # Best Params from Tuning
    # {'n_estimators': 743, 'learning_rate': 0.021021798363784013, 'num_leaves': 63, 'max_depth': 12, 'min_child_samples': 46, 'subsample': 0.6322302302003736, 'colsample_bytree': 0.7249925769273673, 'reg_alpha': 0.0033628467475172083, 'reg_lambda': 0.009033446077309536}
    best_params = {
        'n_estimators': 743,
        'learning_rate': 0.021,
        'num_leaves': 63,
        'max_depth': 12,
        'min_child_samples': 46,
        'subsample': 0.63,
        'colsample_bytree': 0.72,
        'reg_alpha': 0.003,
        'reg_lambda': 0.009,
        'objective': 'binary',
        'metric': 'auc', # Optimize AUC during training, pick threshold for F1
        'is_unbalance': False, # We are oversampling, so set to False? Or keep True? Usually False if balanced.
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))
    
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply SMOTE ONLY on Training Data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        
        # Find fold optimal threshold
        thresh, f1 = find_optimal_threshold(y_val, val_proba)
        fold_f1s.append(f1)
        print(f"Fold {fold+1} F1: {f1:.4f} (Thresh: {thresh:.3f})")
        
        test_preds_accum += model.predict_proba(X_test)[:, 1]
        
    avg_f1 = np.mean(fold_f1s)
    print(f"Average CV F1 (Augmented): {avg_f1:.4f}")
    
    # Global Optimal Threshold
    best_thresh, best_f1 = find_optimal_threshold(y, oof_preds)
    print(f"Global Optimal Threshold: {best_thresh:.3f} with F1: {best_f1:.4f}")
    
    # Final Predictions
    avg_test_proba = test_preds_accum / 5
    final_preds = (avg_test_proba >= best_thresh).astype(int)
    
    # Save
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_preds})
    sub_df.to_csv('submission_physics_augmented.csv', index=False)
    print("Saved submission_physics_augmented.csv")
    
    # Save Probs too
    prob_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': avg_test_proba})
    prob_df.to_csv('submission_physics_augmented_probs.csv', index=False)

if __name__ == "__main__":
    main()

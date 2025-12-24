
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import logging

# Suppress LightGBM logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def objective(trial):
    # Load data (assuming features are already generated)
    train_df = pd.read_csv('physics_features_train_full.csv')
    
    # Handle NaNs (simple fill)
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target']]
    X = train_df[feature_cols]
    y = train_df['target']
    
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'is_unbalance': True,
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        
        _, f1 = find_optimal_threshold(y_val, preds)
        f1_scores.append(f1)

    return np.mean(f1_scores)

if __name__ == "__main__":
    print("Starting Optuna Study...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) 
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    with open("best_params_physics.txt", "w") as f:
        f.write(str(trial.params))

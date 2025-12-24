
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
    print("Training Multi-Class Model to improve TDE detection...")
    
    # 1. Load Feature Data (Physics Features are best so far)
    try:
        train_df = pd.read_csv('physics_features_train_full.csv')
        test_df = pd.read_csv('physics_features_test_full.csv')
    except:
        # Fallback to creating meaningful features if file missing
        # But we assume they exist as we used them before.
        # If not, we might need to rerun extraction (not shown here for brevity)
        print("Feature files missing!")
        return

    # 2. Get SpecType Labels
    log_df = pd.read_csv('train_log.csv')
    
    # Merge SpecType into train_df
    train_df = train_df.merge(log_df[['object_id', 'SpecType']], on='object_id', how='left')
    
    # 3. Label Encoding
    # We want to group rare classes to avoid issues?
    # SpecType distribution: AGN(1786), SN Ia(790), SN II(163), TDE(148), others < 50
    # Let's group others into "Other"
    
    def simplify_class(cls):
        if cls in ['AGN', 'SN Ia', 'SN II', 'TDE']:
            return cls
        return 'Other'
        
    train_df['simple_class'] = train_df['SpecType'].apply(simplify_class)
    
    print("Simplified Class Distribution:")
    print(train_df['simple_class'].value_counts())
    
    le = LabelEncoder()
    y_multi = le.fit_transform(train_df['simple_class'])
    
    # Identify which index is TDE
    tde_idx = list(le.classes_).index('TDE')
    print(f"TDE Class Index: {tde_idx}")
    
    # 4. Prepare Data
    # Fill NaNs
    median_values = train_df.median(numeric_only=True)
    train_df = train_df.fillna(median_values)
    test_df = test_df.fillna(median_values)
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
    X = train_df[feature_cols]
    # Prediction Target is y_multi
    
    # 5. Train Multi-Class LGBM
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'objective': 'multiclass',
        'num_class': 5, # AGN, Other, SN Ia, SN II, TDE
        'metric': 'multi_logloss',
        'class_weight': 'balanced', # IMPORTANT: Give higher weight to rare classes like TDE
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(X))
    test_probs_accum = np.zeros(len(test_df))
    
    fold_f1s = []
    
    # Binary Target for validation
    y_binary = (train_df['simple_class'] == 'TDE').astype(int)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multi)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_multi[train_idx], y_multi[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Predict Probabilities
        val_pred_proba = model.predict_proba(X_val) # Shape (N, 5)
        tde_prob = val_pred_proba[:, tde_idx]
        
        oof_probs[val_idx] = tde_prob
        
        # Calculate F1 for TDE Class Binary
        y_val_binary = (y_val == tde_idx).astype(int)
        
        thresh, f1 = find_optimal_threshold(y_val_binary, tde_prob)
        fold_f1s.append(f1)
        print(f"Fold {fold+1} TDE F1: {f1:.4f} (Thresh: {thresh:.2f})")
        
        # Test Preds
        test_pred_proba = model.predict_proba(test_df[feature_cols])
        test_probs_accum += test_pred_proba[:, tde_idx]
        
    avg_f1 = np.mean(fold_f1s)
    print(f"Average Multi-Class Derived F1: {avg_f1:.4f}")
    
    # Global Optimal
    best_thresh, best_f1 = find_optimal_threshold(y_binary, oof_probs)
    print(f"Global Optimal Threshold: {best_thresh:.3f} with F1: {best_f1:.4f}")
    
    # Generate Output
    avg_test_proba = test_probs_accum / 5
    final_preds = (avg_test_proba >= best_thresh).astype(int)
    
    sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_preds})
    sub.to_csv('submission_multiclass_tde.csv', index=False)
    
    # Save Probs
    prob_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': avg_test_proba})
    prob_df.to_csv('submission_multiclass_probs.csv', index=False)
    
    print("Saved submission_multiclass_tde.csv")
    print(sub['prediction'].value_counts())

if __name__ == "__main__":
    main()

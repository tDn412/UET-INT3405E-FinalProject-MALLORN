
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

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
    print("Starting Pseudo-Labeling Process...")
    
    # 1. Load Original Data
    train_df = pd.read_csv('physics_features_train_full.csv')
    test_df = pd.read_csv('physics_features_test_full.csv')
    
    # Fill median
    median_values = train_df.median(numeric_only=True)
    train_df = train_df.fillna(median_values)
    test_df = test_df.fillna(median_values)
    
    # 2. Load Ensemble Probabilities (Best Source)
    try:
        ens_probs = pd.read_csv('submission_ensemble_probs.csv')
    except FileNotFoundError:
        print("Ensemble probabilities not found!")
        return

    # 3. Select Pseudo-Labels
    # High confidence thresholds
    POS_THRESH = 0.90  # Slightly lower to get more TDEs samples
    NEG_THRESH = 0.05
    
    pseudo_pos = ens_probs[ens_probs['prediction_prob'] > POS_THRESH]
    pseudo_neg = ens_probs[ens_probs['prediction_prob'] < NEG_THRESH]
    
    print(f"Found {len(pseudo_pos)} confident TDEs and {len(pseudo_neg)} confident Non-TDEs in Test set.")
    
    # Create Pseudo-labeled dataframe
    pseudo_test_df = test_df[test_df['object_id'].isin(pd.concat([pseudo_pos, pseudo_neg])['object_id'])].copy()
    
    # Assign labels
    # We need to map object_id to the predicted probability to assign label 1 or 0
    id_to_prob = dict(zip(ens_probs['object_id'], ens_probs['prediction_prob']))
    
    def get_pseudo_label(oid):
        prob = id_to_prob.get(oid)
        if prob > POS_THRESH: return 1
        if prob < NEG_THRESH: return 0
        return -1 # Should not happen
        
    pseudo_test_df['target'] = pseudo_test_df['object_id'].apply(get_pseudo_label)
    
    # Drop samples where target is -1 (just safety)
    pseudo_test_df = pseudo_test_df[pseudo_test_df['target'] != -1]
    
    # 4. Combine Train + Pseudo
    # Make sure columns match
    cols = [c for c in train_df.columns if c in pseudo_test_df.columns]
    
    augmented_train_df = pd.concat([train_df[cols], pseudo_test_df[cols]], axis=0)
    
    print(f"Original Train: {len(train_df)}. Augmented Train: {len(augmented_train_df)}")
    
    # 5. Retrain Physics Model
    feature_cols = [c for c in augmented_train_df.columns if c not in ['object_id', 'target', 'split', 'fold']]
    X_aug = augmented_train_df[feature_cols]
    y_aug = augmented_train_df['target']
    X_test = test_df[feature_cols]
    
    # Use Best Params
    # Tuning results showed:
    best_params = {
        'n_estimators': 1000, # Increased for larger data
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 12,
        'min_child_samples': 46,
        'subsample': 0.63,
        'colsample_bytree': 0.72,
        'reg_alpha': 0.003,
        'reg_lambda': 0.009,
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True, # Keep True as we added mostly negatives likely
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Train on Full Augmented Data
    print("Training on Full Augmented Data...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_aug, y_aug)
    
    # Predict on Test
    test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Use Physics Threshold (safe bet) or maybe slightly higher since model is more confident?
    # Let's stick to 0.154 from pure physics tuning to be safe, or recalculate if we could.
    # Since we changed training distribution, threshold might shift.
    # Actually, we can check distribution of probs compared to original.
    
    thresh = 0.154
    test_preds = (test_proba >= thresh).astype(int)
    
    # Save
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': test_preds})
    sub_df.to_csv('submission_pseudo_final.csv', index=False)
    print("Saved submission_pseudo_final.csv")
    
    # Save Probs
    prob_df = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': test_proba})
    prob_df.to_csv('submission_pseudo_final_probs.csv', index=False)

if __name__ == "__main__":
    main()

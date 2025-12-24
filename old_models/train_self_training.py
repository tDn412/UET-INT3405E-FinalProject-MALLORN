
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def main():
    print("Starting Iterative Self-Training (Teacher-Student) Loop...")
    
    # 1. Load Data
    train_df = pd.read_csv('features_combined_train.csv')
    test_df = pd.read_csv('features_combined_test.csv')
    log_df = pd.read_csv('train_log.csv')
    
    # Load Teacher Predictions (The 0.5057 model probabilities)
    teacher_probs = pd.read_csv('submission_multiclass_refined_probs.csv')
    
    # 2. Add Labels to Train
    train_df = train_df.merge(log_df[['object_id', 'SpecType']], on='object_id', how='left')
    
    def simplify_class(cls):
        if cls in ['AGN', 'SN Ia', 'SN II', 'TDE']:
            return cls
        return 'Other'
    train_df['simple_class'] = train_df['SpecType'].apply(simplify_class)
    
    # 3. Create Pseudo-Labeled Test Set
    # We trust the top 300 as TDEs (High Precision)
    # We trust the bottom 6000 as Non-TDEs (High Precision)
    
    # Merge teacher probs
    test_df_labeled = test_df.merge(teacher_probs[['object_id', 'prob']], on='object_id')
    
    # Sort
    test_df_labeled = test_df_labeled.sort_values('prob', ascending=False)
    
    # Label Top 300 as 'TDE'
    top_tde = test_df_labeled.head(300).copy()
    top_tde['simple_class'] = 'TDE'
    
    # Label Bottom 6000 as 'Other' (or we could try to guess class, but 'Other' is safer distinction from TDE)
    # Actually, let's just label them as 'Other' to reinforce "Not TDE" nature.
    # OR, we can ignore them.
    # IMPORTANT: We need to be careful not to label "Other" as "Other" if they look like SN Ia.
    # Safest bet: Only add Confident TDEs to validation signal.
    # But adding Negatives helps reduce False Positives.
    # Let's take bottom 5000 as 'Other'.
    
    bottom_other = test_df_labeled.tail(5000).copy()
    bottom_other['simple_class'] = 'Other'
    
    # Combine
    pseudo_test = pd.concat([top_tde, bottom_other])
    print(f"Pseudo-Labeled Data: {len(pseudo_test)} samples ({len(top_tde)} TDEs, {len(bottom_other)} Others)")
    
    # 4. Augmented Training Set
    # We need to ensure columns match
    cols = [c for c in train_df.columns if c in pseudo_test.columns]
    
    # We need to drop 'prob' from pseudo_test
    pseudo_test = pseudo_test[cols]
    
    train_aug = pd.concat([train_df[cols], pseudo_test])
    
    # 5. Prepare for Training
    le = LabelEncoder()
    y_aug = le.fit_transform(train_aug['simple_class'])
    tde_idx = list(le.classes_).index('TDE')
    
    feature_cols = [c for c in train_aug.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
    X_aug = train_aug[feature_cols]
    
    # Fill NaNs
    median_values = X_aug.median(numeric_only=True)
    X_aug = X_aug.fillna(median_values)
    
    X_test = test_df[feature_cols].fillna(median_values)
    
    print(f"Training Student Model on {len(X_aug)} samples...")
    
    # 6. Train Student Model
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.015,
        'num_leaves': 31,
        'max_depth': -1,
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
        'colsample_bytree': 0.65, 
        'subsample': 0.65,
        'random_state': 42,
        'verbose': -1
    }
    
    # We don't have a clean validation set anymore since we mixed test data.
    # We will use the original TRAIN data as validation features? No, leakage.
    # We will just train on full AUG data and use early stopping with a small holdout from ORIGINAL TRAIN.
    
    # Let's split ORIGINAL TRAIN into train/val
    # And add PSUEDO data to TRAIN only.
    
    final_test_probs = np.zeros(len(X_test))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We map indices to original train_df
    orig_n = len(train_df)
    orig_indices = np.arange(orig_n)
    
    y_orig = le.transform(train_df['simple_class'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_orig)):
        # Train = Fold Train + All Pseudo
        # Val = Fold Val (Pure Original Data)
        
        orig_train = train_df.iloc[train_idx]
        orig_val = train_df.iloc[val_idx]
        
        # Concat predictions
        X_fold_train = pd.concat([orig_train[feature_cols], pseudo_test[feature_cols]]).fillna(median_values)
        y_fold_train = np.concatenate([y_orig[train_idx], le.transform(pseudo_test['simple_class'])])
        
        X_fold_val = orig_val[feature_cols].fillna(median_values)
        y_fold_val = y_orig[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
        )
        
        probs = model.predict_proba(X_test)
        final_test_probs += probs[:, tde_idx]
        print(f"Fold {fold+1} Student Trained.")
        
    avg_test_proba = final_test_probs / 5
    
    # 7. Generate Submissions (Top N)
    # We stick to the winning strategy: Top 350, but also try neighbors.
    
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prob': avg_test_proba})
    sub_df = sub_df.sort_values('prob', ascending=False)
    
    for count in [325, 350, 375]:
        sub_df['prediction'] = 0
        sub_df.iloc[:count, sub_df.columns.get_loc('prediction')] = 1
        
        fname = f'submission_self_train_top{count}.csv'
        sub_df[['object_id', 'prediction']].to_csv(fname, index=False)
        print(f"Saved {fname}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import warnings
import os

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# TARGET CLEANING
# ---------------------------------------------------------
def clean_target(spectype):
    s = str(spectype).strip()
    if 'TDE' in s:
        return 3
    elif 'AGN' in s:
        return 0
    elif 'SN Ia' in s or 'SN Iax' in s: # Matches SN Ia, SN Ia-pec, SN Iax...
        return 1
    else:
        # SN II, SN Ib, SN Ic, SLSN, etc.
        return 2 

target_names = {0: 'AGN', 1: 'SN_Ia', 2: 'SN_Other', 3: 'TDE'}

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
phys_train = pd.read_csv('physics_features_train.csv')
train_log = pd.read_csv('train_log.csv')

# Map Targets
train_log['target_id'] = train_log['SpecType'].apply(clean_target)

# Merge Physics with Targets
# We use inner join to ensure we only have labelled data
train_df = pd.merge(phys_train, train_log[['object_id', 'target_id']], on='object_id', how='inner')
y = train_df['target_id'].values
X = train_df.drop(['object_id', 'target_id'], axis=1)

print(f"Training on {len(X)} samples.")
print(f"Class distribution: {np.bincount(y)}")

# ---------------------------------------------------------
# GLOBAL PARAMS
# ---------------------------------------------------------
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03, # Lower LR for better generalization
    'num_leaves': 31,
    'max_depth': 7, # Limit depth to prevent overfitting
    'min_child_samples': 30, # Increase regularization
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'class_weight': 'balanced' 
    # Note: LGBM python-package doesn't support 'focal_loss' directly in objective string for multiclass 
    # without custom objective, but we can try class_weight='balanced' first with robust params.
    # Alternatively, use 'multiclassova' which allows is_unbalance.
    # Let's stick to 'multiclass' but with stricter regularization first.
}

# ---------------------------------------------------------
# FOCAL LOSS (Custom Objective) - Optional for advanced users
# For now, let's stick to robust regularized LightGBM.
# ---------------------------------------------------------


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------
# 1. PHYSICS ONLY MODEL
# ---------------------------------------------------------
print("\n" + "="*30)
print("PHYSICS ONLY MODEL")
print("="*30)

oof_preds = np.zeros((len(X), 4))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
    oof_preds[val_idx] = clf.predict_proba(X_val)

oof_class = np.argmax(oof_preds, axis=1)
macro_f1 = f1_score(y, oof_class, average='macro')
weighted_f1 = f1_score(y, oof_class, average='weighted')

print(f"PHYSICS ONLY MACRO F1: {macro_f1:.5f}")
print(f"PHYSICS ONLY WEIGHTED F1: {weighted_f1:.5f}")
print(classification_report(y, oof_class, target_names=[target_names[i] for i in range(4)]))


# ---------------------------------------------------------
# 2. COMBINED MODEL
# ---------------------------------------------------------
print("\n" + "="*30)
print("COMBINED MODEL")
print("="*30)

lc_file_train = None
lc_file_test = None

# Logic to pick best available features
if os.path.exists('features_combined_train.csv') and os.path.exists('features_combined_test.csv'):
    print("Found 'features_combined_*.csv'. Using these extended features.")
    lc_file_train = 'features_combined_train.csv'
    lc_file_test = 'features_combined_test.csv'
elif os.path.exists('lightcurve_features_train.csv') and os.path.exists('lightcurve_features_test.csv'):
    print("Combined features not found. Falling back to 'lightcurve_features_*.csv'.")
    lc_file_train = 'lightcurve_features_train.csv'
    lc_file_test = 'lightcurve_features_test.csv'
else:
    print("WARNING: No matching train/test feature sets found. Skipping combined model.")

if lc_file_train:
    # Load LC Features
    lc_train_df = pd.read_csv(lc_file_train)
    
    # Load GP Features
    if os.path.exists('gp_residual_features_train.csv'):
        print("Loading GP Residual Features...")
        gp_train = pd.read_csv('gp_residual_features_train.csv')
        lc_train_df = pd.merge(lc_train_df, gp_train, on='object_id', how='left')

    # Drop potential leakage columns BUT KEEP Z (Redshift)
    # Z is critical. We will handle overfitting via regularization and stronger features (GP + Physics)
    leakage_cols = ['target', 'SpecType', 'label', 'class', 'split', 'target_id']
    lc_train_df = lc_train_df.drop(columns=[c for c in leakage_cols if c in lc_train_df.columns], errors='ignore')

    # Merge: Logic is careful here.
    # Start with our labelled set 'train_df' (id, target)
    # Merge Physics features (already in train_df? No, X is phys only)
    # Actually train_df has phys features + target.
    
    # Merge LC features onto train_df
    combined_train = pd.merge(train_df, lc_train_df, on='object_id', how='left')
    
    # Prepare X_comb, y_comb
    y_comb = combined_train['target_id'].values
    X_comb = combined_train.drop(['object_id', 'target_id'], axis=1)
    X_comb = X_comb.fillna(-999)
    
    # Check shape
    print(f"Combined Train Shape: {X_comb.shape}")
    
    # CV
    oof_preds_comb = np.zeros((len(X_comb), 4))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_comb, y_comb)):
        X_train, y_train = X_comb.iloc[train_idx], y_comb[train_idx]
        X_val, y_val = X_comb.iloc[val_idx], y_comb[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
        oof_preds_comb[val_idx] = clf.predict_proba(X_val)

    oof_class_comb = np.argmax(oof_preds_comb, axis=1)
    macro_f1_comb = f1_score(y_comb, oof_class_comb, average='macro')
    print(f"COMBINED MACRO F1: {macro_f1_comb:.5f}")
    print(classification_report(y_comb, oof_class_comb, target_names=[target_names[i] for i in range(4)]))
    
    # Feature Importance
    print("\n" + "="*30)
    print("TOP 20 FEATURES")
    print("="*30)
    # Fit a quick model on full data to get importance
    clf_imp = lgb.LGBMClassifier(**params)
    clf_imp.fit(X_comb, y_comb)
    imp_df = pd.DataFrame({'feature': X_comb.columns, 'importance': clf_imp.feature_importances_})
    print(imp_df.sort_values(by='importance', ascending=False).head(20))
    imp_df.sort_values(by='importance', ascending=False).to_csv('feature_importance.csv', index=False)
    
    # Check Physics Features Rank
    print("\n--- PHYSICS FEATURE RANK ---")
    phys_cols_of_interest = phys_train.columns.tolist() # From original physics file
    phys_ranks = imp_df[imp_df['feature'].isin(phys_cols_of_interest)].sort_values(by='importance', ascending=False)
    print(phys_ranks)


    # ---------------------------------------------------------
    # 3. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\n" + "-"*30)
    print("Generating Submission...")
    
    if lc_file_test:
        lc_test_df = pd.read_csv(lc_file_test)
        phys_test = pd.read_csv('physics_features_test.csv')
        
        # Load GP Test
        if os.path.exists('gp_residual_features_test.csv'):
             gp_test = pd.read_csv('gp_residual_features_test.csv')
             lc_test_df = pd.merge(lc_test_df, gp_test, on='object_id', how='left')

        # Merge Physics + LC for Test
        combined_test = pd.merge(lc_test_df, phys_test, on='object_id', how='left')
        test_ids = combined_test['object_id']
        X_test = combined_test.drop(['object_id'], axis=1)
        
        # Align Columns
        # X_comb columns must match X_test columns
        # Add missing cols to test, drop extra
        train_cols = X_comb.columns.tolist()
        
        # Build X_test ensuring strictly same columns
        X_test_aligned = pd.DataFrame(index=X_test.index)
        for col in train_cols:
            if col in X_test.columns:
                X_test_aligned[col] = X_test[col]
            else:
                X_test_aligned[col] = -999 # Missing in test
        
        X_test_aligned = X_test_aligned.fillna(-999)
        
        # Train Full Model
        print("Training full combined model...")
        clf_full = lgb.LGBMClassifier(**params)
        clf_full.fit(X_comb, y_comb)
        
        # Predict
        probs = clf_full.predict_proba(X_test_aligned)
        
        # Save
        sub = pd.DataFrame()
        sub['object_id'] = test_ids
        
        # Convert to Binary (Class 3 is TDE -> 1, Others -> 0)
        preds_multiclass = np.argmax(probs, axis=1)
        
        # Check Distribution
        n_tde = np.sum(preds_multiclass == 3)
        stats_msg = f"Test Set Predictions:\nPredicted TDEs: {n_tde} out of {len(preds_multiclass)} ({n_tde/len(preds_multiclass)*100:.2f}%)\nTraining TDE %: {100*148/3043:.2f}%"
        print(stats_msg)
        with open('prediction_stats.txt', 'w') as f:
            f.write(stats_msg)
        
        sub['prediction'] = (preds_multiclass == 3).astype(int)
        
        sub['prediction'] = (preds_multiclass == 3).astype(int)
        
        sub.to_csv('submission_physics_v4_GP_withZ.csv', index=False)
        print("Submission saved to submission_physics_v4_GP_withZ.csv (Binary Format)")
        

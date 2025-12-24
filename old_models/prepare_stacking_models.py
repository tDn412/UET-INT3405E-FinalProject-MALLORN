"""
Prepare diverse base models for Stacking Ensemble.
Models:
1. KNN (K-Nearest Neighbors) - fast, captures local structure
2. Logistic Regression - linear baseline, good for high-dim if regularized
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb

def main():
    print("="*80)
    print("TRAINING DIVERSE BASE MODELS FOR STACKING (REFINED)")
    print("="*80)
    
    # 1. Load Data
    try:
        # Load Refined Features (67 cols)
        refined_train = pd.read_csv('features_combined_train.csv')
        refined_test = pd.read_csv('features_combined_test.csv')
        
        # Load Poly Features
        poly_train = pd.read_csv('features_poly_train.csv')
        poly_test = pd.read_csv('features_poly_test.csv')
        
        # Merge Poly into Refined
        refined_train = refined_train.merge(poly_train, on='object_id', how='left')
        refined_test = refined_test.merge(poly_test, on='object_id', how='left')
        
        # Load Wavelet Features
        freq_train = pd.read_csv('frequency_features_train.csv')
        freq_test = pd.read_csv('frequency_features_test.csv')
        
        log_train = pd.read_csv('train_log.csv')
    except FileNotFoundError:
        print("Features not found. Check file paths.")
        return

    # Merge Targets for Refined
    if 'SpecType' not in refined_train.columns:
        refined_train = refined_train.merge(log_train[['object_id', 'SpecType']], on='object_id', how='left')
        
    # Simplify Classes (Crucial for Refined Performance)
    def simplify_class(cls):
        if cls in ['AGN', 'SN Ia', 'SN II', 'TDE']:
            return cls
        return 'Other'
    
    refined_train['simple_class'] = refined_train['SpecType'].apply(simplify_class)
    
    # Encode Target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_multi = le.fit_transform(refined_train['simple_class'])
    
    # Check if TDE is in classes
    if 'TDE' in le.classes_:
        tde_idx = list(le.classes_).index('TDE')
        print(f"TDE Class Index: {tde_idx}")
    else:
        print("Error: TDE class not found!")
        return
        
    y_binary = log_train[log_train['object_id'].isin(refined_train['object_id'])]['target'].values
    
    # Prepare Feature Sets
    # Refined Features
    ref_cols = [c for c in refined_train.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
    X_refined = refined_train[ref_cols].fillna(refined_train[ref_cols].median())
    X_refined_test = refined_test[ref_cols].fillna(refined_train[ref_cols].median())
    
    # Wavelet Features (Merge with log_train to align rows)
    # Be careful with row alignment!
    # refined_train might be sorted differently.
    # Let's align everything by object_id
    
    # Create master alignment DF
    master_train = refined_train[['object_id']].copy()
    master_train = master_train.merge(freq_train, on='object_id', how='left')
    
    wavelet_cols = [c for c in freq_train.columns if c != 'object_id']
    X_wavelet = master_train[wavelet_cols].fillna(master_train[wavelet_cols].median())
    
    master_test = refined_test[['object_id']].copy()
    master_test = master_test.merge(freq_test, on='object_id', how='left')
    X_wavelet_test = master_test[wavelet_cols].fillna(master_train[wavelet_cols].median())

    print(f"Refined Features: {X_refined.shape[1]}")
    print(f"Wavelet Features: {X_wavelet.shape[1]}")
    print(f"Target Distribution: {np.bincount(y_multi)}")

    # 2. Define Models
    
    # KNN on Wavelet (Target: Binary TDE? or Multi?)
    # KNN works better on Binary usually, or Multi. Let's stick to Binary for KNN OOF prob.
    # But wait, Stacking is easier if all output P(TDE).
    knn_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))
    ])
    
    # LogReg on Refined (Strong Linear Baseline)
    logreg_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
        ('logreg', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, multi_class='ovr', random_state=42))
    ])
    
    # LGBM on Refined (The STAR model)
    lgbm_params = {
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': -1,
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    
    # 3. Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store P(TDE)
    oof_knn = np.zeros(len(refined_train))
    oof_logreg = np.zeros(len(refined_train))
    oof_lgbm = np.zeros(len(refined_train))
    
    print("\nTraining KNN (Wavelet, Binary Target)...")
    # For KNN, let's use binary target y_binary for simplicity/focus
    y_bin = (y_multi == tde_idx).astype(int)
    
    for fold, (tr, val) in enumerate(skf.split(X_wavelet, y_multi)): # Split statified by Multi-class
        knn_pipe.fit(X_wavelet.iloc[tr], y_bin[tr])
        probs = knn_pipe.predict_proba(X_wavelet.iloc[val])[:, 1]
        oof_knn[val] = probs
    print(f"KNN OOF F1: {find_optimal_f1(y_bin, oof_knn):.4f}")
    
    print("\nTraining LogReg (Refined, Multi-Class)...")
    for fold, (tr, val) in enumerate(skf.split(X_refined, y_multi)):
        logreg_pipe.fit(X_refined.iloc[tr], y_multi[tr])
        probs = logreg_pipe.predict_proba(X_refined.iloc[val])
        # P(TDE)
        oof_logreg[val] = probs[:, tde_idx]
    print(f"LogReg OOF F1: {find_optimal_f1(y_bin, oof_logreg):.4f}")
    
    print("\nTraining LGBM (Refined, Multi-Class)...")
    for fold, (tr, val) in enumerate(skf.split(X_refined, y_multi)):
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_refined.iloc[tr], y_multi[tr],
            eval_set=[(X_refined.iloc[val], y_multi[val])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        probs = model.predict_proba(X_refined.iloc[val])
        oof_lgbm[val] = probs[:, tde_idx]
        
    print(f"LGBM OOF F1: {find_optimal_f1(y_bin, oof_lgbm):.4f}")
    
    # 4. Save OOFs
    res_df = pd.DataFrame({
        'object_id': refined_train['object_id'],
        'target': y_bin,
        'knn_prob': oof_knn,
        'logreg_prob': oof_logreg,
        'lgbm_prob': oof_lgbm
    })
    res_df.to_csv('oof_stacking_refined.csv', index=False)
    print("\nSaved OOFs to oof_stacking_refined.csv")
    
    # 5. Generate Test Predictions
    print("\nTraining Full Models & Predicting Test...")
    
    # KNN
    knn_pipe.fit(X_wavelet, y_bin)
    knn_test = knn_pipe.predict_proba(X_wavelet_test)[:, 1]
    
    # LogReg
    logreg_pipe.fit(X_refined, y_multi)
    logreg_test = logreg_pipe.predict_proba(X_refined_test)[:, tde_idx]
    
    # LGBM
    lgbm_full = lgb.LGBMClassifier(**lgbm_params)
    lgbm_full.fit(X_refined, y_multi)
    lgbm_test = lgbm_full.predict_proba(X_refined_test)[:, tde_idx]
    
    sub_df = pd.DataFrame({
        'object_id': refined_test['object_id'],
        'knn_prob': knn_test,
        'logreg_prob': logreg_test,
        'lgbm_prob': lgbm_test
    })
    sub_df.to_csv('test_preds_stacking_refined.csv', index=False)
    print("Saved Test Preds to test_preds_stacking_refined.csv")

def find_optimal_f1(y_true, y_proba):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
    return best_f1

if __name__ == "__main__":
    main()



def find_optimal_f1(y_true, y_proba):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
    return best_f1

if __name__ == "__main__":
    main()

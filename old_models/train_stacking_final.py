"""
Train Stacking Meta-Learner
Input: OOF predictions from Base Models (KNN, LogReg, LGBM)
Output: Final Submission
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1

def main():
    print("="*80)
    print("TRAINING STACKING META-LEARNER")
    print("="*80)
    
    # 1. Load Base Model Predictions (Refined)
    oof_df = pd.read_csv('oof_stacking_refined.csv')
    test_df = pd.read_csv('test_preds_stacking_refined.csv')
    
    print(f"OOF Shape: {oof_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    
    # Feature columns for Meta-Learner
    meta_cols = ['knn_prob', 'logreg_prob', 'lgbm_prob']
    
    X_meta = oof_df[meta_cols]
    y = oof_df['target'].values
    X_test_meta = test_df[meta_cols]
    
    # Check individual performance
    print("\nBase Model OOF Performance:")
    for col in meta_cols:
        t, f1 = find_optimal_threshold(y, X_meta[col].values)
        print(f"  {col}: F1={f1:.4f} (Thresh={t:.2f})")
        
    # 2. Train Meta-Learner (Logistic Regression)
    # Why LogReg? Because we just want to find optimal weights for probabilities
    # Constrain weights to be positive? Scikit-learn doesn't support non-negative least squares directly in LogReg easily,
    # but regular LogReg usually works fine.
    
    print("\nTraining Meta-Learner (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=42)
    
    # Stratified CV for Meta-Model (to estimate Stacking Performance)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof_probs = np.zeros(len(X_meta))
    
    for fold, (tr, val) in enumerate(skf.split(X_meta, y)):
        meta_model.fit(X_meta.iloc[tr], y[tr])
        probs = meta_model.predict_proba(X_meta.iloc[val])[:, 1]
        meta_oof_probs[val] = probs
        
    t_stack, f1_stack = find_optimal_threshold(y, meta_oof_probs)
    print(f"\n✅ STACKING OOF F1: {f1_stack:.4f} (Thresh={t_stack:.3f})")
    
    # 3. Train on Full Data and Predict Test
    meta_model.fit(X_meta, y)
    print(f"\nMeta-Learner Coefficients: {dict(zip(meta_cols, meta_model.coef_[0]))}")
    
    final_test_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    
    # 4. Generate Submission (Top-N Strategy)
    # We found N=400-405 to be optimal for LGBM
    # Let's try N=400, 405, 410, 420
    
    print("\nGenerating Submission Files...")
    sub_df = pd.DataFrame({'object_id': test_df['object_id'], 'prob': final_test_probs})
    
    for n in [400, 405, 410, 420]:
        sub_df['prediction'] = 0
        top_idx = np.argsort(sub_df['prob'].values)[::-1][:n]
        sub_df.loc[top_idx, 'prediction'] = 1
        
        filename = f'submission_stacking_tier2_top{n}.csv'
        submission = sub_df[['object_id', 'prediction']]
        submission.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
        
    print("\nComparison:")
    best_base_f1 = max([find_optimal_threshold(y, X_meta[c].values)[1] for c in meta_cols])
    gain = f1_stack - best_base_f1
    print(f"  Best Base F1: {best_base_f1:.4f}")
    print(f"  Stacking Gain: {gain:+.4f}")

if __name__ == "__main__":
    main()

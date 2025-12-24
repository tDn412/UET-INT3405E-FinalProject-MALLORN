
import pandas as pd
import numpy as np

def main():
    print("Ensembling Physics and XGBoost models...")
    
    # Load probabilities
    phys_df = pd.read_csv('submission_physics_tuned_probs.csv')
    xgb_df = pd.read_csv('submission_xgboost_probs.csv')
    
    # Merge
    ens_df = phys_df.merge(xgb_df, on='object_id', suffixes=('_phys', '_xgb'))
    
    # Weighted Average (Favoring Physics model which had F1 ~0.46 vs ~0.41)
    # Weights: 0.7 Physics, 0.3 XGBoost
    w_phys = 0.7
    w_xgb = 0.3
    
    ens_df['prediction_prob'] = (w_phys * ens_df['prediction_phys']) + (w_xgb * ens_df['prediction_xgb'])
    
    # Threshold from Physics model optimization
    threshold = 0.154
    
    ens_df['prediction'] = (ens_df['prediction_prob'] >= threshold).astype(int)
    
    # Save Ensemble Probabilities (for potential future stacking)
    ens_df[['object_id', 'prediction_prob']].to_csv('submission_ensemble_probs.csv', index=False)
    
    # Save Final Submission
    sub_df = ens_df[['object_id', 'prediction']]
    sub_df.to_csv('submission_ensemble_final.csv', index=False)
    
    print(f"Saved submission_ensemble_final.csv with threshold {threshold}")
    print(sub_df['prediction'].value_counts())

if __name__ == "__main__":
    main()

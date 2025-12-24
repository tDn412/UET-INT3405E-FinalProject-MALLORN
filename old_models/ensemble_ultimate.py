
import pandas as pd

def main():
    print("Generating Ultimate Ensemble (Top-N Strategy)...")
    
    # Load all sources
    multi_df = pd.read_csv('submission_multiclass_probs.csv')
    phys_df = pd.read_csv('submission_physics_tuned_probs.csv')
    xgb_df = pd.read_csv('submission_xgboost_probs.csv')
    
    # Merge
    df = multi_df.merge(phys_df, on='object_id', suffixes=('_multi', '_phys'))
    df = df.merge(xgb_df, on='object_id')
    df = df.rename(columns={'prediction': 'prediction_xgb'})
    
    # Weighted Average
    # Multi-class is semantically strong (35%)
    # Binary Physics is clean (35%)
    # XGBoost is diverse/raw (30%)
    df['final_prob'] = (0.35 * df['prediction_multi']) + (0.35 * df['prediction_phys']) + (0.30 * df['prediction_xgb'])
    
    # Rank-Based Thresholding
    # We want ~750 positives based on historical best (740 gave 0.41)
    # The ensemble count of 535 gave 0.39 (too low)
    # So let's target 700-800. Let's pick 750.
    TARGET_COUNT = 750
    
    # Save Probs for further use
    df[['object_id', 'final_prob']].to_csv('submission_ultimate_probs.csv', index=False)
    
    # Stratified Precision Strategy
    # User's best score (0.5057) came from Top 350 of Single Model.
    # Self-training suggested lower might be better (325 > 350).
    # We test the Ensemble ranking with these counts.
    
    for count in [300, 325, 340, 350, 360]:
        df = df.sort_values('final_prob', ascending=False)
        df['prediction'] = 0
        df.iloc[:count, df.columns.get_loc('prediction')] = 1
        
        fname = f'submission_ultimate_top{count}.csv'
        df[['object_id', 'prediction']].to_csv(fname, index=False)
        print(f"Saved {fname}")
        
    print("Saved submission_ultimate_probs.csv and variants.")
    print("Top 10 Probs:")
    print(df[['object_id', 'final_prob']].head(10))

if __name__ == "__main__":
    main()

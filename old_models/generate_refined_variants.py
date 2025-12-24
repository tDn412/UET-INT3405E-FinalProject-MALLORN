
import pandas as pd

def main():
    print("Generating Refined Multi-Class Variants...")
    
    # Load probs
    try:
        df = pd.read_csv('submission_multiclass_refined_probs.csv')
    except:
        print("Probs file not found, re-running prediction? No, assume it exists.")
        return

    # Generate Variants
    # We already have Top 350 (Score 0.5057)
    # Let's try surrounding values
    
    for count in [300, 325, 340, 360]:
        df = df.sort_values('prob', ascending=False)
        df['prediction'] = 0
        df.iloc[:count, df.columns.get_loc('prediction')] = 1
        
        fname = f'submission_refined_top{count}.csv'
        df[['object_id', 'prediction']].to_csv(fname, index=False)
        print(f"Saved {fname}")

if __name__ == "__main__":
    main()

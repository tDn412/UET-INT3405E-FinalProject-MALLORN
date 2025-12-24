"""
SYSTEMATIC GRID SEARCH: Find optimal Top-N cutoff
Strategy: Test N from 300 to 400 in steps of 10 for BOTH models
"""

import pandas as pd
import os

def generate_topn_grid(probs_file, base_name, n_values):
    """Generate submission files for different N values"""
    
    print(f"\nProcessing {base_name}...")
    
    # Load probabilities
    df = pd.read_csv(probs_file)
    
    # Determine prob column name (handle different naming)
    if 'prob' in df.columns:
        prob_col = 'prob'
    elif 'final_prob' in df.columns:
        prob_col = 'final_prob'
    elif 'prediction' in df.columns:
        prob_col = 'prediction'
    else:
        raise ValueError(f"Cannot find probability column in {probs_file}")
    
    print(f"  Using column: {prob_col}")
    
    # Sort by probability
    df = df.sort_values(prob_col, ascending=False)
    
    generated = []
    
    for n in n_values:
        # Create binary predictions
        df['prediction'] = 0
        df.iloc[:n, df.columns.get_loc('prediction')] = 1
        
        # Save
        filename = f'submission_{base_name}_top{n}.csv'
        df[['object_id', 'prediction']].to_csv(filename, index=False)
        
        generated.append((n, filename, df['prediction'].sum()))
    
    return generated

def main():
    print("=" * 80)
    print("SYSTEMATIC TOP-N GRID SEARCH")
    print("=" * 80)
    
    # Define N values to test
    n_values = list(range(300, 401, 10))  # 300, 310, 320, ..., 400
    
    print(f"\nTesting N values: {n_values}")
    print(f"Total submissions to generate: {len(n_values) * 2} (2 models)")
    
    all_generated = []
    
    # 1. Refined Multi-Class model
    refined_files = generate_topn_grid(
        'submission_multiclass_refined_probs.csv',
        'refined',
        n_values
    )
    all_generated.extend([('Refined', n, f, c) for n, f, c in refined_files])
    
    # 2. Ultimate Ensemble model  
    ultimate_files = generate_topn_grid(
        'submission_ultimate_probs.csv',
        'ultimate',
        n_values
    )
    all_generated.extend([('Ultimate', n, f, c) for n, f, c in ultimate_files])
    
    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    
    print("\nGenerated files:")
    print(f"{'Model':<10} | {'N':<5} | {'File':<40} | {'Count'}")
    print("-" * 80)
    
    for model, n, filename, count in all_generated:
        print(f"{model:<10} | {n:<5} | {filename:<40} | {count}")
    
    print("\n" + "=" * 80)
    print("SUBMISSION STRATEGY")
    print("=" * 80)
    print("""
Submit in this order to maximize learning:

PHASE 1 (Test extremes):
1. refined_top300.csv   - Low N boundary
2. refined_top400.csv   - High N boundary
3. ultimate_top300.csv  - Ultimate low
4. ultimate_top400.csv  - Ultimate high

PHASE 2 (Based on Phase 1 results):
- If 300 > 400: Test 310, 320, 330 (refined)
- If 400 > 300: Test 370, 380, 390 (refined)
- If 350 still best: Test 340, 360 (refined)

PHASE 3 (Fine-tune winner):
- Once you find best N range, test ±5 around it
- E.g., if N=370 is best, test 365, 375
""")
    
    print("\nCurrent best for reference:")
    print("  submission_multiclass_refined.csv (N=350): 0.5057")
    print("\nTarget: Find N that breaks 0.6")

if __name__ == "__main__":
    main()

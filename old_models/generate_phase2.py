"""
Phase 2: Generate higher N values (400-500)
Based on Phase 1 results showing upward trend
"""

import pandas as pd

def generate_topn_variants(probs_file, base_name, n_values):
    """Generate submission files for different N values"""
    
    df = pd.read_csv(probs_file)
    
    # Get prob column
    if 'prob' in df.columns:
        prob_col = 'prob'
    elif 'final_prob' in df.columns:
        prob_col = 'final_prob'
    else:
        prob_col = 'prediction'
    
    # Sort by probability
    df = df.sort_values(prob_col, ascending=False)
    
    generated = []
    
    for n in n_values:
        df['prediction'] = 0
        df.iloc[:n, df.columns.get_loc('prediction')] = 1
        
        filename = f'submission_{base_name}_top{n}.csv'
        df[['object_id', 'prediction']].to_csv(filename, index=False)
        
        count = df['prediction'].sum()
        generated.append((n, filename, count))
        print(f"  Generated: {filename} (N={n})")
    
    return generated

# Generate higher N values for Refined model only
# (Ultimate model failed Phase 1)

print("=" * 80)
print("PHASE 2: GENERATING HIGHER N VALUES")
print("=" * 80)

n_values = list(range(410, 501, 10))  # 410, 420, ..., 500
print(f"\nGenerating N values: {n_values}")

refined_files = generate_topn_variants(
    'submission_multiclass_refined_probs.csv',
    'refined',
    n_values
)

print("\n" + "=" * 80)
print(f"Generated {len(refined_files)} new files")
print("=" * 80)

print("\nPHASE 2 PRIORITY SUBMISSIONS:")
print("1. submission_refined_top420.csv  <- Most likely peak")
print("2. submission_refined_top440.csv  <- Test if trend continues")  
print("3. submission_refined_top410.csv  <- Between best two")
print("\nIf 420 > 400: Submit 430, 440, 450")
print("If 420 < 400: Peak is ~400, fine-tune with 405, 415")

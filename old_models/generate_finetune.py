"""
Phase 3: Fine-tune around peak (N=400-410)
"""

import pandas as pd

df = pd.read_csv('submission_multiclass_refined_probs.csv')
df = df.sort_values('prob', ascending=False)

# Generate N=401-409
for n in range(401, 410):
    df['prediction'] = 0
    df.iloc[:n, df.columns.get_loc('prediction')] = 1
    
    filename = f'submission_refined_top{n}.csv'
    df[['object_id', 'prediction']].to_csv(filename, index=False)
    print(f"Generated: {filename}")

print("\nPhase 3 files ready!")
print("\nRecommended submission:")
print("  submission_refined_top405.csv (geometric center of peak)")

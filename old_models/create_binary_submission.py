import pandas as pd

# Load probabilities
df = pd.read_csv('submission_physics_full.csv')
print(f"Loaded {len(df)} predictions")

# Apply threshold
threshold = 0.070
df['prediction'] = (df['prediction'] >= threshold).astype(int)

# Save
output_file = 'submission_physics_binary_final.csv'
df.to_csv(output_file, index=False)
print(f"Saved binary submission to {output_file}")
print(f"TDE Count: {df['prediction'].sum()}")

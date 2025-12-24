"""
Convert probability submission to binary using optimal threshold
"""

import pandas as pd

# Load probability submission
df = pd.read_csv('submission_physics_full.csv')

# Use optimal threshold from CV: 0.070
OPTIMAL_THRESHOLD = 0.070

# Convert to binary
df['prediction'] = (df['prediction'] >= OPTIMAL_THRESHOLD).astype(int)

# Save binary submission
df.to_csv('submission_physics_binary_final.csv', index=False)

print(f"✓ Created binary submission with threshold {OPTIMAL_THRESHOLD}")
print(f"  Predicted TDEs: {df['prediction'].sum()} / {len(df)} ({df['prediction'].mean()*100:.2f}%)")
print(f"  File: submission_physics_binary_final.csv")

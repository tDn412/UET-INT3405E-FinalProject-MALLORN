"""
Fix submission format - Convert probabilities to binary predictions
"""

import pandas as pd
import numpy as np

print("="*70)
print("Fixing Submission Format")
print("="*70)

# Load the original submission với probabilities
submission_proba = pd.read_csv('submission.csv')

print(f"\nOriginal submission:")
print(f"Shape: {submission_proba.shape}")
print(f"Prediction range: [{submission_proba['prediction'].min():.4f}, {submission_proba['prediction'].max():.4f}]")
print(submission_proba.head(10))

# Convert to binary: threshold = 0.5
threshold = 0.5
submission_binary = submission_proba.copy()
submission_binary['prediction'] = (submission_proba['prediction'] >= threshold).astype(int)

print(f"\n{'='*70}")
print(f"After conversion (threshold={threshold}):")
print(f"{'='*70}")
print(f"Prediction distribution:")
print(submission_binary['prediction'].value_counts())
print(f"\nFirst 20 rows:")
print(submission_binary.head(20))

# Save
submission_binary.to_csv('submission.csv', index=False)
print(f"\n✅ Fixed submission saved to: submission.csv")
print(f"   Format: Binary (0 or 1)")
print(f"   Ready for Kaggle submission!")

# Verification
print(f"\nVerification:")
print(f"Unique values: {submission_binary['prediction'].unique()}")
print(f"Min: {submission_binary['prediction'].min()}")
print(f"Max: {submission_binary['prediction'].max()}")
print("="*70)

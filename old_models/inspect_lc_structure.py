import pickle
import pandas as pd

# Load raw light curves
with open('raw_lightcurves_train.pkl', 'rb') as f:
    train_lcs = pickle.load(f)

print(f"Total train objects: {len(train_lcs)}")
print(f"\nSample object ID: {list(train_lcs.keys())[0]}")

# Inspect structure
sample_lc = train_lcs[list(train_lcs.keys())[0]]
print(f"\nLight curve type: {type(sample_lc)}")

if isinstance(sample_lc, pd.DataFrame):
    print(f"Columns: {sample_lc.columns.tolist()}")
    print(f"Shape: {sample_lc.shape}")
    print("\nFirst 5 rows:")
    print(sample_lc.head())
else:
    print(f"Data type: {sample_lc.dtype if hasattr(sample_lc, 'dtype') else type(sample_lc)}")
    if hasattr(sample_lc, 'shape'):
        print(f"Shape: {sample_lc.shape}")

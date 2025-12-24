"""
STEP 1 (CORRECTED): Load Physics-Corrected Light Curves
Purpose: Aggregate corrected/clean CSVs into pickle for Step 3.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LOADING PHYSICS-CORRECTED LIGHT CURVES")
print("="*70)

# Directories
base_dir = Path('.')
# Dynamic discovery of corrected splits
split_folders = sorted([d for d in base_dir.glob('split_*_corrected') if d.is_dir()])
print(f"\n[1] Found {len(split_folders)} corrected split directories.")

def load_light_curve_file(filepath):
    """Load a single light curve file"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load training light curves
print("\n[2] Loading CORRECTED TRAINING light curves...")

train_lightcurves = {}  # {object_id: DataFrame}
total_observations = 0

for split_dir in tqdm(split_folders, desc="Processing splits"):
    train_file = split_dir / 'train_full_lightcurves.csv'
    
    if train_file.exists():
        df = load_light_curve_file(train_file)
        if df is not None:
             # Group by object_id
            for obj_id, group in df.groupby('object_id'):
                if obj_id not in train_lightcurves:
                    train_lightcurves[obj_id] = group
                else:
                    train_lightcurves[obj_id] = pd.concat([train_lightcurves[obj_id], group])
                
                total_observations += len(group)

print(f"\nTraining set (Corrected):")
print(f"  Unique objects: {len(train_lightcurves)}")
print(f"  Total observations: {total_observations:,}")

# Load test light curves
print("\n[3] Loading CORRECTED TEST light curves...")

test_lightcurves = {}
total_test_obs = 0

for split_dir in tqdm(split_folders, desc="Processing splits"):
    test_file = split_dir / 'test_full_lightcurves.csv'
    
    if test_file.exists():
        df = load_light_curve_file(test_file)
        if df is not None:
            for obj_id, group in df.groupby('object_id'):
                if obj_id not in test_lightcurves:
                    test_lightcurves[obj_id] = group
                else:
                    test_lightcurves[obj_id] = pd.concat([test_lightcurves[obj_id], group])
                
                total_test_obs += len(group)

print(f"\nTest set (Corrected):")
print(f"  Unique objects: {len(test_lightcurves)}")
print(f"  Total observations: {total_test_obs:,}")

# Save to pickle
print("\n[4] Saving to pickle files...")

with open('raw_lightcurves_train_corrected.pkl', 'wb') as f:
    pickle.dump(train_lightcurves, f)
    
with open('raw_lightcurves_test_corrected.pkl', 'wb') as f:
    pickle.dump(test_lightcurves, f)

print(f"✓ Saved raw_lightcurves_train_corrected.pkl")
print(f"✓ Saved raw_lightcurves_test_corrected.pkl")

print("\n" + "="*70)
print("CORRECTED DATA READY FOR FEATURE EXTRACTION!")
print("="*70)

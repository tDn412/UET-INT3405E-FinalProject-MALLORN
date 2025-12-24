"""
STEP 1: Load Raw Light Curves from All Splits
Purpose: Get actual time series data instead of just statistical features

CONCEPT: Time Series Data
- Light curve = brightness over time
- Each object has multiple observations: (time, flux, error, filter)
- 6 filters (bands): u, g, r, i, z, y (different wavelengths)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LOADING RAW LIGHT CURVES")
print("="*70)

# Directories
base_dir = Path('.')
split_dirs = [f'split_{i:02d}' for i in range(1, 21)]

print(f"\n[1] Scanning {len(split_dirs)} split directories...")

# Check which splits exist
existing_splits = []
for split_dir in split_dirs:
    if (base_dir / split_dir).exists():
        existing_splits.append(split_dir)

print(f"Found {len(existing_splits)} splits")

def load_light_curve_file(filepath):
    """Load a single light curve file"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load training light curves
print("\n[2] Loading TRAINING light curves...")

train_lightcurves = {}  # {object_id: DataFrame}
total_observations = 0

for split_dir in tqdm(existing_splits, desc="Processing splits"):
    train_file = base_dir / split_dir / 'train_full_lightcurves.csv'
    
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

print(f"\nTraining set:")
print(f"  Unique objects: {len(train_lightcurves)}")
print(f"  Total observations: {total_observations:,}")
print(f"  Avg observations per object: {total_observations/len(train_lightcurves):.1f}")

# Analyze light curve structure
print("\n[3] Analyzing light curve structure...")

sample_id = list(train_lightcurves.keys())[0]
sample_lc = train_lightcurves[sample_id]

print(f"\nSample object: {sample_id}")
print(f"Columns: {list(sample_lc.columns)}")
print(f"Number of observations: {len(sample_lc)}")
print(f"\nFirst few rows:")
print(sample_lc.head())

# Statistics
print("\n[4] Light curve statistics:")

n_obs_per_object = [len(lc) for lc in train_lightcurves.values()]

print(f"  Min observations: {min(n_obs_per_object)}")
print(f"  Max observations: {max(n_obs_per_object)}")
print(f"  Median: {np.median(n_obs_per_object):.0f}")
print(f"  Mean: {np.mean(n_obs_per_object):.0f}")

# Filter distribution
all_filters = []
for lc in list(train_lightcurves.values())[:100]:  # Sample 100
    all_filters.extend(lc['Filter'].unique())

filter_counts = pd.Series(all_filters).value_counts()
print(f"\nFilter distribution (sample):")
for filt, count in filter_counts.items():
    print(f"  {filt}: {count}")

# Load test light curves
print("\n[5] Loading TEST light curves...")

test_lightcurves = {}
total_test_obs = 0

for split_dir in tqdm(existing_splits, desc="Processing splits"):
    test_file = base_dir / split_dir / 'test_full_lightcurves.csv'
    
    if test_file.exists():
        df = load_light_curve_file(test_file)
        if df is not None:
            for obj_id, group in df.groupby('object_id'):
                if obj_id not in test_lightcurves:
                    test_lightcurves[obj_id] = group
                else:
                    test_lightcurves[obj_id] = pd.concat([test_lightcurves[obj_id], group])
                
                total_test_obs += len(group)

print(f"\nTest set:")
print(f"  Unique objects: {len(test_lightcurves)}")
print(f"  Total observations: {total_test_obs:,}")
print(f"  Avg observations per object: {total_test_obs/len(test_lightcurves):.1f}")

# Save to pickle for fast loading later
print("\n[6] Saving to pickle files...")

with open('raw_lightcurves_train.pkl', 'wb') as f:
    pickle.dump(train_lightcurves, f)
    
with open('raw_lightcurves_test.pkl', 'wb') as f:
    pickle.dump(test_lightcurves, f)

print(f"✓ Saved raw_lightcurves_train.pkl ({len(train_lightcurves)} objects)")
print(f"✓ Saved raw_lightcurves_test.pkl ({len(test_lightcurves)} objects)")

# Load labels
print("\n[7] Merging with labels...")

train_log = pd.read_csv('train_log.csv')

labeled_count = 0
tde_count = 0

for obj_id in train_lightcurves.keys():
    if obj_id in train_log['object_id'].values:
        labeled_count += 1
        label = train_log[train_log['object_id'] == obj_id]['target'].values[0]
        if label == 1:
            tde_count += 1

print(f"\nLabeled objects: {labeled_count}/{len(train_lightcurves)}")
print(f"TDE: {tde_count}, Non-TDE: {labeled_count - tde_count}")
print(f"Class ratio: 1:{(labeled_count - tde_count)/tde_count:.1f}")

print("\n" + "="*70)
print("RAW LIGHT CURVES LOADED SUCCESSFULLY!")
print("="*70)

print("""
NEXT STEPS:
1. Preprocess sequences (normalize, pad)
2. Build LSTM model
3. Build CNN model
4. Train models
5. Ensemble predictions

KEY CONCEPT LEARNED:
- Raw light curves = time series data
- Variable length sequences (10-500+ obs)
- Multi-band data (6 filters)
- This preserves temporal information lost in statistical features!
""")

"""
STEP 2: Preprocess Light Curves for Deep Learning
Purpose: Convert variable-length sequences to fixed format for LSTM/CNN

CONCEPT: Sequence Preprocessing
- Padding: Make all sequences same length
- Normalization: Scale values to similar ranges
- Multi-band: Handle 6 different filters as channels
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PREPROCESSING LIGHT CURVES")
print("="*70)

# Load raw light curves
print("\n[1] Loading raw data...")
with open('raw_lightcurves_train.pkl', 'rb') as f:
    train_lcs = pickle.load(f)
    
with open('raw_lightcurves_test.pkl', 'rb') as f:
    test_lcs = pickle.load(f)

train_log = pd.read_csv('train_log.csv')

print(f"Train objects: {len(train_lcs)}")
print(f"Test objects: {len(test_lcs)}")

# Parameters
MAX_TIMESTEPS = 200  # Pad/truncate to this length
N_BANDS = 6  # u, g, r, i, z, y
BAND_MAPPING = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}

print(f"\nParameters:")
print(f"  Max timesteps: {MAX_TIMESTEPS}")
print(f"  Number of bands: {N_BANDS}")

def preprocess_light_curve(lc_df, max_len=MAX_TIMESTEPS):
    """
    Convert light curve DataFrame to 3D tensor
    
    Input: DataFrame with columns [Time (MJD), Flux, Flux_err, Filter]
    Output: array of shape (max_len, N_BANDS, 3)
           3 = [flux, flux_err, time_delta]
           
    CONCEPT: Multi-variate Time Series
    - Each timestep has multiple features
    - Each band tracked separately
    - Missing data filled with zeros + mask
    """
    
    # Sort by time
    lc_df = lc_df.sort_values('Time (MJD)').reset_index(drop=True)
    
    # Initialize 3D array: (timesteps, bands, features)
    sequence = np.zeros((max_len, N_BANDS, 3))
    mask = np.zeros((max_len, N_BANDS))  # 1 = valid, 0 = padded
    
    # Get unique times
    unique_times = sorted(lc_df['Time (MJD)'].unique())
    
    # Truncate if too long
    if len(unique_times) > max_len:
        unique_times = unique_times[:max_len]
    
    # Fill sequence
    for t_idx, time in enumerate(unique_times):
        # Get all observations at this time
        time_obs = lc_df[lc_df['Time (MJD)'] == time]
        
        for _, row in time_obs.iterrows():
            band = row['Filter']
            if band in BAND_MAPPING:
                band_idx = BAND_MAPPING[band]
                
                # Store [flux, flux_err, time_delta]
                if t_idx == 0:
                    time_delta = 0
                else:
                    time_delta = time - unique_times[0]
                
                sequence[t_idx, band_idx, 0] = row['Flux']
                sequence[t_idx, band_idx, 1] = row['Flux_err']
                sequence[t_idx, band_idx, 2] = time_delta
                mask[t_idx, band_idx] = 1
    
    return sequence, mask

print("\n[2] Preprocessing training data...")

X_train = []
y_train = []
masks_train = []

successful = 0
failed = 0

for obj_id, lc_df in train_lcs.items():
    try:
        # Get label
        label = train_log[train_log['object_id'] == obj_id]['target'].values[0]
        
        # Preprocess
        sequence, mask = preprocess_light_curve(lc_df)
        
        X_train.append(sequence)
        y_train.append(label)
        masks_train.append(mask)
        successful += 1
        
    except Exception as e:
        failed += 1
        if failed < 5:  # Show first few errors
            print(f"  Error processing {obj_id}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)
masks_train = np.array(masks_train)

print(f"\nTraining data:")
print(f"  Shape: {X_train.shape}")  # (n_samples, timesteps, bands, features)
print(f"  Labels: {y_train.shape}")
print(f"  Masks: {masks_train.shape}")
print(f"  Successful: {successful}, Failed: {failed}")
print(f"  TDE: {(y_train == 1).sum()}, Non-TDE: {(y_train == 0).sum()}")

print("\n[3] Preprocessing test data...")

X_test = []
test_ids = []

for obj_id, lc_df in test_lcs.items():
    try:
        sequence, mask = preprocess_light_curve(lc_df)
        X_test.append(sequence)
        test_ids.append(obj_id)
    except:
        pass

X_test = np.array(X_test)

print(f"\nTest data:")
print(f"  Shape: {X_test.shape}")
print(f"  Objects: {len(test_ids)}")

# Normalize fluxes
print("\n[4] Normalizing fluxes...")

"""
CONCEPT: Normalization
Why? Neural networks work best with values near 0
- Flux values vary widely (-100 to +1000)
- Normalize to mean=0, std=1
- Do separately for each band
"""

# Flatten to normalize per band
for band_idx in range(N_BANDS):
    # Get all flux values for this band (where mask=1)
    train_flux = X_train[:, :, band_idx, 0]
    train_mask = masks_train[:, :, band_idx]
    
    valid_flux = train_flux[train_mask == 1]
    
    if len(valid_flux) > 0:
        mean = valid_flux.mean()
        std = valid_flux.std()
        
        if std > 0:
            # Normalize train
            X_train[:, :, band_idx, 0] = (X_train[:, :, band_idx, 0] - mean) / std
            # Normalize test with same params
            X_test[:, :, band_idx, 0] = (X_test[:, :, band_idx, 0] - mean) / std
            
            print(f"  Band {list(BAND_MAPPING.keys())[band_idx]}: mean={mean:.2f}, std={std:.2f}")

# Normalize time deltas
print("\nNormalizing time deltas...")
all_times = X_train[:, :, :, 2].flatten()
valid_times = all_times[all_times > 0]

if len(valid_times) > 0:
    time_max = valid_times.max()
    X_train[:, :, :, 2] = X_train[:, :, :, 2] / time_max
    X_test[:, :, :, 2] = X_test[:, :, :, 2] / time_max
    print(f"  Max time: {time_max:.2f} days")

# Save preprocessed data
print("\n[5] Saving preprocessed data...")

np.save('X_train_sequences.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test_sequences.npy', X_test)
np.save('masks_train.npy', masks_train)

with open('test_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)

print(f"✓ Saved X_train_sequences.npy: {X_train.shape}")
print(f"✓ Saved y_train.npy: {y_train.shape}")
print(f"✓ Saved X_test_sequences.npy: {X_test.shape}")
print(f"✓ Saved test_ids.pkl")

# Summary statistics
print("\n[6] Summary statistics:")

print(f"\nData shape breakdown:")
print(f"  (samples, timesteps, bands, features)")
print(f"  {X_train.shape} <- Training")
print(f"  {X_test.shape} <- Test")

print(f"\nFeatures per timestep:")
print(f"  0: Flux (normalized)")
print(f"  1: Flux error")
print(f"  2: Time delta (normalized)")

print(f"\nBands (channels):")
for band, idx in BAND_MAPPING.items():
    print(f"  {idx}: {band}-band")

# Sample
sample_idx = 0
sample_seq = X_train[sample_idx]
sample_mask = masks_train[sample_idx]

n_valid = sample_mask.sum()
print(f"\nSample object (index {sample_idx}):")
print(f"  Valid observations: {int(n_valid)}/{MAX_TIMESTEPS * N_BANDS}")
print(f"  Label: {y_train[sample_idx]} ({'TDE' if y_train[sample_idx]==1 else 'Non-TDE'})")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)

print("""
KEY CONCEPTS LEARNED:

1. SEQUENCE PADDING
   - Variable lengths → Fixed length (200 timesteps)
   - Shorter sequences padded with zeros
   - Mask tracks valid vs padded data

2. MULTI-BAND TIME SERIES
   - 6 channels (u,g,r,i,z,y filters)
   - Each band tracked independently
   - Preserves color information

3. NORMALIZATION
   - Flux: mean=0, std=1 (per band)
   - Time: scaled to [0,1]
   - Helps neural network training

4. 3D TENSOR FORMAT
   - Shape: (samples, timesteps, bands, features)
   - Ready for LSTM/CNN input!

NEXT: Build LSTM model!
""")

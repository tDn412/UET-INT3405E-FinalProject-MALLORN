"""
MALLORN TDE Classification - Light Curve Feature Extraction
Extract comprehensive features from time series data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Light Curve Feature Extraction")
print("="*70)

def extract_light_curve_features(lc_df):
    """
    Extract comprehensive features from a single object's light curve
    
    Args:
        lc_df: DataFrame with columns ['Time (MJD)', 'Flux', 'Flux_err', 'Filter']
    
    Returns:
        dict of features
    """
    features = {}
    
    if len(lc_df) == 0:
        return features
    
    # Sort by time
    lc_df = lc_df.sort_values('Time (MJD)')
    
    # ===========================
    # BASIC STATISTICS
    # ===========================
    features['n_observations'] = len(lc_df)
    features['flux_mean'] = lc_df['Flux'].mean()
    features['flux_std'] = lc_df['Flux'].std()
    features['flux_median'] = lc_df['Flux'].median()
    features['flux_min'] = lc_df['Flux'].min()
    features['flux_max'] = lc_df['Flux'].max()
    features['flux_range'] = features['flux_max'] - features['flux_min']
    
    # Skewness and kurtosis
    features['flux_skew'] = lc_df['Flux'].skew()
    features['flux_kurtosis'] = lc_df['Flux'].kurtosis()
    
    # Percentiles
    features['flux_p25'] = lc_df['Flux'].quantile(0.25)
    features['flux_p75'] = lc_df['Flux'].quantile(0.75)
    features['flux_iqr'] = features['flux_p75'] - features['flux_p25']
    
    # ===========================
    # TIME-BASED FEATURES
    # ===========================
    time_diff = lc_df['Time (MJD)'].diff().dropna()
    features['observation_duration'] = lc_df['Time (MJD)'].max() - lc_df['Time (MJD)'].min()
    features['mean_time_interval'] = time_diff.mean() if len(time_diff) > 0 else 0
    features['std_time_interval'] = time_diff.std() if len(time_diff) > 0 else 0
    
    # ===========================
    # VARIABILITY FEATURES
    # ===========================
    # Standard deviation / mean (coefficient of variation)
    if features['flux_mean'] != 0:
        features['flux_cv'] = features['flux_std'] / abs(features['flux_mean'])
    else:
        features['flux_cv'] = 0
    
    # Median Absolute Deviation
    features['flux_mad'] = np.median(np.abs(lc_df['Flux'] - features['flux_median']))
    
    # Peak-to-peak variability
    features['flux_amplitude'] = (features['flux_max'] - features['flux_min']) / 2
    
    # ===========================
    # FLUX ERROR FEATURES
    # ===========================
    features['flux_err_mean'] = lc_df['Flux_err'].mean()
    features['flux_err_median'] = lc_df['Flux_err'].median()
    features['signal_to_noise_mean'] = (lc_df['Flux'] / lc_df['Flux_err']).mean() if (lc_df['Flux_err'] > 0).all() else 0
    
    # ===========================
    # TREND FEATURES
    # ===========================
    # Linear trend (slope)
    if len(lc_df) > 1:
        time_normalized = (lc_df['Time (MJD)'] - lc_df['Time (MJD)'].min()) / (lc_df['Time (MJD)'].max() - lc_df['Time (MJD)'].min() + 1e-10)
        slope = np.polyfit(time_normalized, lc_df['Flux'], 1)[0] if len(time_normalized) > 1 else 0
        features['flux_trend_slope'] = slope
    else:
        features['flux_trend_slope'] = 0
    
    # ===========================
    # FILTER-SPECIFIC FEATURES
    # ===========================
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    for filt in filters:
        filt_data = lc_df[lc_df['Filter'] == filt]
        features[f'n_obs_{filt}'] = len(filt_data)
        features[f'flux_mean_{filt}'] = filt_data['Flux'].mean() if len(filt_data) > 0 else 0
        features[f'flux_std_{filt}'] = filt_data['Flux'].std() if len(filt_data) > 0 else 0
        features[f'flux_max_{filt}'] = filt_data['Flux'].max() if len(filt_data) > 0 else 0
    
    # ===========================
    # COLOR FEATURES (band differences)
    # ===========================
    color_pairs = [('g', 'r'), ('r', 'i'), ('i', 'z')]
    for band1, band2 in color_pairs:
        mean1 = features[f'flux_mean_{band1}']
        mean2 = features[f'flux_mean_{band2}']
        if mean2 != 0:
            features[f'color_{band1}_{band2}'] = mean1 - mean2
        else:
            features[f'color_{band1}_{band2}'] = 0
    
    # ===========================
    # DETECTION-BASED FEATURES
    # ===========================
    # Count significant detections (SNR > 3)
    significant = (lc_df['Flux'] / lc_df['Flux_err']) > 3
    features['n_significant_detections'] = significant.sum()
    features['frac_significant_detections'] = significant.mean()
    
    # Peak detection
    flux_values = lc_df['Flux'].values
    if len(flux_values) > 2:
        # Find local maxima
        is_peak = (flux_values[1:-1] > flux_values[:-2]) & (flux_values[1:-1] > flux_values[2:])
        features['n_peaks'] = is_peak.sum()
    else:
        features['n_peaks'] = 0
    
    return features

def load_all_light_curves(split_folders, is_train=True):
    """Load light curves from all split folders"""
    all_data = []
    
    file_pattern = 'train_full_lightcurves.csv' if is_train else 'test_full_lightcurves.csv'
    
    for split_folder in split_folders:
        file_path = Path(split_folder) / file_pattern
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"  ✓ Loaded {file_path.name} from {split_folder.name}: {len(df)} observations")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total observations: {len(combined):,}")
        print(f"✓ Unique objects: {combined['object_id'].nunique()}")
        return combined
    else:
        return pd.DataFrame()

def extract_features_for_all_objects(lc_data):
    """Extract features for all objects in the dataset"""
    print("\nExtracting features...")
    
    features_list = []
    object_ids = lc_data['object_id'].unique()
    
    total = len(object_ids)
    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} objects ({(i+1)/total*100:.1f}%)")
        
        obj_lc = lc_data[lc_data['object_id'] == obj_id]
        features = extract_light_curve_features(obj_lc)
        features['object_id'] = obj_id
        features_list.append(features)
    
    print(f"  ✓ Extracted features for {len(features_list)} objects")
    
    return pd.DataFrame(features_list)

# ===========================
# MAIN EXECUTION
# ===========================
print("\n[1] Finding split folders...")
# Point to the CORRECTED folder
base_dir = Path('.')
# Dynamic discovery of corrected splits
split_folders = sorted([d for d in base_dir.glob('split_*_corrected') if d.is_dir()])
print(f"Found {len(split_folders)} corrected split folders: {[f.name for f in split_folders]}")

# ===========================
# LOAD TRAINING DATA
# ===========================
print("\n[2] Loading TRAINING light curves...")
train_lc = load_all_light_curves(split_folders, is_train=True)

print("\n[3] Extracting TRAINING features...")
train_features = extract_features_for_all_objects(train_lc)

# Save
train_features.to_csv('lightcurve_features_train_corrected.csv', index=False)
print(f"\n✓ Saved training features: {train_features.shape}")
print(f"  Features: {len(train_features.columns) - 1}")  # -1 for object_id
print(f"  File: lightcurve_features_train_corrected.csv")

# ===========================
# LOAD TEST DATA
# ===========================
print("\n[4] Loading TEST light curves...")
test_lc = load_all_light_curves(split_folders, is_train=False)

print("\n[5] Extracting TEST features...")
test_features = extract_features_for_all_objects(test_lc)

# Save
test_features.to_csv('lightcurve_features_test_corrected.csv', index=False)
print(f"\n✓ Saved test features: {test_features.shape}")
print(f"  Features: {len(test_features.columns) - 1}")
print(f"  File: lightcurve_features_test_corrected.csv")

# ===========================
# SUMMARY
# ===========================
print("\n" + "="*70)
print("FEATURE EXTRACTION COMPLETE!")
print("="*70)
print(f"\nExtracted features:")
print(f"  - Basic statistics: mean, std, median, min, max, range")
print(f"  - Variability: CV, MAD, amplitude, skewness, kurtosis")
print(f"  - Time-based: duration, intervals")
print(f"  - Filter-specific: per-band statistics (ugrizy)")
print(f"  - Colors: g-r, r-i, i-z")
print(f"  - Detection: SNR-based features, peak counting")
print(f"  - Trend: linear slope")

print(f"\nTotal features extracted: {len(train_features.columns) - 1}")

# Show sample
print("\nSample features (first object):")
print(train_features.iloc[0].head(20))

print("\n" + "="*70)
print("Next: Merge with metadata and train improved model")
print("="*70)

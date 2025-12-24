"""
MALLORN TDE Classification - Physics-Informed Feature Extraction
Extract features based on astrophysical properties of TDEs

Features designed to distinguish TDEs based on:
1. Color temperature (flux ratios u/g, g/r) - TDEs stay hot/blue longer
2. Temporal asymmetry (rise/fade ratio) - TDEs have fast rise, slow t^-5/3 decay
3. Light curve morphology (skewness, peak properties)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# LSST filter extinction coefficients (R_λ) from Fitzpatrick 1999
EXTINCTION_COEFFS = {
    'u': 4.239, 'g': 3.303, 'r': 2.285,
    'i': 1.698, 'z': 1.263, 'y': 1.086
}

def apply_extinction_correction(df):
    """
    Apply extinction correction inline to lightcurve dataframe.
    Formula: F_intrinsic = F_observed × 10^(0.4 × R_λ × EBV)
    """
    print("  Applying extinction correction...")
    
    correction_factors = df.apply(
        lambda row: 10 ** (0.4 * EXTINCTION_COEFFS[row['Filter']] * row['EBV']),
        axis=1
    )
    
    df['Flux_corrected'] = df['Flux'] * correction_factors
    return df


def extract_flux_ratio_features(lc_group):
    """
    Extract color/temperature features from flux ratios.
    TDEs are hot (blue) → high u/g and g/r ratios.
    """
    features = {}
    
    # Get max flux for each filter (corrected values)
    filter_max = {}
    for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
        filt_data = lc_group[lc_group['Filter'] == filt]
        if len(filt_data) > 0:
            filter_max[filt] = filt_data['Flux_corrected'].max()
        else:
            filter_max[filt] = np.nan
    
    # Flux ratios at peak (key discriminator)
    features['flux_ratio_u_g'] = filter_max['u'] / (filter_max['g'] + 1e-10)
    features['flux_ratio_g_r'] = filter_max['g'] / (filter_max['r'] + 1e-10)
    features['flux_ratio_r_i'] = filter_max['r'] / (filter_max['i'] + 1e-10)
    features['flux_ratio_i_z'] = filter_max['i'] / (filter_max['z'] + 1e-10)
    
    # Color evolution - change in flux ratio over time
    # Compare peak to +30 days later (if available)
    try:
        peak_time = lc_group.loc[lc_group['Flux_corrected'].idxmax(), 'Time (MJD)']
        
        # Get u/g ratio at peak and +30 days
        peak_data = lc_group[np.abs(lc_group['Time (MJD)'] - peak_time) < 5]
        late_data = lc_group[(lc_group['Time (MJD)'] > peak_time + 25) & (lc_group['Time (MJD)'] < peak_time + 35)]
        
        if len(peak_data) > 0 and len(late_data) > 0:
            u_peak = peak_data[peak_data['Filter'] == 'u']['Flux_corrected'].max()
            g_peak = peak_data[peak_data['Filter'] == 'g']['Flux_corrected'].max()
            u_late = late_data[late_data['Filter'] == 'u']['Flux_corrected'].max()
            g_late = late_data[late_data['Filter'] == 'g']['Flux_corrected'].max()
            
            ratio_peak = u_peak / (g_peak + 1e-10)
            ratio_late = u_late / (g_late + 1e-10)
            features['color_evolution_ug'] = ratio_late / (ratio_peak + 1e-10)
        else:
            features['color_evolution_ug'] = np.nan
    except:
        features['color_evolution_ug'] = np.nan
    
    return features


def extract_temporal_features(lc_group):
    """
    Extract temporal morphology features.
    TDEs have high asymmetry: fast rise, slow decay.
    """
    features = {}
    
    # Use corrected flux
    flux = lc_group['Flux_corrected'].values
    time = lc_group['Time (MJD)'].values
    flux_err = lc_group['Flux_err'].values
    
    # Peak properties
    peak_idx = np.argmax(flux)
    peak_flux = flux[peak_idx]
    peak_time = time[peak_idx]
    
    features['max_flux_corrected'] = peak_flux
    
    # Signal-to-noise
    features['snr_peak'] = peak_flux / (np.median(flux_err) + 1e-10)
    
    # Number of significant detections
    snr = np.abs(flux) / (flux_err + 1e-10)
    features['n_detections'] = np.sum(snr > 5)
    
    # Rise time: first detection to peak
    detections = time[snr > 3]
    if len(detections) > 0:
        first_detection = detections.min()
        features['rise_time'] = peak_time - first_detection
    else:
        features['rise_time'] = np.nan
    
    # Fade time: peak to half-max
    post_peak_mask = time > peak_time
    if np.sum(post_peak_mask) > 0:
        post_peak_flux = flux[post_peak_mask]
        post_peak_time = time[post_peak_mask]
        
        half_max = peak_flux / 2
        # Find when flux drops below half-max
        below_half = post_peak_flux < half_max
        if np.sum(below_half) > 0:
            fade_idx = np.where(below_half)[0][0]
            features['fade_time'] = post_peak_time[fade_idx] - peak_time
        else:
            features['fade_time'] = np.nan
    else:
        features['fade_time'] = np.nan
    
    # Asymmetry ratio (KEY FEATURE for TDE)
    if not np.isnan(features['rise_time']) and not np.isnan(features['fade_time']) and features['fade_time'] > 0:
        features['asymmetry_ratio'] = features['rise_time'] / features['fade_time']
    else:
        features['asymmetry_ratio'] = np.nan
    
    # Skewness (TDEs have positive skew)
    if len(flux) > 3:
        features['flux_skewness'] = pd.Series(flux).skew()
    else:
        features['flux_skewness'] = np.nan
    
    # Duration of event
    features['duration'] = time.max() - time.min()
    
    return features


def process_split(split_num, dataset='train'):
    """Process a single split and extract features."""
    
    base_dir = Path(__file__).parent
    split_dir = base_dir / f"split_{split_num:02d}"
    
    # Lightcurves are in split folders
    lc_file_gz = split_dir / f"{dataset}_full_lightcurves.csv.gz"
    lc_file_csv = split_dir / f"{dataset}_full_lightcurves.csv"
    
    if lc_file_gz.exists():
        lc_file = lc_file_gz
    elif lc_file_csv.exists():
        lc_file = lc_file_csv
    else:
        print(f"  File not found: {lc_file_gz} or {lc_file_csv}")
        return None
    
    # Log files are in root directory
    log_file = base_dir / f"{dataset}_log.csv"
    
    if not log_file.exists():
        print(f"  Log file not found: {log_file}")
        return None
    
    print(f"  Loading {lc_file.name}...")
    lc_df = pd.read_csv(lc_file)
    meta_df = pd.read_csv(log_file)
    
    print(f"  Objects: {meta_df['object_id'].nunique():,}, Observations: {len(lc_df):,}")
    
    # Keep only objects from this split
    split_object_ids = lc_df['object_id'].unique()
    meta_df = meta_df[meta_df['object_id'].isin(split_object_ids)]
    
    print(f"  Objects in split: {len(meta_df):,}")
    
    # Merge to get EBV for extinction correction
    lc_df = lc_df.merge(meta_df[['object_id', 'EBV']], on='object_id', how='left')
    
    # Apply extinction correction
    lc_df = apply_extinction_correction(lc_df)
    
    # Extract features per object
    print("  Extracting physics features...")
    all_features = []
    
    for obj_id, group in tqdm(lc_df.groupby('object_id'), desc="  Processing"):
        features = {'object_id': obj_id}
        
        # Color/spectral features
        features.update(extract_flux_ratio_features(group))
        
        # Temporal features
        features.update(extract_temporal_features(group))
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Add metadata
    features_df = features_df.merge(meta_df[['object_id', 'Z', 'EBV']], on='object_id', how='left')
    
    # Add target if training
    if dataset == 'train' and 'target' in meta_df.columns:
        features_df = features_df.merge(meta_df[['object_id', 'target']], on='object_id', how='left')
    
    return features_df




def main():
    """Extract physics features from all splits."""
    
    print("="*70)
    print("PHYSICS-INFORMED FEATURE EXTRACTION")
    print("="*70)
    print("\nFeatures based on TDE astrophysics:")
    print("  1. Flux Ratios (u/g, g/r) - TDEs stay blue/hot")
    print("  2. Color Evolution - TDEs cool slower than SNe")
    print("  3. Temporal Asymmetry - TDEs: fast rise, slow t^-5/3 decay")
    print("  4. Skewness - TDEs have asymmetric light curves")
    print("="*70)
    
    # Process split_01 for train and test
    print("\n[SPLIT 01 - TRAIN]")
    train_features = process_split(1, 'train')
    
    if train_features is not None:
        output_file = 'physics_features_train.csv'
        train_features.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        print(f"  Features: {train_features.shape[1]-1} (excluding object_id)")
        print(f"  Objects: {len(train_features)}")
        
        # Show feature summary
        print("\n  Feature completeness:")
        for col in train_features.columns:
            if col not in ['object_id', 'target']:
                missing_pct = train_features[col].isna().mean() * 100
                print(f"    {col}: {100-missing_pct:.1f}% complete")
    
    print("\n[SPLIT 01 - TEST]")
    test_features = process_split(1, 'test')
    
    if test_features is not None:
        output_file = 'physics_features_test.csv'
        test_features.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        print(f"  Objects: {len(test_features)}")
    
    print("\n" + "="*70)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("="*70)
    print("\nNext: Train LightGBM with F1 optimization")


if __name__ == "__main__":
    main()

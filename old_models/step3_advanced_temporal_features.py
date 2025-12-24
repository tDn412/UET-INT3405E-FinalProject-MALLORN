"""
STEP 3: Advanced Time-Series Features (Alternative to LSTM)
Since TensorFlow install failed, we'll extract ADVANCED time-series features
that capture temporal patterns (similar to what LSTM would learn)

CONCEPT: Time-Series Feature Engineering
- Instead of deep learning, extract features that capture:
  * Trends (rise/decay patterns)
  * Periodicity
  * Auto-correlation
  * Peak characteristics
  * Multi-band evolution
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED TIME-SERIES FEATURE EXTRACTION")
print("="*70)

def extract_advanced_temporal_features(sequence, lc_df):
    """
    Extract advanced time-series features from light curve
    
    CONCEPTS:
    1. RISE/DECAY TIME - TDE signature
    2. ASYMMETRY - Peak shape
    3. COLOR EVOLUTION - How colors change over time
    4. AUTOCORRELATION - Periodic patterns
    5. PEAK CHARACTERISTICS - Number, width, prominence
    """
    features = {}
    
    # Sort by time
    lc_sorted = lc_df.sort_values('Time (MJD)').reset_index(drop=True)
   
    if len(lc_sorted) < 5:  # Need minimum data
        return {f'ts_{i}': 0 for i in range(30)}
    
    times = lc_sorted['Time (MJD)'].values
    fluxes = lc_sorted['Flux'].values
    
    # 1. RISE AND DECAY TIMES (TDE characteristic!)
    try:
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        start_time = times[0]
        end_time = times[-1]
        
        rise_time = peak_time - start_time
        decay_time = end_time - peak_time
        
        features['ts_rise_time'] = rise_time
        features['ts_decay_time'] = decay_time
        features['ts_asymmetry'] = rise_time / (decay_time + 1e-5)
    except:
        features['ts_rise_time'] = 0
        features['ts_decay_time'] = 0
        features['ts_asymmetry'] = 0
    
    # 2. PEAK CHARACTERISTICS
    try:
        peaks, properties = find_peaks(fluxes, prominence=1.0)
        features['ts_n_peaks'] = len(peaks)
        features['ts_peak_prominence'] = np.mean(properties['prominences']) if len(peaks) > 0 else 0
    except:
        features['ts_n_peaks'] = 0
        features['ts_peak_prominence'] = 0
    
    # 3. TREND ANALYSIS
    try:
        # Linear trend
        slope, intercept, r_value, _, _ = stats.linregress(times, fluxes)
        features['ts_slope'] = slope
        features['ts_r_squared'] = r_value ** 2
        
        # Polynomial fit (captures curvature)
        if len(times) >= 3:
            poly_coef = np.polyfit(times, fluxes, 2)
            features['ts_curvature'] = poly_coef[0]
        else:
            features['ts_curvature'] = 0
    except:
        features['ts_slope'] = 0
        features['ts_r_squared'] = 0
        features['ts_curvature'] = 0
    
    # 4. AUTOCORRELATION (periodicity)
    try:
        if len(fluxes) > 10:
            lag1_corr = np.corrcoef(fluxes[:-1], fluxes[1:])[0, 1]
            features['ts_autocorr_lag1'] = lag1_corr
        else:
            features['ts_autocorr_lag1'] = 0
    except:
        features['ts_autocorr_lag1'] = 0
    
    # 5. FLUX EVOLUTION METRICS
    try:
        flux_diffs = np.diff(fluxes)
        features['ts_mean_change'] = np.mean(flux_diffs)
        features['ts_std_change'] = np.std(flux_diffs)
        features['ts_max_jump'] = np.max(np.abs(flux_diffs))
    except:
        features['ts_mean_change'] = 0
        features['ts_std_change'] = 0
        features['ts_max_jump'] = 0
    
    # 6. COLOR EVOLUTION (multi-band)
    try:
        bands = ['g', 'r', 'i']
        colors = {}
        
        for band in bands:
            band_data = lc_sorted[lc_sorted['Filter'] == band]
            if len(band_data) > 0:
                colors[band] = band_data['Flux'].mean()
            else:
                colors[band] = 0
        
        # Color indices
        features['ts_g_minus_r'] = colors['g'] - colors['r']
        features['ts_r_minus_i'] = colors['r'] - colors['i']
        
        # Color change over time
        if 'g' in lc_sorted['Filter'].values and 'r' in lc_sorted['Filter'].values:
            g_flux = lc_sorted[lc_sorted['Filter'] == 'g']['Flux'].values
            r_flux = lc_sorted[lc_sorted['Filter'] == 'r']['Flux'].values
            
            if len(g_flux) > 1 and len(r_flux) > 1:
                g_change = np.std(g_flux)
                r_change = np.std(r_flux)
                features['ts_color_variability'] = abs(g_change - r_change)
            else:
                features['ts_color_variability'] = 0
        else:
            features['ts_color_variability'] = 0
    except:
        features['ts_g_minus_r'] = 0
        features['ts_r_minus_i'] = 0
        features['ts_color_variability'] = 0
    
    # 7. DURATION AND CADENCE
    features['ts_duration'] = times[-1] - times[0]
    features['ts_mean_cadence'] = np.mean(np.diff(times)) if len(times) > 1 else 0
    features['ts_cadence_std'] = np.std(np.diff(times)) if len(times) > 1 else 0
    
    # 8. FLUX STATISTICS OVER TIME
    # Early vs late flux
    mid_idx = len(fluxes) // 2
    early_flux = fluxes[:mid_idx]
    late_flux = fluxes[mid_idx:]
    
    features['ts_early_late_ratio'] = np.mean(early_flux) / (np.mean(late_flux) + 1e-5)
    features['ts_early_std'] = np.std(early_flux)
    features['ts_late_std'] = np.std(late_flux)
    
    # 9. STETSON INDEX (multi-band variability)
    try:
        stetson_values = []
        for i in range(len(fluxes)-1):
            delta1 = (fluxes[i] - np.mean(fluxes)) / (np.std(fluxes) + 1e-5)
            delta2 = (fluxes[i+1] - np.mean(fluxes)) / (np.std(fluxes) + 1e-5)
            stetson_values.append(delta1 * delta2)
        
        features['ts_stetson_index'] = np.mean(stetson_values) if len(stetson_values) > 0 else 0
    except:
        features['ts_stetson_index'] = 0
    
    # 10. TDE PHYSICS: POWER LAW & COLOR
    try:
        peak_idx = np.argmax(fluxes)
        peaks, _ = find_peaks(fluxes, prominence=1.0)
        
        # A. Power Law Decay: Flux ~ t^(-5/3)
        # We check fit quality (R2) specifically for -5/3 slope
        if peak_idx < len(fluxes) - 3:
            post_peak_flux = fluxes[peak_idx:]
            post_peak_times = times[peak_idx:] - times[peak_idx]
            
            # Filter positive values for log
            mask = (post_peak_times > 0) & (post_peak_flux > 0)
            if np.sum(mask) > 3:
                log_time = np.log10(post_peak_times[mask])
                log_flux = np.log10(post_peak_flux[mask])
                
                # Free fit
                slope, intercept, r_value, _, _ = stats.linregress(log_time, log_flux)
                features['ts_powerlaw_slope'] = slope
                features['ts_powerlaw_r2'] = r_value ** 2
                features['ts_decay_deviation'] = abs(slope - (-1.67)) # Deviation from theoretical -5/3
                
                # Forced fit (slope = -5/3) residuals
                # log_F = -1.67 * log_t + C  => C = log_F + 1.67 * log_t
                c_vals = log_flux + 1.67 * log_time
                c_mean = np.mean(c_vals)
                predicted_flux_log = -1.67 * log_time + c_mean
                mse_forced = np.mean((log_flux - predicted_flux_log)**2)
                features['ts_tde_fit_mse'] = mse_forced
            else:
                features['ts_powerlaw_slope'] = 0
                features['ts_powerlaw_r2'] = 0
                features['ts_decay_deviation'] = 99
                features['ts_tde_fit_mse'] = 99
        else:
            features['ts_powerlaw_slope'] = 0
            features['ts_powerlaw_r2'] = 0
            features['ts_decay_deviation'] = 99
            features['ts_tde_fit_mse'] = 99
            
        # B. TDE Color Locus
        # TDEs are hot blue bodies (Temp ~ 20000K) -> g-r approx -0.2 to 0.0
        if 'g' in colors and 'r' in colors and colors['g'] != 0 and colors['r'] != 0:
            g_r = colors['g'] - colors['r']
            # Distance from theoretical TDE color (-0.1 roughly)
            features['ts_color_tde_dist'] = abs(g_r - (-0.1))
        else:
            features['ts_color_tde_dist'] = 99
            
    except:
        features['ts_powerlaw_slope'] = 0
        features['ts_powerlaw_r2'] = 0
        features['ts_decay_deviation'] = 99
        features['ts_tde_fit_mse'] = 99
        features['ts_color_tde_dist'] = 99
    
    return features


if __name__ == "__main__":
    
    # Load raw light curves for additional features
    print("\n[1] Loading CORRECTED raw light curves...")
    with open('raw_lightcurves_train_corrected.pkl', 'rb') as f:
        train_lcs = pickle.load(f)
    print(f"  Training objects: {len(train_lcs)}")
        
    with open('raw_lightcurves_test_corrected.pkl', 'rb') as f:
        test_lcs = pickle.load(f)
    print(f"  Test objects: {len(test_lcs)}")
    
    print("\n[2] Extracting advanced features for CORRECTED TRAINING set...")
    
    train_advanced_features = []
    obj_ids_train = list(train_lcs.keys())
    
    for i, obj_id in enumerate(obj_ids_train):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(obj_ids_train)}...")
        
        lc_df = train_lcs[obj_id]
        dummy_seq = np.ones((1, 1, 1)) 
        
        features = extract_advanced_temporal_features(dummy_seq, lc_df)
        features['object_id'] = obj_id # Ensure ID is preserved
        train_advanced_features.append(features)
    
    train_advanced_df = pd.DataFrame(train_advanced_features)
    print(f"\n✓ Extracted {len(train_advanced_df.columns)} advanced features")
    
    print("\n[3] Extracting advanced features for CORRECTED TEST set...")
    
    test_advanced_features = []
    obj_ids_test = list(test_lcs.keys())
    
    for i, obj_id in enumerate(obj_ids_test):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(obj_ids_test)}...")
        
        lc_df = test_lcs[obj_id]
        dummy_seq = np.ones((1, 1, 1))
        
        features = extract_advanced_temporal_features(dummy_seq, lc_df)
        features['object_id'] = obj_id
        test_advanced_features.append(features)
    
    test_advanced_df = pd.DataFrame(test_advanced_features)
    print(f"\n✓ Extracted features for {len(test_advanced_df)} test objects")
    
    # Fill NaN and Inf
    train_advanced_df = train_advanced_df.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
    test_advanced_df = test_advanced_df.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Save
    train_advanced_df.to_csv('advanced_temporal_features_train_corrected.csv', index=False)
    test_advanced_df.to_csv('advanced_temporal_features_test_corrected.csv', index=False)
    
    print(f"\n✓ Saved advanced_temporal_features_train_corrected.csv")
    print(f"✓ Saved advanced_temporal_features_test_corrected.csv")
    
    print("\n" + "="*70)
    print("ADVANCED TEMPORAL FEATURES (PHYSICS CORRECTED) COMPLETE!")
    print("="*70)

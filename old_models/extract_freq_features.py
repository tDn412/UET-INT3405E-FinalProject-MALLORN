"""
Extract Wavelet and Fourier Features for Light Curves.
This addresses the "Frequency Domain" gap in our feature set.

Features to extract per band:
1. Fourier (FFT):
   - Dominant frequency
   - Power spectrum entropy (spectral entropy)
   - Ratio of power in high vs low frequencies
2. Wavelet (CWT) using Ricker (Mexican Hat) or Morlet wavelet:
   - Max power at different scales (short, medium, long duration)
   - Mean/Std of coefficients
   - Energy distribution
"""

import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy.signal import find_peaks
import pywt
import warnings
warnings.filterwarnings('ignore')

def extract_frequency_features(df):
    """
    Extracts frequency domain features for a set of light curves.
    Assumes df has columns: ['object_id', 'mjd', 'flux', 'passband']
    """
    
    # pivot to get time-series per band
    # But data is irregular. We need to handle that.
    # For FFT/Wavelet, it's best to interpolate to a regular grid first.
    
    object_ids = df['object_id'].unique()
    features = []
    
    print(f"Processing {len(object_ids)} objects...")
    
    count = 0
    for obj_id in object_ids:
        obj_data = df[df['object_id'] == obj_id]
        
        obj_feats = {'object_id': obj_id}
        
        # Process each passband
        for band in sorted(df['passband'].unique()):
            band_data = obj_data[obj_data['passband'] == band].sort_values('mjd')
            
            if len(band_data) < 3:
                # Fill with NaNs if not enough data
                for k in ['dominant_freq', 'spectral_entropy', 'wavelet_max', 'wavelet_energy']:
                    obj_feats[f'{band}_{k}'] = np.nan
                continue
                
            t = band_data['mjd'].values
            x = band_data['flux'].values
            
            # Normalize time to start at 0
            t = t - t[0]
            
            # 1. Fourier Transform (Simple Periodogram for irregular sampling approximation)
            # Better: Lomb-Scargle, but for speed let's try FFT on interpolated grid
            # Interpolate to regular grid
            if t[-1] > 0:
                # Create regular grid with step size ~1 day
                t_reg = np.arange(0, t[-1], 1.0)
                if len(t_reg) > 2:
                    x_reg = np.interp(t_reg, t, x)
                    
                    # FFT
                    fft_vals = np.fft.rfft(x_reg)
                    fft_power = np.abs(fft_vals)**2
                    freqs = np.fft.rfftfreq(len(x_reg), d=1.0)
                    
                    # Dominant frequency (excluding 0-freq DC component)
                    if len(fft_power) > 1:
                        idx_max = np.argmax(fft_power[1:]) + 1
                        obj_feats[f'{band}_fft_dom_freq'] = freqs[idx_max]
                        obj_feats[f'{band}_fft_dom_power'] = fft_power[idx_max]
                        
                        # Spectral Entropy
                        psd_norm = fft_power[1:] / np.sum(fft_power[1:])
                        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))
                        obj_feats[f'{band}_fft_entropy'] = entropy
                    else:
                        obj_feats[f'{band}_fft_dom_freq'] = 0
                        obj_feats[f'{band}_fft_entropy'] = 0
                        
                    # 2. Wavelet Transform (CWT)
                    # Use scales corresponding to typical transient durations
                    # TDEs are "long" (~30-100 days), random flares are "short"
                    scales = [2, 5, 10, 20, 50]
                    cwt_coeffs, freqs_cwt = pywt.cwt(x_reg, scales, 'mexh') # Mexican Hat wavelet
                    
                    # Extract stats for each scale
                    for i, scale in enumerate(scales):
                        coeffs = cwt_coeffs[i]
                        obj_feats[f'{band}_cwt_scale{scale}_max'] = np.max(np.abs(coeffs))
                        obj_feats[f'{band}_cwt_scale{scale}_energy'] = np.sum(coeffs**2)
                        
                    # Global wavelet stats
                    obj_feats[f'{band}_cwt_total_energy'] = np.sum(cwt_coeffs**2)
                    
                else:
                     # Fallback
                    for k in ['fft_dom_freq', 'fft_entropy', 'cwt_total_energy']:
                        obj_feats[f'{band}_{k}'] = np.nan
            else:
                 # Fallback
                for k in ['fft_dom_freq', 'fft_entropy', 'cwt_total_energy']:
                    obj_feats[f'{band}_{k}'] = np.nan

        features.append(obj_feats)
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} objects")
            
    return pd.DataFrame(features)

if __name__ == "__main__":
    print("EXTRACTING WAVELET & FOURIER FEATURES")
    print("="*80)
    
    import pickle
    import os
    
    # 1. Load Raw Light Curves from Pickle
    print("Loading raw light curves from pickle...")
    if os.path.exists('raw_lightcurves_train.pkl'):
        with open('raw_lightcurves_train.pkl', 'rb') as f:
            train_lcs = pickle.load(f)
        print(f"Loaded {len(train_lcs)} training light curves")
    else:
        print("ERROR: raw_lightcurves_train.pkl not found!")
        print("Please run step1_load_raw_lightcurves.py first.")
        exit()

    if os.path.exists('raw_lightcurves_test.pkl'):
        with open('raw_lightcurves_test.pkl', 'rb') as f:
            test_lcs = pickle.load(f)
        print(f"Loaded {len(test_lcs)} test light curves")
    else:
        print("ERROR: raw_lightcurves_test.pkl not found!")
        exit()
        
    # Helper to convert dictionary format to DataFrame expected by extraction function
    # The dictionary maps object_id -> DataFrame
    # We can iterate through the dict directly instead of converting to giant DataFrame
    
    def extract_features_from_dict(lc_dict):
        features = []
        count = 0
        total = len(lc_dict)
        
        for obj_id, df in lc_dict.items():
            obj_feats = {'object_id': obj_id}
            
            # Ensure columns are correct (Flux, MJD, Filter/passband)
            # step1 script says columns: likely 'mjd', 'flux', 'flux_err', 'filter' or similar
            # Let's normalize column names if needed
            cols = list(df.columns)
            col_map = {c.lower(): c for c in cols}
            
            # Correct column mapping based on inspect_pickle.py
            flux_col = 'Flux'
            mjd_col = 'Time (MJD)'
            passband_col = 'Filter'
            
            if passband_col not in df.columns or flux_col not in df.columns:
                 # Try fallback
                cols = list(df.columns)
                col_map = {c.lower(): c for c in cols}
                flux_col = col_map.get('flux', 'Flux')
                mjd_col = col_map.get('time (mjd)', col_map.get('mjd', 'Time (MJD)'))
                passband_col = col_map.get('filter', 'Filter')
                
                if passband_col not in df.columns:
                    continue
                
            for band in df[passband_col].unique():
                band_data = df[df[passband_col] == band].sort_values(mjd_col)
                
                if len(band_data) < 3:
                     for k in ['fft_dom_freq', 'fft_entropy', 'cwt_total_energy']:
                        obj_feats[f'{band}_{k}'] = np.nan
                     continue
                
                t = band_data[mjd_col].values
                x = band_data[flux_col].values
                
                # Normalize time
                t = t - t[0]
                
                # Interpolate to regular grid
                if t[-1] > 0:
                    t_reg = np.arange(0, t[-1], 1.0) # 1-day step
                    if len(t_reg) > 2:
                        x_reg = np.interp(t_reg, t, x)
                        
                        # fillna if any
                        x_reg = np.nan_to_num(x_reg)
                        
                        # FFT
                        fft_vals = np.fft.rfft(x_reg)
                        fft_power = np.abs(fft_vals)**2
                        # safe divide
                        sum_power = np.sum(fft_power[1:])
                        if sum_power > 0:
                            psd_norm = fft_power[1:] / sum_power
                            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))
                            
                            freqs = np.fft.rfftfreq(len(x_reg), d=1.0)
                            idx_max = np.argmax(fft_power[1:]) + 1
                            dom_freq = freqs[idx_max]
                        else:
                            entropy = 0
                            dom_freq = 0
                            
                        obj_feats[f'{band}_fft_dom_freq'] = dom_freq
                        obj_feats[f'{band}_fft_entropy'] = entropy
                        
                        # Wavelet (MexH)
                        scales = [2, 5, 10, 20] # Short to long scales
                        try:
                            cwt_coeffs, _ = pywt.cwt(x_reg, scales, 'mexh')
                            for i, scale in enumerate(scales):
                                coeffs = cwt_coeffs[i]
                                obj_feats[f'{band}_cwt_scale{scale}_max'] = np.max(np.abs(coeffs))
                                obj_feats[f'{band}_cwt_scale{scale}_std'] = np.std(coeffs)
                                obj_feats[f'{band}_cwt_scale{scale}_energy'] = np.sum(coeffs**2)
                        except Exception:
                            pass

                # Handle failures
                if f'{band}_fft_entropy' not in obj_feats:
                     obj_feats[f'{band}_fft_entropy'] = np.nan
            
            features.append(obj_feats)
            count += 1
            if count % 500 == 0:
                print(f"  Processed {count}/{total}")
                
        return pd.DataFrame(features)

    print("\nExtracting Train Features...")
    train_feats = extract_features_from_dict(train_lcs)
    train_feats.to_csv('frequency_features_train.csv', index=False)
    print(f"Saved {len(train_feats)} train features")
    
    print("\nExtracting Test Features...")
    test_feats = extract_features_from_dict(test_lcs)
    test_feats.to_csv('frequency_features_test.csv', index=False)
    print(f"Saved {len(test_feats)} test features")




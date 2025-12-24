"""
STEP 5: EXTRACT FEATURES FROM SYNTHETIC DATA
Goal: Convert raw synthetic light curves into tabular features for the ensemble.
Input: raw_lightcurves_synthetic.pkl
Output: lightcurve_features_synthetic.csv, advanced_temporal_features_synthetic.csv
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXTRACTING FEATURES FROM SYNTHETIC DATA")
print("="*70)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n[1] Loading Synthetic Light Curves...")
with open('raw_lightcurves_synthetic.pkl', 'rb') as f:
    syn_lcs = pickle.load(f)
print(f"  Loaded {len(syn_lcs)} objects.")

# ==========================================
# 2. FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_basic_features(lc_df):
    """From extract_lightcurve_features.py"""
    features = {}
    if len(lc_df) == 0: return features
    
    lc_df = lc_df.sort_values('Time (MJD)')
    
    # Basic Stats
    features['n_observations'] = len(lc_df)
    features['flux_mean'] = lc_df['Flux'].mean()
    features['flux_std'] = lc_df['Flux'].std()
    features['flux_median'] = lc_df['Flux'].median()
    features['flux_min'] = lc_df['Flux'].min()
    features['flux_max'] = lc_df['Flux'].max()
    features['flux_range'] = features['flux_max'] - features['flux_min']
    features['flux_skew'] = lc_df['Flux'].skew()
    features['flux_kurtosis'] = lc_df['Flux'].kurtosis()
    features['flux_p25'] = lc_df['Flux'].quantile(0.25)
    features['flux_p75'] = lc_df['Flux'].quantile(0.75)
    features['flux_iqr'] = features['flux_p75'] - features['flux_p25']
    
    # Time
    features['observation_duration'] = lc_df['Time (MJD)'].max() - lc_df['Time (MJD)'].min()
    time_diff = lc_df['Time (MJD)'].diff().dropna()
    features['mean_time_interval'] = time_diff.mean() if len(time_diff) > 0 else 0
    features['std_time_interval'] = time_diff.std() if len(time_diff) > 0 else 0
    
    # Variability
    if features['flux_mean'] != 0:
        features['flux_cv'] = features['flux_std'] / abs(features['flux_mean'])
    else:
        features['flux_cv'] = 0
    features['flux_mad'] = np.median(np.abs(lc_df['Flux'] - features['flux_median']))
    features['flux_amplitude'] = (features['flux_max'] - features['flux_min']) / 2
    
    # Error
    features['flux_err_mean'] = lc_df['Flux Error'].mean() # Note: 'Flux Error' in synthetic, check if 'Flux_err' in real
    features['flux_err_median'] = lc_df['Flux Error'].median()
    # Handle potentially missing columns or names
    
    # Trend
    if len(lc_df) > 1:
        time_norm = (lc_df['Time (MJD)'] - lc_df['Time (MJD)'].min()) / (features['observation_duration'] + 1e-10)
        slope = np.polyfit(time_norm, lc_df['Flux'], 1)[0]
        features['flux_trend_slope'] = slope
    else:
        features['flux_trend_slope'] = 0
        
    # Per Filter
    for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
        filt_data = lc_df[lc_df['Filter'] == filt]
        if len(filt_data) > 0:
            features[f'flux_mean_{filt}'] = filt_data['Flux'].mean()
            features[f'flux_max_{filt}'] = filt_data['Flux'].max()
            features[f'n_obs_{filt}'] = len(filt_data)
        else:
            features[f'flux_mean_{filt}'] = 0
            features[f'flux_max_{filt}'] = 0
            features[f'n_obs_{filt}'] = 0
            
    # Colors
    def get_color(b1, b2):
        if features[f'flux_mean_{b1}'] != 0 and features[f'flux_mean_{b2}'] != 0:
            return features[f'flux_mean_{b1}'] - features[f'flux_mean_{b2}']
        return -999
        
    features['color_g_r'] = get_color('g', 'r')
    features['color_r_i'] = get_color('r', 'i')
    features['color_i_z'] = get_color('i', 'z')
    
    return features

def extract_advanced_features(lc_df):
    """From step3_advanced_temporal_features.py"""
    features = {}
    lc_sorted = lc_df.sort_values('Time (MJD)').reset_index(drop=True)
   
    if len(lc_sorted) < 5: 
        return {f'ts_{i}': 0 for i in range(30)}
    
    times = lc_sorted['Time (MJD)'].values
    fluxes = lc_sorted['Flux'].values
    
    # 1. Rise/Decay
    try:
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        rise_time = peak_time - times[0]
        decay_time = times[-1] - peak_time
        features['ts_rise_time'] = rise_time
        features['ts_decay_time'] = decay_time
        features['ts_asymmetry'] = rise_time / (decay_time + 1e-5)
    except:
        features['ts_rise_time'] = 0; features['ts_decay_time'] = 0; features['ts_asymmetry'] = 0

    # 2. Peaks
    try:
        peaks, props = find_peaks(fluxes, prominence=1.0)
        features['ts_n_peaks'] = len(peaks)
        features['ts_peak_prominence'] = np.mean(props['prominences']) if len(peaks)>0 else 0
    except:
        features['ts_n_peaks'] = 0; features['ts_peak_prominence'] = 0

    # 3. Trend
    try:
        slope, _, r_val, _, _ = stats.linregress(times, fluxes)
        features['ts_slope'] = slope
        features['ts_r_squared'] = r_val**2
        if len(times)>=3: features['ts_curvature'] = np.polyfit(times, fluxes, 2)[0]
        else: features['ts_curvature'] = 0
    except:
        features['ts_slope']=0; features['ts_r_squared']=0; features['ts_curvature']=0

    # 4. Autocorr
    try:
        if len(fluxes)>10:
            features['ts_autocorr_lag1'] = np.corrcoef(fluxes[:-1], fluxes[1:])[0,1]
        else: features['ts_autocorr_lag1'] = 0
    except: features['ts_autocorr_lag1'] = 0

    # 5. Flux metrics
    try:
        diffs = np.diff(fluxes)
        features['ts_mean_change'] = np.mean(diffs)
        features['ts_std_change'] = np.std(diffs)
        features['ts_max_jump'] = np.max(np.abs(diffs))
    except:
        features['ts_mean_change']=0; features['ts_std_change']=0; features['ts_max_jump']=0

    # 6. Color Evo
    try:
        bands = ['g','r','i']
        colors = {}
        for b in bands:
            bd = lc_sorted[lc_sorted['Filter']==b]
            colors[b] = bd['Flux'].mean() if len(bd)>0 else 0
        features['ts_g_minus_r'] = colors['g'] - colors['r']
        features['ts_r_minus_i'] = colors['r'] - colors['i']
        
        # Color var
        if 'g' in lc_sorted['Filter'].values and 'r' in lc_sorted['Filter'].values:
            g_flux = lc_sorted[lc_sorted['Filter']=='g']['Flux'].values
            r_flux = lc_sorted[lc_sorted['Filter']=='r']['Flux'].values
            if len(g_flux)>1 and len(r_flux)>1:
                features['ts_color_variability'] = abs(np.std(g_flux)-np.std(r_flux))
            else: features['ts_color_variability'] = 0
        else: features['ts_color_variability'] = 0
    except:
        features['ts_g_minus_r']=0; features['ts_r_minus_i']=0; features['ts_color_variability']=0

    # 7. Duration
    features['ts_duration'] = times[-1]-times[0]
    features['ts_mean_cadence'] = np.mean(np.diff(times)) if len(times)>1 else 0
    features['ts_cadence_std'] = np.std(np.diff(times)) if len(times)>1 else 0

    # 8. Stats over time
    mid = len(fluxes)//2
    features['ts_early_late_ratio'] = np.mean(fluxes[:mid]) / (np.mean(fluxes[mid:]) + 1e-5)
    features['ts_early_std'] = np.std(fluxes[:mid])
    features['ts_late_std'] = np.std(fluxes[mid:])

    # 9. Stetson
    try:
        stet = []
        for i in range(len(fluxes)-1):
            d1 = (fluxes[i]-np.mean(fluxes))/(np.std(fluxes)+1e-5)
            d2 = (fluxes[i+1]-np.mean(fluxes))/(np.std(fluxes)+1e-5)
            stet.append(d1*d2)
        features['ts_stetson_index'] = np.mean(stet) if len(stet)>0 else 0
    except: features['ts_stetson_index']=0

    # 10. TDE Physics
    try:
        peak_idx = np.argmax(fluxes)
        if peak_idx < len(fluxes)-3:
            post_peak_flux = fluxes[peak_idx:]
            post_peak_times = times[peak_idx:] - times[peak_idx]
            mask = (post_peak_times>0) & (post_peak_flux>0)
            if np.sum(mask)>3:
                l_t = np.log10(post_peak_times[mask])
                l_f = np.log10(post_peak_flux[mask])
                slope, _, r_val, _, _ = stats.linregress(l_t, l_f)
                features['ts_powerlaw_slope'] = slope
                features['ts_powerlaw_r2'] = r_val**2
                features['ts_decay_deviation'] = abs(slope - (-1.67))
                
                # Forced fit MSE
                c_vals = l_f + 1.67*l_t
                pred = -1.67*l_t + np.mean(c_vals)
                features['ts_tde_fit_mse'] = np.mean((l_f-pred)**2)
            else:
                 features['ts_powerlaw_slope']=0; features['ts_powerlaw_r2']=0
                 features['ts_decay_deviation']=99; features['ts_tde_fit_mse']=99
        else:
             features['ts_powerlaw_slope']=0; features['ts_powerlaw_r2']=0
             features['ts_decay_deviation']=99; features['ts_tde_fit_mse']=99
             
        # TDE Color dist
        if colors['g']!=0 and colors['r']!=0:
            features['ts_color_tde_dist'] = abs((colors['g']-colors['r']) - (-0.1))
        else:
            features['ts_color_tde_dist'] = 99
    except:
        features['ts_powerlaw_slope']=0; features['ts_powerlaw_r2']=0
        features['ts_decay_deviation']=99; features['ts_tde_fit_mse']=99
        features['ts_color_tde_dist']=99
        
    return features


# ==========================================
# 3. EXTRACTION LOOP
# ==========================================
print("\n[2] Extracting...")
basic_list = []
adv_list = []

for i, (oid, lc) in enumerate(syn_lcs.items()):
    if i % 100 == 0: print(f"  {i}/{len(syn_lcs)}")
    
    # Basic
    bf = extract_basic_features(lc)
    bf['object_id'] = oid
    basic_list.append(bf)
    
    # Advanced
    af = extract_advanced_features(lc)
    af['object_id'] = oid
    adv_list.append(af)

# ==========================================
# 4. SAVE
# ==========================================
basic_df = pd.DataFrame(basic_list)
adv_df = pd.DataFrame(adv_list)

# Fill NaNs
basic_df = basic_df.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
adv_df = adv_df.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"\nExtracted {len(basic_df)} basic feature rows.")
print(f"Extracted {len(adv_df)} advanced feature rows.")

basic_df.to_csv('lightcurve_features_synthetic.csv', index=False)
adv_df.to_csv('advanced_temporal_features_synthetic.csv', index=False)

print("✓ Saved lightcurve_features_synthetic.csv")
print("✓ Saved advanced_temporal_features_synthetic.csv")

"""
STEP 1 V3 (FULL POWER): AUGMENT/GENERATE FEATURES
=================================================

Restores FULL feature complexity (FFT, Polyfit, Slopes, Percentiles)
to match original performance (~0.6 F1).
Keeps robust Kaggle paths & data alignment.
"""

import pandas as pd
import numpy as np
import os
import sys
import gc
import warnings
import glob
from scipy.stats import skew, kurtosis, linregress
from scipy.optimize import curve_fit
from scipy.fft import rfft
from numpy.polynomial.polynomial import polyfit

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 1 V3: FULL FEATURE EXTRACTION (HIGH PERFORMANCE)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
KAGGLE_INPUT = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
EXTINCTION = {"u": 4.81, "g": 3.64, "r": 2.70, "i": 2.06, "z": 1.58, "y": 1.31}

# ============================================================================
# PART 1: FEATURE LOGIC (RESTORED FROM ORIGINAL)
# ============================================================================

def fit_tde_shape(t, f, f_peak, t_peak):
    mask = (t > t_peak) & (f > 0.05 * f_peak)
    if np.sum(mask) < 3: return -1.0
    t_dec, f_dec = t[mask], f[mask]
    func = lambda ti, t0: f_peak * np.power(np.maximum((ti - t0) / (t_peak - t0), 1e-6), -5/3)
    try:
        popt, _ = curve_fit(func, t_dec, f_dec, p0=[t_peak - 20], 
                            bounds=([t_peak - 1000], [t_peak - 0.5]), maxfev=100)
        return np.sum((f_dec - func(t_dec, *popt))**2) / len(t_dec)
    except: return -1.0

def get_fft_features(flux):
    if len(flux) < 5: return [0.0]*4
    f_fft = np.abs(rfft(flux))
    vals = list(f_fft[1:5]) if len(f_fft) >= 5 else list(f_fft[1:])
    return vals + [0.0]*(4-len(vals))

def extract_band_features(time, flux, err, filt, z):
    f = {}
    n = len(flux)
    if n < 3: return f, None, None
    
    # BASICS
    f_mean, f_std = np.mean(flux), np.std(flux)
    f_max, f_min = np.max(flux), np.min(flux)
    t_min, t_max = time.min(), time.max()
    t_peak = time[np.argmax(flux)]
    
    f[f"{filt}_mean"] = f_mean
    f[f"{filt}_std"] = f_std
    f[f"{filt}_max"] = f_max
    f[f"{filt}_skew"] = skew(flux)
    f[f"{filt}_kurt"] = kurtosis(flux)
    f[f"{filt}_mad"] = np.median(np.abs(flux - np.median(flux)))
    f[f"{filt}_amp"] = (f_max - f_min) / (f_mean + 1e-9)
    f[f"{filt}_cv"] = f_std / (f_mean + 1e-9)

    # PERCENTILES
    for p in [0, 5, 20, 50, 80, 95, 100]:
        f[f"{filt}_p{p}"] = np.percentile(flux, p)
    
    # VON NEUMANN
    diff = np.diff(flux)
    f[f"{filt}_von_neumann"] = np.sum(diff**2) / (np.sum((flux - f_mean)**2) + 1e-9)
    
    # TIME / SLOPE
    f[f"{filt}_time_to_peak"] = t_peak - t_min
    z_eff = max(float(z), 0.0)
    t_rest = (time - t_peak) / (1.0 + z_eff)
    
    # Rise/Fall Slope
    mask_rise = t_rest < 0
    mask_fall = t_rest > 0
    f[f"{filt}_slope_rise"] = linregress(t_rest[mask_rise], flux[mask_rise])[0] if np.sum(mask_rise) > 1 else 0
    f[f"{filt}_slope_fall"] = linregress(t_rest[mask_fall], flux[mask_fall])[0] if np.sum(mask_fall) > 1 else 0
    
    # FFT
    for i, val in enumerate(get_fft_features(flux)): 
        f[f"{filt}_fft_{i}"] = val
    
    # POLYFIT SHAPE
    try:
        p = polyfit(t_rest, flux, 2)
        f[f"{filt}_poly2_a"] = p[2]
        f[f"{filt}_poly2_b"] = p[1]
    except:
        f[f"{filt}_poly2_a"] = 0
        f[f"{filt}_poly2_b"] = 0

    # TDE FIT (g, r only)
    if filt in ["g", "r"]:
        f[f"{filt}_tde_chisq"] = fit_tde_shape(time, flux, f_max, t_peak)
        
    return f, t_peak, f_max

def process_raw_data(log_df, lc_paths):
    print(f"   -> Processing {len(lc_paths)} files...")
    
    # Robust Load
    dfs = []
    for f in lc_paths:
        try: dfs.append(pd.read_csv(f))
        except: pass
    df_lc = pd.concat(dfs, ignore_index=True)
    
    # Columns
    time_col = 'Time (MJD)' if 'Time (MJD)' in df_lc.columns else 'Time'
    if 'Time' not in df_lc.columns and 'mjd' in df_lc.columns: time_col = 'mjd'
    
    # Force String ID
    df_lc['object_id'] = df_lc['object_id'].astype(str)
    log_df['object_id'] = log_df['object_id'].astype(str)
    
    # Merge
    df = df_lc.merge(log_df[["object_id", "EBV", "Z"]], on="object_id", how="left").fillna(0)
    df["R"] = df["Filter"].map(EXTINCTION).fillna(0)
    # Clip EBV to avoid explosion
    df["Flux_Corr"] = df["Flux"] * (10 ** (0.4 * df["R"] * df["EBV"].clip(0, 3.0)))

    features = []
    unique_ids = df['object_id'].unique()
    
    for idx, obj_id in enumerate(unique_ids):
        if idx % 500 == 0: print(f"      {idx}/{len(unique_ids)}")
        grp = df[df['object_id'] == obj_id].sort_values(time_col)
        z = grp["Z"].iloc[0]
        row = {"object_id": obj_id}
        
        band_stats = {}
        for filt in ["u", "g", "r", "i", "z", "y"]:
            g = grp[grp["Filter"] == filt]
            if len(g) < 3: continue
            feats, t_peak, f_max = extract_band_features(g[time_col].values, g["Flux_Corr"].values, g["Flux_err"].values, filt, z)
            row.update(feats)
            if t_peak is not None: band_stats[filt] = {"t": t_peak, "f": f_max}
            
        # CROSS BAND
        bands = ["u", "g", "r", "i", "z", "y"]
        for i in range(len(bands)):
            for j in range(i+1, len(bands)):
                b1, b2 = bands[i], bands[j]
                if b1 in band_stats and b2 in band_stats:
                    row[f"{b1}_{b2}_lag"] = band_stats[b1]["t"] - band_stats[b2]["t"]
                    row[f"{b1}_{b2}_max_diff"] = band_stats[b1]["f"] - band_stats[b2]["f"]
                    row[f"{b1}_{b2}_max_ratio"] = band_stats[b1]["f"] / (band_stats[b2]["f"] + 1e-9)

        # Global
        row['n_obs'] = len(grp)
        row['max_flux_global'] = grp['Flux_Corr'].max()
        
        features.append(row)
        
    return pd.DataFrame(features).fillna(-999)

# ============================================================================
# PART 2: EXECUTION
# ============================================================================

print("\n📥 GENERATING FEATURES...")
trainlog = None
raw_dir = None

for loc in [KAGGLE_INPUT, '.']:
    for root, _, files in os.walk(loc):
        if 'train_log.csv' in files:
            raw_dir = root; break
    if raw_dir: break

if not raw_dir: sys.exit("❌ Train Log Not Found")

# TRAIN
print("   Generating TRAIN...")
trainlog = pd.read_csv(os.path.join(raw_dir, 'train_log.csv'))
tr_files = glob.glob(os.path.join(raw_dir, "**", "*train*lightcurves*.csv"), recursive=True)
if not tr_files: tr_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if 'train' in f and 'lightcurve' in f]
Xfull = process_raw_data(trainlog, tr_files)

# TEST
print("   Generating TEST...")
testlog = pd.read_csv(os.path.join(raw_dir, 'test_log.csv'))
te_files = glob.glob(os.path.join(raw_dir, "**", "*test*lightcurves*.csv"), recursive=True)
if not te_files: te_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if 'test' in f and 'lightcurve' in f]
Xtest = process_raw_data(testlog, te_files)

# MERGE TARGETS ALERT
print("   Merging Targets...")
if 'objectid' not in Xfull.columns: Xfull.rename(columns={'object_id': 'objectid'}, inplace=True)
if 'objectid' not in Xtest.columns: Xtest.rename(columns={'object_id': 'objectid'}, inplace=True)
trainlog['object_id'] = trainlog['object_id'].astype(str)
if 'objectid' not in trainlog.columns: trainlog.rename(columns={'object_id': 'objectid'}, inplace=True)

Xfull = Xfull.merge(trainlog[['objectid', 'target']], on='objectid', how='left')

# Check Quality
numeric_cols = Xfull.select_dtypes(include=[np.number]).columns
print(f"\n   Top Correlations:\n{Xfull[numeric_cols].corrwith(Xfull['target']).abs().sort_values(ascending=False).head(5)}")

# Save
print("\n💾 SAVING...")
Xfull.to_csv(os.path.join(OUTPUT_DIR, 'lightcurve_features_improved_train.csv'), index=False)
Xtest.to_csv(os.path.join(OUTPUT_DIR, 'lightcurve_features_improved_test.csv'), index=False)
print("✅ Done. Run Step 2 & 3 now!")

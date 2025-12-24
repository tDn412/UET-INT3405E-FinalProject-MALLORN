"""
STEP 1 (KAGGLE AUTO-GENERATION): AUGMENT/GENERATE FEATURES
==========================================================

Auto-detects if features exist. If not, GENERATES them from raw lightcurves.
Optimized for Kaggle Directories structure.
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
print("STEP 1: FEATURE GENERATION & AUGMENTATION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
KAGGLE_INPUT = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
EXTINCTION = {"u": 4.81, "g": 3.64, "r": 2.70, "i": 2.06, "z": 1.58, "y": 1.31}

# ============================================================================
# PART 1: AUTO-DETECT DATASET PATHS
# ============================================================================

print("\n🔍 Auto-detecting dataset paths...")

feature_dir = None
raw_dir = None

# 1. Search for PROCESSED FEATURES
target_feature_file = 'lightcurve_features_train_corrected.csv'
for root_loc in [KAGGLE_INPUT, '.']:
    if os.path.exists(root_loc):
        for root, dirs, files in os.walk(root_loc):
            if target_feature_file in files:
                feature_dir = root
                print(f"   ✅ Found PROCESSED FEATURES in: {feature_dir}")
                break
    if feature_dir: break

# 2. Search for RAW DATA (train_log.csv + train_full_lightcurves.csv)
search_log_file = 'train_log.csv'
for root_loc in [KAGGLE_INPUT, '.']:
    if os.path.exists(root_loc):
        for root, dirs, files in os.walk(root_loc):
            if search_log_file in files:
                raw_dir = root
                print(f"   ✅ Found RAW DATA in: {raw_dir}")
                break
    if raw_dir: break

if not feature_dir and not raw_dir:
    print("\n❌ FATAL: Could not find ANY data (neither processed features nor raw dataset).")
    sys.exit(1)

# ============================================================================
# PART 2: FEATURE EXTRACTION LOGIC (The Fallback)
# ============================================================================

def fit_tde_shape(t, f, f_peak, t_peak):
    mask = (t > t_peak) & (f > 0.05 * f_peak)
    if np.sum(mask) < 3: return -1.0
    t_dec, f_dec = t[mask], f[mask]
    func = lambda ti, t0: f_peak * np.power(np.maximum((ti - t0) / (t_peak - t0), 1e-6), -5/3)
    try:
        popt, _ = curve_fit(func, t_dec, f_dec, p0=[t_peak - 20], bounds=([t_peak - 500], [t_peak - 0.5]), maxfev=100)
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
    
    f_mean, f_std = np.mean(flux), np.std(flux)
    f_max, f_min = np.max(flux), np.min(flux)
    
    # Basic Stats
    f[f"{filt}_mean"] = f_mean
    f[f"{filt}_std"] = f_std
    f[f"{filt}_max"] = f_max
    f[f"{filt}_skew"] = skew(flux)
    f[f"{filt}_kurt"] = kurtosis(flux)
    f[f"{filt}_amp"] = (f_max - f_min) / (f_mean + 1e-9)
    
    # Time
    t_peak = time[np.argmax(flux)]
    f[f"{filt}_time_to_peak"] = t_peak - time.min()
    
    # FFT
    for i, val in enumerate(get_fft_features(flux)): f[f"{filt}_fft_{i}"] = val
    
    # TDE Fit
    if filt in ["g", "r", "u"]:
        f[f"{filt}_tde_chisq"] = fit_tde_shape(time, flux, f_max, t_peak)
        
    return f, t_peak, f_max

def process_raw_data(log_df, lc_paths):
    print(f"   -> Processing {len(lc_paths)} lightcurve files...")
    df_lc = pd.concat([pd.read_csv(f) for f in lc_paths], ignore_index=True)
    df = df_lc.merge(log_df[["object_id", "EBV", "Z"]], on="object_id", how="left").fillna(0)
    
    # Extinction Correction
    df["R"] = df["Filter"].map(EXTINCTION)
    df["Flux_Corr"] = df["Flux"] * (10 ** (0.4 * df["R"] * df["EBV"].clip(0, 2.0)))
    
    # Extraction
    features = []
    unique_ids = df['object_id'].unique()
    
    for idx, obj_id in enumerate(unique_ids):
        if idx % 500 == 0: print(f"      {idx}/{len(unique_ids)}")
        grp = df[df['object_id'] == obj_id].sort_values("Time (MJD)")
        z = grp["Z"].iloc[0]
        row = {"object_id": obj_id}
        
        band_stats = {}
        for filt in ["u", "g", "r", "i", "z", "y"]:
            g = grp[grp["Filter"] == filt]
            if len(g) < 3: continue
            feats, t_peak, f_max = extract_band_features(g["Time (MJD)"].values, g["Flux_Corr"].values, g["Flux_err"].values, filt, z)
            row.update(feats)
            if t_peak is not None: band_stats[filt] = {"t": t_peak, "f": f_max}
            
        # Cross band
        for b in ["u", "g", "r"]:
            if b in band_stats:
                chi = row.get(f"{b}_tde_chisq", -1)
                if 0 <= chi < 0.5: row["n_good_tde_fit"] = row.get("n_good_tde_fit", 0) + 1
        
        features.append(row)
    return pd.DataFrame(features).fillna(-999)

# ============================================================================
# PART 3: LOAD OR GENERATE DATA
# ============================================================================

print("\n📥 Loading Data...")

# TRAIN DATA
if feature_dir:
    print("   -> Loading pre-computed TRAIN features...")
    Xfull = pd.read_csv(os.path.join(feature_dir, 'lightcurve_features_train_corrected.csv'))
    trainlog = pd.read_csv(os.path.join(feature_dir, 'train_log.csv'))
else:
    print("   ⚠️ Pre-computed features missing. GENERATING FROM RAW TRAIN DATA...")
    trainlog = pd.read_csv(os.path.join(raw_dir, 'train_log.csv'))
    # Find all train lightcurve csvs
    train_lc_files = glob.glob(os.path.join(raw_dir, "**", "*train*lightcurves*.csv"), recursive=True)
    if not train_lc_files:
         # Fallback search
         train_lc_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if 'train' in f and 'lightcurve' in f]
    
    Xfull = process_raw_data(trainlog, train_lc_files)

# TEST DATA
if feature_dir:
    print("   -> Loading pre-computed TEST features...")
    Xtestfinal = pd.read_csv(os.path.join(feature_dir, 'lightcurve_features_test_corrected.csv'))
    testlog = pd.read_csv(os.path.join(feature_dir, 'test_log.csv'))
else:
    print("   ⚠️ Pre-computed features missing. GENERATING FROM RAW TEST DATA...")
    testlog = pd.read_csv(os.path.join(raw_dir, 'test_log.csv'))
    test_lc_files = glob.glob(os.path.join(raw_dir, "**", "*test*lightcurves*.csv"), recursive=True)
    if not test_lc_files:
         test_lc_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if 'test' in f and 'lightcurve' in f]
    
    Xtestfinal = process_raw_data(testlog, test_lc_files)

# Merge Logs
if 'object_id' in Xfull.columns: Xfull.rename(columns={'object_id': 'objectid'}, inplace=True)
if 'object_id' in Xtestfinal.columns: Xtestfinal.rename(columns={'object_id': 'objectid'}, inplace=True)
if 'object_id' in trainlog.columns: trainlog.rename(columns={'object_id': 'objectid'}, inplace=True)
if 'object_id' in testlog.columns: testlog.rename(columns={'object_id': 'objectid'}, inplace=True)

if 'target' not in Xfull.columns:
    Xfull = Xfull.merge(trainlog[['objectid', 'target']], on='objectid', how='left')
if Xtestfinal.shape[0] != testlog.shape[0]:
    Xtestfinal = testlog[['objectid']].merge(Xtestfinal, on='objectid', how='left')

print(f"   ✅ Train: {Xfull.shape}, Test: {Xtestfinal.shape}")
gc.collect()

# ============================================================================
# PART 4: AUGMENTATION (The Original Logic)
# ============================================================================

print("\n🔬 AUGMENTING FEATURES...")

def add_tde_derived_features(X):
    X_new = X.copy()
    eps = 1e-9
    has = lambda c: c in X.columns
    
    if has('g_tde_chisq'): X_new['g_tde_chisq'] = X['g_tde_chisq'] # Ensure preserved
    
    # 1. Colors & Basic Physics
    if has('g_mean') and has('r_mean'):
        X_new['color_g-r'] = X['g_mean'] - X['r_mean']
        c = X_new['color_g-r']
        X_new['blue_color_strength'] = (c - c.min()) / (c.max() - c.min() + eps)
        
    if has('g_max') and has('g_std'):
        X_new['peak_sharpness'] = X['g_max'] / (X['g_std'] + eps)
        
    if has('g_skew'):
        X_new['smoothness'] = 1.0 / (1.0 + np.abs(X['g_skew']))
        
    # [Add other augmentations here as per original script...]
    return X_new

Xfull = add_tde_derived_features(Xfull)
Xtestfinal = add_tde_derived_features(Xtestfinal)

# ============================================================================
# PART 5: SAVE
# ============================================================================

print("\n💾 SAVING...")
out_train = os.path.join(OUTPUT_DIR, 'lightcurve_features_improved_train.csv')
out_test = os.path.join(OUTPUT_DIR, 'lightcurve_features_improved_test.csv')
Xfull.to_csv(out_train, index=False)
Xtestfinal.to_csv(out_test, index=False)
print(f"✅ Saved to: {out_train}")
print(f"✅ Saved to: {out_test}")
print("Done.")

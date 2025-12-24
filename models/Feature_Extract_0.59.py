import pandas as pd
import numpy as np
import glob
import os
import gc
from scipy.stats import skew, kurtosis, linregress, entropy
from scipy.optimize import curve_fit
from scipy.fft import rfft
import warnings

warnings.filterwarnings('ignore')

# --- 1. TỰ ĐỘNG TÌM DỮ LIỆU ---
print("📂 Đang tìm dữ liệu...")
input_dirs = glob.glob('/kaggle/input/*')
BASE_PATH = ''
for path in input_dirs:
    if 'mallorn' in path.lower() and os.path.isdir(path):
        BASE_PATH = path
        break
if BASE_PATH == '' and len(input_dirs) > 0: BASE_PATH = input_dirs[0]

# --- 2. CÁC HÀM HỖ TRỢ (TDE FIT & FFT features) ---

def fit_tde_shape(t, f, f_peak, t_peak):
    """Fit nhanh mô hình TDE power-law"""
    mask = (t > t_peak) & (f > 0.05 * f_peak)
    if np.sum(mask) < 3: return -1.0
    t_dec = t[mask]
    f_dec = f[mask]
    
    def func(ti, t0):
        dt = ti - t0
        return f_peak * np.power(np.maximum(dt / (t_peak - t0), 1e-6), -5/3)

    try:
        popt, _ = curve_fit(func, t_dec, f_dec, p0=[t_peak - 20], 
                            bounds=([t_peak - 500], [t_peak - 0.5]), maxfev=100)
        # Trả về Chisq
        resid = (f_dec - func(t_dec, *popt))
        return np.sum(resid**2) / len(t_dec)
    except: return -1.0

def get_fft_features(flux):
    """Lấy features tần số (Fourier)"""
    if len(flux) < 5: return [0]*4
    f_fft = np.abs(rfft(flux))
    if len(f_fft) < 3: return [0]*4
    # Trả về biên độ của 4 tần số đầu tiên (bỏ DC component ở index 0)
    return list(f_fft[1:5]) if len(f_fft) >= 5 else list(f_fft[1:]) + [0]*(5-len(f_fft))

# --- 3. HÀM TRÍCH XUẤT "MASSIVE" ---
def extract_massive_features(time, flux, err, filt):
    f = {}
    n = len(flux)
    if n < 3: return {}

    # --- A. Basic Stats (15 features) ---
    f_max = np.max(flux)
    f_min = np.min(flux)
    f_mean = np.mean(flux)
    f_std = np.std(flux)
    
    f[f'{filt}_mean'] = f_mean
    f[f'{filt}_std'] = f_std
    f[f'{filt}_max'] = f_max
    f[f'{filt}_min'] = f_min
    f[f'{filt}_skew'] = skew(flux)
    f[f'{filt}_kurt'] = kurtosis(flux)
    f[f'{filt}_mad'] = np.median(np.abs(flux - np.median(flux)))
    f[f'{filt}_amp'] = (f_max - f_min) / (f_mean + 1e-9)
    f[f'{filt}_cv'] = f_std / (f_mean + 1e-9) # Coeff of Variation
    
    # --- B. Percentiles CHI TIẾT (15 features) ---
    # Lấy dày đặc để bắt hình dạng
    pcts = np.percentile(flux, [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    for i, p in zip([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], pcts):
        f[f'{filt}_p{i:02d}'] = p
    
    # Ratios giữa các percentile (Quan trọng cho Shape)
    f[f'{filt}_ratio_p95_p05'] = pcts[11] / (pcts[1] + 1e-9)
    f[f'{filt}_ratio_p95_median'] = pcts[11] / (pcts[6] + 1e-9)

    # --- C. FLUX DIFF & DERIVATIVES (10 features) ---
    diff = np.diff(flux)
    f[f'{filt}_diff_mean'] = np.mean(diff)
    f[f'{filt}_diff_std']  = np.std(diff)
    f[f'{filt}_diff_max']  = np.max(diff)
    f[f'{filt}_diff_min']  = np.min(diff)
    # Von Neumann ratio (độ mượt)
    f[f'{filt}_von_neumann'] = np.sum(diff**2) / (np.sum((flux - f_mean)**2) + 1e-9)

    # --- D. Cumulative stats STATS (Energy) (5 features) ---
    cum_flux = np.cumsum(np.abs(flux))
    f[f'{filt}_energy'] = cum_flux[-1]
    # Linear trend of cumulative flux
    try:
        slope_cum = linregress(np.arange(n), cum_flux)[0]
        f[f'{filt}_cum_slope'] = slope_cum
    except: f[f'{filt}_cum_slope'] = 0

    # --- E. TIME & SHAPE FEATURES (10 features) ---
    t_peak = time[np.argmax(flux)]
    f[f'{filt}_time_to_peak'] = t_peak - time.min()
    f[f'{filt}_time_from_peak'] = time.max() - t_peak
    
    # Slope Rise/Fall
    mask_rise = (time < t_peak)
    if np.sum(mask_rise) > 1:
        f[f'{filt}_slope_rise'] = linregress(time[mask_rise], flux[mask_rise])[0]
    else: f[f'{filt}_slope_rise'] = 0
        
    mask_fall = (time > t_peak)
    if np.sum(mask_fall) > 1:
        f[f'{filt}_slope_fall'] = linregress(time[mask_fall], flux[mask_fall])[0]
    else: f[f'{filt}_slope_fall'] = 0

    # --- F. FFT features COMPONENTS (4 features) ---
    fft_vals = get_fft_features(flux)
    for i, val in enumerate(fft_vals):
        f[f'{filt}_fft_{i}'] = val

    # --- G. TDE SPECIFIC FIT (1 feature) ---
    if filt in ['g', 'r', 'u']:
        f[f'{filt}_tde_chisq'] = fit_tde_shape(time, flux, f_max, t_peak)

    return f

# --- 4. PIPELINE ---
def process_pipeline_massive(df_lc, df_log):
    print("   -> Bắt đầu trích xuất MASSIVE features...")
    
    # De-extinction
    EXTINCTION = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}
    df = df_lc.merge(df_log[['object_id', 'EBV']], on='object_id', how='left').fillna(0)
    df['R'] = df['Filter'].map(EXTINCTION)
    df['Flux_Corr'] = df['Flux'] * (10 ** (0.4 * df['R'] * df['EBV']))
    
    features = []
    
    for obj_id, grp in df.groupby('object_id'):
        row = {'object_id': obj_id}
        grp = grp.sort_values('Time (MJD)')
        
        band_stats = {} # Lưu tạm để tính Cross-band
        
        # 1. PER-FILTER EXTRACTION
        for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
            filt_dat = grp[grp['Filter'] == filt]
            if len(filt_dat) < 3: continue
            
            t = filt_dat['Time (MJD)'].values
            fl = filt_dat['Flux_Corr'].values
            er = filt_dat['Flux_err'].values
            
            # Extract ~60 features per band
            feats = extract_massive_features(t, fl, er, filt)
            row.update(feats)
            
            # Lưu các chỉ số chính để tính color
            if f'{filt}_mean' in feats:
                band_stats[filt] = {
                    'mean': feats[f'{filt}_mean'],
                    'max': feats[f'{filt}_max'],
                    'peak_time': feats[f'{filt}_time_to_peak'] + t.min()
                }

        # 2. CROSS-BAND (COLORS) (~15 cặp x 4 tính chất = 60 features)
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                b1 = bands[i]
                b2 = bands[j]
                if b1 in band_stats and b2 in band_stats:
                    # Delta Mean (Color)
                    row[f'{b1}_{b2}_mean_diff'] = band_stats[b1]['mean'] - band_stats[b2]['mean']
                    # Delta Max
                    row[f'{b1}_{b2}_max_diff'] = band_stats[b1]['max'] - band_stats[b2]['max']
                    # Delta Ratio
                    row[f'{b1}_{b2}_max_ratio'] = band_stats[b1]['max'] / (band_stats[b2]['max'] + 1e-9)
                    # Time Lag (Quan trọng cho TDE vs SN)
                    row[f'{b1}_{b2}_lag'] = band_stats[b1]['peak_time'] - band_stats[b2]['peak_time']

        features.append(row)
        
    df_feats = pd.DataFrame(features)
    return df_feats.fillna(-999) # Fill missing

# --- CHẠY CHƯƠNG TRÌNH ---
print("📥 Loading Data...")
train_files = glob.glob(os.path.join(BASE_PATH, '**', 'train_full_lightcurves.csv'), recursive=True)
test_files = glob.glob(os.path.join(BASE_PATH, '**', 'test_full_lightcurves.csv'), recursive=True)
train_log = pd.read_csv(os.path.join(BASE_PATH, 'train_log.csv'))
test_log = pd.read_csv(os.path.join(BASE_PATH, 'test_log.csv'))

train_lc = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True).dropna(subset=['Flux'])
test_lc = pd.concat((pd.read_csv(f) for f in test_files), ignore_index=True).dropna(subset=['Flux'])

print("⚙️ Generatine Massive Train Features...")
train_feats = process_pipeline_massive(train_lc, train_log)
print("⚙️ Generatine Massive Test Features...")
test_feats = process_pipeline_massive(test_lc, test_log)

print("🔗 Merging...")
X_full = train_log.merge(train_feats, on='object_id', how='left').fillna(-999)
X_test_final = test_log.merge(test_feats, on='object_id', how='left').fillna(-999)

# Đồng bộ cột
cols = [c for c in X_full.columns if c not in ['object_id', 'target', 'split', 'SpecType', 'English Translation'] 
        and pd.api.types.is_numeric_dtype(X_full[c])]

for c in cols:
    if c not in X_test_final.columns: X_test_final[c] = -999

print(f"✅ XONG. Tổng số features tạo ra: {len(cols)}")
del train_lc, test_lc
gc.collect()
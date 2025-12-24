import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import gc
import warnings
from scipy.stats import skew, kurtosis, linregress
from scipy.optimize import curve_fit
from scipy.fft import rfft
from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = 'report_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Font tiếng Việt (fallback)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma', 'Verdana', 'sans-serif']
plt.style.use('ggplot')
sns.set_palette("husl")

FEATURE_FILE = 'friend_features_train.csv'
TOP_K_FEATURES = 140
SCALE_POS_BASE = 1.5

# =========================================================
# PART 1: FEATURE EXTRACTION LOGIC (From file1.py)
# =========================================================

def fit_tde_shape(t, f, f_peak, t_peak):
    """Chi-squared fit cho TDE decay ~ (t - t0)^(-5/3)."""
    mask = (t > t_peak) & (f > 0.05 * f_peak)
    if np.sum(mask) < 3:
        return -1.0
    t_dec = t[mask]
    f_dec = f[mask]

    def func(ti, t0):
        dt = ti - t0
        return f_peak * np.power(np.maximum(dt / (t_peak - t0), 1e-6), -5 / 3)

    try:
        popt, _ = curve_fit(
            func,
            t_dec,
            f_dec,
            p0=[t_peak - 20],
            bounds=([t_peak - 500], [t_peak - 0.5]),
            maxfev=200,
        )
        resid = f_dec - func(t_dec, *popt)
        return np.sum(resid ** 2) / len(t_dec)
    except Exception:
        return -1.0

def get_fft_features(flux):
    if len(flux) < 5:
        return [0.0] * 4
    f_fft = np.abs(rfft(flux))
    if len(f_fft) < 3:
        return [0.0] * 4
    vals = list(f_fft[1:5]) if len(f_fft) >= 5 else list(f_fft[1:])
    if len(vals) < 4:
        vals += [0.0] * (4 - len(vals))
    return vals

def extract_band_features(time, flux, err, filt, z):
    f = {}
    n = len(flux)
    if n < 3:
        return f, None, None

    # ---- BASIC STATS ----
    f_max = np.max(flux)
    f_min = np.min(flux)
    f_mean = np.mean(flux)
    f_std = np.std(flux)

    f[f"{filt}_mean"] = f_mean
    f[f"{filt}_std"] = f_std
    f[f"{filt}_max"] = f_max
    f[f"{filt}_min"] = f_min
    f[f"{filt}_skew"] = skew(flux)
    f[f"{filt}_kurt"] = kurtosis(flux)
    f[f"{filt}_mad"] = np.median(np.abs(flux - np.median(flux)))
    f[f"{filt}_amp"] = (f_max - f_min) / (f_mean + 1e-9)
    f[f"{filt}_cv"] = f_std / (f_mean + 1e-9)

    # ---- PERCENTILES ----
    pcts = np.percentile(flux, [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    for i, p in zip([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], pcts):
        f[f"{filt}_p{i:02d}"] = p
    f[f"{filt}_ratio_p95_p05"] = pcts[11] / (pcts[1] + 1e-9)
    f[f"{filt}_ratio_p95_median"] = pcts[11] / (pcts[6] + 1e-9)

    # ---- DIFF / VON NEUMANN ----
    diff = np.diff(flux)
    f[f"{filt}_diff_mean"] = np.mean(diff)
    f[f"{filt}_diff_std"] = np.std(diff)
    f[f"{filt}_diff_max"] = np.max(diff)
    f[f"{filt}_diff_min"] = np.min(diff)
    f[f"{filt}_von_neumann"] = np.sum(diff ** 2) / (np.sum((flux - f_mean) ** 2) + 1e-9)

    # ---- CUMULATIVE ----
    cum_flux = np.cumsum(np.abs(flux))
    f[f"{filt}_energy"] = cum_flux[-1]
    try:
        slope_cum = linregress(np.arange(n), cum_flux)[0]
        f[f"{filt}_cum_slope"] = slope_cum
    except Exception:
        f[f"{filt}_cum_slope"] = 0.0

    # ---- TIME SHAPE (OBS + REST) ----
    t_peak = time[np.argmax(flux)]
    t_min = time.min()
    t_max = time.max()
    f[f"{filt}_time_to_peak"] = t_peak - t_min
    f[f"{filt}_time_from_peak"] = t_max - t_peak

    z_eff = max(float(z), 0.0)
    t_rest = (time - t_peak) / (1.0 + z_eff)
    mask_rise = t_rest < 0
    mask_fall = t_rest > 0

    if np.sum(mask_rise) > 1:
        f[f"{filt}_slope_rise_rest"] = linregress(t_rest[mask_rise], flux[mask_rise])[0]
    else:
        f[f"{filt}_slope_rise_rest"] = 0.0
    if np.sum(mask_fall) > 1:
        f[f"{filt}_slope_fall_rest"] = linregress(t_rest[mask_fall], flux[mask_fall])[0]
    else:
        f[f"{filt}_slope_fall_rest"] = 0.0

    thr = 0.25 * f_max
    above = time[flux >= thr]
    if len(above) >= 2:
        f[f"{filt}_dur_025"] = (above.max() - above.min()) / (1.0 + z_eff)
    else:
        f[f"{filt}_dur_025"] = 0.0

    # ---- LOG-DECAY SLOPE ----
    log_flux = np.log10(np.maximum(flux, 1e-6))
    if np.sum(mask_fall) > 2:
        f[f"{filt}_logdecay_slope"] = linregress(time[mask_fall], log_flux[mask_fall])[0]
    else:
        f[f"{filt}_logdecay_slope"] = 0.0

    # ---- POLYFIT BẬC 2 ----
    t_norm = (time - time.mean()) / (1.0 + z_eff)
    if len(t_norm) >= 6:
        c0, c1, c2 = polyfit(t_norm, flux, 2)
        f[f"{filt}_poly2_c0"] = c0
        f[f"{filt}_poly2_c1"] = c1
        f[f"{filt}_poly2_c2"] = c2

    # ---- FFT ----
    fft_vals = get_fft_features(flux)
    for i, val in enumerate(fft_vals):
        f[f"{filt}_fft_{i}"] = val

    # ---- SNR ----
    snr = flux / (err + 1e-9)
    f[f"{filt}_snr_mean"] = np.mean(snr)
    f[f"{filt}_snr_max"] = np.max(snr)
    f[f"{filt}_snr_p90"] = np.percentile(snr, 90)
    f[f"{filt}_n_snr_gt5"] = np.sum(snr > 5)

    # ---- TDE POWER-LAW ----
    if filt in ["g", "r", "u"]:
        f[f"{filt}_tde_chisq"] = fit_tde_shape(time, flux, f_max, t_peak)

    return f, t_peak, f_max

def process_pipeline(df_lc, df_log):
    print(" -> Processing Lightcurves...")
    EXTINCTION = {"u": 4.81, "g": 3.64, "r": 2.70, "i": 2.06, "z": 1.58, "y": 1.31}

    df = df_lc.merge(df_log[["object_id", "EBV", "Z"]], on="object_id", how="left").fillna(0)
    df["R"] = df["Filter"].map(EXTINCTION)
    df["Flux_Corr"] = df["Flux"] * (10 ** (0.4 * df["R"] * df["EBV"].clip(0, 2.0)))

    features = []
    # Loop qua từng object
    # Lưu ý: trên data lớn, nên dùng multiprocessing. Ở đây demo sequential.
    unique_ids = df['object_id'].unique()
    total = len(unique_ids)
    
    for idx, obj_id in enumerate(unique_ids):
        if idx % 500 == 0:
            print(f"    ... {idx}/{total} objects")
            
        grp = df[df['object_id'] == obj_id]
        row = {"object_id": obj_id}
        grp = grp.sort_values("Time (MJD)")
        z = float(grp["Z"].iloc[0])

        band_stats = {}
        peak_info = {}

        for filt in ["u", "g", "r", "i", "z", "y"]:
            g = grp[grp["Filter"] == filt]
            if len(g) < 3:
                continue
            t = g["Time (MJD)"].values
            fl = g["Flux_Corr"].values
            er = g["Flux_err"].values
            feats, t_peak, f_peak = extract_band_features(t, fl, er, filt, z)
            row.update(feats)
            if t_peak is not None:
                band_stats[filt] = {
                    "mean": feats.get(f"{filt}_mean", 0.0),
                    "max": feats.get(f"{filt}_max", 0.0),
                    "peak_time": t_peak,
                }
                peak_info[filt] = (t_peak, f_peak)

        # CROSS-BAND
        bands = ["u", "g", "r", "i", "z", "y"]
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                b1, b2 = bands[i], bands[j]
                if b1 in band_stats and b2 in band_stats:
                    row[f"{b1}_{b2}_mean_diff"] = band_stats[b1]["mean"] - band_stats[b2]["mean"]
                    row[f"{b1}_{b2}_max_diff"] = band_stats[b1]["max"] - band_stats[b2]["max"]
                    row[f"{b1}_{b2}_max_ratio"] = band_stats[b1]["max"] / (
                        band_stats[b2]["max"] + 1e-9
                    )
                    row[f"{b1}_{b2}_lag"] = band_stats[b1]["peak_time"] - band_stats[b2]["peak_time"]

        # Good TDE Count
        good_tde_fit = 0
        for flt in ["g", "r", "u"]:
            chi = row.get(f"{flt}_tde_chisq", -1.0)
            if 0 <= chi < 0.5:
                good_tde_fit += 1
        row["n_good_tde_fit"] = good_tde_fit

        row["baseline_rest"] = ((grp["Time (MJD)"].max() - grp["Time (MJD)"].min()) / (1.0 + max(z, 0.0)))
        row["global_snr_max"] = (grp["Flux_Corr"] / (grp["Flux_err"] + 1e-9)).max()
        
        if "r" in peak_info:
            t_peak_r, _ = peak_info["r"]
            row["frac_after_peak_r"] = (grp["Time (MJD)"] > t_peak_r).mean()
        else:
            row["frac_after_peak_r"] = 0.0

        row["n_obs"] = len(grp)
        row["n_filters_used"] = grp["Filter"].nunique()
        row["frac_neg_flux"] = (grp["Flux_Corr"] < 0).mean()

        features.append(row)

    df_feats = pd.DataFrame(features)
    return df_feats.fillna(-999)

def find_and_load_data():
    """Tự động tìm file features hoặc raw data."""
    # 1. Tìm file đã xử lý
    possible_paths = [
        FEATURE_FILE,
        os.path.join('/kaggle/input', FEATURE_FILE), 
        # Add recursive search later if needed
    ]
    # Recursive search in /kaggle/input
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if file == FEATURE_FILE:
                possible_paths.append(os.path.join(root, file))
    
    for p in possible_paths:
        if os.path.exists(p):
            print(f"✅ Found feature file: {p}")
            return pd.read_csv(p)
            
    print("⚠️ Pre-computed features NOT FOUND. Attempting to generate from RAW data...")
    
    # 2. Tìm Raw Data (competition dataset)
    # Giả định cấu trúc thư mục Kaggle
    raw_dirs = glob.glob("/kaggle/input/*")
    base_path = ""
    for path in raw_dirs:
        # Tìm folder chứa train_log.csv
        if os.path.exists(os.path.join(path, "train_log.csv")):
            base_path = path
            break
            
    if not base_path:
        # Thử tìm đệ quy
        for root, dirs, files in os.walk('/kaggle/input'):
            if "train_log.csv" in files:
                base_path = root
                break
                
    if not base_path:
        raise FileNotFoundError("❌ Cannot find 'train_log.csv' in /kaggle/input. Please attach the competition dataset!")
        
    print(f"📂 Found Raw Data at: {base_path}")
    
    # Load Raw
    train_log = pd.read_csv(os.path.join(base_path, "train_log.csv"))
    # Load lightcurves (có thể chia thành nhiều file hoặc 1 file)
    lc_files = glob.glob(os.path.join(base_path, "**", "train_full_lightcurves.csv"), recursive=True)
    # Nếu không tìm thấy file cụ thể, thử tìm file .csv lớn
    if not lc_files:
         lc_files = glob.glob(os.path.join(base_path, "*.csv"))
         # Filter logic if needed
    
    if not lc_files:
         raise FileNotFoundError("❌ Found log but no lightcurve CSVs!")
         
    print(f"   Loading {len(lc_files)} lightcurve files...")
    train_lc = pd.concat([pd.read_csv(f) for f in lc_files], ignore_index=True).dropna(subset=["Flux"])
    
    # Run Pipeline
    print("⚙️ Running Feature Extraction Pipeline (this may take a few minutes)...")
    train_feats = process_pipeline(train_lc, train_log)
    
    # Merge
    print("🔗 Merging...")
    X_full = train_log.merge(train_feats, on="object_id", how="left").fillna(-999)
    
    # Save for future
    print(f"💾 Saving {FEATURE_FILE} for future use...")
    X_full.to_csv(FEATURE_FILE, index=False)
    
    return X_full

def save_plot(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot: {path}")

# =========================================================
# MAIN EXECUTION
# =========================================================

# 1. LOAD / GENERATE DATA
try:
    df_full = find_and_load_data()
    feature_cols = [c for c in df_full.columns if c not in ['object_id', 'target', 'split', 'SpecType', 'English Translation'] and pd.api.types.is_numeric_dtype(df_full[c])]
    X = df_full[feature_cols]
    y = df_full['target']
    print(f"📊 Data Loaded. Shape: {X.shape}. Class Balance: {y.value_counts().to_dict()}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("👉 HINT: Make sure you have added the Competition Dataset to this Notebook!")
    exit(1)

# 2. CLASS DISTRIBUTION PLOT
plt.figure(figsize=(6, 6))
counts = y.value_counts()
labels = [f'Others ({counts.get(0, 0)})', f'TDE ({counts.get(1, 0)})']
colors = ['#3498db', '#e74c3c'] if 1 in counts else ['#3498db']
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=[0.1]*len(counts))
plt.title('Tỷ lệ mẫu TDE vs Các sự kiện khác', fontsize=14)
save_plot('dist_class.png')

# 3. FEATURE SELECTION & TRAINING (LightGBM)
print("\n🤖 Training LightGBM for Visualization...")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Imbalance handling
n_pos = sum(y_train == 1)
n_neg = sum(y_train == 0)
scale_pos_weight = (n_neg / n_pos) * SCALE_POS_BASE if n_pos > 0 else 1.0
print(f"   -> scale_pos_weight: {scale_pos_weight:.2f}")

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=63,
    objective='binary',
    scale_pos_weight=scale_pos_weight,
    importance_type='gain',
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(100)])

# 4. PLOT FEATURE IMPORTANCE
imp = pd.DataFrame({'feature': feature_cols, 'gain': model.feature_importances_})
imp = imp.sort_values('gain', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='gain', y='feature', data=imp, palette='viridis')
plt.title('Top 20 Feature Importance (LightGBM Gain)', fontsize=14)
plt.xlabel('Gain Value')
plt.ylabel('Feature Name')
save_plot('ranking_features.png')

# 5. METRICS & CURVES
y_prob = model.predict_proba(X_val)[:, 1]

# Find best threshold
best_f1 = 0
best_thresh = 0.5
for t in np.arange(0.1, 0.9, 0.02):
    y_pred_t = (y_prob >= t).astype(int)
    score = f1_score(y_val, y_pred_t)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"\n🏆 Best Valid F1: {best_f1:.4f} @ Threshold: {best_thresh:.2f}")
y_pred_final = (y_prob >= best_thresh).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred Other', 'Pred TDE'],
            yticklabels=['True Other', 'True TDE'])
plt.title(f'Confusion Matrix (Thresh={best_thresh:.2f})', fontsize=14)
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
save_plot('matrix_confusion.png')

# ROC & PR Curves
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_val, y_prob)
pr_auc = auc(rec, prec)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax1.set_title('ROC Curve')
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.legend(loc="lower right")

ax2.plot(rec, prec, color='green', lw=2, label=f'AUC = {pr_auc:.2f}')
ax2.set_title('Precision-Recall Curve')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend(loc="lower left")
save_plot('curve_roc_pr.png')

# 6. FEATURE DISTRIBUTIONS
# Plot distribution of top 4 features
top_feats = imp['feature'].head(4).tolist()
plt.figure(figsize=(12, 8))
for i, f in enumerate(top_feats):
    plt.subplot(2, 2, i+1)
    # Clip outliers for better viz
    low, high = np.percentile(X[f], [1, 99])
    
    sns.kdeplot(X.loc[y==0, f].clip(low, high), shade=True, color='blue', label='Others', alpha=0.3)
    sns.kdeplot(X.loc[y==1, f].clip(low, high), shade=True, color='red', label='TDE', alpha=0.3)
    plt.title(f)
    plt.legend()
save_plot('dist_features.png')

print("\n✅ DONE. All plots saved to 'report_images/'")

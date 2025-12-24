import pandas as pd
import numpy as np
import glob, os, gc

from scipy.stats import skew, kurtosis, linregress
from scipy.optimize import curve_fit
from scipy.fft import rfft
from numpy.polynomial.polynomial import polyfit
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. TÌM ĐƯỜNG DỮ LIỆU =====================

print("📂 Đang tìm dữ liệu MALLORN...")
input_dirs = glob.glob("/kaggle/input/*")
BASE_PATH = ""
for path in input_dirs:
    if "mallorn" in path.lower() and os.path.isdir(path):
        BASE_PATH = path
        break
if BASE_PATH == "" and len(input_dirs) > 0:
    BASE_PATH = input_dirs[0]
print("BASE_PATH:", BASE_PATH)

# ===================== 2. HỖ TRỢ: TDE FIT + BIẾN ĐỔI FOURIER =====================

def fit_tde_shape(t, f, f_peak, t_peak):
    """Khớp Chi-bình phương cho phân rã TDE ~ (t - t0)^(-5/3)."""
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

# ===================== 3. HÀM TRÍCH XUẤT FEATURE =====================

def extract_band_features(time, flux, err, filt, z):
    f = {}
    n = len(flux)
    if n < 3:
        return f, None, None

    # ---- THỐNG KÊ CƠ BẢN ----
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

    # ---- CÁC PHÂN VỊ ----
    pcts = np.percentile(flux, [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    for i, p in zip([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], pcts):
        f[f"{filt}_p{i:02d}"] = p
    f[f"{filt}_ratio_p95_p05"] = pcts[11] / (pcts[1] + 1e-9)
    f[f"{filt}_ratio_p95_median"] = pcts[11] / (pcts[6] + 1e-9)

    # ---- SAI PHÂN / VON NEUMANN ----
    diff = np.diff(flux)
    f[f"{filt}_diff_mean"] = np.mean(diff)
    f[f"{filt}_diff_std"] = np.std(diff)
    f[f"{filt}_diff_max"] = np.max(diff)
    f[f"{filt}_diff_min"] = np.min(diff)
    f[f"{filt}_von_neumann"] = np.sum(diff ** 2) / (np.sum((flux - f_mean) ** 2) + 1e-9)

    # ---- TÍCH LŨY ----
    cum_flux = np.cumsum(np.abs(flux))
    f[f"{filt}_energy"] = cum_flux[-1]
    try:
        slope_cum = linregress(np.arange(n), cum_flux)[0]
        f[f"{filt}_cum_slope"] = slope_cum
    except Exception:
        f[f"{filt}_cum_slope"] = 0.0

    # ---- DẠNG THỜI GIAN (OBS + REST) ----
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

    # ---- ĐỘ DỐC LOG-DECAY ----
    log_flux = np.log10(np.maximum(flux, 1e-6))
    if np.sum(mask_fall) > 2:
        f[f"{filt}_logdecay_slope"] = linregress(time[mask_fall], log_flux[mask_fall])[0]
    else:
        f[f"{filt}_logdecay_slope"] = 0.0

    # ---- ĐA THỨC BẬC 2 ----
    t_norm = (time - time.mean()) / (1.0 + z_eff)
    if len(t_norm) >= 6:
        c0, c1, c2 = polyfit(t_norm, flux, 2)
        f[f"{filt}_poly2_c0"] = c0
        f[f"{filt}_poly2_c1"] = c1
        f[f"{filt}_poly2_c2"] = c2

    # ---- BIẾN ĐỔI FOURIER ----
    fft_vals = get_fft_features(flux)
    for i, val in enumerate(fft_vals):
        f[f"{filt}_fft_{i}"] = val

    # ---- TỈ SỐ TÍN HIỆU TRÊN NHIỄU ----
    snr = flux / (err + 1e-9)
    f[f"{filt}_snr_mean"] = np.mean(snr)
    f[f"{filt}_snr_max"] = np.max(snr)
    f[f"{filt}_snr_p90"] = np.percentile(snr, 90)
    f[f"{filt}_n_snr_gt5"] = np.sum(snr > 5)

    # ---- LUẬT MŨ TDE ----
    if filt in ["g", "r", "u"]:
        f[f"{filt}_tde_chisq"] = fit_tde_shape(time, flux, f_max, t_peak)

    return f, t_peak, f_max

# ===================== 4. QUY TRÌNH XỬ LÝ TỪNG VẬT THỂ =====================

def process_pipeline(df_lc, df_log):
    print(" -> Trích xuất features...")
    EXTINCTION = {"u": 4.81, "g": 3.64, "r": 2.70, "i": 2.06, "z": 1.58, "y": 1.31}

    df = df_lc.gộp(df_log[["object_id", "EBV", "Z"]], on="object_id", how="left").fillna(0)
    df["R"] = df["Filter"].map(EXTINCTION)
    df["Flux_Corr"] = df["Flux"] * (10 ** (0.4 * df["R"] * df["EBV"].clip(0, 2.0)))

    features = []
    for obj_id, grp in df.groupby("object_id"):
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

        # CROSS-BAND COLORS + TIME LAG
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

        # Đếm số band fit TDE tốt
        good_tde_fit = 0
        for flt in ["g", "r", "u"]:
            chi = row.get(f"{flt}_tde_chisq", -1.0)
            if 0 <= chi < 0.5:
                good_tde_fit += 1
        row["n_good_tde_fit"] = good_tde_fit

        # Global baseline & TỈ SỐ TÍN HIỆU TRÊN NHIỄU max
        row["baseline_rest"] = (
            (grp["Time (MJD)"].max() - grp["Time (MJD)"].min()) / (1.0 + max(z, 0.0))
        )
        row["global_snr_max"] = (grp["Flux_Corr"] / (grp["Flux_err"] + 1e-9)).max()

        # frac after peak r (đuôi dài)
        if "r" in peak_info:
            t_peak_r, _ = peak_info["r"]
            row["frac_after_peak_r"] = (grp["Time (MJD)"] > t_peak_r).mean()
        else:
            row["frac_after_peak_r"] = 0.0

        # Global counts
        row["n_obs"] = len(grp)
        row["n_filters_used"] = grp["Filter"].nunique()
        row["frac_neg_flux"] = (grp["Flux_Corr"] < 0).mean()

        features.append(row)

    df_feats = pd.DataFrame(features)
    return df_feats.fillna(-999)

# ===================== 5. THỰC HIỆN TRÍCH XUẤT =====================

print("📥 Đang tải lightcurves...")
train_files = glob.glob(os.path.join(BASE_PATH, "**", "train_full_lightcurves.csv"), recursive=True)
test_files = glob.glob(os.path.join(BASE_PATH, "**", "test_full_lightcurves.csv"), recursive=True)

train_log = pd.read_csv(os.path.join(BASE_PATH, "train_log.csv"))
test_log = pd.read_csv(os.path.join(BASE_PATH, "test_log.csv"))

train_lc = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True).dropna(subset=["Flux"])
test_lc = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True).dropna(subset=["Flux"])

print("⚙️ Đang tạo đặc trưng tập Train...")
train_feats = process_pipeline(train_lc, train_log)
print("⚙️ Đang tạo đặc trưng tập Test...")
test_feats = process_pipeline(test_lc, test_log)

print("🔗 Đang gộp thông tin log...")
X_full = train_log.gộp(train_feats, on="object_id", how="left").fillna(-999)
X_test_final = test_log.gộp(test_feats, on="object_id", how="left").fillna(-999)

cols = [
    c for c in X_full.columns
    if c not in ["object_id", "target", "split", "SpecType", "English Translation"]
    and pd.api.types.is_numeric_dtype(X_full[c])
]
for c in cols:
    if c not in X_test_final.columns:
        X_test_final[c] = -999

print(f"✅ XONG. Tổng số features: {len(cols)}")

del train_lc, test_lc
gc.collect()

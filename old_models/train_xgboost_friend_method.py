import pandas as pd
import numpy as np
import glob
import os
import gc
from scipy.stats import skew, kurtosis, linregress, pearsonr
from scipy.optimize import curve_fit

# --- 1. SETUP ---
print("📂 Đang tìm dữ liệu...")
# Modified for local environment
BASE_PATH = '.' 

# --- 2. HÀM FIT TDE ---
def get_tde_chisq(time, flux, err):
    if len(flux) < 5: return 999.0
    try:
        idx_max = np.argmax(flux)
        t_peak = time[idx_max]
        f_peak = flux[idx_max]
        mask = (time > t_peak) & (flux > 0.05 * f_peak) # Lấy đuôi dài hơn chút
        if len(time[mask]) < 3: return 999.0
        
        def func(t, t0): return f_peak * np.power((t - t0)/(t_peak - t0), -5/3)
        popt, _ = curve_fit(func, time[mask], flux[mask], p0=[t_peak-20], 
                            bounds=([t_peak-500], [t_peak-0.1]), maxfev=100)
        return np.mean(((flux[mask] - func(time[mask], *popt)) / err[mask])**2)
    except: return 999.0

# --- 3. HÀM TÍNH FEATURE CHI TIẾT ---
def get_full_stats(time, flux, err, filt):
    f = {}
    n = len(flux)
    if n < 2: return {}
    
    # --- A. Weighted Stats ---
    w = 1.0 / (err**2 + 1e-9)
    w_sum = np.sum(w)
    w_mean = np.sum(flux * w) / w_sum
    w_std = np.sqrt(np.sum(w * (flux - w_mean)**2) / w_sum)
    f[f'{filt}_w_mean'] = w_mean
    f[f'{filt}_w_std'] = w_std
    
    # --- B. Percentiles & Distribution (Rất chi tiết) ---
    # Quét từ 0% đến 100% với bước nhảy nhỏ
    pcts = np.percentile(flux, [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    f[f'{filt}_min'] = pcts[0]
    f[f'{filt}_p05'] = pcts[1]
    f[f'{filt}_p10'] = pcts[2]
    f[f'{filt}_p20'] = pcts[3]
    f[f'{filt}_p30'] = pcts[4]
    f[f'{filt}_p40'] = pcts[5]
    f[f'{filt}_median'] = pcts[6]
    f[f'{filt}_p60'] = pcts[7]
    f[f'{filt}_p70'] = pcts[8]
    f[f'{filt}_p80'] = pcts[9]
    f[f'{filt}_p90'] = pcts[10]
    f[f'{filt}_p95'] = pcts[11]
    f[f'{filt}_max'] = pcts[12]
    
    f[f'{filt}_skew'] = skew(flux)
    f[f'{filt}_kurt'] = kurtosis(flux)
    
    # --- C. Ratios & Amplitudes ---
    # So sánh đỉnh với nền
    f[f'{filt}_amp_high'] = (pcts[11] - pcts[1]) / (np.abs(pcts[6]) + 1e-6) # (p95-p05)/med
    f[f'{filt}_amp_low']  = (pcts[9] - pcts[3]) / (np.abs(pcts[6]) + 1e-6)  # (p80-p20)/med
    f[f'{filt}_flux_ratio_1'] = pcts[12] / (pcts[6] + 1e-6) # Max / Median
    f[f'{filt}_flux_ratio_2'] = (pcts[12] - pcts[6]) / (pcts[6] - pcts[0] + 1e-6) # Skewness proxy
    
    # --- D. Time Series Structure ---
    # TDE Fit
    f[f'{filt}_tde_chisq'] = get_tde_chisq(time, flux, err)
    
    # Rise / Fall
    idx_max = np.argmax(flux)
    t_peak = time[idx_max]
    t_rise = t_peak - time.min()
    t_fall = time.max() - t_peak
    f[f'{filt}_rise_time'] = t_rise
    f[f'{filt}_fall_time'] = t_fall
    f[f'{filt}_rise_fall_ratio'] = t_rise / (t_fall + 1e-6)
    
    # Slopes
    if idx_max > 0:
        f[f'{filt}_rise_slope'] = (flux[idx_max] - flux[0]) / (t_rise + 1e-6)
    else: f[f'{filt}_rise_slope'] = 0
    
    if idx_max < n - 1:
        f[f'{filt}_fall_slope'] = (flux[idx_max] - flux[-1]) / (t_fall + 1e-6)
    else: f[f'{filt}_fall_slope'] = 0
    
    # Variability check
    f[f'{filt}_mad'] = np.median(np.abs(flux - pcts[6]))
    f[f'{filt}_beyond_1std'] = np.sum(np.abs(flux - w_mean) > w_std) / n
    f[f'{filt}_stetson_k'] = (1/np.sqrt(n)) * np.sum(np.abs((flux-w_mean)/err)) / np.sqrt(np.mean(((flux-w_mean)/err)**2))

    return f

def process_pipeline(df_lc, df_log):
    print("   -> Đang tạo ~500 features...")
    
    # De-extinction
    EXTINCTION = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}
    df = df_lc.merge(df_log[['object_id', 'EBV']], on='object_id', how='left').fillna(0)
    df['R'] = df['Filter'].map(EXTINCTION)
    factor = 10 ** (0.4 * df['R'] * df['EBV'])
    df['Flux'] = df['Flux'] * factor
    df['Flux_err'] = df['Flux_err'] * factor
    
    features = []
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    
    for obj, grp in df.groupby('object_id'):
        f = {'object_id': obj}
        grp = grp.sort_values('Time (MJD)')
        
        band_stats = {filt: {} for filt in filters}
        band_fluxes = {}
        
        # 1. PER-BAND FEATURES
        for filt in filters:
            d = grp[grp['Filter'] == filt]
            if len(d) < 3: continue
            
            t = d['Time (MJD)'].values
            fl = d['Flux'].values
            er = d['Flux_err'].values
            
            stats = get_full_stats(t, fl, er, filt)
            f.update(stats)
            
            # Lưu để tính cross-band
            band_stats[filt] = stats
            band_fluxes[filt] = fl
            
        # 2. CROSS-BAND FEATURES (ALL PAIRS - 15 cặp)
        # Tính tất cả các tổ hợp màu có thể để không bỏ sót
        for i in range(len(filters)):
            for j in range(i + 1, len(filters)):
                b1 = filters[i]
                b2 = filters[j]
                
                # Kiểm tra xem có dữ liệu không
                if f'{b1}_w_mean' in f and f'{b2}_w_mean' in f:
                    # Color Mean
                    f[f'{b1}_{b2}_col_mean'] = f[f'{b1}_w_mean'] - f[f'{b2}_w_mean']
                    # Color Max
                    f[f'{b1}_{b2}_col_max'] = f[f'{b1}_max'] - f[f'{b2}_max']
                    # Flux Ratio
                    f[f'{b1}_{b2}_ratio'] = f[f'{b1}_max'] / (f[f'{b2}_max'] + 1e-6)
                    # Time Lag (Trễ giữa 2 đỉnh)
                    # (Cần lấy từ rise_time + min_time, ở đây ước lượng qua rise_time nếu min_time gần nhau)
                    # Cách tốt nhất là lưu peak time, nhưng để đơn giản ta dùng hiệu rise_time
                    f[f'{b1}_{b2}_lag'] = f[f'{b1}_rise_time'] - f[f'{b2}_rise_time']
                else:
                    f[f'{b1}_{b2}_col_mean'] = 0
                    f[f'{b1}_{b2}_col_max'] = 0
                    f[f'{b1}_{b2}_ratio'] = 0
                    f[f'{b1}_{b2}_lag'] = 0

        features.append(f)
        
    return pd.DataFrame(features).fillna(0)

# --- CHẠY ---
print("📥 Đang tải dữ liệu...")
train_files = glob.glob(os.path.join(BASE_PATH, '**', 'train_full_lightcurves.csv'), recursive=True)
test_files = glob.glob(os.path.join(BASE_PATH, '**', 'test_full_lightcurves.csv'), recursive=True)

# Filter out 'corrected' files to avoid duplication and double-correction
train_files = [f for f in train_files if 'corrected' not in f]
test_files = [f for f in test_files if 'corrected' not in f]

# Sắp xếp để đảm bảo thứ tự (optional)
train_files.sort()
test_files.sort()

print(f"   -> Tìm thấy {len(train_files)} files train và {len(test_files)} files test.")

train_lc = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True)
test_lc = pd.concat((pd.read_csv(f) for f in test_files), ignore_index=True)
train_log = pd.read_csv(os.path.join(BASE_PATH, 'train_log.csv'))
test_log = pd.read_csv(os.path.join(BASE_PATH, 'test_log.csv'))

# Lọc nhiễu
train_lc = train_lc.dropna(subset=['Flux'])
test_lc = test_lc.dropna(subset=['Flux'])

print("⚙️ Feature Engineering (Massive)...")
train_feats = process_pipeline(train_lc, train_log)
test_feats = process_pipeline(test_lc, test_log)

print("🔗 Merging...")
X_full = train_log.merge(train_feats, on='object_id', how='left').fillna(0)
X_test_final = test_log.merge(test_feats, on='object_id', how='left').fillna(0)

# --- 3. LUMINOSITY & Z INTERACTIONS ---
if 'Z' in X_full.columns:
    print("✨ Adding Z-interactions...")
    for filt in ['u', 'g', 'r', 'i']:
        if f'{filt}_max' in X_full.columns:
            X_full[f'{filt}_lum'] = X_full[f'{filt}_max'] * (X_full['Z']**2)
            X_test_final[f'{filt}_lum'] = X_test_final[f'{filt}_max'] * (X_test_final['Z']**2)

# Lọc cột số
features_col = [c for c in X_full.columns if c not in ['object_id', 'target', 'split', 'SpecType', 'English Translation'] and pd.api.types.is_numeric_dtype(X_full[c])]

# Đồng bộ cột
for col in features_col:
    if col not in X_test_final.columns:
        X_test_final[col] = 0.0

print(f"✅ Đã tạo tổng cộng: {len(features_col)} features.")
del train_lc, test_lc
gc.collect()

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Chuẩn bị dữ liệu đầy đủ
X = X_full[features_col]
y = X_full['target']
X_test_raw = X_test_final[features_col]

print(f"📦 Dữ liệu gốc: {X.shape[1]} features.")

# --- GIAI ĐOẠN 1: FEATURE SELECTION BẰNG XGBOOST GAIN ---
print("\n🔍 Giai đoạn 1: Huấn luyện sơ bộ để tính Feature Importance (Gain)...")

# Cấu hình XGBoost để tính Gain (cần max_depth vừa phải để không overfit quá sớm)
selector_params = {
    'n_estimators': 500,      # Ít cây thôi, chủ yếu để đo độ quan trọng
    'learning_rate': 0.05,
    'max_depth': 6,
    'scale_pos_weight': 4.0,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
    'importance_type': 'gain' # QUAN TRỌNG: Chọn theo Gain (chất lượng), không phải Weight (số lần dùng)
}

selector_model = XGBClassifier(**selector_params)
selector_model.fit(X, y)

# Lấy Feature Importance
importances = selector_model.feature_importances_
feature_imp_df = pd.DataFrame({'feature': features_col, 'gain': importances})
feature_imp_df = feature_imp_df.sort_values('gain', ascending=False)

# Chọn Top K features
K_BEST = 350
top_features = feature_imp_df['feature'].head(K_BEST).tolist()

print(f"✅ Đã chọn {K_BEST} features có Gain cao nhất.")
print(f"   -> Top 5 Features mạnh nhất: {top_features[:5]}")
print(f"   -> Các features bị loại bỏ (Ví dụ): {feature_imp_df['feature'].tail(5).tolist()}")

# Lọc dữ liệu theo features đã chọn
X_selected = X[top_features]
X_test_selected = X_test_raw[top_features]

# --- GIAI ĐOẠN 2: HUẤN LUYỆN CHÍNH THỨC (FINAL TRAINING) ---
print(f"\n🚀 Giai đoạn 2: Huấn luyện XGBoost Final trên {K_BEST} features...")

final_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,    # Học chậm để tối ưu
    'max_depth': 6,
    'min_child_weight': 5,    # Chống nhiễu
    'gamma': 1.0,             # Regularization
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'scale_pos_weight': 4.0,
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10 Fold

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test_selected))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[tr_idx], X_selected.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]

    model = XGBClassifier(**final_params, early_stopping_rounds=150)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Dự đoán
    oof_preds[va_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test_selected)[:, 1]
    
    print(f"   -> Fold {fold+1} xong.")

test_preds /= kf.n_splits

# --- ĐÁNH GIÁ & SUBMISSION ---
print("\n🔍 Đang tìm ngưỡng tối ưu...")
best_f1 = 0
best_t = 0
for t in np.arange(0.1, 0.9, 0.01):
    score = f1_score(y, (oof_preds > t).astype(int))
    if score > best_f1:
        best_f1 = score
        best_t = t

print(f"🏆 BEST F1 (XGBoost Gain): {best_f1:.4f} tại ngưỡng {best_t:.2f}")
print(classification_report(y, (oof_preds > best_t).astype(int)))

# Vẽ Feature Importance của Model Final
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20, importance_type='gain', title='Top 20 Features (Final Model)')
plt.savefig('xgboost_feature_importance.png') 
# plt.show() # Comment out plt.show() for headless environment

# Submission
sub = pd.DataFrame({
    'object_id': X_test_final['object_id'],
    'target': (test_preds > best_t).astype(int)
})
sub.to_csv("submission_v19_xgboost_gain.csv", index=False)
print("✅ Đã lưu file submission_v19_xgboost_gain.csv")

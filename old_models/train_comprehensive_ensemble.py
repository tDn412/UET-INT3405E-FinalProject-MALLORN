import pandas as pd
import numpy as np
import glob
import os
import gc
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# --- 1. SETUP ---
BASE_PATH = '.' 

# --- 2. FEATURE ENGINEERING FUNCTIONS (REUSED) ---
def get_tde_chisq(time, flux, err):
    if len(flux) < 5: return 999.0
    try:
        idx_max = np.argmax(flux)
        t_peak = time[idx_max]
        f_peak = flux[idx_max]
        mask = (time > t_peak) & (flux > 0.05 * f_peak) 
        if len(time[mask]) < 3: return 999.0
        
        def func(t, t0): return f_peak * np.power((t - t0)/(t_peak - t0), -5/3)
        popt, _ = curve_fit(func, time[mask], flux[mask], p0=[t_peak-20], 
                            bounds=([t_peak-500], [t_peak-0.1]), maxfev=100)
        return np.mean(((flux[mask] - func(time[mask], *popt)) / err[mask])**2)
    except: return 999.0

def get_full_stats(time, flux, err, filt):
    f = {}
    n = len(flux)
    if n < 2: return {}
    w = 1.0 / (err**2 + 1e-9)
    w_sum = np.sum(w)
    w_mean = np.sum(flux * w) / w_sum
    w_std = np.sqrt(np.sum(w * (flux - w_mean)**2) / w_sum)
    f[f'{filt}_w_mean'] = w_mean
    f[f'{filt}_w_std'] = w_std
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
    f[f'{filt}_amp_high'] = (pcts[11] - pcts[1]) / (np.abs(pcts[6]) + 1e-6)
    f[f'{filt}_amp_low']  = (pcts[9] - pcts[3]) / (np.abs(pcts[6]) + 1e-6)
    f[f'{filt}_flux_ratio_1'] = pcts[12] / (pcts[6] + 1e-6)
    f[f'{filt}_flux_ratio_2'] = (pcts[12] - pcts[6]) / (pcts[6] - pcts[0] + 1e-6)
    f[f'{filt}_tde_chisq'] = get_tde_chisq(time, flux, err)
    idx_max = np.argmax(flux)
    t_peak = time[idx_max]
    t_rise = t_peak - time.min()
    t_fall = time.max() - t_peak
    f[f'{filt}_rise_time'] = t_rise
    f[f'{filt}_fall_time'] = t_fall
    f[f'{filt}_rise_fall_ratio'] = t_rise / (t_fall + 1e-6)
    if idx_max > 0: f[f'{filt}_rise_slope'] = (flux[idx_max] - flux[0]) / (t_rise + 1e-6)
    else: f[f'{filt}_rise_slope'] = 0
    if idx_max < n - 1: f[f'{filt}_fall_slope'] = (flux[idx_max] - flux[-1]) / (t_fall + 1e-6)
    else: f[f'{filt}_fall_slope'] = 0
    f[f'{filt}_mad'] = np.median(np.abs(flux - pcts[6]))
    f[f'{filt}_beyond_1std'] = np.sum(np.abs(flux - w_mean) > w_std) / n
    return f

def process_pipeline(df_lc, df_log):
    print("   -> Processing light curves...")
    EXTINCTION = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}
    df = df_lc.merge(df_log[['object_id', 'EBV']], on='object_id', how='left').fillna(0)
    df['R'] = df['Filter'].map(EXTINCTION)
    factor = 10 ** (0.4 * df['R'] * df['EBV'])
    df['Flux'] = df['Flux'] * factor
    df['Flux_err'] = df['Flux_err'] * factor
    
    features = []
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    
    # Process by object
    for obj, grp in df.groupby('object_id'):
        f = {'object_id': obj}
        grp = grp.sort_values('Time (MJD)')
        
        band_stats = {}
        
        # Per-band
        for filt in filters:
            d = grp[grp['Filter'] == filt]
            if len(d) < 3: continue
            stats = get_full_stats(d['Time (MJD)'].values, d['Flux'].values, d['Flux_err'].values, filt)
            f.update(stats)
            band_stats[filt] = stats

        # Cross-band
        for i in range(len(filters)):
            for j in range(i + 1, len(filters)):
                b1, b2 = filters[i], filters[j]
                if f'{b1}_w_mean' in f and f'{b2}_w_mean' in f:
                    f[f'{b1}_{b2}_col_mean'] = f[f'{b1}_w_mean'] - f[f'{b2}_w_mean']
                    f[f'{b1}_{b2}_col_max'] = f[f'{b1}_max'] - f[f'{b2}_max']
                    f[f'{b1}_{b2}_ratio'] = f[f'{b1}_max'] / (f[f'{b2}_max'] + 1e-6)
                    f[f'{b1}_{b2}_lag'] = f[f'{b1}_rise_time'] - f[f'{b2}_rise_time']
                else:
                    f[f'{b1}_{b2}_col_mean'] = 0
                    f[f'{b1}_{b2}_col_max'] = 0
                    f[f'{b1}_{b2}_ratio'] = 0
                    f[f'{b1}_{b2}_lag'] = 0
        features.append(f)
        
    return pd.DataFrame(features).fillna(0)

# --- 3. EXECUTION FLOW ---

# CHECK FOR SAVED FEATURES
FEATURE_TRAIN_PATH = 'friend_features_train.csv'
FEATURE_TEST_PATH = 'friend_features_test.csv'

if os.path.exists(FEATURE_TRAIN_PATH) and os.path.exists(FEATURE_TEST_PATH):
    print("📦 Found saved features! Loading...")
    X_full = pd.read_csv(FEATURE_TRAIN_PATH)
    X_test_final = pd.read_csv(FEATURE_TEST_PATH)
else:
    print("⚙️ No saved features found. Running full pipeline...")
    train_files = glob.glob(os.path.join(BASE_PATH, '**', 'train_full_lightcurves.csv'), recursive=True)
    test_files = glob.glob(os.path.join(BASE_PATH, '**', 'test_full_lightcurves.csv'), recursive=True)
    
    # Filter corrected
    train_files = [f for f in train_files if 'corrected' not in f]
    test_files = [f for f in test_files if 'corrected' not in f]
    
    print(f"   -> Loading {len(train_files)} train files and {len(test_files)} test files.")
    
    train_lc = pd.concat((pd.read_csv(f) for f in train_files), ignore_index=True).dropna(subset=['Flux'])
    test_lc = pd.concat((pd.read_csv(f) for f in test_files), ignore_index=True).dropna(subset=['Flux'])
    train_log = pd.read_csv(os.path.join(BASE_PATH, 'train_log.csv'))
    test_log = pd.read_csv(os.path.join(BASE_PATH, 'test_log.csv'))
    
    print("   -> Extracting features (Train)...")
    train_feats = process_pipeline(train_lc, train_log)
    print("   -> Extracting features (Test)...")
    test_feats = process_pipeline(test_lc, test_log)
    
    X_full = train_log.merge(train_feats, on='object_id', how='left').fillna(0)
    X_test_final = test_log.merge(test_feats, on='object_id', how='left').fillna(0)
    
    # Add interactions
    if 'Z' in X_full.columns:
        for filt in ['u', 'g', 'r', 'i']:
            if f'{filt}_max' in X_full.columns:
                X_full[f'{filt}_lum'] = X_full[f'{filt}_max'] * (X_full['Z']**2)
                X_test_final[f'{filt}_lum'] = X_test_final[f'{filt}_max'] * (X_test_final['Z']**2)

    # Save
    print("💾 Saving extracted features for reuse...")
    X_full.to_csv(FEATURE_TRAIN_PATH, index=False)
    X_test_final.to_csv(FEATURE_TEST_PATH, index=False)

# Prepare Data
features_col = [c for c in X_full.columns if c not in ['object_id', 'target', 'split', 'SpecType', 'English Translation'] and pd.api.types.is_numeric_dtype(X_full[c])]
for col in features_col:
    if col not in X_test_final.columns: X_test_final[col] = 0.0

X = X_full[features_col]
y = X_full['target']
X_test_raw = X_test_final[features_col]

# --- 4. MODELING LOOP ---
print(f"\n🚀 STARTING MULTI-MODEL EXPERIMENT")
print(f"Total available features: {len(features_col)}")

# FEATURE RANKING (Once)
print("   -> Ranking features with XGBoost...")
ranker = xgb.XGBClassifier(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1, random_state=42)
ranker.fit(X, y)
imp_df = pd.DataFrame({'feature': features_col, 'gain': ranker.feature_importances_}).sort_values('gain', ascending=False)

# EXPERIMENT SETTINGS
K_VALUES = [150, 300, 450, len(features_col)]
MODELS = {
    'XGB': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, tree_method='hist', eval_metric='aucpr', n_jobs=-1, random_state=42),
    'LGBM': lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=8, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1),
    'CAT': None # Will initialize if available
}
try:
    MODELS['CAT'] = cb.CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, loss_function='Logloss', eval_metric='F1', verbose=0, random_seed=42)
except:
    print("⚠️ CatBoost not installed. Skipping.")
    del MODELS['CAT']

results = []

for k in K_VALUES:
    current_feats = imp_df['feature'].head(k).tolist()
    X_sel = X[current_feats]
    X_test_sel = X_test_raw[current_feats]
    
    print(f"\n🔹 Testing with TOP {k} Features")
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Store OOF preds for this K to ensemble later
    oof_preds = {name: np.zeros(len(X)) for name in MODELS}
    test_preds = {name: np.zeros(len(X_test_sel)) for name in MODELS}
    
    for name, model in MODELS.items():
        print(f"   Running {name}...", end=' ')
        fold_scores = []
        
        for tr_idx, va_idx in kf.split(X_sel, y):
            X_tr, X_va = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            # Reset model for clean fit
            if name == 'XGB':
                clf = xgb.XGBClassifier(**model.get_params())
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
            elif name == 'LGBM':
                clf = lgb.LGBMClassifier(**model.get_params())
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            elif name == 'CAT':
                clf = cb.CatBoostClassifier(**model.get_params())
                clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False)
                
            # Predict
            val_p = clf.predict_proba(X_va)[:, 1]
            oof_preds[name][va_idx] = val_p
            test_preds[name] += clf.predict_proba(X_test_sel)[:, 1] / kf.get_n_splits()
        
        # Optimize Threshold for this model
        best_f1, best_t = 0, 0
        for t in np.arange(0.1, 0.9, 0.05):
            s = f1_score(y, (oof_preds[name] > t).astype(int))
            if s > best_f1: best_f1, best_t = s, t
            
        print(f"-> Best F1: {best_f1:.4f} (Thresh {best_t:.2f})")
        results.append({'Features': k, 'Model': name, 'F1': best_f1, 'Threshold': best_t})

    # ENSEMBLE FOR THIS K (Average)
    ens_oof = np.mean([oof_preds[name] for name in MODELS], axis=0)
    best_ens_f1, best_ens_t = 0, 0
    for t in np.arange(0.1, 0.9, 0.05):
        s = f1_score(y, (ens_oof > t).astype(int))
        if s > best_ens_f1: best_ens_f1, best_ens_t = s, t
    
    print(f"   🏆 Ensemble F1: {best_ens_f1:.4f} (Thresh {best_ens_t:.2f})")
    results.append({'Features': k, 'Model': 'Ensemble_Avg', 'F1': best_ens_f1, 'Threshold': best_ens_t})
    
    # Save Ensemble CSV if it's the best so far
    if best_ens_f1 > 0.5: # Only verify meaningful improvements
        ens_test = np.mean([test_preds[name] for name in MODELS], axis=0)
        sub = pd.DataFrame({'object_id': X_test_final['object_id'], 'target': (ens_test > best_ens_t).astype(int)})
        sub.to_csv(f"submission_ensemble_k{k}.csv", index=False)
        print(f"      -> Saved submission_ensemble_k{k}.csv")

# SUMMARY
print("\n📊 EXPERIMENT SUMMARY")
summary_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(summary_df)
summary_df.to_csv('experiment_results.csv', index=False)

"""
STEP 10: CHI-SQUARE TEMPLATE FITTING (PHYSICS MODELS)
Goal: Distinguish TDE from Supernovae (SN) using Goodness-of-Fit.
Method:
1. Define TDE Model (Power Law Decay).
2. Define SN Model (Bazin Function).
3. Fit BOTH models to every light curve (all bands combined or per band).
4. Extract Feature: Delta Chi-Square = Chi2_TDE - Chi2_SN.
   - If < 0: Fits TDE better.
   - If > 0: Fits SN better.
"""

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHYSICS TEMPLATE FITTING: TDE vs SUPERNOVA")
print("="*70)

# 1. Load Data
print("\n[1] Loading Corrected Data...")
with open('raw_lightcurves_train_corrected.pkl', 'rb') as f:
    train_lcs = pickle.load(f)
with open('raw_lightcurves_test_corrected.pkl', 'rb') as f:
    test_lcs = pickle.load(f)

# 2. Define Physics Models

def tde_func(t, t0, A, rise, decay):
    # Phenomenological TDE: Exponential Rise + Power Law Decay
    # Shifts time so t0 is peak
    dt = t - t0
    
    # Rise: Exp
    with np.errstate(over='ignore', invalid='ignore'):
        flux_rise = A * np.exp(dt / (rise + 1e-3))
        # Decay: Power Law (index -5/3 = -1.67)
        # Standard: L ~ (t - t0 + t_fall)^-1.67
        # We model as: A * ((t - t0)/decay + 1)^-1.67
        flux_decay = A * np.power((dt/decay + 1.0), -1.67)
    
    # Combine
    y = np.where(dt < 0, flux_rise, flux_decay)
    return np.nan_to_num(y)

def bazin_func(t, t0, A, t_fall, t_rise):
    # Standard Supernova Analytic Model (Bazin et al. 2009)
    # F(t) = A * exp(-(t-t0)/t_fall) / (1 + exp((t-t0)/t_rise))
    with np.errstate(over='ignore', invalid='ignore'):
        y = A * np.exp(-(t - t0) / t_fall) / (1 + np.exp((t - t0) / t_rise))
    return np.nan_to_num(y)

def fit_templates(lc_df):
    features = {}
    
    # We fit on the 'flux' column directly.
    # To be robust, we normalize the flux first or handle A.
    # Let's fit mainly on the 'r' band (most sensitive usually) or 'g'.
    # Or combine all bands if we ignore color? Better to fit 'g' and 'r' separately.
    
    best_chisq_tde = 0
    best_chisq_sn = 0
    
    bands = ['g', 'r']
    valid_bands = 0
    
    for b in bands:
        df = lc_df[lc_df['Filter'] == b].sort_values('Time (MJD)')
        if len(df) < 5: 
            features[f'chisq_tde_{b}'] = 9999
            features[f'chisq_sn_{b}'] = 9999
            continue
            
        t = df['Time (MJD)'].values
        y = df['Flux'].values
        y_err = df['Flux Error'].values if 'Flux Error' in df.columns else df['Flux_err'].values # Handle naming
        
        # Normalize t
        t_start = t.min()
        t_norm = t - t_start
        
        # Initial Guesses
        t_peak_guess = t_norm[np.argmax(y)]
        A_guess = np.max(y)
        
        # --- Fit TDE ---
        # Bounds: t0 (0, dur), A (0, max*2), rise (0.1, 50), decay (10, 500)
        try:
            popt_tde, _ = curve_fit(tde_func, t_norm, y, 
                                    p0=[t_peak_guess, A_guess, 5.0, 30.0],
                                    bounds=([0, 0, 0.1, 1.0], [t_norm.max(), A_guess*5, 100, 1000]),
                                    maxfev=1000)
            
            y_pred_tde = tde_func(t_norm, *popt_tde)
            chisq_tde = np.sum(((y - y_pred_tde) / (y_err + 1e-5))**2) / (len(y) - 4) # Reduced Chi2
        except:
            chisq_tde = 9999
            
        # --- Fit SN (Bazin) ---
        # Bounds: t0 (-50, dur), A (0, max*2), fall (5, 200), rise (1, 50)
        try:
            popt_sn, _ = curve_fit(bazin_func, t_norm, y,
                                   p0=[t_peak_guess, A_guess, 50.0, 5.0],
                                   bounds=([-50, 0, 1.0, 0.1], [t_norm.max()*1.5, A_guess*5, 500, 100]),
                                   maxfev=1000)
            y_pred_sn = bazin_func(t_norm, *popt_sn)
            chisq_sn = np.sum(((y - y_pred_sn) / (y_err + 1e-5))**2) / (len(y) - 4)
        except:
            chisq_sn = 9999
            
        features[f'chisq_tde_{b}'] = chisq_tde
        features[f'chisq_sn_{b}'] = chisq_sn
        valid_bands += 1
        
    # Aggregate
    if valid_bands > 0:
        # Sum of Chi2s? Or min?
        # Let's simple use the g and r features directly in the tree model.
        # But we can create a ratio feature here.
        tde_total = features.get('chisq_tde_g', 0) + features.get('chisq_tde_r', 0)
        sn_total = features.get('chisq_sn_g', 0) + features.get('chisq_sn_r', 0)
        
        features['delta_chisq'] = tde_total - sn_total
    else:
        features['delta_chisq'] = 0
        
    return features

# 3. Extraction Loop
print("\n[2] Extracting Template Features (Train)...")
train_feats = []
for idx, (oid, lc) in enumerate(train_lcs.items()):
    if idx % 500 == 0: print(f"  {idx}/{len(train_lcs)}")
    f = fit_templates(lc)
    f['object_id'] = oid
    train_feats.append(f)
    
train_df = pd.DataFrame(train_feats)
train_df.to_csv('template_features_train.csv', index=False)
print("✓ Saved template_features_train.csv")

print("\n[3] Extracting Template Features (Test)...")
test_feats = []
for idx, (oid, lc) in enumerate(test_lcs.items()):
    if idx % 500 == 0: print(f"  {idx}/{len(test_lcs)}")
    f = fit_templates(lc)
    f['object_id'] = oid
    test_feats.append(f)
    
test_df = pd.DataFrame(test_feats)
test_df.to_csv('template_features_test.csv', index=False)
print("✓ Saved template_features_test.csv")

print("\nPhase 11 (Template Fitting) Complete.")

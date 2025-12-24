
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

print("="*70)
print("PHYSICS-BASED FEATURE EXTRACTION")
print("Focus: Blackbody Temperature & SED Shape")
print("="*70)

# LSST Band Effective Wavelengths (Angstroms)
# Source: standard LSST passbands
WAVELENGTHS = {
    'u': 3685.0,
    'g': 4802.0,
    'r': 6231.0,
    'i': 7542.0,
    'z': 8690.0,
    'y': 9736.0
}

def fit_blackbody_scipy(wavelengths_angstrom, fluxes, flux_errors):
    """
    Fit Planck's law using scipy.optimize.curve_fit
    Faster than astropy modeling for simple loops
    """
    
    # Planck's law function: B(lambda, T)
    # B_lambda(T) = (2hc^2 / lambda^5) * (1 / (exp(hc / (lambda k T)) - 1))
    
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    
    # Convert wavelengths to meters
    lam = wavelengths_angstrom * 1e-10
    
    def planck(lam, T, scale):
        # T in Kelvin, scale is a normalization factor related to solid angle/distance
        # Handle potential overflow in exp
        try:
            val = (2 * h * c**2) / (lam**5) * (1 / (np.exp((h * c) / (lam * k * T)) - 1))
            return scale * val
        except:
            return np.zeros_like(lam)

    # Initial guess: T=10000K, scale=arbitrary
    p0 = [10000, 1e-15]
    bounds = ([1000, 0], [100000, np.inf]) # Constrain T between 1000K and 100000K
    
    try:
        popt, pcov = curve_fit(planck, lam, fluxes, sigma=flux_errors, p0=p0, bounds=bounds, maxfev=1000)
        return popt[0], np.sqrt(np.diag(pcov))[0]  # T, T_err
    except:
        return np.nan, np.nan

def extract_physics_features_single(obj_id, lc_df):
    """
    Extract physics features for a single object.
    
    Strategy:
    1. Group by MJD (epoch) to get multi-band SED snapshots.
    2. At each epoch with >= 3 bands, fit Blackbody.
    3. Aggregate temperature statistics.
    4. Calculate aggregated colors and SED shapes.
    """
    
    features = {}
    features['object_id'] = obj_id
    
    # Pre-calculate simple color features (mean flux differences)
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    mean_fluxes = {b: lc_df[lc_df['Filter'] == b]['Flux'].mean() for b in bands}
    
    # Q Parameter (Reddening-free index often used for Quasars/TDEs)
    # Q = (u-g) - 0.8(g-r)
    # Requires substituting magnitudes, here we just use flux ratios or log-flux diffs
    # Converting mean fluxes to approx mag: -2.5 * log10(flux)
    mags = {}
    valid_mags = True
    for b, f in mean_fluxes.items():
        if f > 0:
            mags[b] = -2.5 * np.log10(f)
        else:
            mags[b] = np.nan
            
    if not np.isnan(mags['u']) and not np.isnan(mags['g']):
        features['color_u_g'] = mags['u'] - mags['g']
        
    if not np.isnan(mags['g']) and not np.isnan(mags['r']):
        features['color_g_r'] = mags['g'] - mags['r']
        
    if not np.isnan(mags['u']) and not np.isnan(mags['g']) and not np.isnan(mags['r']):
        features['Q_parameter'] = (mags['u'] - mags['g']) - 0.8 * (mags['g'] - mags['r'])
    else:
        features['Q_parameter'] = np.nan
        
    # UV Excess: Flux_u / (Flux_g + Flux_r)
    if (mean_fluxes['g'] + mean_fluxes['r']) > 0:
        features['uv_excess'] = mean_fluxes['u'] / (mean_fluxes['g'] + mean_fluxes['r'])
    else:
        features['uv_excess'] = 0

    # Group by integer MJD (epochs)
    # Round MJD to integers to group observations taken essentially at the same time
    lc_df['mjd_int'] = lc_df['Time (MJD)'].astype(int)
    
    temps = []
    temp_errs = []
    chisqs = []
    
    valid_epochs = 0
    
    for mjd, group in lc_df.groupby('mjd_int'):
        # Need at least 3 bands to fit 2 parameters (T, scale) reasonable
        if len(group) >= 3:
            # Prepare data arrays
            wavs = []
            flxs = []
            errs = []
            
            for _, row in group.iterrows():
                if row['Filter'] in WAVELENGTHS and row['Flux'] > 0: # Only fit positive fluxes
                    wavs.append(WAVELENGTHS[row['Filter']])
                    flxs.append(row['Flux'])
                    errs.append(row['Flux_err'] if row['Flux_err'] > 0 else 1.0)
            
            if len(wavs) >= 3:
                T, T_err = fit_blackbody_scipy(np.array(wavs), np.array(flxs), np.array(errs))
                
                if not np.isnan(T): # and T < 50000: # Filter failures or unrealistic?
                    temps.append(T)
                    temp_errs.append(T_err)
                    valid_epochs += 1

    # Aggregate Temperature Features
    if temps:
        features['T_mean'] = np.mean(temps)
        features['T_median'] = np.median(temps)
        features['T_std'] = np.std(temps)
        features['T_max'] = np.max(temps)
        features['T_min'] = np.min(temps)
        
        # Evolution: simple slope of T vs time?
        # Requires keeping track of MJDs for the valid fits. 
        # Skipping for simplicity in v1, focus on statistical distribution.
        
        features['T_is_constant'] = 1 if features['T_std'] < 5000 else 0
    else:
        features['T_mean'] = np.nan
        features['T_median'] = np.nan
        features['T_std'] = np.nan
        features['T_max'] = np.nan
        features['T_min'] = np.nan
        features['T_is_constant'] = 0
        
    features['n_valid_bb_fits'] = valid_epochs
    
    return features

def process_lightcurves(pkl_path, output_csv):
    """Load pickle, extract features, save CSV"""
    
    print(f"\nProcessing {pkl_path}...")
    
    if not Path(pkl_path).exists():
        print(f"File not found: {pkl_path}")
        return
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Loaded {len(data)} objects.")
    
    results = []
    
    # Use tqdm
    for obj_id, df in tqdm(data.items()):
        feats = extract_physics_features_single(obj_id, df)
        results.append(feats)
        
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with shape {result_df.shape}")

# MAIN
if __name__ == "__main__":
    
    # Train
    process_lightcurves('raw_lightcurves_train.pkl', 'physics_features_train.csv')
    
    # Test
    process_lightcurves('raw_lightcurves_test.pkl', 'physics_features_test.csv')
    
    print("\nExtraction complete.")

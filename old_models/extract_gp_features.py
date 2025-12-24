"""
STEP 7: GAUSSIAN PROCESS FEATURE EXTRACTION
Goal: Capture the "texture" and correlation length of light curves using GPs.
Method:
1. Fit a Gaussian Process with Matern 3/2 Kernel to 'g' and 'r' bands.
2. Extract Hyperparameters: Length Scale (time correlation) and Amplitude.
3. Calculate Residuals: How well does the smooth GP explain the variability?
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GAUSSIAN PROCESS FEATURE EXTRACTION")
print("="*70)

# 1. Load Data
print("\n[1] Loading Corrected Data...")
with open('raw_lightcurves_train_corrected.pkl', 'rb') as f:
    train_lcs = pickle.load(f)
with open('raw_lightcurves_test_corrected.pkl', 'rb') as f:
    test_lcs = pickle.load(f)

print(f"  Train: {len(train_lcs)} | Test: {len(test_lcs)}")

def extract_gp_features(lc_df):
    """
    Fit GP to 'g' and 'r' bands and extract kernel parameters.
    Kernel: C * Matern(nu=1.5) + WhiteKernel
    """
    features = {}
    
    # Initialize GP
    # Length scale bounds: 1 day to 200 days (typical TDE/SN timescales)
    # Amplitude bounds: 1e-3 to 1e5
    # Noise bounds: 1e-3 to 1e3
    kernel = C(1.0, (1e-3, 1e5)) * Matern(length_scale=20.0, length_scale_bounds=(1.0, 500.0), nu=1.5) \
             + WhiteKernel(noise_level=10.0, noise_level_bounds=(1e-2, 1e4))
             
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
    
    bands = ['g', 'r']
    
    for band in bands:
        band_data = lc_df[lc_df['Filter'] == band].dropna()
        
        if len(band_data) < 5:
            features[f'gp_{band}_length_scale'] = 0
            features[f'gp_{band}_amplitude'] = 0
            features[f'gp_{band}_noise'] = 0
            features[f'gp_{band}_log_likelihood'] = 0
            continue
            
        X = band_data['Time (MJD)'].values.reshape(-1, 1)
        y = band_data['Flux'].values
        
        # Normalize time to start at 0 to help optimizer
        X = X - X.min()
        
        try:
            gp.fit(X, y)
            
            # Extract Params
            params = gp.kernel_.get_params()
            
            # Kernel structure: k1 (C * Matern) + k2 (White)
            # k1: k1 (C) * k2 (Matern)
            
            # Accessing via named steps in the complex kernel is tricky, simpler to inspect directly
            # The structure is Sum(Product(Constant, Matern), White)
            
            # Let's rely on the hyperparameter array order or names
            # Or better, parse the kernel object
            
            # Assuming standard optimization structure:
            k_sum = gp.kernel_ # Sum
            k_prod = k_sum.k1 # Product
            k_white = k_sum.k2 # White
            
            k_const = k_prod.k1 # Constant
            k_matern = k_prod.k2 # Matern
            
            features[f'gp_{band}_length_scale'] = k_matern.length_scale
            features[f'gp_{band}_amplitude'] = k_const.constant_value
            features[f'gp_{band}_noise'] = k_white.noise_level
            features[f'gp_{band}_log_likelihood'] = gp.log_marginal_likelihood()
            
        except Exception as e:
            # print(f"GP Fit Error: {e}")
            features[f'gp_{band}_length_scale'] = 0
            features[f'gp_{band}_amplitude'] = 0
            features[f'gp_{band}_noise'] = 0
            features[f'gp_{band}_log_likelihood'] = 0
            
    return features

# 2. Extract Validation/Training Features
print("\n[2] Extracting GP Features (Train)...")
train_features_list = []
keys = list(train_lcs.keys())

for i, oid in enumerate(keys):
    if i % 500 == 0: print(f"  {i}/{len(keys)}")
    f = extract_gp_features(train_lcs[oid])
    f['object_id'] = oid
    train_features_list.append(f)

train_gp_df = pd.DataFrame(train_features_list)
train_gp_df.to_csv('gp_features_train_corrected.csv', index=False)
print("✓ Saved gp_features_train_corrected.csv")

# 3. Extract Test Features
print("\n[3] Extracting GP Features (Test)...")
test_features_list = []
keys = list(test_lcs.keys())

for i, oid in enumerate(keys):
    if i % 500 == 0: print(f"  {i}/{len(keys)}")
    f = extract_gp_features(test_lcs[oid])
    f['object_id'] = oid
    test_features_list.append(f)

test_gp_df = pd.DataFrame(test_features_list)
test_gp_df.to_csv('gp_features_test_corrected.csv', index=False)
print("✓ Saved gp_features_test_corrected.csv")

print("\nGP EXTRACTION COMPLETE.")

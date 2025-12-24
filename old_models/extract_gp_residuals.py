
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GP RESIDUAL FEATURE EXTRACTION")
print("="*70)

def extract_gp_residuals_single(obj_id, lc_df):
    """
    Fit GP to each band, extract residual statistics.
    """
    features = {}
    features['object_id'] = obj_id
    
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    
    # Combined residuals
    all_residuals = []
    
    for band in bands:
        band_data = lc_df[lc_df['Filter'] == band]
        
        if len(band_data) < 3:
            features[f'gp_res_std_{band}'] = np.nan
            features[f'gp_loglike_{band}'] = np.nan
            continue
            
        X = band_data['Time (MJD)'].values.reshape(-1, 1)
        y = band_data['Flux'].values
        # Normalize y for stable GP
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-10
        y_norm = (y - y_mean) / y_std
        
        # Kernel: RBF (length_scale bounds roughly matching transient duration) + Noise
        kernel = RBF(length_scale=20.0, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=0.1)
        
        try:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=0) # alpha handled by WhiteKernel
            gp.fit(X, y_norm)
            
            y_pred_norm = gp.predict(X)
            residuals_norm = y_norm - y_pred_norm
            
            features[f'gp_res_std_{band}'] = np.std(residuals_norm)
            features[f'gp_loglike_{band}'] = gp.log_marginal_likelihood_value_
            
            all_residuals.extend(residuals_norm)
            
        except Exception:
            features[f'gp_res_std_{band}'] = np.nan
            features[f'gp_loglike_{band}'] = np.nan

    if all_residuals:
        features['gp_res_std_mean'] = np.std(all_residuals)
        features['gp_res_skew'] = pd.Series(all_residuals).skew()
    else:
        features['gp_res_std_mean'] = np.nan
        features['gp_res_skew'] = np.nan
        
    return features

def process_lightcurves_gp(pkl_path, output_csv):
    print(f"\nProcessing {pkl_path}...")
    if not Path(pkl_path).exists():
        return
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    results = []
    # Taking a subset or full? GP is slow. 
    # For draft, we'll try full but it might take hours.
    # User instructions implied this is Tier A (high priority).
    
    for obj_id, df in tqdm(data.items()):
        feats = extract_gp_residuals_single(obj_id, df)
        results.append(feats)
        
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    from joblib import Parallel, delayed
    
    def process_parallel(pkl_path, output_csv):
        print(f"\nProcessing {pkl_path}...")
        if not Path(pkl_path).exists():
            return
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # Parallel extraction
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(extract_gp_residuals_single)(obj_id, df) 
            for obj_id, df in data.items()
        )
            
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"Saved {output_csv}")

    process_parallel('raw_lightcurves_train.pkl', 'gp_residual_features_train.csv')
    process_parallel('raw_lightcurves_test.pkl', 'gp_residual_features_test.csv')

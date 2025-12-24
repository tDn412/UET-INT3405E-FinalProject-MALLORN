"""
STEP 4: GENERATIVE PHYSICS AUGMENTATION
Goal: Generate SYNTHETIC TDE light curves to solve class imbalance.
Method:
1. Analyze properties of REAL TDEs (Rise time, Peak Flux, Color)
2. Fit distributions (Gaussian/KDE) to these parameters
3. Generate new consistent light curves based on TDE physics (Rise + Power Law Decay)
4. Simulate survey cadence (gaps, noise)
"""

import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy

print("="*70)
print("GENERATIVE PHYSICS: SYNTHETIC TDE CREATION")
print("="*70)

# 1. Load Data
print("\n[1] Loading Real TDEs...")
train_meta = pd.read_csv('train_log.csv')
tde_ids = train_meta[train_meta['target'] == 1]['object_id'].values

with open('raw_lightcurves_train_corrected.pkl', 'rb') as f:
    all_lcs = pickle.load(f)

tde_lcs = {oid: all_lcs[oid] for oid in tde_ids if oid in all_lcs}
print(f"  Found {len(tde_lcs)} confirmed TDE light curves.")

# 2. Define Physics Model
def tde_model(t, t_peak, flux_peak, rise_time, decay_time):
    """
    Phenomenological TDE Model:
    - Exponential Rise
    - Power-Law Decay (t^-5/3)
    """
    # Shift time so t_peak is 0 for calculation
    dt = t - t_peak
    
    # Rise phase (t < t_peak)
    # Exponential rise: F ~ exp((t - t_peak) / rise_time)
    rise_mask = dt < 0
    flux_rise = flux_peak * np.exp(dt[rise_mask] / (rise_time + 1e-5))
    
    # Decay phase (t >= t_peak)
    # Power law: F ~ (t - t_peak + t0)^(-5/3)
    # We match flux at peak: flux_peak * (1 + dt/decay_scale)^(-5/3)
    decay_mask = ~rise_mask
    flux_decay = flux_peak * (1 + dt[decay_mask] / (decay_time + 1e-5))**(-1.67)
    
    flux = np.zeros_like(t)
    flux[rise_mask] = flux_rise
    flux[decay_mask] = flux_decay
    
    return flux

# 3. Parameter Estimation
print("\n[2] Extracting physical parameters from Real TDEs...")

params_list = []

for oid, lc in tde_lcs.items():
    if len(lc) < 5: continue
    
    # Process per band to get colors
    times = lc['Time (MJD)'].values
    fluxes = lc['Flux'].values
    filters = lc['Filter'].values
    
    # Estimate basic params
    peak_idx = np.argmax(fluxes)
    t_peak = times[peak_idx]
    f_peak_max = fluxes[peak_idx]
    
    # Estimate Rise Time (approximate)
    # Time from first detection to peak
    t_start = times[0]
    rise_time_est = max(1.0, t_peak - t_start)
    
    # Estimate Decay Scale
    # Just a rough guess or calculating from half-max
    decay_time_est = 20.0 # Default
    
    # Get Peak Colors (ratios relative to 'r' band)
    band_peaks = {}
    for b in ['u','g','r','i','z','y']:
        b_data = lc[lc['Filter'] == b]
        if len(b_data) > 0:
            # Use max flux in this band
            band_peaks[b] = b_data['Flux'].max()
        else:
            band_peaks[b] = 0
            
    # Normalize to r-band for "color"
    ref_flux = band_peaks.get('r', 1.0)
    if ref_flux <= 0: ref_flux = 1.0
    
    color_ratios = {b: band_peaks.get(b,0)/ref_flux for b in ['u','g','r','i','z','y']}
    
    params_list.append({
        'rise_time': rise_time_est,
        'decay_time': decay_time_est, # We'll sample this from a distribution
        'log_flux_peak': np.log10(f_peak_max) if f_peak_max > 0 else 1.0,
        'colors': color_ratios
    })

# Convert to DataFrame for stats
df_params = pd.DataFrame(params_list)

# Calculate stats for generation
mean_rise = df_params['rise_time'].mean()
std_rise = df_params['rise_time'].std()
mean_log_flux = df_params['log_flux_peak'].mean()
std_log_flux = df_params['log_flux_peak'].std()

# Average color profile
avg_colors = {}
for b in ['u','g','r','i','z','y']:
    vals = [p['colors'][b] for p in params_list if p['colors'][b] > 0]
    avg_colors[b] = np.mean(vals) if vals else 0.5

print(f"  Mean Rise Time: {mean_rise:.2f} +/- {std_rise:.2f} days")
print(f"  Mean Peak Flux (log10): {mean_log_flux:.2f}")
print(f"  TDE Color Profile (rel to r): {avg_colors}")

# 4. Generate Synthetic TDEs
print("\n[3] Generating 500 Synthetic TDEs...")

n_synthetic = 500
synthetic_lcs = {}

# We need a template for "Cadence" (Time sampling)
# We'll use a random real light curve's time points for each synthetic one
# to mimic realistic survey gaps/seasons.
real_tde_keys = list(tde_lcs.keys())

for i in range(n_synthetic):
    # A. Sample Parameters
    sim_rise = np.abs(np.random.normal(mean_rise, std_rise))
    sim_decay = np.random.uniform(10, 100) # Decay timescale distribution
    sim_peak_flux_log = np.random.normal(mean_log_flux, std_log_flux)
    sim_peak_flux = 10**sim_peak_flux_log
    
    # B. Choose a Cadence Template
    template_oid = np.random.choice(real_tde_keys)
    template_lc = tde_lcs[template_oid]
    template_times = template_lc['Time (MJD)'].values
    template_filters = template_lc['Filter'].values
    
    # Shift times so peak is randomly placed within the window
    # But usually TDEs are detected near peak. fit peak in middle 50%
    t_min, t_max = template_times.min(), template_times.max()
    sim_t_peak = np.random.uniform(t_min + (t_max-t_min)*0.2, t_max - (t_max-t_min)*0.2)
    
    # C. Generate Ideal Fluxes
    sim_fluxes = tde_model(template_times, sim_t_peak, sim_peak_flux, sim_rise, sim_decay)
    
    # D. Apply Colors & Noise
    final_fluxes = []
    final_errors = []
    
    for t, f_ideal, filt in zip(template_times, sim_fluxes, template_filters):
        # Apply color factor
        color_factor = avg_colors.get(filt, 0.5)
        # Add some random variation to color per object
        color_factor *= np.random.uniform(0.8, 1.2)
        
        band_flux = f_ideal * color_factor
        
        # Add Noise (heteroscedastic)
        # Error is roughly proportional to flux + background
        # Let's approximate from real data errors? Or just say 5% + background
        err = 5.0 + 0.05 * band_flux # Base noise level + 5%
        
        noisy_flux = band_flux + np.random.normal(0, err)
        
        final_fluxes.append(noisy_flux)
        final_errors.append(err)
        
    # Create DataFrame
    syn_df = pd.DataFrame({
        'Time (MJD)': template_times,
        'Filter': template_filters,
        'Flux': final_fluxes,
        'Flux Error': final_errors
    })
    
    # ID: synthetic_0, synthetic_1...
    syn_id = f'synthetic_{i}'
    synthetic_lcs[syn_id] = syn_df

print(f"  Generated {len(synthetic_lcs)} synthetic light curves.")

# 5. Save
print("\n[4] Saving Synthetic Data...")
with open('raw_lightcurves_synthetic.pkl', 'wb') as f:
    pickle.dump(synthetic_lcs, f)

# Also create a synthetic metadata log
syn_meta = pd.DataFrame({
    'object_id': list(synthetic_lcs.keys()),
    'target': [1] * n_synthetic,
    # Assign mean Z/EBV or sample
    'Z': np.random.choice(train_meta['Z'].dropna(), n_synthetic),
    'EBV': np.random.choice(train_meta['EBV'].dropna(), n_synthetic)
})
syn_meta.to_csv('synthetic_log.csv', index=False)

print("✓ Saved raw_lightcurves_synthetic.pkl")
print("✓ Saved synthetic_log.csv")
print("\nREADY FOR FEATURE EXTRACTION.")

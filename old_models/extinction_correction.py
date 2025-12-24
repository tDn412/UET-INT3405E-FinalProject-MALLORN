"""
MALLORN TDE Classification - Extinction Correction
Apply physics-based correction to restore intrinsic flux values

Based on Fitzpatrick (1999) extinction law with R_V = 3.1
Formula: F_intrinsic = F_observed × 10^(0.4 × R_λ × EBV)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# LSST filter extinction coefficients (R_λ) from Fitzpatrick 1999
# Source: Friend's technical report + LSST Science Book
EXTINCTION_COEFFS = {
    'u': 4.239,  # λ_eff ~ 364 nm (most affected by dust)
    'g': 3.303,  # λ_eff ~ 470 nm
    'r': 2.285,  # λ_eff ~ 616 nm
    'i': 1.698,  # λ_eff ~ 750 nm
    'z': 1.263,  # λ_eff ~ 869 nm
    'y': 1.086   # λ_eff ~ 1006 nm (least affected)
}

def apply_extinction_correction(flux, ebv, filter_name):
    """
    Apply extinction correction to restore intrinsic flux.
    
    Physics: Dust in Milky Way absorbs blue light more than red (like sunset).
    TDEs are intrinsically blue (hot, ~10^4-5×10^4 K), but appear red if behind dust.
    This correction is CRITICAL for proper color-based classification.
    
    Args:
        flux: Observed flux in μJy (can be negative due to noise)
        ebv: E(B-V) reddening coefficient from metadata
        filter_name: LSST filter ('u', 'g', 'r', 'i', 'z', 'y')
    
    Returns:
        Corrected intrinsic flux
    """
    if filter_name not in EXTINCTION_COEFFS:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    R_lambda = EXTINCTION_COEFFS[filter_name]
    A_lambda = R_lambda * ebv
    
    # Formula: F_int = F_obs × 10^(0.4 × A_λ)
    correction_factor = 10 ** (0.4 * A_lambda)
    
    return flux * correction_factor


def correct_lightcurve_file(lc_path, log_path, output_path):
    """
    Apply extinction correction to a lightcurve file.
    
    Args:
        lc_path: Path to lightcurve CSV
        log_path: Path to metadata log CSV (contains EBV values)
        output_path: Where to save corrected lightcurves
    """
    print(f"Processing: {lc_path.name}")
    
    # Load data
    lc_df = pd.read_csv(lc_path)
    log_df = pd.read_csv(log_path)
    
    print(f"  Lightcurves: {len(lc_df):,} observations")
    print(f"  Objects: {len(log_df):,}")
    
    # Merge to get EBV for each observation
    lc_df = lc_df.merge(log_df[['object_id', 'EBV']], on='object_id', how='left')
    
    # Apply correction per filter
    print("  Applying extinction correction...")
    corrected_flux = []
    
    for idx, row in tqdm(lc_df.iterrows(), total=len(lc_df), desc="  Correcting"):
        flux_corrected = apply_extinction_correction(
            flux=row['Flux'],
            ebv=row['EBV'],
            filter_name=row['Filter']
        )
        corrected_flux.append(flux_corrected)
    
    # Replace flux with corrected values
    lc_df['Flux'] = corrected_flux
    
    # Drop temporary EBV column
    lc_df = lc_df.drop(columns=['EBV'])
    
    # Save corrected lightcurves
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lc_df.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved: {output_path}")
    
    # Stats
    print(f"  Correction factors range: {10**(0.4*log_df['EBV'].min()*EXTINCTION_COEFFS['u']):.3f} to {10**(0.4*log_df['EBV'].max()*EXTINCTION_COEFFS['u']):.3f} (u-band)")
    

def main():
    """Process all splits with extinction correction."""
    
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("EXTINCTION CORRECTION - Restoring Intrinsic Flux Values")
    print("="*70)
    print("\nPhysics Background:")
    print("  - Dust in Milky Way absorbs blue light > red light")
    print("  - TDEs are hot (~10^4 K) → intrinsically BLUE")
    print("  - Without correction, dusty TDEs look like red SNe → misclassification")
    print("\nFormula: F_intrinsic = F_observed × 10^(0.4 × R_λ × EBV)")
    print(f"\nExtinction coefficients (R_λ): {EXTINCTION_COEFFS}")
    print("="*70)
    
    # Dynamic split discovery
    split_folders = sorted([d for d in base_dir.glob('split_*') if d.is_dir()])
    print(f"Found {len(split_folders)} split folders.")

    for split_dir in tqdm(split_folders, desc="Processing Splits"):
        split_name = split_dir.name
        
        for dataset in ['train', 'test']:
            lc_file = f"{dataset}_full_lightcurves.csv" 
            log_file = f"{dataset}_log.csv"
            
            lc_path = split_dir / lc_file
            log_path = base_dir / log_file # FIX: Log files are in root
            
            # Create corrected directory
            output_dir = base_dir / f"{split_name}_corrected"
            output_path = output_dir / lc_file
            
            if not lc_path.exists():
                # print(f"  ⚠ Skipping - file not found: {lc_path}")
                continue
            
            if output_path.exists():
                # print(f"  Skipping existing: {output_path}")
                continue
            
            correct_lightcurve_file(lc_path, log_path, output_path)
    
    print("\n" + "="*70)
    print("✅ EXTINCTION CORRECTION COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Extract physics-informed features from corrected lightcurves")
    print("  2. Train LightGBM classifier")
    print("="*70)


if __name__ == "__main__":
    main()

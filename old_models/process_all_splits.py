"""
MALLORN TDE - Process ALL Splits for Physics Features
Extract physics features from all 20 splits to get full dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from physics_features module
import sys
sys.path.append(str(Path(__file__).parent))
from physics_features import apply_extinction_correction, extract_flux_ratio_features, extract_temporal_features

def process_all_splits_parallel(dataset='train'):
    """Process all 20 splits in parallel and combine."""
    
    base_dir = Path(__file__).parent
    
    # Load metadata once (it's in root)
    log_file = base_dir / f"{dataset}_log.csv"
    meta_df = pd.read_csv(log_file)
    print(f"Total objects in metadata: {len(meta_df):,}")
    
    all_features = []
    splits_found = 0
    
    for split_num in range(1, 21):
        split_dir = base_dir / f"split_{split_num:02d}"
        
        # Try both compressed and uncompressed
        lc_file_gz = split_dir / f"{dataset}_full_lightcurves.csv.gz"
        lc_file_csv = split_dir / f"{dataset}_full_lightcurves.csv"
        
        if lc_file_gz.exists():
            lc_file = lc_file_gz
        elif lc_file_csv.exists():
            lc_file = lc_file_csv
        else:
            continue
        
        splits_found += 1
        print(f"\n[Split {split_num:02d}] Loading {lc_file.name}...")
        
        lc_df = pd.read_csv(lc_file)
        
        # Get object IDs in this split
        split_object_ids = lc_df['object_id'].unique()
        split_meta = meta_df[meta_df['object_id'].isin(split_object_ids)]
        
        print(f"  Objects: {len(split_object_ids):,}, Observations: {len(lc_df):,}")
        
        # Merge metadata for extinction correction
        lc_df = lc_df.merge(split_meta[['object_id', 'EBV']], on='object_id', how='left')
        
        # Apply extinction correction
        print("  Applying extinction correction...")
        lc_df = apply_extinction_correction(lc_df)
        
        # Extract features
        print("  Extracting physics features...")
        for obj_id, group in tqdm(lc_df.groupby('object_id'), desc=f"  Split {split_num:02d}"):
            features = {'object_id': obj_id}
            
            # Color/spectral features
            features.update(extract_flux_ratio_features(group))
            
            # Temporal features
            features.update(extract_temporal_features(group))
            
            all_features.append(features)
    
    print(f"\n✓ Processed {splits_found} splits")
    
    # Combine all features
    features_df = pd.DataFrame(all_features)
    
    # Add metadata
    features_df = features_df.merge(meta_df[['object_id', 'Z', 'EBV']], on='object_id', how='left')
    
    # Add target if training
    if dataset == 'train' and 'target' in meta_df.columns:
        features_df = features_df.merge(meta_df[['object_id', 'target']], on='object_id', how='left')
    
    return features_df


def main():
    print("="*70)
    print("PHYSICS FEATURES - ALL SPLITS")
    print("="*70)
    
    # Process training data
    print("\n[TRAIN - ALL SPLITS]")
    train_features = process_all_splits_parallel('train')
    
    output_file = 'physics_features_train_full.csv'
    train_features.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total objects: {len(train_features):,}")
    print(f"  Features: {train_features.shape[1]-1} (excluding object_id)")
    
    if 'target' in train_features.columns:
        print(f"\n  Target distribution:")
        print(f"    Non-TDE: {(train_features['target']==0).sum():,}")
        print(f"    TDE: {(train_features['target']==1).sum():,}")
        print(f"    TDE ratio: {train_features['target'].mean()*100:.3f}%")
    
    # Process test data
    print("\n[TEST - ALL SPLITS]")
    test_features = process_all_splits_parallel('test')
    
    output_file = 'physics_features_test_full.csv'
    test_features.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total objects: {len(test_features):,}")
    
    print("\n" + "="*70)
    print("✅ ALL SPLITS PROCESSED")
    print("="*70)
    print("\nNext: Train LightGBM on full dataset")


if __name__ == "__main__":
    main()

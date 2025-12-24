
import pandas as pd

def main():
    print("Merging Feature Sets for Robust Multi-Class Training...")
    
    # Load standardized feature sets
    phys_train = pd.read_csv('physics_features_train_full.csv')
    phys_test = pd.read_csv('physics_features_test_full.csv')
    
    lc_train = pd.read_csv('lightcurve_features_train.csv')
    lc_test = pd.read_csv('lightcurve_features_test.csv')
    
    print(f"Physics Train: {phys_train.shape}")
    print(f"LC Train: {lc_train.shape}")
    
    # Merge on object_id
    # Note: lc_train has 'target' and extra cols.
    drop_cols = ['target', 'split', 'label', 'English Translation', 'SpecType', 'Z', 'Z_err', 'EBV']
    lc_train_cln = lc_train.drop(columns=[c for c in drop_cols if c in lc_train.columns])
    lc_test_cln = lc_test.drop(columns=[c for c in drop_cols if c in lc_test.columns])
    
    train_full = phys_train.merge(lc_train_cln, on='object_id', how='left')
    test_full = phys_test.merge(lc_test_cln, on='object_id', how='left')
    
    # Check shape
    print(f"Combined Train: {train_full.shape}")
    
    # Save for training
    train_full.to_csv('features_combined_train.csv', index=False)
    test_full.to_csv('features_combined_test.csv', index=False)
    print("Saved features_combined_train.csv and test.csv")

if __name__ == "__main__":
    main()

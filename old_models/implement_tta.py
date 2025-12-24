"""
Implement Test-Time Augmentation (TTA) for Tabular Data.
Strategy:
1. Train strong LightGBM Multi-Class model (our best baseline).
2. For test set, generate augmented versions:
   - Original
   - Flux scale +5%
   - Flux scale -5%
   - Add Gaussian noise to flux (sigma=0.01)
   - Add Gaussian noise to flux (sigma=0.02)
3. Predict probas for all versions.
4. Average probas (Soft Voting).
5. Generate submissions for N=400, 405, 410.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    print("="*80)
    print("TEST-TIME AUGMENTATION (TTA) IMPLEMENTATION")
    print("="*80)

    # 1. Load Data
    print("Loading data...")
    phys_train = pd.read_csv('physics_features_train_full.csv')
    log_train = pd.read_csv('train_log.csv')
    
    if 'target' in phys_train.columns:
        phys_train = phys_train.drop(columns=['target'])
        
    # Merge training data
    train_data = phys_train.merge(log_train[['object_id', 'target', 'SpecType']], on='object_id', how='inner')
    
    # Load test data
    phys_test = pd.read_csv('physics_features_test_full.csv')

    # Prepare features
    # Exclude metadata like Z unless we are sure (we used Z in best model? No, usually dropped)
    # Check what features were used in refined model. 
    # Usually: physics features + lightcurve stats.
    # Let's use all numeric columns in phys_train except object_id.
    
    feature_cols = [c for c in phys_train.columns if c not in ['object_id', 'target', 'Z', 'EBV', 'Z_err']]
    # Note: If Z was used in best model, we should include it. 
    # Refined model used 67 features.
    # Let's check overlap.
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Feature list sample: {feature_cols[:5]}")
    
    X_train = train_data[feature_cols].fillna(train_data[feature_cols].median())
    
    # 2. Prepare Targets (Multi-Class)
    def simplify_spectype(spec):
        if spec == 'TDE': return 0
        elif 'AGN' in spec: return 1
        elif 'Ia' in spec: return 2
        else: return 3
        
    y_train = train_data['SpecType'].apply(simplify_spectype).values
    
    # 3. Train Model
    print("\nTraining base model (LightGBM Multi-Class)...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective='multiclass',
        num_class=4,
        is_unbalance=True,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # 4. TTA Pipeline
    print("\nStarting TTA Pipeline...")
    
    X_test_base = phys_test[feature_cols].fillna(phys_train[feature_cols].median()) # Use train median for test fill
    
    # Identification of Flux columns for augmentation
    flux_cols = [c for c in feature_cols if 'flux' in c.lower() or 'amp' in c.lower()]
    print(f"Identified {len(flux_cols)} flux-related columns for augmentation.")
    
    predictions = []
    
    # Augmentation 1: Original
    print("  Predicting: Original")
    predictions.append(model.predict_proba(X_test_base))
    
    # Augmentation 2: Flux Scale +5%
    print("  Predicting: Flux +5%")
    X_aug = X_test_base.copy()
    X_aug[flux_cols] = X_aug[flux_cols] * 1.05
    predictions.append(model.predict_proba(X_aug))
    
    # Augmentation 3: Flux Scale -5%
    print("  Predicting: Flux -5%")
    X_aug = X_test_base.copy()
    X_aug[flux_cols] = X_aug[flux_cols] * 0.95
    predictions.append(model.predict_proba(X_aug))
    
    # Augmentation 4: Gaussian Noise (1%)
    print("  Predicting: Noise 1%")
    X_aug = X_test_base.copy()
    noise = np.random.normal(0, 0.01 * X_aug[flux_cols].std(), size=X_aug[flux_cols].shape)
    X_aug[flux_cols] = X_aug[flux_cols] + noise
    predictions.append(model.predict_proba(X_aug))
    
    # Augmentation 5: Gaussian Noise (2%)
    print("  Predicting: Noise 2%")
    X_aug = X_test_base.copy()
    noise = np.random.normal(0, 0.02 * X_aug[flux_cols].std(), size=X_aug[flux_cols].shape)
    X_aug[flux_cols] = X_aug[flux_cols] + noise
    predictions.append(model.predict_proba(X_aug))
    
    # Average Predictions
    print("\nAveraging predictions...")
    avg_probs = np.mean(predictions, axis=0)
    
    # Extract P(TDE) (Class 0)
    p_tde = avg_probs[:, 0]
    
    # 5. Generate Submissions (N=400, 405, 410)
    print("\nGenerating submissions...")
    
    submission_df = pd.DataFrame({'object_id': phys_test['object_id']})
    
    for n in [400, 405, 410]:
        submission_df['prediction'] = 0
        
        # Select Top-N
        top_idx = np.argsort(p_tde)[::-1][:n]
        submission_df.loc[top_idx, 'prediction'] = 1
        
        filename = f'submission_tta_top{n}.csv'
        submission_df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
        
        # Reset for next N
        submission_df['prediction'] = 0

    print("\n✅ TTA Process Complete.")
    print("Recommended: submission_tta_top405.csv")

if __name__ == "__main__":
    main()

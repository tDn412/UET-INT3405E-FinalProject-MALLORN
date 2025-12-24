"""
ANGLE 1: Error Analysis (CRITICAL - Never Done!)
Extract Out-of-Fold predictions and analyze error patterns
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

def main():
    print("=" * 80)
    print("ERROR ANALYSIS: Finding Systematic Patterns")
    print("=" * 80)
    
    # Load physics features (our best working features)
    phys = pd.read_csv('physics_features_train_full.csv')
    log = pd.read_csv('train_log.csv')
    
    # Merge on object_id
    # Physics may have target already, drop it
    if 'target' in phys.columns:
        phys = phys.drop(columns=['target'])
    
    data = phys.merge(log, on='object_id', how='inner')
    
    print(f"Merged data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()[:10]}...")


    
    # Features (exclude non-feature columns, but keep Z for analysis)
    exclude = ['object_id', 'target', 'split', 'SpecType', 'English Translation']
    feats = [c for c in data.columns if c not in exclude and not c.startswith('fold')]
    
    X = data[feats].fillna(data[feats].median())
    y = data['target'].values
    
    print(f"\nDataset: {len(data)} objects, {len(feats)} features")
    print(f"TDEs: {y.sum()}")
    
    # Train with CV to get OOF predictions
    print("\nGenerating Out-of-Fold predictions...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))
    oof_preds = np.zeros(len(X))
    
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5...", end='')
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Get val probabilities
        val_proba = model.predict_proba(X.iloc[val])[:, 1]  # P(TDE)
        oof_probs[val] = val_proba
        
        print(" Done")
    
    # Rank and select Top-400 (our best N)
    sorted_idx = np.argsort(oof_probs)[::-1]
    oof_preds[sorted_idx[:400]] = 1
    
    # Calculate metrics
    f1 = f1_score(y, oof_preds)
    cm = confusion_matrix(y, oof_preds)
    
    print(f"\nOOF Performance:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Analyze errors
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    # False Positives (predicted TDE, actually not)
    fp_mask = (oof_preds == 1) & (y == 0)
    fp_data = data[fp_mask].copy()
    fp_data['prob'] = oof_probs[fp_mask]
    fp_data = fp_data.sort_values('prob', ascending=False)
    
    print(f"\nFALSE POSITIVES: {fp_mask.sum()} objects")
    print("\nTop 20 False Positives:")
    print(fp_data[['object_id', 'prob', 'SpecType', 'Z']].head(20).to_string(index=False))
    
    # Analyze SpecType distribution
    print(f"\nFalse Positive SpecType Distribution:")
    fp_spec_dist = fp_data['SpecType'].value_counts()
    for spec, count in fp_spec_dist.head(10).items():
        pct = count / len(fp_data) * 100
        print(f"  {spec:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Redshift analysis
    print(f"\nFalse Positive Redshift Stats:")
    print(f"  Mean: {fp_data['Z'].mean():.3f}")
    print(f"  Median: {fp_data['Z'].median():.3f}")
    print(f"  Std: {fp_data['Z'].std():.3f}")
    
    # Compare to True Positives
    tp_mask = (oof_preds == 1) & (y == 1)
    tp_data = data[tp_mask].copy()
    
    print(f"\nTrue Positive Redshift Stats (for comparison):")
    print(f"  Mean: {tp_data['Z'].mean():.3f}")
    print(f"  Median: {tp_data['Z'].median():.3f}")
    
    # False Negatives (missed TDEs)
    fn_mask = (oof_preds == 0) & (y == 1)
    fn_data = data[fn_mask].copy()
    fn_data['prob'] = oof_probs[fn_mask]
    fn_data = fn_data.sort_values('prob', ascending=False)
    
    print(f"\n\nFALSE NEGATIVES: {fn_mask.sum()} missed TDEs")
    print("\nTop 20 Missed TDEs (highest prob):")
    print(fn_data[['object_id', 'prob', 'SpecType', 'Z']].head(20).to_string(index=False))
    
    print(f"\nMissed TDE Redshift Stats:")
    print(f"  Mean: {fn_data['Z'].mean():.3f}")
    print(f"  Median: {fn_data['Z'].median():.3f}")
    
    # Save detailed analysis
    fp_data.to_csv('error_analysis_false_positives.csv', index=False)
    fn_data.to_csv('error_analysis_false_negatives.csv', index=False)
    
    print("\n" + "=" * 80)
    print("ACTIONABLE INSIGHTS")
    print("=" * 80)
    
    # Check if FPs are concentrated in specific SpecTypes
    dominant_fp_spec = fp_spec_dist.iloc[0]
    if dominant_fp_spec[1] / len(fp_data) > 0.3:
        print(f"\n⚠️  {dominant_fp_spec[0]} accounts for {dominant_fp_spec[1]/len(fp_data)*100:.1f}% of FPs!")
        print(f"   → Consider adding feature: is_confused_{dominant_fp_spec[0]}")
    
    # Check redshift separation
    z_separation = abs(fp_data['Z'].mean() - tp_data['Z'].mean())
    if z_separation > 0.2:
        print(f"\n⚠️  Large redshift gap: FP_z={fp_data['Z'].mean():.2f} vs TP_z={tp_data['Z'].mean():.2f}")
        print(f"   → Consider redshift-based features")
    
    print("\n✅ Saved detailed analysis:")
    print("   - error_analysis_false_positives.csv")
    print("   - error_analysis_false_negatives.csv")
    print("\nManually inspect these files for patterns!")

if __name__ == "__main__":
    main()

"""
Generate Polynomial Features (Interaction Terms)
Focus: Top 20 features from Refined Model
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

def main():
    print("="*80)
    print("GENERATING POLYNOMIAL FEATURES")
    print("="*80)
    
    # 1. Load Data
    try:
        train_df = pd.read_csv('features_combined_train.csv')
        test_df = pd.read_csv('features_combined_test.csv')
        log_df = pd.read_csv('train_log.csv')
    except FileNotFoundError:
        print("Files not found.")
        return

    # Add labels for feature selection training
    if 'target' in train_df.columns:
        train_df = train_df.drop(columns=['target'])
        
    train_df = train_df.merge(log_df[['object_id', 'target']], on='object_id', how='left')
    
    feature_cols = [c for c in train_df.columns if c not in ['object_id', 'target', 'split', 'fold', 'SpecType', 'simple_class']]
    X = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y = train_df['target'].values
    
    print(f"Base Features: {len(feature_cols)}")
    
    # 2. Identify Top Features via Quick LGBM
    print("\nIdentifying Top Features...")
    model = lgb.LGBMClassifier(n_estimators=500, verbose=-1, random_state=42, is_unbalance=True)
    model.fit(X, y)
    
    imp_df = pd.DataFrame({'feature': feature_cols, 'imp': model.feature_importances_})
    top_feats = imp_df.sort_values('imp', ascending=False).head(15)['feature'].tolist()
    
    print("Top 15 Features:")
    print(top_feats)
    
    # 3. Generate Interactions
    print("\nGenerating Polynomials (Degree 2)...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    X_top = train_df[top_feats].fillna(train_df[top_feats].median())
    X_test_top = test_df[top_feats].fillna(train_df[top_feats].median())
    
    poly_train = poly.fit_transform(X_top)
    poly_test = poly.transform(X_test_top)
    
    new_cols = poly.get_feature_names_out(top_feats)
    print(f"New Features Generated: {len(new_cols)}")
    
    # Create DF
    poly_train_df = pd.DataFrame(poly_train, columns=new_cols)
    poly_test_df = pd.DataFrame(poly_test, columns=new_cols)
    
    # Only keep the pure interaction terms (exclude original features which are included in degree=1 subset of poly, wait, interaction_only=True doesn't include x^2 but might include x? No.
    # PolynomialFeatures(interaction_only=True) gives x1, x2, x1*x2.
    # We already have x1, x2 in the main dataset. We only want interactions.
    # Actually, interaction_only=True with include_bias=False gives x1, x2, x1x2.
    # We should drop the single terms to avoid duplication.
    
    cols_to_keep = [c for c in new_cols if ' ' in c] # Interaction terms have space "x0 x1" usually? No, sklearn names them "feat1 feat2"
    
    # Let's filter manually
    interaction_cols = []
    for c in new_cols:
        if c not in top_feats:
            interaction_cols.append(c)
            
    print(f"Interaction Features (Unique): {len(interaction_cols)}")
    
    poly_train_final = poly_train_df[interaction_cols]
    poly_test_final = poly_test_df[interaction_cols]
    
    # Add object_ids
    poly_train_final['object_id'] = train_df['object_id']
    poly_test_final['object_id'] = test_df['object_id']
    
    # Save
    poly_train_final.to_csv('features_poly_train.csv', index=False)
    poly_test_final.to_csv('features_poly_test.csv', index=False)
    print("Saved features_poly_train.csv and features_poly_test.csv")

if __name__ == "__main__":
    main()

"""
RADICAL APPROACH: Back to Basics + Smart Features
Hypothesis: Statistical light curve features are overfitting
New strategy: Focus on METADATA + simple aggregates only
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RADICAL NEW APPROACH - Metadata Focus")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc, on='object_id', how='inner')
test_full = test_meta.merge(test_lc, on='object_id', how='inner')

# ============================================
# STRATEGY 1: METADATA ONLY (Better Engineering)
# ============================================
print("\n[2] Strategy 1: Metadata-Only (Enhanced)...")

def create_metadata_features(df):
    """Create smart metadata features"""
    features = pd.DataFrame(index=df.index)
    
    # Original
    features['Z'] = df['Z']
    features['EBV'] = df['EBV']
    
    # Ratios and interactions
    features['Z_EBV_ratio'] = df['Z'] / (df['EBV'] + 1e-5)
    features['Z_times_EBV'] = df['Z'] * df['EBV']
    
    # Polynomials
    features['Z_squared'] = df['Z'] ** 2
    features['Z_cubed'] = df['Z'] ** 3
    features['EBV_squared'] = df['EBV'] ** 2
    
    # Log transforms
    features['log_Z'] = np.log1p(df['Z'])
    features['log_EBV'] = np.log1p(df['EBV'])
    
    # Bins/categorical encoding
    features['Z_bin'] = pd.cut(df['Z'], bins=5, labels=False)
    features['EBV_bin'] = pd.cut(df['EBV'], bins=5, labels=False)
    
    return features

X_meta_train = create_metadata_features(train_full)
X_meta_test = create_metadata_features(test_full)
y = train_full['target']

print(f"  Features: {len(X_meta_train.columns)}")

# Train with metadata only
scale_pos = (y == 0).sum() / (y == 1).sum()

model_meta = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8
)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_meta = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_train, y), 1):
    X_tr, X_val = X_meta_train.iloc[train_idx], X_meta_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model_meta.fit(X_tr, y_tr)
    y_pred = model_meta.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    cv_scores_meta.append(score)
    print(f"  Fold {fold}: {score:.4f}")

print(f"  Mean CV: {np.mean(cv_scores_meta):.4f}")

# Train final
model_meta.fit(X_meta_train, y)
meta_proba = model_meta.predict_proba(X_meta_test)[:, 1]

# ============================================
# STRATEGY 2: Metadata + MINIMAL Light Curve
# ============================================
print("\n[3] Strategy 2: Metadata + Top 3 Light Curve Features...")

# Add ONLY these light curve features (simplest aggregates)
def create_minimal_lc_features(df_full, lc_df):
    """Only the most basic light curve info"""
    features = pd.DataFrame(index=df_full.index)
    
    # Just counts and basic stats
    features['n_observations'] = lc_df['n_observations']
    features['observation_duration'] = lc_df['observation_duration']
    features['flux_range'] = lc_df['flux_range']
    
    return features

lc_minimal_train = create_minimal_lc_features(train_full, train_lc)
lc_minimal_test = create_minimal_lc_features(test_full, test_lc)

X_minimal_train = pd.concat([X_meta_train, lc_minimal_train], axis=1)
X_minimal_test = pd.concat([X_meta_test, lc_minimal_test], axis=1)

print(f"  Features: {len(X_minimal_train.columns)}")

model_minimal = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8
)

# CV
cv_scores_minimal = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_minimal_train, y), 1):
    X_tr, X_val = X_minimal_train.iloc[train_idx], X_minimal_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model_minimal.fit(X_tr, y_tr)
    y_pred = model_minimal.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    cv_scores_minimal.append(score)
    print(f"  Fold {fold}: {score:.4f}")

print(f"  Mean CV: {np.mean(cv_scores_minimal):.4f}")

model_minimal.fit(X_minimal_train, y)
minimal_proba = model_minimal.predict_proba(X_minimal_test)[:, 1]

# ============================================
# STRATEGY 3: Ratio Features ONLY
# ============================================
print("\n[4] Strategy 3: Ratio-Based Features...")

def create_ratio_features(df_full, lc_df):
    """Create features that are ratios/relative measures"""
    features = pd.DataFrame(index=df_full.index)
    
    # Metadata ratios
    features['Z_EBV_ratio'] = df_full['Z'] / (df_full['EBV'] + 1e-5)
    
    # Light curve ratios (avoid absolute values)
    features['flux_peak_to_median'] = lc_df['flux_max'] / (lc_df['flux_median'].abs() + 1e-5)
    features['flux_variability_ratio'] = lc_df['flux_std'] / (lc_df['flux_mean'].abs() + 1e-5)
    features['signal_to_noise_ratio'] = lc_df['flux_mean'] / (lc_df['flux_err_mean'] + 1e-5)
    
    # Detection ratios
    features['detection_efficiency'] = lc_df['n_significant_detections'] / (lc_df['n_observations'] + 1)
    
    # Color ratios
    features['color_gr_ratio'] = (lc_df['flux_mean_g'] + 1) / (lc_df['flux_mean_r'] + 1)
    features['color_ri_ratio'] = (lc_df['flux_mean_r'] + 1) / (lc_df['flux_mean_i'] + 1)
    
    return features.fillna(0).replace([np.inf, -np.inf], 0)

X_ratio_train = create_ratio_features(train_full, train_lc)
X_ratio_test = create_ratio_features(test_full, test_lc)

print(f"  Features: {len(X_ratio_train.columns)}")

model_ratio = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8
)

cv_scores_ratio = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ratio_train, y), 1):
    X_tr, X_val = X_ratio_train.iloc[train_idx], X_ratio_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model_ratio.fit(X_tr, y_tr)
    y_pred = model_ratio.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    cv_scores_ratio.append(score)
    print(f"  Fold {fold}: {score:.4f}")

print(f"  Mean CV: {np.mean(cv_scores_ratio):.4f}")

model_ratio.fit(X_ratio_train, y)
ratio_proba = model_ratio.predict_proba(X_ratio_test)[:, 1]

# ============================================
# CREATE SUBMISSIONS
# ============================================
print("\n[5] Creating submissions...")

submissions = [
    ('submission_metadata_only.csv', meta_proba, 0.5),
    ('submission_metadata_minimal_lc.csv', minimal_proba, 0.5),
    ('submission_ratio_features.csv', ratio_proba, 0.5),
]

for filename, proba, threshold in submissions:
    binary = (proba >= threshold).astype(int)
    sub = pd.DataFrame({
        'object_id': test_full['object_id'],
        'prediction': binary
    })
    sub.to_csv(filename, index=False)
    print(f"✓ {filename}: {binary.sum()} TDEs ({binary.mean()*100:.1f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
THREE NEW APPROACHES (all avoiding statistical moments):

1. METADATA ONLY (Enhanced)
   - CV: {np.mean(cv_scores_meta):.4f}
   - 11 features from Z and EBV only
   - No light curves at all!
   
2. METADATA + MINIMAL LC
   - CV: {np.mean(cv_scores_minimal):.4f}
   - Just counts and duration
   - Avoids statistical features
   
3. RATIO FEATURES
   - CV: {np.mean(cv_scores_ratio):.4f}
   - All relative measures
   - No absolute values

HYPOTHESIS:
Statistical light curve features (std, skew, kurtosis) are dataset-specific
and don't transfer to test set!

TRY IN ORDER:
1. submission_metadata_only.csv (if problem is in LC features)
2. submission_ratio_features.csv (relative measures might transfer better)
3. submission_metadata_minimal_lc.csv (hybrid approach)
""")
print("="*70)

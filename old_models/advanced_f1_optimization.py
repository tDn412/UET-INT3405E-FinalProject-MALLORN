"""
Advanced F1 Optimization - Multiple Strategies
Goal: Break 0.42 barrier!
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED F1 OPTIMIZATION")
print("="*70)

# Load data
print("\n[1] Loading data...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

train_full = train_meta.merge(train_lc, on='object_id', how='inner')
test_full = test_meta.merge(test_lc, on='object_id', how='inner')

# Enhanced feature engineering
print("\n[2] Enhanced feature engineering...")

def create_enhanced_features(df_meta, df_lc):
    """Create enhanced features focusing on discriminative power"""
    features = pd.DataFrame(index=df_meta.index)
    
    # Metadata
    features['Z'] = df_meta['Z']
    features['EBV'] = df_meta['EBV']
    features['Z_EBV_ratio'] = df_meta['Z'] / (df_meta['EBV'] + 1e-5)
    features['Z_squared'] = df_meta['Z'] ** 2
    
    # Light curve - focus on VARIABILITY (key for TDE)
    features['flux_range'] = df_lc['flux_range']
    features['flux_std'] = df_lc['flux_std']
    features['flux_mad'] = df_lc['flux_mad']
    features['flux_cv'] = df_lc['flux_cv']
    
    # Peak features (TDE has characteristic peaks)
    features['flux_max'] = df_lc['flux_max']
    features['flux_p75'] = df_lc['flux_p75']
    features['n_peaks'] = df_lc['n_peaks']
    
    # Time-based
    features['observation_duration'] = df_lc['observation_duration']
    features['mean_time_interval'] = df_lc['mean_time_interval']
    
    # Detection quality
    features['frac_significant_detections'] = df_lc['frac_significant_detections']
    features['signal_to_noise_mean'] = df_lc['signal_to_noise_mean']
    
    # Multi-band features (colors are important!)
    features['flux_mean_r'] = df_lc['flux_mean_r']
    features['flux_mean_g'] = df_lc['flux_mean_g']
    features['flux_std_r'] = df_lc['flux_std_r']
    features['color_g_r'] = df_lc['color_g_r']
    
    # Ratios (more robust)
    features['peak_to_median'] = df_lc['flux_max'] / (df_lc['flux_median'].abs() + 1e-5)
    features['std_to_mean'] = df_lc['flux_std'] / (df_lc['flux_mean'].abs() + 1e-5)
    
    return features.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

X = create_enhanced_features(train_full, train_lc)
X_test = create_enhanced_features(test_full, test_lc)
y = train_full['target']

print(f"Features: {len(X.columns)}")
print(f"Selected features: {list(X.columns)[:5]}...")

# ==========================================
# STRATEGY 1: Hyperparameter Tuning for F1
# ==========================================
print("\n[3] Strategy 1: Hyperparameter tuning with F1 metric...")

# Custom F1 scorer
f1_scorer = make_scorer(f1_score)

# Define param grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5, 10],
    'scale_pos_weight': [15, 19.6, 25, 30]  # Around the true ratio
}

base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Randomized search (faster than grid search)
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 combinations
    scoring=f1_scorer,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Running randomized search...")
random_search.fit(X, y)

print(f"\nBest F1 score: {random_search.best_score_:.4f}")
print(f"Best parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

best_model = random_search.best_estimator_

# ==========================================
# STRATEGY 2: Find optimal threshold  
# ==========================================
print("\n[4] Finding optimal threshold for best model...")

# Get CV predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_val_proba = []
all_val_y = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    best_model.fit(X_tr, y_tr)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    all_val_proba.extend(val_proba)
    all_val_y.extend(y_val)

all_val_proba = np.array(all_val_proba)
all_val_y = np.array(all_val_y)

# Find best threshold
thresholds = np.arange(0.05, 0.5, 0.02)
best_f1 = 0
best_threshold = 0.1

for threshold in thresholds:
    y_pred = (all_val_proba >= threshold).astype(int)
    f1 = f1_score(all_val_y, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1 score: {best_f1:.4f}")

# ==========================================
# STRATEGY 3: Train final model
# ==========================================
print("\n[5] Training final optimized model...")

final_model = xgb.XGBClassifier(**random_search.best_params_, random_state=42)
final_model.fit(X, y)

# Predict on test
test_proba = final_model.predict_proba(X_test)[:, 1]

# Create submissions with multiple thresholds
print("\n[6] Creating optimized submissions...")

thresholds_to_try = [
    (best_threshold, 'submission_f1_tuned_optimal.csv'),
    (best_threshold - 0.02, 'submission_f1_tuned_lower.csv'),
    (best_threshold + 0.02, 'submission_f1_tuned_higher.csv'),
]

for threshold, filename in thresholds_to_try:
    binary = (test_proba >= threshold).astype(int)
    sub = pd.DataFrame({
        'object_id': test_full['object_id'],
        'prediction': binary
    })
    sub.to_csv(filename, index=False)
    
    tde_count = binary.sum()
    print(f"✓ {filename}")
    print(f"  Threshold: {threshold:.3f}, TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")

# ==========================================
# Feature importance
# ==========================================
print("\n[7] Top 10 important features:")
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
ADVANCED OPTIMIZATION COMPLETE!

Previous best: 0.4146
Expected improvement: {best_f1:.4f} (CV)

KEY IMPROVEMENTS:
1. Hyperparameter tuning WITH F1 metric (not ROC-AUC!)
2. Enhanced feature engineering (20 best features)
3. Optimal threshold: {best_threshold:.3f}

SUBMIT IN ORDER:
1. submission_f1_tuned_optimal.csv (best threshold)
2. submission_f1_tuned_lower.csv (slightly lower)
3. submission_f1_tuned_higher.csv (slightly higher)

This should improve beyond 0.42!
""")
print("="*70)

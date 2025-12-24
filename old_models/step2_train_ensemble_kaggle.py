"""
STEP 2 (KAGGLE OPTIMIZED): DIVERSE ENSEMBLE TRAINING
====================================================

Fixes: XGBoost early_stopping_rounds issue.
Input: Output from STEP 1 (lightcurve_features_improved_*.csv)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import gc
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 2: ENSEMBLE TRAINING (FIXED)")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA (Robust Path)
# ============================================================================

print("\n📥 Loading features...")
KAGGLE_WORKING = '/kaggle/working'
KAGGLE_INPUT = '/kaggle/input/mallorn-dataset'

paths = [
    KAGGLE_WORKING, # Priority 1: Generated in Step 1
    '.',            # Priority 2: Current dir
    KAGGLE_INPUT    # Priority 3: Uploaded dataset
]

Xfull = None
Xtestfinal = None

for path in paths:
    train_f = os.path.join(path, 'lightcurve_features_improved_train.csv')
    test_f = os.path.join(path, 'lightcurve_features_improved_test.csv')
    if os.path.exists(train_f) and os.path.exists(test_f):
        print(f"   ✅ Found files in: {path}")
        Xfull = pd.read_csv(train_f)
        Xtestfinal = pd.read_csv(test_f)
        break

if Xfull is None:
    print("\n❌ CRITICAL: Input files not found!")
    print("   Please run STEP 1 successfully first.")
    exit(1)

# Prepare data
drop_cols = ['objectid', 'target', 'split', 'SpecType', 'English Translation']
all_features = [c for c in Xfull.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(Xfull[c])]

X = Xfull[all_features]
y = Xfull['target'].astype(int)
Xtest_raw = Xtestfinal[all_features]

print(f"   Train: {X.shape}, Test: {Xtest_raw.shape}")

# ============================================================================
# PART 2: FEATURE SELECTION
# ============================================================================
print("\n🔍 Feature Selection (LGBM)...")
TOP_K = 120

fsmodel = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, 
                             objective='binary', is_unbalanced=True, 
                             random_state=42, n_jobs=-1, verbosity=-1)
fsmodel.fit(X, y)
indices = np.argsort(fsmodel.feature_importances_)[::-1]
top_features = [all_features[i] for i in indices[:TOP_K]]
print(f"   Selected {len(top_features)} features.")

X = X[top_features]
Xtest_raw = Xtest_raw[top_features]

# ============================================================================
# PART 3: ENSEMBLE DEFINITION
# ============================================================================

feature_groups = {
    'all': top_features,
    'time_series': [c for c in top_features if any(k in c.lower() for k in ['duration', 'time', 'rise', 'decay', 'lag'])],
    'colors': [c for c in top_features if any(k in c.lower() for k in ['color', 'g-r', 'u-g'])]
}
# Fallback if specific groups are empty
if not feature_groups['time_series']: feature_groups['time_series'] = top_features
if not feature_groups['colors']: feature_groups['colors'] = top_features

class DiverseEnsemble:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.models = []
        self.weights = {}
        self.oof_pred = None
        
    def get_configs(self):
        return [
            {
                'name': 'LGB_TimeSeries', 'algo': 'lgb', 'group': 'time_series',
                'params': {'num_leaves': 63, 'learning_rate': 0.03, 'lambda_l1': 1.5, 'lambda_l2': 4.0},
                'n_est': 1000
            },
            {
                'name': 'XGB_Colors', 'algo': 'xgb', 'group': 'colors',
                'params': {'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'reg_alpha': 2, 'reg_lambda': 5},
                'n_est': 800
            },
            {
                'name': 'CAT_Robust', 'algo': 'cat', 'group': 'all',
                'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 5, 'subsample': 0.8, 'verbose': False},
                'n_est': 1000
            }
        ]
    
    def fit(self, X, y, ratio_weight):
        configs = self.get_configs()
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        oof_preds = {c['name']: np.zeros(len(X)) for c in configs}
        fold_scores = {c['name']: [] for c in configs}
        self.models = {c['name']: [] for c in configs}
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            print(f"   🔄 Fold {fold+1}/{self.n_folds}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            for cfg in configs:
                ft = feature_groups[cfg['group']]
                Xt_sub, Xv_sub = X_tr[ft], X_va[ft]
                
                # --- MODEL TRAINING ---
                if cfg['algo'] == 'lgb':
                    model = lgb.LGBMClassifier(**cfg['params'], n_estimators=cfg['n_est'], 
                                               objective='binary', scale_pos_weight=ratio_weight, 
                                               verbosity=-1, random_state=42)
                    model.fit(Xt_sub, y_tr, eval_set=[(Xv_sub, y_va)], 
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                    
                elif cfg['algo'] == 'xgb':
                    # FIX: Pass early_stopping_rounds to constructor
                    model = xgb.XGBClassifier(**cfg['params'], n_estimators=cfg['n_est'],
                                              objective='binary:logistic', eval_metric='auc',
                                              scale_pos_weight=ratio_weight, random_state=42,
                                              early_stopping_rounds=100) # <--- MOVED HERE
                    
                    # FIX: Removed early_stopping_rounds from fit()
                    model.fit(Xt_sub, y_tr, eval_set=[(Xv_sub, y_va)], verbose=False)
                    
                elif cfg['algo'] == 'cat':
                    model = CatBoostClassifier(**cfg['params'], iterations=cfg['n_est'],
                                               scale_pos_weight=ratio_weight, random_seed=42)
                    model.fit(Xt_sub, y_tr, eval_set=(Xv_sub, y_va), 
                              early_stopping_rounds=100, verbose=False)
                
                # Predict
                val_prob = model.predict_proba(Xv_sub)[:, 1]
                oof_preds[cfg['name']][va_idx] = val_prob
                score = f1_score(y_va, (val_prob > 0.5).astype(int))
                fold_scores[cfg['name']].append(score)
                self.models[cfg['name']].append((model, ft))
                
        # Weighting
        print("\n📊 Model Weights:")
        for name in fold_scores:
            score = np.mean(fold_scores[name])
            self.weights[name] = score
            print(f"   {name}: {score:.4f}")
            
        total = sum(self.weights.values())
        self.weights = {k: v/total for k,v in self.weights.items()}
        
        self.oof_pred = np.zeros(len(X))
        for name in oof_preds:
            self.oof_pred += oof_preds[name] * self.weights[name]
            
        return self.oof_pred

    def predict(self, X_new):
        final_pred = np.zeros(len(X_new))
        for name, models in self.models.items():
            model_pred = np.zeros(len(X_new))
            for model, feats in models:
                model_pred += model.predict_proba(X_new[feats])[:, 1] / len(models)
            final_pred += model_pred * self.weights[name]
        return final_pred

# ============================================================================
# PART 4: EXECUTION
# ============================================================================

ratio = (len(y)-y.sum())/y.sum()
print(f"\n⚙️ Class Ratio: {ratio:.2f} (Using scale_pos_weight)")

ensemble = DiverseEnsemble()
oof = ensemble.fit(X, y, ratio**1.5)

# Threshold Search
best_f1, best_t = 0, 0.5
for t in np.arange(0.3, 0.8, 0.01):
    score = f1_score(y, (oof > t).astype(int))
    if score > best_f1: best_f1, best_t = score, t

print(f"\n🏆 Best F1: {best_f1:.4f} @ Threshold {best_t:.2f}")

# Inference
print("\n🔮 Predicting Test...")
test_probs = ensemble.predict(Xtest_raw)

# Save
print("\n💾 Saving...")
np.save('oof_predictions_train.npy', oof)
np.save('test_predictions_raw.npy', test_probs)
with open('best_threshold.txt', 'w') as f: f.write(f"{best_t}\n{best_f1}")

print(f"✅ STEP 2 COMPLETE. Check /kaggle/working/")
print(f"   Test Mean Prob: {test_probs.mean():.4f}")

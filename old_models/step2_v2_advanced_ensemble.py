"""
STEP 2 V2 (ADVANCED ENSEMBLE): TREES + KNN
==========================================

Goal: Break 0.6 F1 by adding diversity (KNN).
Input: Output from STEP 1 V3.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import gc
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 2 V2: ADVANCED ENSEMBLE (TREES + KNN)")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================

print("\n📥 Loading Improved Features...")
KAGGLE_WORKING = '/kaggle/working'
KAGGLE_INPUT = '/kaggle/input/mallorn-dataset'

Xfull = None
Xtestfinal = None

# Priority Search
candidates = [
    os.path.join(KAGGLE_WORKING, 'lightcurve_features_improved_train.csv'),
    os.path.join('.', 'lightcurve_features_improved_train.csv'),
    os.path.join(KAGGLE_INPUT, 'lightcurve_features_improved_train.csv')
]

for p in candidates:
    if os.path.exists(p):
        print(f"   ✅ Found Train: {p}")
        Xfull = pd.read_csv(p)
        # Try finding corresponding test
        test_p = p.replace('train.csv', 'test.csv')
        if os.path.exists(test_p):
            Xtestfinal = pd.read_csv(test_p)
        break

if Xfull is None: sys.exit("❌ Features not found! Run Step 1 V3 first.")

# Process -999 to NaN for correct handling in pandas if needed, 
# but Tree models handle -999 fine. KNN needs correct imputation.
Xfull.replace(-999, np.nan, inplace=True)
Xtestfinal.replace(-999, np.nan, inplace=True)

drop_cols = ['objectid', 'target', 'split', 'SpecType', 'English Translation']
all_features = [c for c in Xfull.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(Xfull[c])]

# FILL NA for global simplicity (Mean imputation)
# (Tree models can handle NaN, but consistent input helps ensemble)
print("   Imputing NaNs...")
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(Xfull[all_features]), columns=all_features)
Xtest_raw = pd.DataFrame(imputer.transform(Xtestfinal[all_features]), columns=all_features)
y = Xfull['target'].astype(int)

print(f"   Train: {X.shape}, Test: {Xtest_raw.shape}")

# ============================================================================
# PART 2: FEATURE SELECTION
# ============================================================================

print("\n🔍 Feature Selection...")
# Basic LGBM to select top 100 features
fs = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=42)
fs.fit(X, y)
imp = fs.feature_importances_
top_idx = np.argsort(imp)[::-1][:100] # Top 100
top_features = [all_features[i] for i in top_idx]

print(f"   Selected {len(top_features)} features.")
X = X[top_features]
Xtest_raw = Xtest_raw[top_features]

# ============================================================================
# PART 3: ENSEMBLE
# ============================================================================

class AdvancedEnsemble:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.weights = {}
        self.oof_pred = None
        self.models = [] # List of (model, feature_subset_name)
        
    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        ratio = (len(y) - y.sum()) / y.sum()
        
        # DEFINITIONS
        configs = [
            {'name': 'LGBM', 'type': 'tree'},
            {'name': 'XGB', 'type': 'tree'},
            {'name': 'CAT', 'type': 'tree'},
            {'name': 'KNN', 'type': 'dist'}
        ]
        
        oof_preds = {c['name']: np.zeros(len(X)) for c in configs}
        scores = {c['name']: [] for c in configs}
        self.models_list = {c['name']: [] for c in configs} # Store trained models
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            print(f"   🔄 Fold {fold+1}/{self.n_folds}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            # 1. LGBM
            m1 = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                    scale_pos_weight=ratio**1.2, verbosity=-1, random_state=42)
            m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            p1 = m1.predict_proba(X_va)[:, 1]
            oof_preds['LGBM'][va_idx] = p1
            scores['LGBM'].append(f1_score(y_va, (p1>0.5).astype(int)))
            self.models_list['LGBM'].append(m1)
            
            # 2. XGB
            m2 = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6,
                                   scale_pos_weight=ratio**1.2, random_state=42, 
                                   early_stopping_rounds=100, eval_metric='auc') # FIX APPLIED
            m2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            p2 = m2.predict_proba(X_va)[:, 1]
            oof_preds['XGB'][va_idx] = p2
            scores['XGB'].append(f1_score(y_va, (p2>0.5).astype(int)))
            self.models_list['XGB'].append(m2)
            
            # 3. CATBOOST
            m3 = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6,
                                    scale_pos_weight=ratio**1.2, random_seed=42, verbose=False)
            m3.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
            p3 = m3.predict_proba(X_va)[:, 1]
            oof_preds['CAT'][va_idx] = p3
            scores['CAT'].append(f1_score(y_va, (p3>0.5).astype(int)))
            self.models_list['CAT'].append(m3)
            
            # 4. KNN (Scaled)
            knn_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))
            ])
            knn_pipe.fit(X_tr, y_tr)
            p4 = knn_pipe.predict_proba(X_va)[:, 1]
            oof_preds['KNN'][va_idx] = p4
            scores['KNN'].append(f1_score(y_va, (p4>0.5).astype(int)))
            self.models_list['KNN'].append(knn_pipe)

        # WEIGHTS
        print("\n📊 Model Performance (Avg F1):")
        total_score = 0
        for name in scores:
            mu = np.mean(scores[name])
            print(f"   {name}: {mu:.4f}")
            self.weights[name] = mu**2 # Softmax-like weighting (emphasize better models)
            total_score += self.weights[name]
        
        # Normalize
        self.weights = {k: v/total_score for k,v in self.weights.items()}
        print(f"   Weights: {self.weights}")
        
        # Combine OOF
        self.oof_pred = np.zeros(len(X))
        for name in oof_preds:
            self.oof_pred += oof_preds[name] * self.weights[name]
            
        return self.oof_pred

    def predict(self, X_new):
        final_pred = np.zeros(len(X_new))
        
        for name, models in self.models_list.items():
            model_pred = np.zeros(len(X_new))
            for m in models:
                model_pred += m.predict_proba(X_new)[:, 1] / len(models)
            final_pred += model_pred * self.weights[name]
            
        return final_pred

# ============================================================================
# PART 4: TRAIN & OPTIMIZE
# ============================================================================

ensemble = AdvancedEnsemble()
oof = ensemble.fit(X, y)

# Threshold Search
best_f1, best_t = 0, 0.5
for t in np.arange(0.3, 0.9, 0.01):
    score = f1_score(y, (oof > t).astype(int))
    if score > best_f1: best_f1, best_t = score, t

print(f"\n🏆 Best OOF F1: {best_f1:.4f} @ {best_t:.2f}")

# Inference
print("\n🔮 Predicting Test...")
test_probs = ensemble.predict(Xtest_raw)

# SAVE
np.save(os.path.join(KAGGLE_WORKING, 'oof_predictions_train.npy'), oof)
np.save(os.path.join(KAGGLE_WORKING, 'test_predictions_raw.npy'), test_probs)
with open(os.path.join(KAGGLE_WORKING, 'best_threshold.txt'), 'w') as f: f.write(f"{best_t}\n{best_f1}")

print("✅ Step 2 V2 Complete.")

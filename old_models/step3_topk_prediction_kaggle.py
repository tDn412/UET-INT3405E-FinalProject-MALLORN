"""
STEP 3 (KAGGLE OPTIMIZED): TOP-K PREDICTION & MULTI-SUBMISSION
==============================================================

Optimizes K (Top-K) predictions based on OOF and generates multiple submissions.
Robust file loading from Step 1 & 2 outputs.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("STEP 3: TOP-K PREDICTION & MULTI-SUBMISSION (ROBUST)")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA (Robust Path)
# ============================================================================

print("\n📥 Loading Data...")

# 1. Find NPY files (from Step 2)
npy_search_paths = ['/kaggle/working', '.', '/kaggle/input']
oof_path = None
test_pred_path = None

print("   Searching for predictions...")
for path in npy_search_paths:
    if not os.path.exists(path): continue
    if not oof_path and os.path.exists(os.path.join(path, 'oof_predictions_train.npy')):
        oof_path = os.path.join(path, 'oof_predictions_train.npy')
    if not test_pred_path and os.path.exists(os.path.join(path, 'test_predictions_raw.npy')):
        test_pred_path = os.path.join(path, 'test_predictions_raw.npy')

if not oof_path or not test_pred_path:
    print("❌ CRITICAL: Prediction files (.npy) not found!")
    print("   Please make sure STEP 2 ran successfully.")
    sys.exit(1)

print(f"   ✅ Loaded OOF: {oof_path}")
print(f"   ✅ Loaded Test Preds: {test_pred_path}")
oof_proba = np.load(oof_path)
test_proba = np.load(test_pred_path)

# 2. Find Test IDs (from Step 1 CSV or Test Log)
# We need 'objectid' for submission
test_id_path = None
# Try finding improved features csv first
for path in npy_search_paths + ['/kaggle/input/mallorn-dataset']:
    if not os.path.exists(path): continue
    hits = glob.glob(os.path.join(path, '*lightcurve_features_improved_test.csv'))
    if hits:
        test_id_path = hits[0]
        break

# Fallback to test_log.csv if improved features not found
if not test_id_path:
    print("   ⚠️ Improved features csv not found. Searching for test_log.csv...")
    for root, _, files in os.walk('/kaggle/input'):
        if 'test_log.csv' in files:
            test_id_path = os.path.join(root, 'test_log.csv')
            break

if not test_id_path:
    print("❌ CRITICAL: Could not find Source of Test IDs (features csv or test_log.csv)!")
    sys.exit(1)

print(f"   ✅ Loading Test IDs from: {test_id_path}")
df_test_full = pd.read_csv(test_id_path)
if 'objectid' not in df_test_full.columns and 'object_id' in df_test_full.columns:
    df_test_full.rename(columns={'object_id': 'objectid'}, inplace=True)
test_ids = df_test_full['objectid'].values

# 3. Find Train Targets (for OOF scoring)
train_target_path = None
y_train = None

# PRIORITY 1: Try to load from training features csv (ENSURES ALIGNMENT)
train_feat_hits = glob.glob(os.path.join('/kaggle/working', '*lightcurve_features_improved_train.csv'))
if not train_feat_hits:
    train_feat_hits = glob.glob(os.path.join('.', '*lightcurve_features_improved_train.csv'))

if train_feat_hits:
     print(f"   ✅ Loading Targets from: {train_feat_hits[0]} (Aligned)")
     df_tr = pd.read_csv(train_feat_hits[0])
     y_train = df_tr['target'].values
else:
     # Fallback to train_log (Risky alignment if rows shuffled)
     print("   ⚠️ Feature CSV not found. Falling back to train_log.csv (Check alignment!)")
     for root, _, files in os.walk('/kaggle/input'):
        if 'train_log.csv' in files:
            train_target_path = os.path.join(root, 'train_log.csv')
            break
     
     if train_target_path:
        print(f"   ✅ Loading Targets from: {train_target_path}")
        y_train = pd.read_csv(train_target_path)['target'].values
     else:
        print("❌ CRITICAL: Could not find ANY target file!")
        sys.exit(1)

print(f"\n📊 Summary:")
print(f"   OOF Samples: {len(oof_proba)}")
print(f"   Test Samples: {len(test_proba)}")
print(f"   Train Targets: {len(y_train)}")

# ============================================================================
# PART 2: OPTIMIZE K ON OOF
# ============================================================================

print("\n\n" + "="*80)
print("K OPTIMIZATION ON OOF")
print("="*80)

results = []
# Scan broad range
k_scan = list(range(100, 1500, 10))

for k in k_scan:
    if k >= len(oof_proba): break
    
    # Select Top-K
    top_k_idx = np.argsort(oof_proba)[-k:]
    y_pred = np.zeros(len(oof_proba), dtype=int)
    y_pred[top_k_idx] = 1
    
    f1 = f1_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred)
    rec = recall_score(y_train, y_pred)
    
    results.append({'K': k, 'F1': f1, 'Precision': prec, 'Recall': rec})

results_df = pd.DataFrame(results)

# Find Best
best_idx = results_df['F1'].idxmax()
best_k = int(results_df.loc[best_idx, 'K'])
best_f1 = results_df.loc[best_idx, 'F1']

print(f"\n🏆 BEST PARAMETERS FOUND:")
print(f"   Optimal K: {best_k}")
print(f"   Max OOF F1: {best_f1:.4f}")
print(f"   (Prec: {results_df.loc[best_idx, 'Precision']:.4f}, Rec: {results_df.loc[best_idx, 'Recall']:.4f})")

# Output Dir
OUTPUT_DIR = '/kaggle/working'
if not os.path.exists(OUTPUT_DIR): OUTPUT_DIR = '.'

# Plot
plt.figure(figsize=(10, 5))
plt.plot(results_df['K'], results_df['F1'], label='F1 Score')
plt.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
plt.title(f'F1 Score vs High-Confidence predictions (K)\nBest K={best_k}')
plt.xlabel('K (Top Predictions)')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'topk_optimization_plot.png'))
print(f"   ✅ Saved optimization plot to {OUTPUT_DIR}")

# ============================================================================
# PART 3: GENERATE SUBMISSIONS
# ============================================================================

print("\n\n" + "="*80)
print("GENERATING SUBMISSIONS")
print("="*80)

# 1. Best K Submission
top_k_idx = np.argsort(test_proba)[-best_k:]
test_pred = np.zeros(len(test_proba), dtype=int)
test_pred[top_k_idx] = 1

sub_df = pd.DataFrame({'objectid': test_ids, 'target': test_pred})
best_filename = f'submission_topk_{best_k}_BEST_F1_{best_f1:.4f}.csv'
sub_df.to_csv(os.path.join(OUTPUT_DIR, best_filename), index=False)
print(f"   ✅ Created: {best_filename} (Prioritize this!)")

# 2. Multi-Submission Strategy (Around Best K)
# Walkthrough suggested K=350-450. We center around our found best K.
range_start = max(100, best_k - 50)
range_end = min(len(test_proba), best_k + 51)
step = 10

print(f"\n   Generating variations from K={range_start} to {range_end}...")

for k in range(range_start, range_end, step):
    top_k_idx = np.argsort(test_proba)[-k:]
    test_pred = np.zeros(len(test_proba), dtype=int)
    test_pred[top_k_idx] = 1
    
    sub = pd.DataFrame({'objectid': test_ids, 'target': test_pred})
    fname = f'submission_topk_{k}.csv'
    sub.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    # print(f"      Saved {fname}", end='\r')

print(f"   ✅ All variation submissions saved to {OUTPUT_DIR}")

print("\n" + "="*80)
print("✅ STEP 3 COMPLETE")
print("="*80)
print(f"Summary:")
print(f"1. Download '{best_filename}' and submit to Kaggle.")
print(f"2. Compare score. If lower than {best_f1:.3f}, try neighbor K files (submission_topk_*.csv)")

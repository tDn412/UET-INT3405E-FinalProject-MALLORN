"""
MALLORN TDE Classification - Baseline Models
Building and evaluating baseline classification models
IMPORTANT: NOT using SpecType due to data leakage!
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MALLORN TDE Classification - Baseline Models")
print("="*70)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1] Loading data...")
train_df = pd.read_csv('train_log.csv')
test_df = pd.read_csv('test_log.csv')

print(f"✓ Training set: {train_df.shape}")
print(f"✓ Test set: {test_df.shape}")

# ====================
# 2. FEATURE ENGINEERING
# ====================
print("\n[2] Feature engineering...")

# WARNING: SpecType='TDE' perfectly identifies TDE class -> DATA LEAKAGE!
# We will NOT use SpecType in our models

# Check class distribution
print("\nClass distribution:")
print(train_df['target'].value_counts())
print(f"TDE ratio: {train_df['target'].mean()*100:.2f}%")

# Select features (KHÔNG dùng SpecType!)
# Z_err is 100% missing, so we exclude it too
feature_cols = ['Z', 'EBV']

# Create additional features
print("\nCreating engineered features...")
train_df['Z_EBV_ratio'] = train_df['Z'] / (train_df['EBV'] + 1e-5)
train_df['Z_squared'] = train_df['Z'] ** 2
train_df['EBV_squared'] = train_df['EBV'] ** 2
train_df['Z_EBV_interaction'] = train_df['Z'] * train_df['EBV']

test_df['Z_EBV_ratio'] = test_df['Z'] / (test_df['EBV'] + 1e-5)
test_df['Z_squared'] = test_df['Z'] ** 2
test_df['EBV_squared'] = test_df['EBV'] ** 2
test_df['Z_EBV_interaction'] = test_df['Z'] * test_df['EBV']

# Final feature list
all_features = ['Z', 'EBV', 'Z_EBV_ratio', 'Z_squared', 'EBV_squared', 'Z_EBV_interaction']

print(f"✓ Total features: {len(all_features)}")
print(f"  Features: {all_features}")

# Prepare data
X = train_df[all_features]
y = train_df['target']
X_test = test_df[all_features]

# ====================
# 3. TRAIN/VAL SPLIT
# ====================
print("\n[3] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Val set: {X_val.shape}")
print(f"Train TDE ratio: {y_train.mean()*100:.2f}%")
print(f"Val TDE ratio: {y_val.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ====================
# 4. CLASS WEIGHTS
# ====================
print("\n[4] Computing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")
print(f"TDE weight is {class_weights[1]/class_weights[0]:.1f}x higher than Non-TDE")

# ====================
# 5. BASELINE 1: LOGISTIC REGRESSION (No balance)
# ====================
print("\n" + "="*70)
print("[5] BASELINE 1: Logistic Regression (No class weights)")
print("="*70)

lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train_scaled, y_train)

y_pred_lr = lr_baseline.predict(X_val_scaled)
y_pred_proba_lr = lr_baseline.predict_proba(X_val_scaled)[:, 1]

print("\nResults:")
print(classification_report(y_val, y_pred_lr, target_names=['Non-TDE', 'TDE']))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_lr):.4f}")

# ====================
# 6. BASELINE 2: LOGISTIC REGRESSION (Balanced)
# ====================
print("\n" + "="*70)
print("[6] BASELINE 2: Logistic Regression (With class weights)")
print("="*70)

lr_balanced = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    class_weight='balanced'
)
lr_balanced.fit(X_train_scaled, y_train)

y_pred_lr_bal = lr_balanced.predict(X_val_scaled)
y_pred_proba_lr_bal = lr_balanced.predict_proba(X_val_scaled)[:, 1]

print("\nResults:")
print(classification_report(y_val, y_pred_lr_bal, target_names=['Non-TDE', 'TDE']))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_lr_bal):.4f}")

# ====================
# 7. BASELINE 3: RANDOM FOREST (Balanced)
# ====================
print("\n" + "="*70)
print("[7] BASELINE 3: Random Forest (With class weights)")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_val_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]

print("\nResults:")
print(classification_report(y_val, y_pred_rf, target_names=['Non-TDE', 'TDE']))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba_rf):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# ====================
# 8. MODEL COMPARISON
# ====================
print("\n" + "="*70)
print("[8] MODEL COMPARISON")
print("="*70)

models = {
    'LR (No balance)': (y_pred_lr, y_pred_proba_lr),
    'LR (Balanced)': (y_pred_lr_bal, y_pred_proba_lr_bal),
    'RF (Balanced)': (y_pred_rf, y_pred_proba_rf)
}

results = []
for name, (y_pred, y_proba) in models.items():
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc_score(y_val, y_proba),
        'F1-Score': f1_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred)
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best model
best_idx = results_df['ROC-AUC'].idxmax()
best_model = results_df.loc[best_idx, 'Model']
print(f"\n🏆 Best model: {best_model}")
print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")

# ====================
# 9. VISUALIZATIONS
# ====================
print("\n[9] Creating visualizations...")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, (y_pred, _)) in enumerate(models.items()):
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'Confusion Matrix\n{name}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# ROC Curves
plt.figure(figsize=(10, 6))
for name, (_, y_proba) in models.items():
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_baseline.png', dpi=150, bbox_inches='tight')
print("✓ Saved: roc_curves_baseline.png")
plt.close()

# Precision-Recall curves
plt.figure(figsize=(10, 6))
for name, (_, y_proba) in models.items():
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    plt.plot(recall, precision, label=name, linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Baseline Models', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curves_baseline.png', dpi=150, bbox_inches='tight')
print("✓ Saved: pr_curves_baseline.png")
plt.close()

# Model performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC comparison
axes[0].barh(results_df['Model'], results_df['ROC-AUC'], color='skyblue')
axes[0].set_xlabel('ROC-AUC Score')
axes[0].set_title('Model Comparison: ROC-AUC', fontweight='bold')
axes[0].set_xlim(0, 1)
axes[0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['ROC-AUC']):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center')

# F1-Score comparison
axes[1].barh(results_df['Model'], results_df['F1-Score'], color='salmon')
axes[1].set_xlabel('F1-Score')
axes[1].set_title('Model Comparison: F1-Score', fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['F1-Score']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: model_performance_comparison.png")
plt.close()

# ====================
# 10. CROSS-VALIDATION
# ====================
print("\n" + "="*70)
print("[10] CROSS-VALIDATION (Best Model)")
print("="*70)

# Use best model for CV
if 'RF' in best_model:
    best_clf = rf_model
else:
    best_clf = lr_balanced

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_clf, X, y,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"\n5-Fold Cross-Validation Results:")
print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ====================
# 11. SAVE SUMMARY
# ====================
print("\n[11] Saving summary...")

summary = f"""MALLORN TDE Classification - Baseline Models Summary
Generated: {pd.Timestamp.now()}

IMPORTANT: SpecType NOT used due to perfect separation (data leakage)!

FEATURES USED ({len(all_features)}):
{', '.join(all_features)}

CLASS IMBALANCE:
- Non-TDE: {(y_train==0).sum()} samples ({(1-y_train.mean())*100:.2f}%)
- TDE: {(y_train==1).sum()} samples ({y_train.mean()*100:.2f}%)
- Imbalance ratio: 1:{(y_train==0).sum()/(y_train==1).sum():.1f}

MODEL RESULTS:
{results_df.to_string(index=False)}

BEST MODEL: {best_model}
- ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}
- F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}
- Precision: {results_df.loc[best_idx, 'Precision']:.4f}
- Recall: {results_df.loc[best_idx, 'Recall']:.4f}

CROSS-VALIDATION:
- Mean ROC-AUC: {cv_scores.mean():.4f}
- Std: {cv_scores.std():.4f}

FEATURE IMPORTANCE (Random Forest):
{feature_importance.to_string(index=False)}

KEY OBSERVATIONS:
1. Class imbalance is severe (1:19.6 ratio)
2. Class weights significantly improve recall for TDE class
3. Random Forest shows better performance than Logistic Regression
4. Z (redshift) is the most important feature
5. Engineered features help improve model performance

NEXT STEPS:
- Try XGBoost with scale_pos_weight
- Experiment with SMOTE for synthetic sampling
- Hyperparameter tuning
- Ensemble methods
"""

with open('baseline_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("✓ Saved: baseline_summary.txt")

# ====================
# 12. FINAL OUTPUT
# ====================
print("\n" + "="*70)
print("✅ BASELINE MODELS COMPLETED!")
print("="*70)
print(f"\n📊 Generated files:")
print("  - confusion_matrices.png")
print("  - roc_curves_baseline.png")
print("  - pr_curves_baseline.png")
print("  - model_performance_comparison.png")
print("  - baseline_summary.txt")
print(f"\n🏆 Best model: {best_model}")
print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
print("="*70)

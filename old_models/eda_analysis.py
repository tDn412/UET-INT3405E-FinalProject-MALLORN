"""
MALLORN Astronomical Classification Challenge - EDA (Non-Interactive)
Exploratory Data Analysis for TDE Detection - No Plot Display
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("MALLORN TDE Classification - Exploratory Data Analysis")
print("="*60)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1] Loading data...")
train_df = pd.read_csv('train_log.csv')
test_df = pd.read_csv('test_log.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"✓ Training set shape: {train_df.shape}")
print(f"✓ Test set shape: {test_df.shape}")
print(f"✓ Sample submission shape: {sample_sub.shape}")

# ====================
# 2. BASIC INFO
# ====================
print("\n[2] Dataset Info:")
print("\nColumn names:")
print(train_df.columns.tolist())

print("\nData types:")
print(train_df.dtypes)

print("\nFirst 5 rows:")
print(train_df.head())

print("\nBasic statistics:")
print(train_df.describe())

# ====================
# 3. CLASS DISTRIBUTION (QUAN TRỌNG!)
# ====================
print("\n"+"="*60)
print("[3] CLASS DISTRIBUTION ANALYSIS")
print("="*60)
target_dist = train_df['target'].value_counts().sort_index()
print("\nTarget distribution:")
print(target_dist)

tde_count = target_dist[1]
non_tde_count = target_dist[0]
total = len(train_df)
tde_ratio = (tde_count / total) * 100

print(f"\nNon-TDE (Class 0): {non_tde_count} samples ({100-tde_ratio:.2f}%)")
print(f"TDE (Class 1): {tde_count} samples ({tde_ratio:.2f}%)")
print(f"\n⚠️  IMBALANCE RATIO: 1:{non_tde_count/tde_count:.1f}")
print(f"\n>>> VẤN ĐỀ CHÍNH: Class imbalance nghiêm trọng!")

# Visualize class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
target_dist.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax1)
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Target (0: Non-TDE, 1: TDE)')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Non-TDE', 'TDE'], rotation=0)
ax1.grid(True, alpha=0.3)
for i, v in enumerate(target_dist):
    ax1.text(i, v + 20, str(v), ha='center', fontweight='bold')

# Pie chart
ax2.pie(target_dist, labels=['Non-TDE', 'TDE'], autopct='%1.2f%%',
        colors=['skyblue', 'salmon'], startangle=90)
ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: class_distribution.png")
plt.close()

# ====================
# 4. MISSING VALUES
# ====================
print("\n"+"="*60)
print("[4] MISSING VALUES ANALYSIS")
print("="*60)

# Check all columns for missing
for col in train_df.columns:
    missing_count = train_df[col].isnull().sum()
    if missing_count > 0:
        missing_pct = (missing_count / len(train_df)) * 100
        print(f"{col}: {missing_count} ({missing_pct:.2f}%)")

# Specifically check SpecType
spectype_missing = train_df['SpecType'].isnull().sum()
if spectype_missing > 0:
    print(f"\n⚠️  SpecType has {spectype_missing} missing values ({spectype_missing/len(train_df)*100:.2f}%)")
else:
    print("\n✓ No missing values in SpecType")

# Check Z_err
z_err_missing = train_df['Z_err'].isnull().sum()
if z_err_missing > 0:
    print(f"⚠️  Z_err has {z_err_missing} missing values ({z_err_missing/len(train_df)*100:.2f}%)")

# ====================
# 5. FEATURE ANALYSIS BY TARGET
# ====================
print("\n"+"="*60)
print("[5] FEATURE ANALYSIS BY TARGET CLASS")
print("="*60)

numerical_features = ['Z', 'Z_err', 'EBV']

# Statistical comparison
for feature in numerical_features:
    non_tde_data = train_df[train_df['target']==0][feature].dropna()
    tde_data = train_df[train_df['target']==1][feature].dropna()
    
    print(f"\n{feature}:")
    print(f"  Non-TDE: mean={non_tde_data.mean():.4f}, std={non_tde_data.std():.4f}, "
          f"min={non_tde_data.min():.4f}, max={non_tde_data.max():.4f}")
    print(f"  TDE:     mean={tde_data.mean():.4f}, std={tde_data.std():.4f}, "
          f"min={tde_data.min():.4f}, max={tde_data.max():.4f}")

# Visualizations
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Feature Distributions by Target Class', fontsize=16, fontweight='bold')

for idx, feature in enumerate(numerical_features):
    # Histogram
    ax1 = axes[idx, 0]
    non_tde = train_df[train_df['target']==0][feature].dropna()
    tde = train_df[train_df['target']==1][feature].dropna()
    
    ax1.hist(non_tde, bins=50, alpha=0.6, label='Non-TDE', color='skyblue')
    ax1.hist(tde, bins=50, alpha=0.6, label='TDE', color='salmon')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.set_title(f'{feature} Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2 = axes[idx, 1]
    data_to_plot = [non_tde, tde]
    bp = ax2.boxplot(data_to_plot, labels=['Non-TDE', 'TDE'], patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('salmon')
    ax2.set_ylabel(feature)
    ax2.set_title(f'{feature} Boxplot by Class')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: feature_distributions.png")
plt.close()

# ====================
# 6. SPECTYPE ANALYSIS
# ====================
print("\n"+"="*60)
print("[6] SPECTYPE ANALYSIS")
print("="*60)

# Count by SpecType (handle NaN)
spectype_counts = train_df['SpecType'].value_counts()
print("\nSpecType distribution (top 10):")
print(spectype_counts.head(10))

# SpecType by target
print("\nSpecType vs Target (top 10 SpecTypes):")
top_specs = spectype_counts.head(10).index
filtered_df = train_df[train_df['SpecType'].isin(top_specs)]
spec_target = pd.crosstab(filtered_df['SpecType'], filtered_df['target'])
print(spec_target)

#  Visualize SpecType by target
spec_dist = train_df.groupby(['SpecType', 'target']).size().unstack(fill_value=0)
top_10_specs = spec_dist.sum(axis=1).nlargest(10).index
spec_dist_top = spec_dist.loc[top_10_specs]

plt.figure(figsize=(14, 6))
spec_dist_top.plot(kind='bar', color=['skyblue', 'salmon'], width=0.8)
plt.title('SpecType Distribution by Target Class (Top 10)', fontsize=14, fontweight='bold')
plt.xlabel('SpecType')
plt.ylabel('Count')
plt.legend(['Non-TDE', 'TDE'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('spectype_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: spectype_distribution.png")
plt.close()

# ====================
# 7. CORRELATION ANALYSIS
# ====================
print("\n"+"="*60)
print("[7] CORRELATION ANALYSIS")
print("="*60)

# Select numerical features (excluding those with NaN)
corr_features = ['Z', 'EBV', 'target']
corr_matrix = train_df[corr_features].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: correlation_matrix.png")
plt.close()

# ====================
# 8. SPLIT DISTRIBUTION
# ====================
print("\n"+"="*60)
print("[8] DATA SPLIT ANALYSIS")
print("="*60)

split_dist = train_df['split'].value_counts().sort_index()
print("\nData distribution across splits:")
print(split_dist)

# Target distribution per split
split_target = pd.crosstab(train_df['split'], train_df['target'])
split_target['Total'] = split_target.sum(axis=1)
split_target['TDE_ratio_%'] = (split_target[1] / split_target['Total']) * 100

print("\nTDE ratio per split:")
print(split_target)

# ====================
# 9. SUMMARY & INSIGHTS
# ====================
print("\n" + "="*60)
print("SUMMARY & KEY INSIGHTS")
print("="*60)

print(f"""
1. DATASET SIZE:
   - Training: {len(train_df):,} samples
   - Test: {len(test_df):,} samples
   - Features: {train_df.shape[1] - 1} (excluding target)

2. CLASS IMBALANCE (⚠️ VẤN ĐỀ CHÍNH!):
   - TDE (Class 1): {tde_count} samples ({tde_ratio:.2f}%)
   - Non-TDE (Class 0): {non_tde_count} samples ({100-tde_ratio:.2f}%)
   - Imbalance ratio: 1:{non_tde_count/tde_count:.1f}
   
   >>> ĐÂY LÀ THÁCH THỨC CHÍNH CẦN GIẢI QUYẾT!
   >>> Cần sử dụng: Class weights, SMOTE, hoặc ensemble methods

3. FEATURES:
   - Z (Redshift): Khoảng cách vũ trụ
     Mean: Non-TDE={train_df[train_df['target']==0]['Z'].mean():.3f}, TDE={train_df[train_df['target']==1]['Z'].mean():.3f}
   - Z_err: Sai số của Z (có missing values: {z_err_missing})
   - EBV: Extinction (hấp thụ ánh sáng)
     Mean: Non-TDE={train_df[train_df['target']==0]['EBV'].mean():.3f}, TDE={train_df[train_df['target']==1]['EBV'].mean():.3f}
   - SpecType: Loại phổ (categorical, {spectype_missing} missing values)

4. KEY OBSERVATIONS:
   - TDE events are extremely rare (~{tde_ratio:.1f}%)
   - Features show some differentiation between classes
   - Data split across {len(split_dist)} folders
   - Need to handle class imbalance carefully

5. NEXT STEPS:
   ✓ Feature engineering (tạo features mới)
   ✓ Xử lý class imbalance (Class weights, SMOTE)
   ✓ Build baseline model (Logistic Regression)
   ✓ Try advanced models (XGBoost với scale_pos_weight)
   ✓ Cross-validation để đánh giá
   ✓ Hyperparameter tuning
""")

print("="*60)
print("✅ EDA COMPLETED!")
print("📊 Generated plots:")
print("   - class_distribution.png")
print("   - feature_distributions.png")
print("   - spectype_distribution.png")
print("   - correlation_matrix.png")
print("="*60)

# Save summary to file
summary_text = f"""MALLORN TDE Classification - EDA Summary
Generated: {pd.Timestamp.now()}

DATASET INFO:
- Training samples: {len(train_df):,}
- Test samples: {len(test_df):,}
- Features: {train_df.shape[1] - 1}

CLASS DISTRIBUTION:
- Non-TDE (0): {non_tde_count} ({100-tde_ratio:.2f}%)
- TDE (1): {tde_count} ({tde_ratio:.2f}%)
- Imbalance ratio: 1:{non_tde_count/tde_count:.1f}

FEATURE STATISTICS:
{train_df[numerical_features + ['target']].groupby('target').describe().to_string()}

CONCLUSIONS:
1. Severe class imbalance - TDE is very rare ({tde_ratio:.2f}%)
2. Need to use imbalance handling techniques
3. Features show some signal for classification
4. Recommended approach: XGBoost with scale_pos_weight
"""

with open('eda_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)
    
print("\n✓ Saved: eda_summary.txt")

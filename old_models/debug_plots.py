
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('physics_features_train_full.csv')
# Clip outliers for plotting
df['asymmetry_ratio'] = df['asymmetry_ratio'].clip(0, 50)
df['rise_time'] = df['rise_time'].clip(0, 1000)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(data=df, x='asymmetry_ratio', hue='target', common_norm=False, fill=True)
plt.title('Asymmetry Ratio Distribution (TDE vs Non-TDE)')
plt.xlim(0, 20)

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='flux_ratio_u_g', y='flux_ratio_g_r', hue='target', alpha=0.6)
plt.title('Color-Color Diagram (u/g vs g/r)')
plt.xlim(-2, 10)
plt.ylim(-2, 10)

plt.tight_layout()
plt.savefig('physics_debug_plots.png')
print("Saved physics_debug_plots.png")

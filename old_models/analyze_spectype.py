
import pandas as pd

df = pd.read_csv('train_log.csv')
print("SpecType Distribution:")
print(df['SpecType'].value_counts(dropna=False))

print("\nTarget by SpecType:")
print(df.groupby('SpecType')['target'].mean())

print("\nCount of TDEs by SpecType:")
print(df[df['target']==1]['SpecType'].value_counts())

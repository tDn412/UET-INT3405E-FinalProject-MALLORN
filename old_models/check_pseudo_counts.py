
import pandas as pd

df = pd.read_csv('submission_pseudo_final.csv')
print(df['prediction'].value_counts())

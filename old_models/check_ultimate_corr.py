
import pandas as pd

try:
    ult = pd.read_csv('submission_ultimate.csv')
    best = pd.read_csv('submission_f1_optimal.csv')
    
    # Merge
    df = ult.merge(best, on='object_id', suffixes=('_ult', '_best'))
    
    corr = df['prediction_ult'].corr(df['prediction_best'])
    overlap = (df['prediction_ult'] & df['prediction_best']).sum()
    
    print(f"Correlation with Best (0.4146): {corr:.4f}")
    print(f"Overlap (Both=1): {overlap} out of {ult['prediction_ult'].sum()} predicted.")
    
except Exception as e:
    print(e)

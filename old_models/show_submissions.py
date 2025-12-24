import pandas as pd

files = [
    'submission_conservative.csv',
    'submission_moderate.csv', 
    'submission_aggressive.csv',
    'submission_ratio_based.csv'
]

print("\n" + "="*60)
print("NEW SUBMISSION VARIANTS")
print("="*60)

for f in files:
    try:
        df = pd.read_csv(f)
        tdes = df['prediction'].sum()
        pct = df['prediction'].mean() * 100
        print(f"{f:35s}: {tdes:4d} TDEs ({pct:5.2f}%)")
    except:
        print(f"{f:35s}: NOT FOUND")

print("="*60)
print("\nRECOMMENDATION:")
print("Try submission_moderate.csv first - balanced approach")
print("If too low, try submission_aggressive.csv")
print("If too high precision needed, try submission_conservative.csv")
print("="*60)

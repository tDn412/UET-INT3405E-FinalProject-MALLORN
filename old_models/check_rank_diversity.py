
import pandas as pd

ult = pd.read_csv('submission_ultimate_top350.csv')
ref = pd.read_csv('submission_multiclass_refined_probs.csv') # Need binary? No I have to make it or load refined main
# Wait, let's load the main refined result which was top 350
ref_main = pd.read_csv('submission_multiclass_refined.csv')

# Merge
df = ult.merge(ref_main, on='object_id', suffixes=('_ult', '_ref'))

overlap = (df['prediction_ult'] & df['prediction_ref']).sum()

print(f"Ultimate 350 Count: {ult['prediction'].sum()}")
print(f"Refined 350 Count: {ref_main['prediction'].sum()}")
print(f"Overlap: {overlap}")
print(f"Difference: {350 - overlap} objects.")
print(f"Percent Agreement: {overlap / 350 * 100:.1f}%")

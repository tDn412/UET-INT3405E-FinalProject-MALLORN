
import pandas as pd

ref = pd.read_csv('submission_multiclass_refined.csv')
ult = pd.read_csv('submission_ultimate.csv')

# Merge
df = ref.merge(ult, on='object_id', suffixes=('_ref', '_ult'))

overlap = (df['prediction_ref'] & df['prediction_ult']).sum()

print(f"Refined Count: {ref['prediction'].sum()}")
print(f"Ultimate Count: {ult['prediction'].sum()}")
print(f"Overlap: {overlap}")
print(f"Percent of Refined in Ultimate: {overlap / ref['prediction'].sum() * 100:.1f}%")

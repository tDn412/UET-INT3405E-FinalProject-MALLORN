
import pandas as pd

teacher = pd.read_csv('submission_multiclass_refined.csv')
student = pd.read_csv('submission_self_train_top350.csv')

# Merge
df = teacher.merge(student, on='object_id', suffixes=('_teach', '_stud'))

overlap = (df['prediction_teach'] & df['prediction_stud']).sum()

print(f"Teacher Count: {teacher['prediction'].sum()}")
print(f"Student Count: {student['prediction'].sum()}")
print(f"Overlap: {overlap}")
print(f"Percent Agreement: {overlap / teacher['prediction'].sum() * 100:.1f}%")

# Check if Student is more confident
prob_teach = pd.read_csv('submission_multiclass_refined_probs.csv')
prob_stud = pd.read_csv('submission_self_train_top350.csv')  # Wait, this file has binary preds, need probs if I saved them? 
# Ah, I didn't save probs in main loop for student explicitly other than for ranking.
# But we can check overlap identity.

# Let's check what new objects entered the top 350
new_objs = df[(df['prediction_stud'] == 1) & (df['prediction_teach'] == 0)]
dropped_objs = df[(df['prediction_stud'] == 0) & (df['prediction_teach'] == 1)]

print(f"\nNew Objects in Top 350: {len(new_objs)}")
print(f"Dropped Objects from Top 350: {len(dropped_objs)}")

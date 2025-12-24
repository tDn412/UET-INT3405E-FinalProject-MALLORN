import pickle
import pandas as pd

print("Inspecting pickle...")
with open('raw_lightcurves_train.pkl', 'rb') as f:
    data = pickle.load(f)

obj_id = list(data.keys())[0]
df = data[obj_id]

print(f"Object ID: {obj_id}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

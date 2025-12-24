
import pandas as pd

# Load the best submission so far
try:
    best_df = pd.read_csv('submission_f1_optimal.csv')
    print(f"Best Submission (0.4146) Count: {best_df['prediction'].sum()}")
except FileNotFoundError:
    print("submission_f1_optimal.csv not found")

# Load the ensemble that just failed
try:
    ens_df = pd.read_csv('submission_ensemble_final.csv')
    print(f"Ensemble Submission (0.3893) Count: {ens_df['prediction'].sum()}")
except:
    pass

# Load Physics model count
try:
    phys_df = pd.read_csv('submission_physics_tuned_final.csv')
    print(f"Physics Model (Untried on LB) Count: {phys_df['prediction'].sum()}")
except:
    pass

# Load Pseudo Label count
try:
    pseudo_df = pd.read_csv('submission_pseudo_final.csv')
    print(f"Pseudo-Label Model (Untried on LB) Count: {pseudo_df['prediction'].sum()}")
except:
    pass


import pandas as pd
from pathlib import Path

def debug():
    print("DEBUGGING SPLITS")
    
    # Check log file
    log_df = pd.read_csv("train_log.csv")
    print(f"Log file objects: {len(log_df)}")
    print(f"Log IDs sample: {log_df['object_id'].head().tolist()}")
    
    sub_dirs = sorted([d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("split_")])
    print(f"Found {len(sub_dirs)} split directories")
    
    for d in sub_dirs[:3]: # Check first 3
        print(f"\nChecking {d.name}...")
        
        # Check files
        files = list(d.glob("train_full_lightcurves.csv*"))
        print(f"  Files: {[f.name for f in files]}")
        
        if not files:
            continue
            
        f = files[0]
        try:
            df = pd.read_csv(f)
            print(f"  Loaded {f.name}: {len(df)} rows")
            ids = df['object_id'].unique()
            print(f"  Unique objects: {len(ids)}")
            
            # Check overlap with log
            overlap = log_df[log_df['object_id'].isin(ids)]
            print(f"  Objects found in log: {len(overlap)}")
            
            if len(overlap) == 0:
                print("  ALARM: No objects from this split found in main log file!")
                print(f"  Split IDs sample: {ids[:5]}")
                
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

if __name__ == "__main__":
    debug()

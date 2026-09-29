import os
import numpy as np
import pandas as pd

# Import your custom modules
import init
from hankslab_db import basicRLtasks_db as bandit_db
import fp_analysis_helpers as fpah

def explore_data(data, name="root", indent=0):
    """Recursively evaluates and prints the structure of any arbitrary data type."""
    spacing = "  " * indent
    
    if isinstance(data, dict):
        print(f"{spacing}['{name}'] -> dict | Keys: {len(data)}")
        for key, value in data.items():
            explore_data(value, name=str(key), indent=indent + 1)
            
    elif isinstance(data, (list, tuple)):
        print(f"{spacing}['{name}'] -> {type(data).__name__} | Length: {len(data)}")
        if len(data) > 0:
            # Inspect the first item to understand what the sequence contains
            explore_data(data[0], name=f"index_0_sample", indent=indent + 1)
            
    elif isinstance(data, np.ndarray):
        print(f"{spacing}['{name}'] -> np.ndarray | Shape: {data.shape} | Dtype: {data.dtype}")
        
    elif isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        col_preview = columns[:4] + ["..."] if len(columns) > 4 else columns
        print(f"{spacing}['{name}'] -> pd.DataFrame | Shape: {data.shape} | Cols: {col_preview}")
        
    elif isinstance(data, pd.Series):
        print(f"{spacing}['{name}'] -> pd.Series | Length: {len(data)} | Dtype: {data.dtype}")
        
    else:
        # Fallback for primitive types (int, float, str) or custom class objects
        val_str = str(data).replace('\n', ' ')
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        print(f"{spacing}['{name}'] -> {type(data).__name__} | Value preview: {val_str}")

def main():
    print("Initializing Bandit Local DB...")
    loc_db = bandit_db.LocalDB_BasicRLTasks("twoArmBandit")
    
    # Replace with a specific subject/session ID you know exists
    test_session_id = "116498" 
    print(f"\nAttempting to load FP data for session: {test_session_id}\n")
    fp_data = fpah.load_fp_data(loc_db, test_session_id)

if __name__ == "__main__":
    main()
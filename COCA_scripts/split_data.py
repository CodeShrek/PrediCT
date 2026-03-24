import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

# Config
BASE_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2")
CSV_PATH = BASE_DIR / "data_canonical" / "tables" / "scan_index.csv"
DATA_DIR = BASE_DIR / "data_resampled"
OUTPUT_DIR = BASE_DIR / "gsoc_split"

def create_split():
    df = pd.read_csv(CSV_PATH)
    
    # Filter out those 2-3 failed scans that don't exist in data_resampled
    df['exists'] = df['scan_id'].apply(lambda x: (DATA_DIR / x).exists())
    df = df[df['exists'] == True]

    # Create a "Stratification Category"
    # 0: No Calcium, 1: Low Calcium (<500 voxels), 2: High Calcium (>500 voxels)
    def categorize(v):
        if v == 0: return 0
        if v < 500: return 1
        return 2
    
    df['strat_cat'] = df['voxels'].apply(categorize)

    # 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['strat_cat'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['strat_cat'], random_state=42)

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_path = OUTPUT_DIR / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Moving {len(split_df)} scans to {split_name}...")
        for _, row in split_df.iterrows():
            # We copy instead of move so we don't break your data_resampled folder
            src = DATA_DIR / row['scan_id']
            dst = split_path / row['scan_id']
            if not dst.exists():
                shutil.copytree(src, dst)

    print("\nSplit Complete!")
    print(f"Check your results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_split()
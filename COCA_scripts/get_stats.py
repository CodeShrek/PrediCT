import pandas as pd
from pathlib import Path

CSV_PATH = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/data_canonical/tables/scan_index.csv")
SPLIT_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split")

df = pd.read_csv(CSV_PATH)

print("--- DATASET STATISTICS ---")
for split in ['train', 'val', 'test']:
    split_scans = [p.name for p in (SPLIT_DIR / split).iterdir() if p.is_dir()]
    split_df = df[df['scan_id'].isin(split_scans)]
    
    avg_calcium = split_df['voxels'].mean()
    zero_calcium = len(split_df[split_df['voxels'] == 0])
    
    print(f"\n[{split.upper()} SET]")
    print(f"Total Scans: {len(split_df)}")
    print(f"Zero Calcium Cases: {zero_calcium} ({zero_calcium/len(split_df):.1%})")
    print(f"Avg Calcium Voxels: {avg_calcium:.2f}")
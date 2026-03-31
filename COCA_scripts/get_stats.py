import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION (Absolute Paths)
# ==========================================
BASE_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2")
CSV_PATH = BASE_DIR / "data_canonical" / "tables" / "scan_index.csv"
SPLIT_DIR = BASE_DIR / "gsoc_split"

def generate_official_stats():
    # 2. Safety Check: Verify the master CSV exists
    if not CSV_PATH.exists():
        print(f" Error: {CSV_PATH} not found. Please ensure COCA_processor.py has finished.")
        return

    # 3. Load the data populated by the COCAProcessor
    df = pd.read_csv(CSV_PATH)
    
    print("="*50)
    print(" OFFICIAL DATASET STATISTICS (XML GROUND TRUTH)")
    print("="*50)

    # 4. Iterate through each split to calculate set-specific metrics
    for split in ['train', 'val', 'test']:
        split_path = SPLIT_DIR / split
        
        if not split_path.exists():
            print(f"\n[!] Split folder '{split}' not found at {split_path}")
            continue
            
        # Identify the specific scans physically present in this split folder
        split_scans = [p.name for p in split_path.iterdir() if p.is_dir()]
        split_df = df[df['scan_id'].isin(split_scans)]
        
        if len(split_df) == 0:
            print(f"\n[{split.upper()} SET] - No matching scans found in index.")
            continue
            
        # Perform aggregations on the XML-derived 'voxels' data
        total_scans = len(split_df)
        zero_calcium_count = len(split_df[split_df['voxels'] == 0])
        avg_calcium_voxels = split_df['voxels'].mean()
        
        print(f"\n[{split.upper()} SET]")
        print(f"Total Scans:         {total_scans}")
        print(f"Zero Calcium Cases:  {zero_calcium_count} ({zero_calcium_count/total_scans:.1%})")
        print(f"Avg Calcium Voxels:  {avg_calcium_voxels:.2f}")

if __name__ == "__main__":
    generate_official_stats()

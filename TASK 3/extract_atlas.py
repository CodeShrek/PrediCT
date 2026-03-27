import os
import subprocess

# Paths based on your Mac setup
data_dir = "data"
# The 'head' of the split archive
zip_head = os.path.join(data_dir, "1-200.change2zip.zip")
output_dir = "atlas_extracted"

def space_safe_extract():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    # 7z automatically finds .z01, .z02 if they are in the same folder as the .zip
    result = subprocess.run(["7z", "x", zip_head, f"-o{output_dir}", "-y"])
    
    if result.returncode == 0:
        print(f" Success! Files extracted to: {output_dir}")
        # List the first few files to verify
        print("Files found:", os.listdir(output_dir)[:5])
    else:
        print(" Extraction failed. Ensure 7zip is installed ('brew install p7zip')")

if __name__ == "__main__":
    space_safe_extract()

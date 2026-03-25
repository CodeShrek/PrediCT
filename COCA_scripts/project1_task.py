import os
import subprocess
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.transforms import SpatialPadd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord
)
from monai.data import DataLoader, Dataset

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_SCAN_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train")
BASE_LABEL_DIR = Path("/Users/karan/Desktop/COCA/project1_labels")
MODEL_SAVE_PATH = "project1_heart_model.pth"

# Apple Silicon GPU Acceleration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# MODULE 1: GROUND TRUTH GENERATION
# ==========================================
def generate_ground_truth(limit=35):
    print(f"\n[1/3] Starting High-Res Ground Truth Generation...")
    BASE_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort scans to ensure consistent progress
    scans = sorted([d for d in INPUT_SCAN_DIR.iterdir() if d.is_dir()])[:limit]
    
    for i, scan_path in enumerate(scans):
        scan_id = scan_path.name
        img_file = scan_path / f"{scan_id}_img.nii.gz"
        out_dir = BASE_LABEL_DIR / scan_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if fusion is already done to allow resuming
        fused_file = out_dir / "fused_heart_mask.nii.gz"
        if fused_file.exists():
            print(f"[{i+1}/{limit}] Skipping {scan_id} (Already exists)")
            continue

        print(f"\n{'#'*60}")
        print(f" PROCESSING SCAN {i+1}/{limit}: {scan_id}")
        print(f"{'#'*60}\n")
        
        # FIXED: Removed --fast to resolve the ValueError
        '''cmd = [
            "TotalSegmentator",
            "-i", str(img_file),
            "-o", str(out_dir),
            "-ta", "heartchambers_highres"
        ]'''
        cmd = [
            "TotalSegmentator",
            "-i", str(img_file),
            "-o", str(out_dir),
            "-ta", "total",
            "--fast",
            "--roi_subset", "heart" # Only segments the heart to save more time
        ]
        
        # Streams TotalSegmentator's output directly to your terminal
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(f"  [TotalSeg]: {line.strip()}")
            
            process.wait()
            if process.returncode != 0:
                print(f"!! Error processing {scan_id}. Moving to next...")
                continue

        fuse_chambers(out_dir)
        print(f"SUCCESS: Fused mask created for {scan_id}")

'''def fuse_chambers(patient_dir):
    chambers = ["heart_atrium_left.nii.gz", "heart_atrium_right.nii.gz", 
                "heart_ventricle_left.nii.gz", "heart_ventricle_right.nii.gz",
                "aorta.nii.gz", "myocardium.nii.gz"]
    fused_mask, affine = None, None
    for f in chambers:
        p = patient_dir / f
        if p.exists():
            img = nib.load(str(p))
            if fused_mask is None:
                fused_mask = np.zeros_like(img.get_fdata())
                affine = img.affine
            fused_mask[img.get_fdata() > 0] = 1
    
    if fused_mask is not None:
        nib.save(nib.Nifti1Image(fused_mask.astype(np.uint8), affine), str(patient_dir / "fused_heart_mask.nii.gz"))'''
def fuse_chambers(patient_dir):
    # For the 'total' task, the file is simply named heart.nii.gz
    heart_file = patient_dir / "heart.nii.gz"
    if heart_file.exists():
        # Rename it to our expected name so the rest of the script works
        fused_path = patient_dir / "fused_heart_mask.nii.gz"
        os.rename(heart_file, fused_path)


# ==========================================
# MODULE 2: DATASET & TRAINING
# ==========================================
def get_loader():
    data_list = []
    for p_dir in INPUT_SCAN_DIR.iterdir():
        if p_dir.is_dir():
            img = p_dir / f"{p_dir.name}_img.nii.gz"
            lbl = BASE_LABEL_DIR / p_dir.name / "fused_heart_mask.nii.gz"
            if img.exists() and lbl.exists():
                data_list.append({"image": str(img), "label": str(lbl)})
    
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        
        # --- NEW FIX: Add padding here ---
        # This ensures every scan is at least 96x96x96 before cropping
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label", 
            spatial_size=(96, 96, 96), 
            pos=1, neg=1, num_samples=2 # Reduced samples to 2 for M2 memory safety
        ),
        ToTensord(keys=["image", "label"]),
    ])
    return DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1, shuffle=True)
    
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1),
        ToTensord(keys=["image", "label"]),
    ])
    return DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1, shuffle=True)

def train(loader, epochs=20):
    print(f"\n[3/3] Training 3D U-Net on {DEVICE}...")
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, 
                 channels=(16, 32, 64, 128), strides=(2, 2, 2)).to(DEVICE)
    loss_func = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        step_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Mean Loss: {step_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nCOMPLETED: Weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    generate_ground_truth(limit=35) 
    train_loader = get_loader()
    if len(train_loader) > 0:
        train(train_loader)
    else:
        print("\nCould not find paired images/labels. Check Step 1 progress.")
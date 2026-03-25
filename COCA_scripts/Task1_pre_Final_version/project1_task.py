import os
import subprocess
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

# MONAI Imports
from monai.networks.nets import AttentionUnet  
from monai.losses import TverskyLoss           
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord,
    SpatialPadd, RandGaussianNoised, RandAdjustContrastd, RandRotated 
)
from monai.data import DataLoader, Dataset

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_SCAN_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train")
BASE_LABEL_DIR = Path("/Users/karan/Desktop/COCA/project1_labels")
MODEL_SAVE_PATH = "project1_heart_model.pth"

# Apple Silicon GPU Acceleration (M2)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# MODULE 1: GROUND TRUTH GENERATION
# ==========================================
def generate_ground_truth(limit=35):
    print(f"\n[1/3] Verifying Ground Truth Data...")
    BASE_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    scans = sorted([d for d in INPUT_SCAN_DIR.iterdir() if d.is_dir()])[:limit]
    
    for i, scan_path in enumerate(scans):
        scan_id = scan_path.name
        img_file = scan_path / f"{scan_id}_img.nii.gz"
        out_dir = BASE_LABEL_DIR / scan_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if (out_dir / "fused_heart_mask.nii.gz").exists():
            continue

        print(f"\nPROCESSING SCAN {i+1}/{limit}: {scan_id}")
        # Using 'total' task with '--fast' for GSoC Project 1 efficiency
        cmd = [
            "TotalSegmentator", "-i", str(img_file), "-o", str(out_dir),
            "-ta", "total", "--fast", "--roi_subset", "heart"
        ]
        
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(f"  [TotalSeg]: {line.strip()}")
            process.wait()

        fuse_chambers(out_dir)

def fuse_chambers(patient_dir):
    heart_file = patient_dir / "heart.nii.gz"
    if heart_file.exists():
        fused_path = patient_dir / "fused_heart_mask.nii.gz"
        os.rename(heart_file, fused_path)

# ==========================================
# MODULE 2: DATASET & AUGMENTED LOADER
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
        
        # Data Augmentation for Robustness
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),
        RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.2),
        
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label", 
            spatial_size=(96, 96, 96), 
            pos=1, neg=1, num_samples=2 
        ),
        ToTensord(keys=["image", "label"]),
    ])
    return DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1, shuffle=True)

# ==========================================
# MODULE 3: UPGRADED TRAINING
# ==========================================
def train(loader, epochs=30): 
    print(f"\n[3/3] Training Attention U-Net on {DEVICE}...")
    
    model = AttentionUnet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1, 
        channels=(16, 32, 64, 128), 
        strides=(2, 2, 2)
    ).to(DEVICE)
    
    # Tversky Loss (alpha=0.5, beta=0.5) balances precision and recall
    loss_func = TverskyLoss(sigmoid=True, alpha=0.5, beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

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
        train(train_loader, epochs=100) 
    else:
        print("\nCould not find paired images/labels.")
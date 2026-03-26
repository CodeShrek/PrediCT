import os
import subprocess
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

#Monai imports
from monai.networks.nets import AttentionUnet  
from monai.losses import TverskyLoss  
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord,
    SpatialPadd, RandGaussianNoised, RandAdjustContrastd, RandRotated,
    CenterSpatialCropd
)
from monai.data import DataLoader, Dataset

# ==========================================
# CUSTOM LOSS: Focal Tversky Implementation 
# ==========================================

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma=1.3, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, input, target):
        t_loss = super().forward(input, target)
        return torch.pow(t_loss, 1.0 / self.gamma)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_SCAN_DIR = Path("/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train")
BASE_LABEL_DIR = Path("/Users/karan/Desktop/COCA/project1_labels")
MODEL_SAVE_PATH = "project1_heart_model.pth"

# Apple Silicon GPU Acceleration (M2)- important mention
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
        
        # FOCUS: Center 160x160mm axial crop to exclude ribcage and focus on heart region
        CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 160, -1)), 
        
        # AUGMENTATION: Creating 'synthetic' variety
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
# MODULE 3: FINE-TUNING TRAINING
# ==========================================
def train(loader, epochs=100): #100epochs for boundary refinement(better than previous 35)
    print(f"\n[3/3] Fine-tuning Attention U-Net on {DEVICE}...")
    
    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1, 
        channels=(16, 32, 64, 128), strides=(2, 2, 2)
    ).to(DEVICE)

    # LOAD EXISTING PROGRESS: Start from 0.81 can be seen at Task1_pre_Final_version in repository, now we are fine-tuning to get better boundary refinement and hopefully reach 0.85+ dice score
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Resuming from current weights: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # LOSS: Custom Focal Tversky with gamma=1.3 focuses gradients on blurry edges
    loss_func = FocalTverskyLoss(sigmoid=True, alpha=0.7, beta=0.3, gamma=1.3)
    
    # OPTIMIZER: 1e-5  for precise boundary refinement
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 

    for epoch in range(epochs):
        model.train()
        step_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Mean Loss: {step_loss/len(loader):.4f}")
        
        # Auto-save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nCOMPLETED: Final weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    generate_ground_truth(limit=35) 
    train_loader = get_loader()
    if len(train_loader) > 0:
        train(train_loader, epochs=100) 
    else:
        print("\nCould not find paired images/labels.")
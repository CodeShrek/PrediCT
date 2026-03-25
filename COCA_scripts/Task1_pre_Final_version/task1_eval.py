import torch
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from skimage.measure import label as sk_label
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Orientationd, ToTensord, SpatialPadd
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = Path("project1_heart_model.pth")
TEST_PATIENT = "15ae0189e37b" 
IMG_PATH = f"/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train/{TEST_PATIENT}/{TEST_PATIENT}_img.nii.gz"
LBL_PATH = f"/Users/karan/Desktop/COCA/project1_labels/{TEST_PATIENT}/fused_heart_mask.nii.gz"

from scipy.ndimage import binary_erosion, binary_dilation

def post_process_mask(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8).squeeze()
    
    # 1. Aggressive Erosion to break 'bridges' to the chest wall
    # This disconnects the heart from the rib noise
    eroded = binary_erosion(mask_np, iterations=3)
    
    # 2. Largest Connected Component (on the cleaned volume)
    labels = sk_label(eroded)
    if labels.max() == 0: return mask_tensor
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    heart_only = (labels == largest_label)
    
    # 3. Dilate back to original size to recover the heart boundaries
    final_mask = binary_dilation(heart_only, iterations=3).astype(np.float32)
    
    return torch.from_numpy(final_mask[np.newaxis, np.newaxis, ...]).to(DEVICE)

def evaluate():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model weights not found at {MODEL_PATH}. Train the model first!")
        return

    print(f"--- Evaluating 100-Epoch Attention U-Net Model ---")
    print(f"Loading weights from: {MODEL_PATH}")
    
    # Architecture must match project1_task.py exactly
    model = AttentionUnet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1, 
        channels=(16, 32, 64, 128), 
        strides=(2, 2, 2)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Define transforms (Standardized to training params)
    eval_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image", "label"]),
    ])

    print(f"Processing patient: {TEST_PATIENT}...")
    data = eval_transforms({"image": IMG_PATH, "label": LBL_PATH})
    image = data["image"].unsqueeze(0).to(DEVICE)
    label = data["label"].unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        # Overlap=0.5 (4 samples) provides better boundary smoothing
        output = sliding_window_inference(image, (96, 96, 96), 4, model)
        output = (torch.sigmoid(output) > 0.5).float()
        
        # Apply the thorough cleaning logic
        print("Applying post-processing (Closing + LCC)...")
        output = post_process_mask(output)
        
    inf_time = time.time() - start_time

    # Calculate Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=output, y=label)
    dice_score = dice_metric.aggregate().item()

    print(f"\n{'='*40}")
    print(f"--- FINAL PERFORMANCE METRICS ---")
    print(f"{'='*40}")
    print(f"Final Dice Score:  {dice_score:.4f}")
    print(f"Inference Time:    {inf_time:.2f}s")
    print(f"{'='*40}")

    # Visualization of the central heart slice
    img_slice = image.cpu().numpy()[0, 0]
    lbl_slice = label.cpu().numpy()[0, 0]
    pred_slice = output.cpu().numpy()[0, 0]
    z = img_slice.shape[2] // 2
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice[:, :, z], cmap="gray")
    plt.title("Cardiac CT Scan")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(lbl_slice[:, :, z], cmap="Reds", alpha=0.5)
    plt.title("Ground Truth (TotalSeg)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_slice[:, :, z], cmap="Greens", alpha=0.5)
    plt.title(f"PrediCT Result (Dice: {dice_score:.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("final_gsoc_result_100epochs.png")
    print("\nVisualization saved to final_gsoc_result_100epochs.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
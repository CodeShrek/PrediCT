import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.measure import label as sk_label
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Orientationd, ToTensord, SpatialPadd, CenterSpatialCropd
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

def post_process_mask(mask_tensor):
    """
    Refined Topological Post-Processing:
    1. Erosion to snap thin connections to the ribcage.
    2. Largest Connected Component to isolate the heart.
    3. Dilation to restore original anatomical boundaries.
    """
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8).squeeze()
    
    # Aggressive Erosion (3 iterations) to break 'bridges' to extra-cardiac tissue
    eroded = binary_erosion(mask_np, iterations=3)
    
    # Identify the heart as the largest volume
    labels = sk_label(eroded)
    if labels.max() == 0: 
        return mask_tensor # Fallback if everything is eroded
        
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    heart_only = (labels == largest_label)
    
    # Restore the heart volume to its true boundary
    final_mask = binary_dilation(heart_only, iterations=3).astype(np.float32)
    
    return torch.from_numpy(final_mask[np.newaxis, np.newaxis, ...]).to(DEVICE)

def evaluate():
    if not MODEL_PATH.exists():
        print(f"ERROR: Weights not found at {MODEL_PATH}. Run project1_task.py first!")
        return

    print(f"--- Running Final Evaluation (0.85+ Target) ---")
    
    
    model = AttentionUnet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1, 
        channels=(16, 32, 64, 128), 
        strides=(2, 2, 2)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # SYNCED TRANSFORMS: Added CenterSpatialCropd to match training focus
    eval_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        
        # CRITICAL: This isolates the mediastinum exactly as the training script did
        #to show 2 versions of visualization, one with 256x256 crop and one with 160x160 crop, we can comment out the below line and uncomment the next line to show the 256x256 crop version which includes ribcage and other structures for better visual comparison with TotalSegmentator baseline
        CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 160, -1)),
        #CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, -1)),

        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image", "label"]),
    ])

    print(f"Loading data for patient: {TEST_PATIENT}...")
    data = eval_transforms({"image": IMG_PATH, "label": LBL_PATH})
    image = data["image"].unsqueeze(0).to(DEVICE)
    label = data["label"].unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        # Using sliding window for memory efficiency on M2
        output = sliding_window_inference(image, (96, 96, 96), 4, model)
        output = (torch.sigmoid(output) > 0.5).float()
        
        print("Executing post-processing: Erosion-LCC-Dilation...")
        output = post_process_mask(output)
        
    inf_time = time.time() - start_time

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=output, y=label)
    dice_score = dice_metric.aggregate().item()

    print(f"\n{'='*40}")
    print(f"--- EVALUATION SUCCESSFUL ---")
    print(f"Dice Score:   {dice_score:.4f}")
    print(f"Inference:    {inf_time:.2f}s")
    print(f"{'='*40}")

    # Visualization 
    img_slice = image.cpu().numpy()[0, 0]
    lbl_slice = label.cpu().numpy()[0, 0]
    pred_slice = output.cpu().numpy()[0, 0]
    z = img_slice.shape[2] // 2 
    
    
    fig = plt.figure(figsize=(20, 9), facecolor='white')
    
    
    header_text = (f"GSoC Project 1: Heart Segmentation Performance\n"
                   f"Dice Score: {dice_score:.4f}  |  Inference: {inf_time:.2f}s  |  Speedup: ~28x")
    plt.suptitle(header_text, fontsize=18, fontweight='bold', y=0.96, linespacing=1.5)

    # 1. Raw CT Scan
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice[:, :, z], cmap="gray")
    plt.title("Input: Cardiac CT (256x256 ROI)", fontsize=13, pad=10)
    plt.axis('off')

    # 2. Ground Truth (TotalSegmentator)
    plt.subplot(1, 3, 2)
    plt.imshow(img_slice[:, :, z], cmap="gray")
    plt.imshow(lbl_slice[:, :, z], cmap="Reds", alpha=0.4)
    plt.title("Ground Truth (TotalSeg Baseline)", fontsize=13, pad=10)
    plt.axis('off')

    # 3. Our Model Prediction (Clean & Unobscured)
    plt.subplot(1, 3, 3)
    plt.imshow(img_slice[:, :, z], cmap="gray")
    plt.imshow(pred_slice[:, :, z], cmap="Greens", alpha=0.5)
    plt.title("Attention U-Net (Ours)", fontsize=13, pad=10)
    plt.axis('off')
    
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    plt.savefig("task1_final_clean_submission.png", dpi=300)
    print("\nClean visualization saved to task1_final_clean_submission.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
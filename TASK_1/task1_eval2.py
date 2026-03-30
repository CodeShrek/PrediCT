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

# ==========================================
# VALIDATION METRICS & PLOTTING
# ==========================================

def compute_slice_wise_dice(y_pred, y_true):
    """Calculates Dice score for every axial slice to identify z-axis accuracy trends."""
    pred = (y_pred.cpu().numpy() > 0.5).astype(np.uint8).squeeze()
    true = (y_true.cpu().numpy() > 0.5).astype(np.uint8).squeeze()
    
    slice_dice = []
    for z in range(pred.shape[2]):
        p_slice = pred[:, :, z]
        t_slice = true[:, :, z]
        intersection = np.sum(p_slice * t_slice)
        union = np.sum(p_slice) + np.sum(t_slice)
        slice_dice.append((2.0 * intersection) / union if union != 0 else 1.0)
    return slice_dice

def visualize_mentor_requirements(image, label, output, dice_scores, patient_id):
    """Generates a clean, non-overlapping validation suite using explicit subplot spacing."""
    img_np = image.cpu().numpy()[0, 0]
    lbl_np = label.cpu().numpy()[0, 0]
    pred_np = output.cpu().numpy()[0, 0]
    
    # Calcium threshold: mapped -150 to 500 into 0 to 1 range (130 HU approx 0.43) 
    calcium_mask = (img_np > 0.43).astype(np.float32)
    z_mid = img_np.shape[2] // 2
    
    # Create figure with high DPI and explicit sizing
    fig = plt.figure(figsize=(18, 14), facecolor='white', dpi=100)
    
    # Use GridSpec for absolute control over spacing 
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    plt.suptitle(f"GSoC Task 1: Heart Segmentation Anatomical Validation\nPatient ID: {patient_id}", 
                 fontsize=22, fontweight='bold', y=0.96)

    # --- 1. Z-axis Trend Plot ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(dice_scores, color='#1f77b4', linewidth=2.5, alpha=0.8)
    ax0.set_title("A. Heart Segmentation Accuracy (Z-Axis Trend)", fontsize=16, pad=15)
    ax0.set_xlabel("Slice Index (Apex -> Base)", fontsize=12)
    ax0.set_ylabel("Dice Score", fontsize=12)
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Calcium Containment Check ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_np[:, :, z_mid], cmap='gray')
    ax1.imshow(pred_np[:, :, z_mid], cmap='Greens', alpha=0.35)
    # Overlay calcium in bright autumn colors for high visibility
    ax1.imshow(np.ma.masked_where(calcium_mask[:, :, z_mid] == 0, calcium_mask[:, :, z_mid]), 
               cmap='autumn', alpha=1.0)
    ax1.set_title("B. Calcium Containment Check\n(Yellow = Calcium, Green = Predicted Mask)", fontsize=16, pad=15)
    ax1.axis('off')

    # --- 3. Error Map (False Positives vs False Negatives) ---
    ax2 = fig.add_subplot(gs[1, 0])
    error_map = np.zeros_like(pred_np[:, :, z_mid])
    error_map[(pred_np[:, :, z_mid] == 0) & (lbl_np[:, :, z_mid] == 1)] = 1 # FN (Missed)
    error_map[(pred_np[:, :, z_mid] == 1) & (lbl_np[:, :, z_mid] == 0)] = 2 # FP (Extra)
    
    ax2.imshow(img_np[:, :, z_mid], cmap='gray')
    ax2.imshow(np.ma.masked_where(error_map == 0, error_map), cmap='coolwarm', alpha=0.85)
    ax2.set_title("C. Spatial Error Map\n(Blue = Missed Heart, Red = Extra Segment)", fontsize=16, pad=15)
    ax2.axis('off')

    # --- 4. Coverage Statistics & Metrics ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("D. Validation Metrics Summary", fontsize=16, pad=15)
    
    calc_outside = np.sum((calcium_mask == 1) & (pred_np == 0))
    total_calc = np.sum(calcium_mask)
    coverage = (1.0 - (calc_outside/total_calc)) * 100 if total_calc > 0 else 100.0
    
    metrics_text = (
        f"Calcium Coverage: {coverage:.2f}%\n\n"
        f"False Negative Voxels: {int(np.sum(error_map==1))}\n"
        f"False Positive Voxels: {int(np.sum(error_map==2))}\n\n"
        f"Coverage Result: {'✅ SUCCESS' if coverage > 99 else '⚠️ MARGINAL'}"
    )
    
    # Place text in the center of the panel with a clean background box
    ax3.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=18, 
             bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6', boxstyle='round,pad=1.5'))
    ax3.axis('off')

    # Final spacing adjustment to ensure no overlap with the suptitle
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    save_path = f"GSoC_Task1_Validation_{patient_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n--- ✅ Clean Validation Suite saved as: {save_path} ---")
    plt.show()

# ==========================================
# POST-PROCESSING & CORE LOGIC
# ==========================================

def post_process_mask(mask_tensor):
    """Refined post-processing to isolate heart and clean ribcage noise."""
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8).squeeze()
    # Erosion breaks thin 'bridges' to extra-cardiac structures 
    eroded = binary_erosion(mask_np, iterations=3)
    labels = sk_label(eroded)
    if labels.max() == 0: return mask_tensor
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    heart_only = (labels == largest_label)
    # Dilation restores the anatomical boundary 
    final_mask = binary_dilation(heart_only, iterations=3).astype(np.float32)
    return torch.from_numpy(final_mask[np.newaxis, np.newaxis, ...]).to(DEVICE)

def evaluate():
    if not MODEL_PATH.exists():
        print(f"ERROR: Weights not found at {MODEL_PATH}.")
        return

    print(f"--- ⚡ Starting Project 1 Robust Evaluation ---")
    
    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1, 
        channels=(16, 32, 64, 128), strides=(2, 2, 2)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    eval_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 160, -1)),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image", "label"]),
    ])

    print(f"Loading data for patient: {TEST_PATIENT}...")
    data = eval_transforms({"image": IMG_PATH, "label": LBL_PATH})
    image, label = data["image"].unsqueeze(0).to(DEVICE), data["label"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Balanced sw_batch_size for M2 Mac RAM stability 
        output = sliding_window_inference(image, (96, 96, 96), 4, model, overlap=0.25)
        output = (torch.sigmoid(output) > 0.5).float()
        output = post_process_mask(output)
        
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=output, y=label)
    global_dice = dice_metric.aggregate().item()

    print(f"Global Dice Score: {global_dice:.4f}")
    
    # Calculate Trends and Generate Plot
    print("Generating refined validation suite...")
    z_dice_scores = compute_slice_wise_dice(output, label)
    visualize_mentor_requirements(image, label, output, z_dice_scores, TEST_PATIENT)

if __name__ == "__main__":
    evaluate()

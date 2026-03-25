import torch
import time
import nibabel as nib
from skimage.measure import label as sk_label
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology  # Added for noise removal
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    Orientationd, ToTensord, SpatialPadd
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from pathlib import Path

# --- SETTINGS ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "project1_heart_model.pth"
# Use a scan the model hasn't seen or the last one from your list
TEST_PATIENT = "15ae0189e37b" 
IMG_PATH = f"/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train/{TEST_PATIENT}/{TEST_PATIENT}_img.nii.gz"
LBL_PATH = f"/Users/karan/Desktop/COCA/project1_labels/{TEST_PATIENT}/fused_heart_mask.nii.gz"

def post_process_mask(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8).squeeze()
    
    # Label each connected "blob" with a different number
    labels = sk_label(mask_np)
    
    if labels.max() == 0: # Nothing was predicted
        return mask_tensor
        
    # Find which label has the most pixels (the heart)
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    
    # Keep only that label
    cleaned_np = (labels == largest_label).astype(np.float32)
    
    # Put back into original shape [1, 1, D, H, W]
    cleaned_np = cleaned_np[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(cleaned_np).to(DEVICE)

def evaluate():
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, 
                 channels=(16, 32, 64, 128), strides=(2, 2, 2)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Setup Transform
    eval_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-150, a_max=500, b_min=0, b_max=1, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image", "label"]),
    ])

    print("Pre-processing test image...")
    data = eval_transforms({"image": IMG_PATH, "label": LBL_PATH})
    image = data["image"].unsqueeze(0).to(DEVICE)
    label = data["label"].unsqueeze(0).to(DEVICE)

    # 3. Measure Inference Time
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        # Sliding window ensures we handle full volumes without memory errors
        output = sliding_window_inference(image, (96, 96, 96), 4, model)
        output = (torch.sigmoid(output) > 0.5).float()
    
    # --- ADDED POST-PROCESSING STEP ---
    print("Post-processing (removing noise blobs)...")
    output = post_process_mask(output)
    
    inf_time = time.time() - start_time

    # 4. Calculate Dice
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=output, y=label)
    dice_score = dice_metric.aggregate().item()

    print(f"\n{'='*30}")
    print(f"--- EVALUATION RESULTS ---")
    print(f"{'='*30}")
    print(f"Final Dice Score: {dice_score:.4f}")
    print(f"U-Net Inference Time: {inf_time:.2f} seconds")
    print(f"TotalSegmentator Time (approx): 180.00 seconds")
    print(f"Speedup: {180/inf_time:.1f}x faster")

    # 5. Save Visualization
    img_slice = image.cpu().numpy()[0, 0]
    lbl_slice = label.cpu().numpy()[0, 0]
    pred_slice = output.cpu().numpy()[0, 0]
    
    # Take the middle slice where the heart is most visible
    z = img_slice.shape[2] // 2
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice[:, :, z], cmap="gray")
    plt.title("CT Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(lbl_slice[:, :, z], cmap="Reds", alpha=0.5)
    plt.title("TotalSeg (GT)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # We use Greens to show your model's prediction
    plt.imshow(pred_slice[:, :, z], cmap="Greens", alpha=0.5)
    plt.title(f"Prediction (Dice: {dice_score:.2f})")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("task1_visual_result_cleaned.png")
    print(f"\nVisualization saved to task1_visual_result_cleaned.png")

if __name__ == "__main__":
    evaluate()
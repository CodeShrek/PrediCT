import SimpleITK as sitk
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
ATLAS_DIR = "/Users/karan/Desktop/PrediCT/task3/atlas_extracted/1-200"
PATIENT_IMG = "/Users/karan/Desktop/COCA/cocacoronarycalciumandchestcts-2/gsoc_split/train/15ae0189e37b/15ae0189e37b_img.nii.gz"
HEART_MASK = "/Users/karan/Desktop/COCA/project1_labels/15ae0189e37b/fused_heart_mask.nii.gz"

def resample_to_spacing(itk_image, target_spacing=[2.0, 2.0, 2.0], is_label=False):
    """Resamples to 2.0mm to provide the high-detail necessary for a 10x10x10 BSpline."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    
    return sitk.Resample(itk_image, new_size, sitk.Transform(), 
                         sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
                         itk_image.GetOrigin(), target_spacing, itk_image.GetDirection(), 0.0, itk_image.GetPixelID())

def run_task3_master():
    # ---  FILE DISCOVERY ---
    atlas_images = glob.glob(os.path.join(ATLAS_DIR, "*.img.nii.gz"))
    if not atlas_images:
        print(f" Error: No atlas files found in {ATLAS_DIR}")
        return

    moving_path = atlas_images[0]
    label_path = moving_path.replace(".img.nii.gz", ".label.nii.gz")
    print(f"---  Processing: {os.path.basename(moving_path)} ---")
    
    # --- ⚡ LOAD & PRE-PROCESS ---
    fixed_orig = sitk.DICOMOrient(sitk.ReadImage(PATIENT_IMG, sitk.sitkFloat32), 'LPS')
    moving_orig = sitk.DICOMOrient(sitk.ReadImage(moving_path, sitk.sitkFloat32), 'LPS')
    mask_orig = sitk.DICOMOrient(sitk.ReadImage(HEART_MASK, sitk.sitkUInt8), 'LPS')

    print("Resampling images to 2.0mm High-Resolution...")
    fixed = resample_to_spacing(fixed_orig, [2.0, 2.0, 2.0])
    moving = resample_to_spacing(moving_orig, [2.0, 2.0, 2.0])
    mask = resample_to_spacing(mask_orig, [2.0, 2.0, 2.0], is_label=True)

    # ---  REGISTRATION ENGINE ---
    reg = sitk.ImageRegistrationMethod()
    
    # High-fidelity metric settings
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10) # 10% sampling for detail
    reg.SetMetricFixedMask(mask)

    # --- STAGE 1: MULTI-LEVEL AFFINE ---
    print("Initializing alignment via Moments (Anti-Slingshot)...")
    initial_tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), 
                                                   sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg.SetInitialTransform(initial_tx)

    # 3-Level Pyramid for a strong global foundation
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    # Controlled learning rate for stability
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=100)
    reg.SetOptimizerScalesFromPhysicalShift() 

    print("Executing Stage 1: Deep Affine Alignment...")
    affine_tx = reg.Execute(fixed, moving)

    # --- STAGE 2: DEEP BSPLINE (DEFORMABLE) ---
    print("Stage 2: Initializing Dense 10x10x10 BSpline Grid...")
    bspline_tx = sitk.BSplineTransformInitializer(fixed, [10, 10, 10])
    
    composite_tx = sitk.CompositeTransform([affine_tx, bspline_tx])
    reg.SetInitialTransform(composite_tx, inPlace=False)
    
    # Keep 2-level pyramid for BSpline to prevent local minima
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    
    reg.SetOptimizerAsLBFGS2(solutionAccuracy=1e-5, numberOfIterations=100)
    
    print("Executing Stage 2: Chained Deformable Registration...")
    final_tx = reg.Execute(fixed, moving)

    # ---  VALIDATION ---
    print("Finalizing High-Res Result...")
    moving_lbl = sitk.ReadImage(label_path, sitk.sitkUInt8)
    resampled_lbl = sitk.Resample(moving_lbl, fixed_orig, final_tx, sitk.sitkNearestNeighbor, 0.0)
    
    binary_lbl = sitk.Cast(resampled_lbl > 0, sitk.sitkUInt8)
    dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(binary_lbl, squaredDistance=False, useImageSpacing=True))
    dist_np = sitk.GetArrayFromImage(dist_map)
    img_np = sitk.GetArrayFromImage(fixed_orig)
    heart_mask_np = sitk.GetArrayFromImage(mask_orig)
    
    calc_mask = (img_np > 130) & (heart_mask_np > 0)
    num_calc = np.sum(calc_mask)
    
    if num_calc > 0:
        accuracy = (np.sum(dist_np[calc_mask] <= 10.0) / num_calc) * 100
        print(f"\n SUCCESS: Accuracy is {accuracy:.2f}%")
    else:
        accuracy = 0.0

    # ---  VISUALIZATION ---
    z = img_np.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img_np[z], cmap='gray'); plt.title("Patient (NCCT)"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img_np[z], cmap='gray')
    plt.imshow(sitk.GetArrayFromImage(resampled_lbl)[z], cmap='jet', alpha=0.3)
    plt.title(f"Atlas Registered (Acc: {accuracy:.1f}%)"); plt.axis('off')

    plt.savefig("task3_50percent_result.png")
    plt.show()

if __name__ == "__main__":
    start = time.time()
    run_task3_master()
    print(f"Process Complete. Total Time: {time.time() - start:.2f}s")

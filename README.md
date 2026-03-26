# GSoC 2026 – PrediCT Evaluation (Shriyam Baloni)

## 🚀 My Contribution (Start Here)

This repository contains my implementation of the PrediCT GSoC 2026 evaluation tasks, focusing on building a scalable preprocessing pipeline for coronary calcium analysis.

### ✅ Completed: Common Task

* DICOM → NIfTI conversion using SimpleITK
* XML annotation parsing → calcium masks
* Anisotropic resampling (0.7 × 0.7 × 3.0 mm)
* Stratified dataset split (Null / Emergent / Established)
* HU windowing (-150 to 500)
* MONAI-compatible data pipeline

---

## 🔬 Design Rationale

Data volumes were resampled to an anisotropic resolution of 0.7 × 0.7 × 3.0 mm, explicitly prioritizing in-plane spatial resolution to capture sub-millimeter coronary calcifications while preserving the native 3.0 mm axial slice thickness. By avoiding z-axis interpolation, this design mitigates partial volume artifacts and avoids cross-slice blurring, thereby maintaining geometric fidelity for downstream multi-modal registration (NCCT-to-CCTA) in Project 3. This reflects a deliberate trade-off between voxel isotropy and anatomical integrity, preserving the sharp intensity gradients required for reliable Agatston scoring.

To address the pronounced class imbalance and long-tail distribution of calcium burden, a stratified cohort partitioning strategy was employed. Scans were grouped into clinically meaningful strata—Null, Emergent (<500 voxels), and Established (≥500 voxels)—ensuring that validation and test sets capture the full spectrum of disease progression rather than being dominated by the zero-calcium majority. This stratification enables more faithful performance estimation under clinically realistic distributions.

The preprocessing pipeline incorporates a domain-specific Hounsfield Unit (HU) window of [-150, 500], isolating the relevant radiodensity range for epicardial fat, myocardium, and calcified plaque. For Project 1 (segmentation), the training pipeline employs MONAI’s RandCropByPosNegLabeld strategy to enforce balanced sampling of foreground (calcified) and background regions at the patch level. This targeted sampling mitigates the extreme spatial sparsity of positive voxels, stabilizes optimization in 3D U-Net training, and improves boundary sensitivity in low-signal regimes typical of early-stage calcification.

---

## 📊 Dataset Summary

* Total scans: ~787
* Excluded scans: 2–3 (invalid depth)
* Output: Train / Validation / Test splits

---
## 🚀 Task 1: Heart Segmentation Pipeline (Progress)

### Objective
Built a scalable preprocessing and segmentation pipeline for the COCA dataset, replacing slow silver-standard labeling (TotalSegmentator) with a lightweight, fast 3D U-Net(Attention-Gated 3D U-Net now).

---

- **Ground Truth**
  - Generated heart masks using TotalSegmentator (fast mode) on ~35 scans  

- **Model**
  - 3D Residual U-Net (MONAI)
  - Multi-stage encoder-decoder with volumetric context awareness  

- **Hardware**
  - Optimized for Apple Silicon (MPS)
  - Fast GPU-based inference  


---

### 🚧 Next Steps
- Project 3 (task3)implementation
- 
## 🚧 Ongoing Work

* 🔄 Project 1: Heart Segmentation (top priority)- completed can be viewed in TASK_1 folder alogn with its report
* 🔄 Project 3: NCCT ↔ CCTA Registration(in progress)

---



#Final submission for Task1 is present at TASK_1
- its intital files are present in COCA_scripts just for version control ; along with a folder named Task1_pre_final_version which has prefinal version which yielded a score of 0.81 Dice.



##ADIOS!!!
### SEE U AROUND

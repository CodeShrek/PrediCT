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

## 🚧 Ongoing Work

* 🔄 Project 1: Heart Segmentation (MONAI 3D U-Net)(top priority)
* 🔄 Project 3: NCCT ↔ CCTA Registration

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 📚 Documentation

* Dataset instructions → `docs/` folder

---

## 📦 About PrediCT (Original Description Below)



##ADIOS!!!
### SEE U AROUND


# PrediCT
A project to enhance predictive power of routine non-contrast CT scans

Problem:

Current coronary artery calcium (CAC) tests are widely used in clinical practice but have limited predictive power regarding when and where future occlusions may occur. While more advanced imaging techniques such as Intravascular Ultrasound (IVUS) and optical coherence tomography (OCT) offer greater predictive capabilities and resolution, they are more resource-intensive, time-consuming, and less practical for routine use. At present, there is no diagnostic tool in cardiology that combines strong predictive ability with high clinical usability and accessibility.
________________________________________
Hypothesis:

Machine learning models, when trained on a large set of CT scans, can detect predictive patterns in calcium deposition that are not readily identifiable by human interpretation. Such models could uncover hidden features that improve risk prediction beyond traditional scoring systems, discover correlations between clinical and image data, and enhance the predictive capability of these scans through powerful feature detection.
________________________________________
Current Direction:

While we await data transfer of major adverse cardiovascular event (MACE) endpoints + NCCT from Kettering Health Network, we will focus on constructing a robust front end for our model using open source data (Stanford COCA, ImageCAS, etc.). This includes a calcium segmentation head that combines speed with accuracy while preserving accurate spatial relations. We also aim to map the heart anatomy to properly localize calcium deposits. We will perform extensive feature extraction and analysis, with the goal of creating distinct calcium phenotypes that may map to MACE endpoints. Eventually, with expert annotation, we hope to integrate other features such as EAT, Heart Chamber Volume, etc. To supplement our current efforts we are also exploring data augmentation using simulation-based synthetic calcium generation and placement into empty CAC scans.

Clinical Translation (Long-Term Goal)

o	Develop a clinician-facing tool that, based on a simple CAC scan, outputs:

	Predicted time-dependent risk levels for MACE,
	Likely anatomical regions of future occlusion?,
	Confidence intervals for predictions.

o	This tool could improve patient outcomes by assisting providers in balancing the risks of cardiac events with the risks of further tests, treatments, or surgeries.

o	Longer-term, this framework could serve as a model for applying machine learning to preventive medicine more broadly.



# PrediCT
A project to enhance predictive power of routine non-contrast CT scans

Problem:

Current coronary artery calcium (CAC) tests are widely used in clinical practice but have limited predictive power regarding when and where future occlusions may occur. While more advanced imaging techniques such as Intravascular Ultrasound (IVUS) and optical coherence tomography (OCT) offer greater predictive capabilities and resolution, they are more resource-intensive, time-consuming, and less practical for routine use. At present, there is no diagnostic tool in cardiology that combines strong predictive ability with high clinical usability and accessibility.
________________________________________
Hypothesis:

Machine learning models, when trained on a large set of CT scans, can detect predictive patterns in calcium deposition that are not readily identifiable by human interpretation. Such models could uncover hidden features that improve risk prediction beyond traditional scoring systems, discover correlations between clinical and image data, and enhance the predictive capability of these scans through powerful feature detection.
________________________________________
Current Direction:

While we await data transfer of major adverse cardiovascular event (MACE) endpoints + NCCT from Kettering Health Network, we will focus on constructing a robust front end for our model using open source data (Stanford COCA, ImageCAS, etc.). This includes a calcium segmentation head that combines speed with accuracy while preserving accurate spatial relations. We also aim to map the heart anatomy to properly localize calcium deposits. We will perform extensive feature extraction and analysis, with the goal of creating distinct calcium phenotypes that may map to MACE endpoints. Eventually, with expert annotation, we hope to integrate other features such as EAT, Heart Chamber Volume, etc. To supplement our current efforts we are also exploring data augmentation using simulation-based synthetic calcium generation and placement into empty CAC scans.

Clinical Translation (Long-Term Goal)

o	Develop a clinician-facing tool that, based on a simple CAC scan, outputs:

	Predicted time-dependent risk levels for MACE,
	Likely anatomical regions of future occlusion?,
	Confidence intervals for predictions.

o	This tool could improve patient outcomes by assisting providers in balancing the risks of cardiac events with the risks of further tests, treatments, or surgeries.

o	Longer-term, this framework could serve as a model for applying machine learning to preventive medicine more broadly.

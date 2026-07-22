## Initial Registration Pipeline Failure Analysis

The first implementation of the registration pipeline produced anatomically plausible results in several cases but failed to maintain consistent coronary alignment across the complete cardiac volume. The primary limitations identified during development are summarized below.

- **Limited Preprocessing:** The initial preprocessing pipeline lacked robust anatomical normalization, intensity standardization, and adaptive ROI generation, making registration highly sensitive to patient-to-patient anatomical variation.( New preprocessing was done after analysis of the dataset)

- **Global Registration Bias:** Registration was performed over the entire thoracic volume, allowing surrounding structures such as the lungs, ribs and spine to dominate the optimization instead of the coronary anatomy. 

- **Affine Registration Instability:** Multiple registrations failed during the affine stage due to insufficient overlap between the fixed and moving images, producing the SimpleITK *"All samples map outside moving image buffer"* error. 

- **Transform Management:** Early iterations required refinement of rigid-to-affine transform initialization and composition to maintain a consistent physical coordinate system throughout the registration pipeline. 

- **Distorted Vessel Geometry:** The stretched and deformed coronary masks were primarily caused by accumulated registration inaccuracies during atlas warping rather than incorrect vessel segmentation.

- **Single-Atlas Generalization:** Using a single anatomical atlas limited robustness to variations in coronary dominance, cardiac orientation and patient morphology, reducing vessel coverage in challenging cases. :contentReference[oaicite:3]{index=3}

- **Elastix Investigation:** Elastix was evaluated during development as an alternative registration framework; however, experiments indicated that preprocessing quality, ROI definition, and transform management had a greater impact on registration accuracy than the registration backend itself.

## Resolution

These limitations were addressed through a redesigned hierarchical **Rigid → Affine** registration pipeline incorporating improved preprocessing, adaptive cardiac ROI generation, metric masking, standardized physical coordinate handling, and multi-atlas fusion. These changes significantly improved registration stability, anatomical consistency, and coronary vessel coverage across the evaluation cohort. 

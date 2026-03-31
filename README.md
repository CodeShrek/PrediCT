

# 😊 GSoC 2026 Evaluation: PrediCT Project (Shriyam Baloni)

This repository contains the evaluation tasks and technical proposals for the Google Summer of Code 2026 **PrediCT** project under **ML4Sci**. The work focuses on standardizing the Stanford COCA dataset and developing deep learning frameworks for coronary artery calcium (CAC) segmentation and physics-informed plaque growth simulation.

---

###  Dataset Overview: Stanford COCA
The pipeline processes the **Stanford Coronary Artery Calcium (COCA)** dataset, standardizing 1,155 non-contrast CT scans. The cohort was partitioned using a **stratified split** to ensure balanced representation of calcium burden across training, validation, and testing sets.

| Cohort | Total Scans | Zero Calcium Cases (%) | Avg Calcium Burden |
| :--- | :---: | :---: | :---: |
| **TRAIN SET** | 714 | 306 (42.9%) | 607.19 voxels |
| **VAL SET** | 217 | 90 (41.5%) | 621.07 voxels |
| **TEST SET** | 224 | 87 (38.8%) | 630.06 voxels |
| **TOTAL** | **1,155** | **483 (41.8%)** | **614.21 voxels** |

---

### ✅ Completed: 
*  Project 1: Heart Segmentation (top priority)- completed can be viewed in TASK_1 folder along with its report
*  Project 3 (task3)implementation- Completed task 3 and its results can be viewed in TASK 3 folder .
*  Common task is present inside COCA_scripts as it was implemented on the original code .
---
### <--Repository Structure-->

```bash
├── COCA_scripts/           # Common Task: Preprocessing Suite
│   
├── TASK_1/                 # Project 1: Heart Segmentation
│
├── TASK 3/                 # Project 3: Coronary Atlas Registration
│
├── proposal/               # GSoC 2026 Formal Submissions
    ├── proposal 1/
    └── proposal 2/        
```


#Development reports of each fo the TASK(1&3) and common task  are present in their respective folders . 
additionally, common task's documentation is simply this readme File(The Design Rationale Section).

## ADIOS!!!
### SEE U AROUND 

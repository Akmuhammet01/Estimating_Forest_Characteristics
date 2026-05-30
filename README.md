# Estimating Forest Structural Characteristics from Satellite Imagery Using AlphaEarth Foundation Models

> Master's Thesis — Skolkovo Institute of Science and Technology (Skoltech), 2026  
> **Author:** Akmuhammet Gurbangeldiyev  
> **Advisor:** Svetlana Illarionova, PhD, Head of Research Group

---

## Overview

This repository contains the full experimental pipeline for predicting primary forest structural attributes — **stand age**, **tree height**, **basal area**, and **timber stock** (m³/ha) — from satellite imagery over the **Sakhalin region, Russia**.

The core idea is to fuse high-dimensional latent embeddings from the **AlphaEarth Foundation (AEF)** model with standard **Sentinel-2 (S2)** multispectral data, then benchmark two tabular ML architectures — **XGBoost** and **TabNet** — across all prediction targets. A SHAP-based feature selection stage further reduces the feature space to compact, target-specific AEF band subsets.

---

## Key Results

| Target | Model | Dataset | RMSE / Accuracy |
|---|---|---|---|
| Stand Age | TabNet | AEF + S2 | **0.94 accuracy, 0.93 F1-macro** |
| Stand Age | XGBoost | AEF + S2 | 0.92 accuracy |
| Tree Height | TabNet | AEF + S2 | **RMSE = 1.274 m** |
| Basal Area | TabNet | AEF + S2 | **RMSE = 6.147 m²/ha** |
| Timber Stock | TabNet | AEF + S2 | **RMSE = 22.972 m³/ha** |

All models trained on S2-only data served as baselines (e.g., Height RMSE = 2.989, Age accuracy = 0.63), highlighting the substantial gain from AEF embeddings.

---

## Repository Structure

```
.
├── data_preparation/
│   ├── Preperation_from_raster_to_csv.ipynb          # Raster → tabular pipeline (zonal statistics)
│   └── targets_distribution_plots.ipynb              # EDA and target distribution analysis
│
├── xgboost_training/
│   ├── XGBoost_Training_AEF_S2_Target_Age_GridSearch.ipynb
│   ├── XGBoost_Training_AEF_S2_Target_Height_GridSearch.ipynb
│   ├── XGBoost_Training_AEF_S2_Target_Basal_Area_GridSearch.ipynb
│   └── XGBoost_Training_AEF_S2_Target_Stock_GridSearch.ipynb
│
├── tabnet_training/
│   ├── Tabnet_Training_AEF_S2_Target_Age_Bayesian_Optimization.ipynb
│   ├── Tabnet_Training_AEF_S2_Target_Height_Bayesian_Optimization.ipynb
│   ├── Tabnet_Training_AEF_S2_Target_Basal_Area_Bayesian_Optimization.ipynb
│   └── Tabnet_Training_AEF_S2_Target_Stock_Bayesian_Optimization.ipynb
│
├── feature_selection_and_reduced_training/
│   ├── Tabnet_XGBoost_Training_with_selected_features_Target_Age.ipynb
│   ├── Tabnet_XGBoost_Training_with_selected_features_Target_Height.ipynb
│   ├── Tabnet_XGBoost_Training_with_selected_features_Target_Basal_Area.ipynb
│   └── Tabnet_XGBoost_Training_with_selected_features_Target_Stock.ipynb
│
├── comparison_and_feature_importance/
│   ├── Comparison_and_Feature_Importance_Target_Age.ipynb
│   ├── Comparison_and_Feature_Importance_Target_Height.ipynb
│   ├── Comparison_and_Feature_Importance_Target_Basal_Area.ipynb
│   └── Comparison_and_Feature_Importance_Target_Stock.ipynb
│
└── README.md
```

---

## Methodology

### Study Area
Three forestry districts in the **Sakhalin region** of Russia: Korsakovskoye (KSK), Nevelskoye (NVL), and Kholmskoye (KLM). Forest inventory data compiled in 2018 covers 81,143 Individual Forest Stand (IFS) polygons across the three districts.

### Data Sources

**Sentinel-2 imagery** — 10 spectral bands (B02–B08, B8A, B11, B12) plus three computed vegetation indices:

$$NDVI = \frac{B08 - B04}{B08 + B04}, \quad EVI = 2.5 \times \frac{B08 - B04}{B08 + 6 \cdot B04 - 7.5 \cdot B02 + 1}, \quad GNDVI = \frac{B08 - B03}{B08 + B03}$$

**AEF embeddings** — 64-dimensional L2-normalized latent vectors derived from multi-year (2017–2024) multi-modal satellite time series.

### Feature Engineering (Stand-Based)
For each IFS polygon, four zonal statistics (min, max, mean, std) were extracted per band using `rasterstats`, producing:
- **S2 baseline dataset:** 52 features (13 layers × 4 stats)
- **Combined AEF+S2 dataset:** 306 features (77 layers × 4 stats)

Total dataset: **393,243 polygon records** — split 70% train / 15% validation / 15% test.

### Models

| Model | Hyperparameter Search | Class Imbalance Handling |
|---|---|---|
| XGBoost | Grid Search | Random Oversampling (age only) |
| TabNet | Bayesian Optimization (Optuna / TPE) | Root Square Transformer weights |

### Feature Importance & Selection
SHAP values (TreeExplainer for XGBoost, KernelExplainer for TabNet) were used to rank features. Target-specific AEF-only subsets were identified for efficient retraining:

| Target | Selected AEF Bands | Count |
|---|---|---|
| Stand Age | 3, 5, 7, 8, 9, 15, 17, 21–24, 28, 30, 31, 34, 37–39, 41, 43, 45, 49, 51–53, 55, 57, 62, 63 | 29 |
| Stand Height | 2, 3, 4, 7–9, 15–17, 19, 21–23, 31–33, 37–39, 41, 43, 45, 47, 49, 51, 55, 56, 59, 60, 63 | 30 |
| Basal Area | 2–4, 7–9, 11, 17–19, 21, 26, 28, 30–34, 37–43, 47, 49–51, 55, 57, 59, 61–64 | 36 |
| Timber Stock | 1–3, 5, 7–11, 13–15, 17, 19, 21, 24, 27–31, 33, 35–37, 39–43, 45, 47–53, 55, 57, 62–64 | 43 |

---

## Setup & Requirements

```bash
git clone https://github.com/Akmuhammet01/Estimating_Forest_Characteristics.git
cd Estimating_Forest_Characteristics
pip install -r requirements.txt
```

### Core Dependencies

```
python>=3.9
xgboost
pytorch-tabnet
optuna
shap
rasterstats
geopandas
scikit-learn
imbalanced-learn
numpy
pandas
matplotlib
```

---

## Reproducing Experiments

### 1. Data Preparation
Run `Preperation_from_raster_to_csv.ipynb` to extract zonal statistics from raster files and produce the tabular datasets.

### 2. Training

**XGBoost (Grid Search):**
```
xgboost_training/XGBoost_Training_AEF_S2_Target_<TARGET>_GridSearch.ipynb
```

**TabNet (Bayesian Optimization):**
```
tabnet_training/Tabnet_Training_AEF_S2_Target_<TARGET>_Bayesian_Optimization.ipynb
```
Replace `<TARGET>` with `Age`, `Height`, `Basal_Area`, or `Stock`.

### 3. Feature Importance & Comparison
```
comparison_and_feature_importance/Comparison_and_Feature_Importance_Target_<TARGET>.ipynb
```

### 4. Retraining on Reduced Feature Space
```
feature_selection_and_reduced_training/Tabnet_XGBoost_Training_with_selected_features_Target_<TARGET>.ipynb
```

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{gurbangeldiyev2026forest,
  author    = {Akmuhammet Gurbangeldiyev},
  title     = {Estimating Forest Structural Characteristics from Satellite Imagery Using AlphaEarth Foundation Models},
  school    = {Skolkovo Institute of Science and Technology},
  year      = {2026},
  address   = {Moscow, Russia}
}
```

---

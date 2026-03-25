# 0.338 Method Reconstruction

## Status
This file documents the closest recoverable pipeline for the previously observed online score around `0.338`, reconstructed from the assets still present in the workspace on 2026-03-10.

## Main Entry Point
- `src/models/reproduce_0338_method.py`
- Implementation lives in `src/models/advanced_stacking_pipeline.py`

## Command
```powershell
C:\Users\cocol\miniconda3\envs\ds-base\python.exe src\models\reproduce_0338_method.py
```

## Data Inputs
- `data/processed/landsat_api_training.csv`
- `data/processed/landsat_api_validation.csv`
- `resources/code/general/landsat_features_training.csv`
- `resources/code/general/landsat_features_validation.csv`
- `resources/code/general/terraclimate_features_training.csv`
- `resources/code/general/terraclimate_features_validation.csv`
- `resources/code/general/water_quality_training_dataset.csv`
- `resources/code/general/submission_template.csv`

## Reconstructed Method
1. Merge API Landsat with official Landsat fallback features and TerraClimate.
2. Build engineered optical, hydro-climate, and quality-control features.
3. Use target-specific feature spaces:
   - Total Alkalinity: full common feature block plus `Latitude`, `Longitude`
   - Electrical Conductance: full common feature block without direct coordinates
   - Dissolved Reactive Phosphorus: full common feature block plus `Latitude`
4. Train a three-stage target chain:
   - `TA` first
   - `EC <- TA prediction`
   - `DRP <- TA prediction + EC prediction`
5. For each target, train three base learners (`XGBoost`, `LightGBM`, `CatBoost`) under `GroupKFold` using `spatial_group`.
6. Stack out-of-fold base predictions with a `Ridge` meta-model.
7. Restrict `EC` and `DRP` training to rows with complete API optical bands.
8. Weight rows by API completeness, fallback availability, and `days_offset` quality.

## Current Reproduced Local Metrics
- Total Alkalinity: `0.4085`
- Electrical Conductance: `0.3280`
- Dissolved Reactive Phosphorus: `0.1774`

These metrics are stored in `cv_metrics.csv` beside the submission file.

## Outputs
- `output/submission_advanced_stacking/submission_ensemble.csv`
- `output/submission_advanced_stacking/cv_metrics.csv`
- `output/submission_advanced_stacking/feature_summary.txt`

## Important Note
The exact historical intermediate folders mentioned earlier in conversation were not present in the current workspace. This reconstruction is therefore based on the strongest recoverable pipeline still available on disk, rather than a byte-for-byte restoration of missing historical artifacts.

# Project Guide

## Objective

Predict the following targets for the 200 validation locations in `resources/code/general/submission_template.csv`:

- `Total Alkalinity`
- `Electrical Conductance`
- `Dissolved Reactive Phosphorus`

The competition metric is the average R2 across these three targets.

## Core Data Sources

Official challenge data under `resources/code/general`:

- `water_quality_training_dataset.csv`: labels and sample metadata
- `submission_template.csv`: validation rows to score
- `landsat_features_training.csv`, `landsat_features_validation.csv`: official Landsat feature tables
- `terraclimate_features_training.csv`, `terraclimate_features_validation.csv`: official TerraClimate feature tables

Additional processed data under `data/processed`:

- `landsat_api_training.csv`, `landsat_api_validation.csv`: cleaner API-derived optical features
- `landsat_context_training.csv`, `landsat_context_validation.csv`: multi-window, multi-scale contextual Landsat summaries

## Modeling Families In This Repo

### 1. Main recovered strong baseline

File: `src/models/advanced_stacking_pipeline.py`

Key ideas:

- Merge API Landsat, official Landsat fallback features, and TerraClimate
- Build engineered optical and hydro-climate features
- Use a target chain: `TA -> EC -> DRP`
- For each target, fit `XGBoost`, `LightGBM`, and `CatBoost`
- Stack base-model OOF predictions with a ridge meta-model
- Restrict `EC` and `DRP` to rows with complete API optical coverage
- Use row weights based on API completeness, fallback availability, and date offset quality

Output folder:

- `output/submission_advanced_stacking`

Recovered local CV metrics:

- `TA = 0.4085`
- `EC = 0.3280`
- `DRP = 0.1774`
- `avg spatial = 0.3046`

### 2. Stricter but more conservative stack

File: `src/models/advanced_strict_hybrid_pipeline.py`

Purpose:

- reduce optimistic meta-stacking bias
- keep a stricter OOF structure
- blend DRP chain and DRP specialist conservatively

Output folder:

- `output/submission_advanced_strict_hybrid`

### 3. Online-oriented DRP hybrid

File: `src/models/advanced_online_hybrid_pipeline.py`

Purpose:

- keep strong `TA/EC` from `advanced_stacking`
- scan specialist DRP blends and thresholds for better online behavior

Known result:

- online score came back as `0.3489`, below the `0.3529` anchor

### 4. Target-wise hybrid

File: `src/models/targetwise_hybrid_pipeline.py`

Purpose:

- use `advanced_no_quality_flags` style features for `TA/EC`
- use `advanced_no_old_optics` style features for `DRP`
- blend DRP chain with a specialist head

Output folder:

- `output/submission_targetwise_hybrid`

Current local CV metrics:

- `TA = 0.4141`
- `EC = 0.3442`
- `DRP = 0.1809`
- `avg spatial = 0.3131`

This is structurally strong, but it still needs online confirmation.

## Local Evaluation

Main evaluator:

- `src/evaluation/evaluate_local.py`

What it now does:

- reads saved CV metrics from submission folders
- estimates public score from known online anchors
- uses submission-level proximity to known anchors when `submission_ensemble.csv` is available
- is calibrated against at least these confirmed online anchors:
  - `output/submission_advanced_stacking/submission_ensemble.csv -> 0.3459`
  - `output/submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv -> 0.3529`
  - `output/submission_advanced_online_hybrid/submission_ensemble.csv -> 0.3489`

Important limitation:

- the evaluator is useful for ranking nearby variants
- it is not a guarantee of online score
- large distribution shifts or new feature families still require real submissions to validate

## Contextual Data Pipeline

File:

- `src/data/fetch_planetary_contextual.py`

What it fetches:

- Landsat scenes from Planetary Computer
- windows: `15`, `30`, `60` days around each sample date
- scales: `fine`, `wide`
- per-scene band statistics plus derived indices
- summarized scene distributions for each window-scale pair

Current status:

- validation contextual file is complete: `200/200`
- training contextual file is partial: `400/9319`
- this is currently the biggest unfinished data improvement path in the repo

## Suggested Workflow

1. Use `output/submission_advanced_stacking` as the reference baseline.
2. Use `output/submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv` as the online anchor.
3. Compare any new run with `src/evaluation/evaluate_local.py`.
4. Prefer target-wise changes over global model changes when only one target is unstable.
5. Finish contextual training data before expecting a major jump beyond the current online best.

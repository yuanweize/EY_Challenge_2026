# EY Open Science Data Challenge 2026

This repository contains our working code, local evaluation tooling, experiment outputs, and documentation for the EY Open Science Data Challenge 2026 task: predicting three water-quality targets in South Africa from satellite, climate, and derived contextual features.

## Current Status

- Best confirmed online score so far: `0.3529`
- Best confirmed submission file so far: `output/submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv`
- Best reconstructed main pipeline on disk: `src/models/advanced_stacking_pipeline.py`
- Latest stronger target-wise experiment on disk: `src/models/targetwise_hybrid_pipeline.py`
- Local evaluator now uses both CV metrics and submission-level nearest-anchor calibration: `src/evaluation/evaluate_local.py`
- Contextual Landsat data exists for validation (`200/200`) but training is still partial (`400/9319`): `data/processed/landsat_context_validation.csv`, `data/processed/landsat_context_training.csv`

## Documentation Map

- `docs/PROJECT_GUIDE.md`: project layout, datasets, pipelines, and command reference
- `docs/RULES_AND_SKILLS.md`: challenge rules, repo working rules, and available Codex skills
- `docs/EXPERIMENT_STATUS.md`: current best runs, what is trustworthy, and the next highest-value improvements
- `resources/docs/challenge_rules_faq.md`: condensed challenge rules and FAQ
- `resources/docs/snowflake_guide.md`: official Snowflake-oriented getting-started notes

## Environment

Primary working environment in this repo:

```powershell
C:/Users/cocol/miniconda3/envs/ds-base/python.exe
```

Install the core dependencies with:

```powershell
pip install -r resources/code/general/requirements.txt
pip install scikit-learn xgboost lightgbm catboost pystac-client odc-stac planetary-computer
```

## Main Commands

Run the main reconstructed stacking pipeline:

```powershell
C:/Users/cocol/miniconda3/envs/ds-base/python.exe -m src.models.advanced_stacking_pipeline
```

Run the latest target-wise hybrid experiment:

```powershell
C:/Users/cocol/miniconda3/envs/ds-base/python.exe -m src.models.targetwise_hybrid_pipeline
```

Score a saved submission directory with the calibrated local evaluator:

```powershell
C:/Users/cocol/miniconda3/envs/ds-base/python.exe src/evaluation/evaluate_local.py --submission-dir output/submission_advanced_stacking
```

Resume contextual data fetching from Planetary Computer:

```powershell
C:/Users/cocol/miniconda3/envs/ds-base/python.exe src/data/fetch_planetary_contextual.py
```

## Repository Layout

- `src/data`: data extraction and feature-building scripts
- `src/models`: model pipelines and experiment scripts
- `src/evaluation`: local evaluation and diagnostics
- `resources/code/general`: official challenge datasets and notebooks
- `data/processed`: fetched API/contextual features
- `output`: saved submissions, CV metrics, scans, and experiment artifacts
- `docs`: maintained project documentation

## Recommended Starting Point

1. Read `docs/PROJECT_GUIDE.md`.
2. Read `docs/EXPERIMENT_STATUS.md`.
3. Use `output/submission_advanced_stacking` as the baseline reference.
4. Treat `output/submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv` as the current online anchor.
5. Prioritize completing contextual training data before expecting another large score jump.

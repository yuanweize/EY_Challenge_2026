# Experiment Status

## Confirmed Online Anchors

### Best current online result

- file: `output/submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv`
- online score: `0.3529`
- structure: keep `advanced_stacking` TA/EC, regularize only DRP with a light specialist blend

### Recovered strong baseline

- folder: `output/submission_advanced_stacking`
- online score: `0.3459`
- local CV: `0.3046`

### Recent online miss

- folder: `output/submission_advanced_online_hybrid`
- online score: `0.3489`
- lesson: a slightly better local DRP CV does not automatically improve online score

## Strong Local-Only Experiments

### `advanced_no_old_optics`

- local CV:
  - `TA = 0.4105`
  - `EC = 0.3311`
  - `DRP = 0.1798`
- signal:
  - removing old optical fallback features can help DRP stability

### `advanced_no_quality_flags`

- local CV:
  - `TA = 0.4141`
  - `EC = 0.3442`
  - `DRP = 0.1666`
- signal:
  - many quality/missingness flags help less than expected for TA/EC

### `submission_targetwise_hybrid`

- local CV:
  - `TA = 0.4141`
  - `EC = 0.3442`
  - `DRP = 0.1809`
  - `avg spatial = 0.3131`
- local calibrated public estimate:
  - about `0.3476`
- interpretation:
  - likely promising but not yet strong enough to displace the `0.3529` online anchor without real submission evidence

## Biggest Unfinished Opportunity

### Contextual Landsat training data

The highest-upside unfinished improvement path is not another tiny blend search. It is completing the contextual training dataset and injecting those features into the main pipeline.

Why this matters:

- current models mainly rely on point-wise or near-point features
- DRP appears especially sensitive to time context and local heterogeneity
- contextual windows can add stability that simple blend tuning cannot

Current blocker:

- `data/processed/landsat_context_training.csv` currently has only `400` rows out of `9319`

## Recommended Next Steps

1. Finish `landsat_context_training.csv`.
2. Build a context-aware variant of `advanced_stacking` or `targetwise_hybrid`.
3. Test context features first on `DRP`, then decide whether to expose them to `TA/EC`.
4. Keep using the updated local evaluator for ranking, but validate any serious candidate online.
5. Preserve every strong candidate submission file with a stable name and notes.

## What Not To Over-Invest In Right Now

- pure DRP threshold sweeps without new data
- tiny blend-weight changes that do not materially alter prediction distributions
- overinterpreting local calibrated public scores as if they were leaderboard truth

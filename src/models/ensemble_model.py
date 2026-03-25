"""
EY Challenge 2026 — Phase T Ensemble Pipeline
Integration of Target Chain + Ridge Meta-Stacking + DRP Specialist + Sample Weighting
Adapted from liuxinyi15's 0.3599 approach, cherry-picked into our robust data pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

N_SPLITS = 5
RANDOM_STATE = 42
DRP_SPECIALIST_THRESHOLD = 144.0
DRP_SPECIALIST_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Base Model Factories (Phase T: Updated Hyperparameters)
# ---------------------------------------------------------------------------
def build_base_models():
    return {
        'xgb': XGBRegressor(
            n_estimators=700, learning_rate=0.025, max_depth=5,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, missing=np.nan,
        ),
        'lgb': LGBMRegressor(
            n_estimators=700, learning_rate=0.025, num_leaves=28,
            max_depth=5, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        'cat': CatBoostRegressor(
            iterations=700, learning_rate=0.025, depth=5,
            l2_leaf_reg=3.0, random_strength=1.0,
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        ),
    }


# ---------------------------------------------------------------------------
# Feature Engineering (Phase T: Expanded with interaction terms + quality flags)
# ---------------------------------------------------------------------------
def add_features(df):
    """Unified feature engineering for both train and validation."""
    df = df.copy()

    # Temporal
    df['date_parsed'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df['month'] = df['date_parsed'].dt.month
    df['quarter'] = df['date_parsed'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Data quality flags
    optical_cols = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
    old_cols = ['nir', 'green_old', 'swir16_old', 'swir22_old']
    climate_cols = ['pet', 'ppt', 'tmax', 'tmin', 'q']

    df['days_offset_filled'] = df['days_offset'].fillna(61).clip(0, 61) if 'days_offset' in df.columns else 61.0
    df['days_offset_missing'] = (df['days_offset'].isna().astype(int)) if 'days_offset' in df.columns else 1
    df['api_optical_complete'] = df[optical_cols].notna().all(axis=1).astype(int)
    has_old = [c for c in old_cols if c in df.columns]
    df['old_optical_available'] = df[has_old].notna().all(axis=1).astype(int) if has_old else 0
    df['api_preferred'] = ((df['api_optical_complete'] == 1) & (df['days_offset_filled'] <= 10)).astype(int) if 'days_offset' in df.columns else df['api_optical_complete']

    # Spectral indices
    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)
    df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + 1e-8)
    df['turbidity_ratio'] = df['red'] / (df['blue'] + 1e-8)
    df['green_red_ratio'] = df['green'] / (df['red'] + 1e-8)
    df['nir_swir_ratio'] = df['nir08'] / (df['swir16'] + 1e-8)
    df['swir_ratio'] = df['swir16'] / (df['swir22'] + 1e-8)

    # Climate interaction terms
    df['ppt_pet_ratio'] = df['ppt'] / (df['pet'] + 1e-8)
    df['temp_range'] = df['tmax'] - df['tmin']
    df['runoff_ratio'] = df['q'] / (df['ppt'] + 1e-8)
    df['hydro_stress'] = df['pet'] / (df['ppt'] + 1e-8)
    df['thermal_runoff'] = df['temp_range'] * df['q']

    # Climate missingness flags
    for col in climate_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)

    # Old optical indices (if available)
    if 'NDMI' not in df.columns and 'nir' in df.columns and 'swir16_old' in df.columns:
        df['NDMI'] = (df['nir'] - df['swir16_old']) / (df['nir'] + df['swir16_old'] + 1e-8)
    if 'MNDWI' not in df.columns and 'green_old' in df.columns and 'swir16_old' in df.columns:
        df['MNDWI'] = (df['green_old'] - df['swir16_old']) / (df['green_old'] + df['swir16_old'] + 1e-8)

    # Spatial grouping
    df['lat_bin'] = np.round(df['Latitude'], 1)
    df['lon_bin'] = np.round(df['Longitude'], 1)
    df['spatial_group'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)

    return df


def build_feature_sets():
    """Target-specific feature lists."""
    common = [
        'blue', 'green', 'red', 'nir08', 'swir16', 'swir22',
        'pet', 'ppt', 'tmax', 'tmin', 'q',
        'soil', 'vpd', 'srad', 'water_def',
        'month_sin', 'month_cos', 'quarter',
        'days_offset_filled', 'days_offset_missing',
        'api_optical_complete', 'old_optical_available', 'api_preferred',
        'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI', 'NDTI',
        'turbidity_ratio', 'green_red_ratio', 'nir_swir_ratio', 'swir_ratio',
        'ppt_pet_ratio', 'temp_range', 'runoff_ratio', 'hydro_stress', 'thermal_runoff',
        'pet_is_missing', 'ppt_is_missing', 'tmax_is_missing', 'tmin_is_missing', 'q_is_missing',
    ]
    # Add old optical columns if they exist
    old_optical = ['nir', 'green_old', 'swir16_old', 'swir22_old', 'NDMI', 'MNDWI']
    return {
        'Total Alkalinity': common + old_optical + ['Latitude', 'Longitude'],
        'Electrical Conductance': common + old_optical,
        'Dissolved Reactive Phosphorus': common + old_optical + ['Latitude'],
    }


# ---------------------------------------------------------------------------
# Sample Weighting (Phase T: Penalizes low-quality data sources)
# ---------------------------------------------------------------------------
def sample_weights(df):
    source_w = np.where(
        df['api_optical_complete'] == 1, 1.0,
        np.where(df['old_optical_available'] == 1, 0.92, 0.85)
    )
    offset_w = 1.0 - (df['days_offset_filled'] / 61.0) * 0.18
    climate_w = np.where(df['ppt'].notna(), 1.0, 0.95) if 'ppt' in df.columns else 1.0
    return source_w * offset_w * climate_w


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
def load_frames():
    """Load and merge all data sources for training and validation."""
    # Training
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    tc_tr = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
    ls_api_tr = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_training.csv')).set_index('Index')
    old_tr = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))

    train = wq.join(ls_api_tr, how='left')
    train = train.merge(old_tr, on=MERGE_KEYS, how='left', suffixes=('', '_old'))
    train = train.merge(tc_tr, on=MERGE_KEYS, how='left')

    tc_extra_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra_training.csv')
    if os.path.exists(tc_extra_path):
        tc_extra = pd.read_csv(tc_extra_path)
        train = train.merge(tc_extra, on=MERGE_KEYS, how='left')
        print("  [+] Extra climate features loaded (soil/vpd/srad/water_def)")

    # Validation
    sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    ls_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
    tc_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))
    ls_val_api = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_validation.csv')).set_index('Index')

    val = ls_val[MERGE_KEYS].join(ls_val_api, how='left')
    val = val.merge(ls_val, on=MERGE_KEYS, how='left', suffixes=('', '_old'))
    val = val.merge(tc_val, on=MERGE_KEYS, how='left')

    tc_api_val_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_api_validation.csv')
    if os.path.exists(tc_api_val_path):
        tc_api_val = pd.read_csv(tc_api_val_path).rename(columns={'pr': 'ppt', 'ro': 'q'})
        val = val.merge(tc_api_val, on=MERGE_KEYS, how='left')

    tc_extra_val_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra_validation.csv')
    if os.path.exists(tc_extra_val_path):
        tc_extra_val = pd.read_csv(tc_extra_val_path)
        val = val.merge(tc_extra_val, on=MERGE_KEYS, how='left')

    return train, val, sub


# ---------------------------------------------------------------------------
# Target Chain with Ridge Meta-Stacking (Phase T Core Innovation)
# ---------------------------------------------------------------------------
def fit_target_chain(train_df, val_df, target, feature_cols,
                     extra_train=None, extra_val=None, filter_complete=False):
    """
    Train XGB/LGB/CAT base models with Spatial K-Fold, then stack with Ridge.
    Supports target chaining via extra_train/extra_val predictions.
    """
    train_mask = pd.Series(True, index=train_df.index)
    if filter_complete:
        train_mask &= train_df['api_optical_complete'] == 1

    tr = train_df.loc[train_mask].reset_index(drop=True)
    va = val_df.reset_index(drop=True).copy()

    # Filter feature_cols to only those that exist in the dataframe
    available_features = [c for c in feature_cols if c in tr.columns]
    X = tr[available_features].copy()
    X_val = va[available_features].copy()

    if extra_train is not None:
        for name, values in extra_train.items():
            X[name] = values.loc[train_mask].reset_index(drop=True)
    if extra_val is not None:
        for name, values in extra_val.items():
            X_val[name] = values.reset_index(drop=True)

    y = tr[target].copy()
    groups = tr['spatial_group']
    weights = sample_weights(tr)
    gkf = GroupKFold(n_splits=N_SPLITS)

    base_keys = list(build_base_models().keys())
    oof_base = {name: np.zeros(len(tr)) for name in base_keys}
    val_fold_preds = {name: [] for name in base_keys}

    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = weights[tr_idx]

        for name, model in build_base_models().items():
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_base[name][va_idx] = model.predict(X_va)
            val_fold_preds[name].append(model.predict(X_val))

    # Ridge Meta-Stacking
    stack_train = pd.DataFrame({f'{name}_pred': pred for name, pred in oof_base.items()})
    stack_val = pd.DataFrame({f'{name}_pred': np.mean(preds, axis=0) for name, preds in val_fold_preds.items()})

    passthrough_cols = ['days_offset_filled', 'api_optical_complete', 'old_optical_available',
                        'Latitude', 'Longitude', 'month_sin', 'month_cos']
    for col in passthrough_cols:
        if col in X.columns:
            stack_train[col] = X[col].values
            stack_val[col] = X_val[col].values

    if extra_train is not None:
        for name, values in extra_train.items():
            stack_train[name] = values.loc[train_mask].reset_index(drop=True).values
    if extra_val is not None:
        for name, values in extra_val.items():
            stack_val[name] = values.reset_index(drop=True).values

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_train, y, sample_weight=weights)
    oof = meta_model.predict(stack_train)

    # Final models retrained on 100% data
    final_models = {}
    for name, model in build_base_models().items():
        model.fit(X, y, sample_weight=weights)
        final_models[name] = model

    final_stack_val = pd.DataFrame({f'{name}_pred': model.predict(X_val) for name, model in final_models.items()})
    for col in passthrough_cols:
        if col in X.columns:
            final_stack_val[col] = X_val[col].values
    if extra_val is not None:
        for name, values in extra_val.items():
            final_stack_val[name] = values.reset_index(drop=True).values
    val_pred = meta_model.predict(final_stack_val)

    return {
        'train_mask': train_mask,
        'oof_pred': pd.Series(oof, index=tr.index),
        'val_pred': pd.Series(val_pred, index=va.index),
        'CV_R2': float(r2_score(y, oof)),
        'CV_RMSE': float(np.sqrt(mean_squared_error(y, oof))),
        'rows_used': len(tr),
    }


# ---------------------------------------------------------------------------
# DRP Specialist: CatBoost classifier splits high/low regimes
# ---------------------------------------------------------------------------
def fit_drp_specialist_oof(train_df, val_df, features, threshold):
    """Specialist for DRP: classifies high/low, trains separate regressors, blends by P(high)."""
    mask = train_df['api_optical_complete'] == 1
    tr = train_df.loc[mask].reset_index(drop=True)
    va = val_df.reset_index(drop=True).copy()

    available_features = [c for c in features if c in tr.columns]
    X = tr[available_features].copy()
    X_val = va[available_features].copy()
    y = tr['Dissolved Reactive Phosphorus'].copy()
    groups = tr['spatial_group']
    w = sample_weights(tr)
    oof = np.zeros(len(tr))
    val_parts = []

    def blend_preds(preds_dict):
        vals = list(preds_dict.values())
        return np.mean(vals, axis=0)

    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = w[tr_idx]
        label = (y_tr >= threshold).astype(int)

        clf = CatBoostClassifier(
            iterations=250, learning_rate=0.03, depth=5,
            l2_leaf_reg=3.0, random_strength=1.0,
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        clf.fit(X_tr, label, sample_weight=w_tr)
        p_high = clf.predict_proba(X_va)[:, 1]

        low_mask = y_tr < threshold
        high_mask = y_tr >= threshold

        # Train separate regressors for low and high regimes
        low_preds, high_preds = {}, {}
        for name, model in build_base_models().items():
            if low_mask.sum() > 5:
                model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
                low_preds[name] = model.predict(X_va)
            else:
                low_preds[name] = np.full(len(X_va), y_tr.loc[low_mask].mean() if low_mask.sum() > 0 else y_tr.mean())

        for name, model in build_base_models().items():
            if high_mask.sum() > 5:
                model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])
                high_preds[name] = model.predict(X_va)
            else:
                high_preds[name] = np.full(len(X_va), y_tr.loc[high_mask].mean() if high_mask.sum() > 0 else y_tr.mean())

        oof[va_idx] = (1.0 - p_high) * blend_preds(low_preds) + p_high * blend_preds(high_preds)

        # Validation fold prediction
        clf_v = CatBoostClassifier(
            iterations=250, learning_rate=0.03, depth=5,
            l2_leaf_reg=3.0, random_strength=1.0,
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        clf_v.fit(X_tr, label, sample_weight=w_tr)
        p_high_val = clf_v.predict_proba(X_val)[:, 1]

        low_val, high_val = {}, {}
        for name, model in build_base_models().items():
            if low_mask.sum() > 5:
                model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
                low_val[name] = model.predict(X_val)
        for name, model in build_base_models().items():
            if high_mask.sum() > 5:
                model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])
                high_val[name] = model.predict(X_val)

        if low_val and high_val:
            val_parts.append((1.0 - p_high_val) * blend_preds(low_val) + p_high_val * blend_preds(high_val))

    val_pred = np.mean(val_parts, axis=0) if val_parts else np.zeros(len(va))
    return {
        'train_mask': mask,
        'oof_pred': pd.Series(oof, index=tr.index),
        'val_pred': pd.Series(val_pred, index=va.index),
        'CV_R2': float(r2_score(y, oof)),
        'CV_RMSE': float(np.sqrt(mean_squared_error(y, oof))),
        'rows_used': len(tr),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_ensemble_pipeline():
    print("=" * 70)
    print("  EY Challenge 2026 — Phase T Pipeline")
    print("  Target Chain + Ridge Meta-Stacking + DRP Specialist")
    print("=" * 70)

    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)

    feature_sets = build_feature_sets()
    all_feats = set()
    for v in feature_sets.values():
        all_feats.update(v)
    print(f"\n[1/5] Feature universe: {len(all_feats)} unique features")

    # ---------------------------------------------------------------
    # Step 1: Predict Total Alkalinity
    # ---------------------------------------------------------------
    print("\n[2/5] Target Chain: Total Alkalinity...")
    ta_feats = feature_sets['Total Alkalinity']
    ta = fit_target_chain(train_df, val_df, 'Total Alkalinity', ta_feats, filter_complete=False)
    ta_oof_full = pd.Series(np.nan, index=train_df.index, name='ta_pred')
    ta_oof_full.loc[ta['train_mask']] = ta['oof_pred'].values
    ta_val = ta['val_pred'].rename('ta_pred')
    print(f"  -> TA Spatial CV R2: {ta['CV_R2']:.4f}  RMSE: {ta['CV_RMSE']:.2f}")

    # ---------------------------------------------------------------
    # Step 2: Predict EC, using TA predictions as extra feature
    # ---------------------------------------------------------------
    print("\n[3/5] Target Chain: Electrical Conductance (with ta_pred)...")
    ec_feats = feature_sets['Electrical Conductance']
    ec = fit_target_chain(
        train_df, val_df, 'Electrical Conductance', ec_feats,
        extra_train={'ta_pred': ta_oof_full},
        extra_val={'ta_pred': ta_val},
        filter_complete=True,
    )
    ec_oof_full = pd.Series(np.nan, index=train_df.index, name='ec_pred')
    ec_oof_full.loc[ec['train_mask']] = ec['oof_pred'].values
    ec_val = ec['val_pred'].rename('ec_pred')
    print(f"  -> EC Spatial CV R2: {ec['CV_R2']:.4f}  RMSE: {ec['CV_RMSE']:.2f}")

    # ---------------------------------------------------------------
    # Step 3: Predict DRP, using both TA + EC as extra features
    # ---------------------------------------------------------------
    print("\n[4/5] Target Chain: Dissolved Reactive Phosphorus (with ta_pred + ec_pred)...")
    drp_feats = feature_sets['Dissolved Reactive Phosphorus']

    # Main chain prediction
    drp_chain = fit_target_chain(
        train_df, val_df, 'Dissolved Reactive Phosphorus', drp_feats,
        extra_train={'ta_pred': ta_oof_full, 'ec_pred': ec_oof_full},
        extra_val={'ta_pred': ta_val, 'ec_pred': ec_val},
        filter_complete=True,
    )

    # DRP Specialist
    specialist_features = [
        'blue', 'green', 'red', 'nir08', 'swir16', 'swir22',
        'pet', 'ppt', 'tmax', 'tmin', 'q',
        'days_offset_filled', 'days_offset_missing',
        'api_optical_complete', 'api_preferred',
        'month_sin', 'month_cos',
        'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI', 'NDTI',
        'turbidity_ratio', 'ppt_pet_ratio', 'temp_range', 'runoff_ratio',
        'Latitude',
    ]
    drp_specialist = fit_drp_specialist_oof(train_df, val_df, specialist_features, threshold=DRP_SPECIALIST_THRESHOLD)

    # Blend chain + specialist
    w = DRP_SPECIALIST_WEIGHT
    drp_oof = (1.0 - w) * drp_chain['oof_pred'].values + w * drp_specialist['oof_pred'].values
    drp_val = (1.0 - w) * drp_chain['val_pred'].values + w * drp_specialist['val_pred'].values

    drp_mask = train_df['api_optical_complete'] == 1
    drp_y = train_df.loc[drp_mask, 'Dissolved Reactive Phosphorus'].reset_index(drop=True)
    drp_r2 = float(r2_score(drp_y, drp_oof))
    drp_rmse = float(np.sqrt(mean_squared_error(drp_y, drp_oof)))
    print(f"  -> DRP Chain CV R2: {drp_chain['CV_R2']:.4f}")
    print(f"  -> DRP Specialist CV R2: {drp_specialist['CV_R2']:.4f}")
    print(f"  -> DRP Final (blended) CV R2: {drp_r2:.4f}  RMSE: {drp_rmse:.2f}")

    # ---------------------------------------------------------------
    # Summary & Submission
    # ---------------------------------------------------------------
    avg_r2 = np.mean([ta['CV_R2'], ec['CV_R2'], drp_r2])
    print(f"\n[5/5] CV Summary:")
    metrics = pd.DataFrame([
        {'Parameter': 'Total Alkalinity', 'CV_R2': ta['CV_R2'], 'CV_RMSE': ta['CV_RMSE'], 'rows': ta['rows_used']},
        {'Parameter': 'Electrical Conductance', 'CV_R2': ec['CV_R2'], 'CV_RMSE': ec['CV_RMSE'], 'rows': ec['rows_used']},
        {'Parameter': 'Dissolved Reactive Phosphorus', 'CV_R2': drp_r2, 'CV_RMSE': drp_rmse, 'rows': drp_specialist['rows_used']},
    ])
    print(metrics.to_string(index=False))
    print(f"  Average Spatial CV R2: {avg_r2:.4f}")

    submission = sub[MERGE_KEYS].copy()
    submission['Total Alkalinity'] = ta['val_pred'].values
    submission['Electrical Conductance'] = ec['val_pred'].values
    submission['Dissolved Reactive Phosphorus'] = drp_val

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'submission_ensemble.csv')
    submission.to_csv(out_path, index=False)
    print(f"\nPhase T pipeline complete. Saved to: {out_path}")


if __name__ == '__main__':
    run_ensemble_pipeline()

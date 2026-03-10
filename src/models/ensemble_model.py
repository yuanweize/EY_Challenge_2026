import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

BASE_FEATURES = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'pet']
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

N_SPLITS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def execute_feature_engineering_train(df):
    df = df.copy()
    
    df['date_parsed'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')
    if df['date_parsed'].isnull().any():
        df['date_parsed'] = pd.to_datetime(df['Sample Date'], errors='coerce')
    df['month'] = df['date_parsed'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Phase F: Spectral indices
    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)
    
    # Phase M: Water quality indices
    df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + 1e-8)
    df['FAI'] = df['nir08'] - (df['red'] + (df['swir16'] - df['red']) * (865-655)/(1610-655))
    df['CDOM'] = df['blue'] / (df['green'] + 1e-8)
    df['Turbidity'] = df['red'] / (df['blue'] + 1e-8)
    df['BSI'] = ((df['swir16'] + df['red']) - (df['nir08'] + df['blue'])) / \
                ((df['swir16'] + df['red']) + (df['nir08'] + df['blue']) + 1e-8)
    
    # Phase Q: All features — geographic, temporal, spectral, climate
    # ppt/tmax/tmin/q from original tc CSV; soil/vpd/srad/water_def, aet/pdsi/vap/ws from API
    engineered_cols = ['Latitude', 'Longitude', 'month_sin', 'month_cos', 
                       'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI',
                       'NDTI', 'FAI', 'CDOM', 'Turbidity', 'BSI',
                       'ppt', 'tmax', 'tmin', 'q',
                       'soil', 'vpd', 'srad', 'water_def',
                       'aet', 'pdsi', 'vap', 'ws']
    
    final_features = BASE_FEATURES + engineered_cols
    return df, final_features

def execute_feature_engineering_test(df):
    df = df.copy()
    
    df['date_parsed'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')
    if df['date_parsed'].isnull().any():
        df['date_parsed'] = pd.to_datetime(df['Sample Date'], errors='coerce')
    df['month'] = df['date_parsed'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)
    
    df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + 1e-8)
    df['FAI'] = df['nir08'] - (df['red'] + (df['swir16'] - df['red']) * (865-655)/(1610-655))
    df['CDOM'] = df['blue'] / (df['green'] + 1e-8)
    df['Turbidity'] = df['red'] / (df['blue'] + 1e-8)
    df['BSI'] = ((df['swir16'] + df['red']) - (df['nir08'] + df['blue'])) / \
                ((df['swir16'] + df['red']) + (df['nir08'] + df['blue']) + 1e-8)
    
    engineered_cols = ['Latitude', 'Longitude', 'month_sin', 'month_cos',
                       'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI',
                       'NDTI', 'FAI', 'CDOM', 'Turbidity', 'BSI',
                       'ppt', 'tmax', 'tmin', 'q',
                       'soil', 'vpd', 'srad', 'water_def',
                       'aet', 'pdsi', 'vap', 'ws']
    
    final_features = BASE_FEATURES + engineered_cols
    return df, final_features

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_and_preprocess_training():
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
    ls_api = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_training.csv')).set_index('Index')
    
    merged = wq.join(ls_api, how='left')
    # Original tc has: pet, ppt, tmax, tmin, q
    merged = merged.merge(tc, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Phase P: Load extra climate features (soil, vpd, srad, water_def)
    tc_extra_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra_training.csv')
    if os.path.exists(tc_extra_path):
        tc_extra = pd.read_csv(tc_extra_path)
        merged = merged.merge(tc_extra, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
        print(f"  [+] Extra climate features loaded (soil/vpd/srad/water_def)")
        
    # Phase Q: Load the remaining extra climate features (aet, pdsi, vap, ws)
    tc_extra2_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra2_training.csv')
    if os.path.exists(tc_extra2_path):
        tc_extra2 = pd.read_csv(tc_extra2_path)
        merged = merged.merge(tc_extra2, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
        print(f"  [+] Extra climate features 2 loaded (aet/pdsi/vap/ws)")
    
    merged, final_features = execute_feature_engineering_train(merged)
    
    merged['lat_bin'] = np.round(merged['Latitude'], 1)
    merged['lon_bin'] = np.round(merged['Longitude'], 1)
    merged['spatial_group'] = merged['lat_bin'].astype(str) + '_' + merged['lon_bin'].astype(str)
    
    return merged, final_features

def load_and_preprocess_validation():
    sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    ls_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
    tc_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))
    ls_val_api = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_validation.csv')).set_index('Index')
    
    val_data = ls_val[['Latitude', 'Longitude', 'Sample Date']].join(ls_val_api, how='left')
    val_data = val_data.merge(tc_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Validation tc CSV only has 'pet'. Load API-extracted climate features.
    tc_api_val_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_api_validation.csv')
    if os.path.exists(tc_api_val_path):
        tc_api_val = pd.read_csv(tc_api_val_path).rename(columns={'pr': 'ppt', 'ro': 'q'})
        val_data = val_data.merge(tc_api_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Phase P: Load extra climate features for validation
    tc_extra_val_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra_validation.csv')
    if os.path.exists(tc_extra_val_path):
        tc_extra_val = pd.read_csv(tc_extra_val_path)
        val_data = val_data.merge(tc_extra_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
        
    # Phase Q: Load the remaining extra climate features for validation
    tc_extra2_val_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_extra2_validation.csv')
    if os.path.exists(tc_extra2_val_path):
        tc_extra2_val = pd.read_csv(tc_extra2_val_path)
        val_data = val_data.merge(tc_extra2_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    val_data, final_features = execute_feature_engineering_test(val_data)
    return sub, val_data, final_features

# ---------------------------------------------------------------------------
# Ensembling Engine (Phase P: Reverted to Phase N proven config)
# ---------------------------------------------------------------------------
def cross_validate_and_train_ensemble(df, feature_cols, target_col):
    print(f"\n--- Training {target_col} ---")
    
    X = df[feature_cols].copy()
    
    # Phase P: REVERTED from Phase O. Ridge stacking caused LB regression
    # (0.2879→0.2419) due to intercept overfitting on 0% overlap test set.
    use_log1p = False
    y = df[target_col].copy()
    y_raw = df[target_col].copy()
    groups = df['spatial_group']
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    oof_preds_xgb = np.zeros(len(df))
    oof_preds_lgb = np.zeros(len(df))
    oof_preds_cat = np.zeros(len(df))
    
    # Phase N regularization (proven on LB: 0.2879)
    xgb_params = {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 15, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': RANDOM_STATE, 'n_jobs': -1}
    lgb_params = {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 24, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_samples': 20, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
    cat_params = {'iterations': 500, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 3.0, 'random_strength': 1.0, 'random_seed': RANDOM_STATE, 'verbose': False, 'allow_writing_files': False}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va = X.iloc[val_idx]
        
        m_xgb = XGBRegressor(**xgb_params); m_xgb.fit(X_tr, y_tr)
        m_lgb = LGBMRegressor(**lgb_params); m_lgb.fit(X_tr, y_tr)
        m_cat = CatBoostRegressor(**cat_params); m_cat.fit(X_tr, y_tr)
        
        oof_preds_xgb[val_idx] = m_xgb.predict(X_va)
        oof_preds_lgb[val_idx] = m_lgb.predict(X_va)
        oof_preds_cat[val_idx] = m_cat.predict(X_va)
    
    # Simple averaging (proven safe on LB: 0.2879)
    oof_preds_ens = (oof_preds_xgb * 0.4) + (oof_preds_lgb * 0.3) + (oof_preds_cat * 0.3)
    
    overall_r2 = r2_score(y_raw, oof_preds_ens)
    overall_rmse = np.sqrt(mean_squared_error(y_raw, oof_preds_ens))
    print(f"-> Spatial CV R2  : {overall_r2:.4f}")
    print(f"-> Spatial CV RMSE: {overall_rmse:.4f}")
    
    # Final models retrained on 100% data
    print("Retraining final models on 100% data...")
    final_xgb = XGBRegressor(**xgb_params).fit(X, y)
    final_lgb = LGBMRegressor(**lgb_params).fit(X, y)
    final_cat = CatBoostRegressor(**cat_params).fit(X, y)
    
    metrics = {'Parameter': target_col, 'CV_R2': overall_r2, 'CV_RMSE': overall_rmse}
    
    return {'xgb': final_xgb, 'lgb': final_lgb, 'cat': final_cat, 'use_log1p': use_log1p}, metrics

def run_ensemble_pipeline():
    print("=" * 70)
    print("  EY Challenge 2026 — Phase P Ensemble (Reverted + Extra Climate)")
    print("=" * 70)
    
    train_df, features = load_and_preprocess_training()
    sub_template, val_df, _ = load_and_preprocess_validation()
    
    print(f"[1/4] Features constructed ({len(features)} total):")
    print(f"      {features}")
    
    models = {}
    all_metrics = []
    
    print("\n[2/4] Executing Spatial K-Fold CV & Final Training...")
    for target in TARGET_COLS:
        model_dict, metrics = cross_validate_and_train_ensemble(train_df, features, target)
        models[target] = model_dict
        all_metrics.append(metrics)
    
    print("\n[3/4] CV Results Summary:")
    results_df = pd.DataFrame(all_metrics)
    print(results_df.to_string(index=False))
    
    print("\n[4/4] Generating validation predictions...")
    X_val = val_df[features].copy()
    submission = sub_template[['Latitude', 'Longitude', 'Sample Date']].copy()
    
    for target in TARGET_COLS:
        d = models[target]
        p_xgb = d['xgb'].predict(X_val)
        p_lgb = d['lgb'].predict(X_val)
        p_cat = d['cat'].predict(X_val)
        
        if d['use_log1p']:
            p_xgb = np.expm1(p_xgb)
            p_lgb = np.expm1(p_lgb)
            p_cat = np.expm1(p_cat)
        
        # Simple averaging (Phase N proven config)
        p_ens = (p_xgb * 0.4) + (p_lgb * 0.3) + (p_cat * 0.3)
        submission[target] = p_ens
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'submission_ensemble.csv')
    submission.to_csv(out_path, index=False)
    print(f"\nEnsemble pipeline complete. Saved to: {out_path}")

if __name__ == '__main__':
    run_ensemble_pipeline()

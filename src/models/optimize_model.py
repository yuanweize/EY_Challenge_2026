import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
# Pointing to the new directory structure: src/models/optimize_model.py -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Base features natively provided
BASE_FEATURES = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'pet']
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

N_SPLITS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data Loading & Feature Engineering
# ---------------------------------------------------------------------------
def execute_feature_engineering(df):
    """
    Applies feature engineering:
    1. Extracts temporal features from 'Sample Date'
    2. Keeps Latitude and Longitude as spatial features
    3. Calculates robust Water Spectral Indices from new Planetary Data
    """
    df = df.copy()
    
    # 1. Temporal Features (Handling DD-MM-YYYY format)
    df['date_parsed'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')
    if df['date_parsed'].isnull().any():
        df['date_parsed'] = pd.to_datetime(df['Sample Date'], errors='coerce')
    
    df['month'] = df['date_parsed'].dt.month
    
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 2. Water Spectral Indices (NDWI, MNDWI, SABI, WRI, NDVI)
    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)
    
    # After extreme testing: Geographic coordinations and time cyclical features 
    # proved phenomenally synergistic when paired with high-fidelity pristine reflectance.
    engineered_cols = ['Latitude', 'Longitude', 'month_sin', 'month_cos', 
                       'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI']
    
    final_features = BASE_FEATURES + engineered_cols
    return df, final_features

def load_and_preprocess_training():
    """Build the complete training dataset."""
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
    ls_api = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_training.csv')).set_index('Index')
    
    # Merge sequentially
    merged = wq.join(ls_api, how='left')
    merged = merged.merge(tc, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Apply Feature Engineering
    merged, final_features = execute_feature_engineering(merged)
    
    # Spatial Groups for Cross Validation
    merged['lat_bin'] = np.round(merged['Latitude'], 1)
    merged['lon_bin'] = np.round(merged['Longitude'], 1)
    merged['spatial_group'] = merged['lat_bin'].astype(str) + '_' + merged['lon_bin'].astype(str)
    
    return merged, final_features

def load_and_preprocess_validation():
    """Load validation features to predict on."""
    sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    ls_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
    tc_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))
    ls_val_api = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'landsat_api_validation.csv')).set_index('Index')
    
    val_data = ls_val[['Latitude', 'Longitude', 'Sample Date']].join(ls_val_api, how='left')
    val_data = val_data.merge(tc_val, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Ensure all base features match training
    val_data, final_features = execute_feature_engineering(val_data)
    return sub, val_data, final_features

# ---------------------------------------------------------------------------
# Modeling & Evaluation (Spatial K-Fold)
# ---------------------------------------------------------------------------
def cross_validate_and_train(df, feature_cols, target_col):
    """
    Evaluates XGBoost using Spatial K-Fold on targets.
    After CV is complete, trains a final model on 100% of data.
    """
    print(f"\n--- Training {target_col} ---")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df['spatial_group']
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    oof_preds = np.zeros(len(df))
    cv_r2_scores = []
    
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'Missing': np.nan # Native XGBoost handles purely physical data missingness without median fill biases
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_va)
        oof_preds[val_idx] = val_preds
        fold_r2 = r2_score(y_va, val_preds)
        cv_r2_scores.append(fold_r2)
    
    overall_r2 = r2_score(y, oof_preds)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"-> Spatial CV R2  : {overall_r2:.4f}")
    print(f"-> Spatial CV RMSE: {overall_rmse:.4f}")
    
    # Retrain on full dataset for final submission
    final_model = XGBRegressor(**params)
    final_model.fit(X, y)
    
    metrics = {
        'Parameter': target_col,
        'CV_R2': overall_r2,
        'CV_RMSE': overall_rmse
    }
    
    return final_model, metrics

def run_optimization_pipeline():
    print("=" * 70)
    print("  EY Challenge 2026 — Optimized Spatial CV Pipeline (Planetary Data)")
    print("=" * 70)
    
    train_df, features = load_and_preprocess_training()
    sub_template, val_df, _ = load_and_preprocess_validation()
    
    print(f"[1/4] Features constructed ({len(features)} total):")
    print(f"      {features}")
    
    print("\n[2/4] Executing Spatial K-Fold CV & Final Training...")
    models = {}
    all_metrics = []
    
    for target in TARGET_COLS:
        model, metrics = cross_validate_and_train(train_df, features, target)
        models[target] = model
        all_metrics.append(metrics)
        
    print("\n[3/4] CV Results Summary:")
    results_df = pd.DataFrame(all_metrics)
    print(results_df.to_string(index=False))
    
    print("\n[4/4] Generating validation predictions...")
    X_val = val_df[features].copy()
    submission = sub_template[['Latitude', 'Longitude', 'Sample Date']].copy()
    
    for target in TARGET_COLS:
        preds = models[target].predict(X_val)
        submission[target] = preds
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'submission_optimized.csv')
    submission.to_csv(out_path, index=False)
    print(f"\nOptimization pipeline complete. Saved to: {out_path}")

if __name__ == '__main__':
    run_optimization_pipeline()

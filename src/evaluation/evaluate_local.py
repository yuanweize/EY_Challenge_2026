import os
import argparse
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------
TARGET_COLS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_and_merge_data():
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    
    # NEW PLANETARY DATA
    ls_api_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(DATA_DIR))), 'data', 'processed', 'landsat_api_training.csv')
    ls_api = pd.read_csv(ls_api_path).set_index('Index')
    
    # OLD FALLBACK DATA (for comparison)
    ls_old = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
    
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
    
    df = wq.join(ls_api, how='left')  # Join by index which aligns perfectly with wq
    df = df.merge(tc, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    
    # Merge old data for the benchmark baseline to still function
    df = df.merge(ls_old, on=['Latitude', 'Longitude', 'Sample Date'], suffixes=('', '_old'), how='left')
    
    # Pre-calculate common engineered features
    # NOTE: Our new API data has nir08 instead of nir
    df['pseudo_ndvi'] = (df['nir08'] - df['green']) / (df['nir08'] + df['green'] + 1e-8)
    df['pseudo_ndvi_old'] = (df['nir'] - df['green_old']) / (df['nir'] + df['green_old'] + 1e-8)
    # --- PHASE F: WATER SPECTRAL INDICES ---
    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)
    
    # --- PHASE M: NEW WATER PHYSICS INDICES ---
    df['NDTI'] = (df['red'] - df['green']) / (df['red'] + df['green'] + 1e-8)
    df['FAI'] = df['nir08'] - (df['red'] + (df['swir16'] - df['red']) * (865-655)/(1610-655))
    df['CDOM'] = df['blue'] / (df['green'] + 1e-8)
    df['Turbidity'] = df['red'] / (df['blue'] + 1e-8)
    df['BSI'] = ((df['swir16'] + df['red']) - (df['nir08'] + df['blue'])) / \
                ((df['swir16'] + df['red']) + (df['nir08'] + df['blue']) + 1e-8)
    
    # --- CYCLICAL TIME FEATURES ---
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    df['month'] = df['Sample Date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Spatial Groups for Spatial CV (rounded to 1 decimal, roughly 11km grids)
    lat_bin = np.round(df['Latitude'], 1)
    lon_bin = np.round(df['Longitude'], 1)
    df['spatial_group'] = lat_bin.astype(str) + '_' + lon_bin.astype(str)
    
    return df

# ---------------------------------------------------------
# EVALUATION LOGIC
# ---------------------------------------------------------
def evaluate_model(df, name, feature_cols, model_cls='RF', use_imputer=False):
    print("="*60)
    print(f"  Evaluating Model: {name}")
    print("="*60)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols].copy()
    groups = df['spatial_group'].copy()
    
    spatial_r2_list = []
    random_r2_list = []
    
    for target in TARGET_COLS:
        y = df[target].copy()
        
        # 1. Spatial Group K-Fold (Strict Out-of-Location extrapolation)
        gkf = GroupKFold(n_splits=5)
        oof_spatial = np.zeros(len(df))
        for tr_idx, va_idx in gkf.split(X, y, groups):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            
            if use_imputer:
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_va = imputer.transform(X_va)
                
            if model_cls == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, 
                                     subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
                                     reg_alpha=0.5, reg_lambda=2.0,
                                     random_state=42, n_jobs=-1, Missing=np.nan)
            
            model.fit(X_tr, y_tr)
            oof_spatial[va_idx] = model.predict(X_va)
            
        # 2. Random K-Fold (Measures temporal and in-location variation)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_random = np.zeros(len(df))
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            
            if use_imputer:
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_va = imputer.transform(X_va)
                
            if model_cls == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, 
                                     subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
                                     reg_alpha=0.5, reg_lambda=2.0,
                                     random_state=42, n_jobs=-1, Missing=np.nan)
            
            model.fit(X_tr, y_tr)
            oof_random[va_idx] = model.predict(X_va)
            
        spatial_r2 = max(0, r2_score(y, oof_spatial)) # Cap at 0 mapping to official LB behavior
        random_r2 = max(0, r2_score(y, oof_random))
        
        spatial_r2_list.append(spatial_r2)
        random_r2_list.append(random_r2)
        
        print(f"\n[{target}]")
        print(f"  -> Spatial CV R2 : {spatial_r2:.4f}")
        print(f"  -> Random CV R2  : {random_r2:.4f}")
        
    avg_spatial = np.mean(spatial_r2_list)
    avg_random = np.mean(random_r2_list)
    # Phase M Calibration: LB heavily penalizes spatial extrapolation.
    # Reverse-engineered from 3 confirmed LB scores: weight ≈ 0.8 spatial + 0.2 random
    estimated_lb = (avg_spatial * 0.8) + (avg_random * 0.2)
    
    print("-" * 60)
    print(f"Overall Spatial CV : {avg_spatial:.4f}")
    print(f"Overall Random CV  : {avg_random:.4f}")
    print(f"ESTIMATED LB SCORE : {estimated_lb:.4f}")
    print("="*60 + "\n")

    return estimated_lb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EY Challenge Local Evaluator")
    parser.add_argument('--model', type=str, choices=['benchmark', 'optimized', 'all'], default='all', 
                        help='Which model configuration to evaluate.')
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_and_merge_data()
    
    if args.model in ['benchmark', 'all']:
        evaluate_model(
            df, 
            name="1. Original Benchmark (RandomForest, Median Fill, 4 old feats)",
            feature_cols=['swir22_old', 'NDMI', 'MNDWI', 'pet'],
            model_cls='RF',
            use_imputer=True
        )
        
    if args.model in ['optimized', 'all']:
        evaluate_model(
            df, 
            name="2. Phase M (XGBoost, 21 Features, Tightened Regularization)",
            feature_cols=['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 
                          'pet', 'Latitude', 'Longitude', 'month_sin', 'month_cos', 
                          'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI',
                          'NDTI', 'FAI', 'CDOM', 'Turbidity', 'BSI'],
            model_cls='XGB',
            use_imputer=False
        )

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# We can re-use the exact preprocessing from ensemble_model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_model import load_and_preprocess_training, TARGET_COLS

N_SPLITS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Optuna Objective Functions
# ---------------------------------------------------------------------------
def optimize_xgb(trial, X, y, groups, use_log1p):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    cv_rmse = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_va)
        if use_log1p:
            y_va_real = np.expm1(y_va)
            preds_real = np.expm1(preds)
            rmse = np.sqrt(mean_squared_error(y_va_real, preds_real))
        else:
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            
        cv_rmse.append(rmse)
        
    return np.mean(cv_rmse)

def optimize_lgb(trial, X, y, groups, use_log1p):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    cv_rmse = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_va)
        if use_log1p:
            y_va_real = np.expm1(y_va)
            preds_real = np.expm1(preds)
            rmse = np.sqrt(mean_squared_error(y_va_real, preds_real))
        else:
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            
        cv_rmse.append(rmse)
        
    return np.mean(cv_rmse)

def optimize_cat(trial, X, y, groups, use_log1p):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'allow_writing_files': False
    }
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    cv_rmse = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_va)
        if use_log1p:
            y_va_real = np.expm1(y_va)
            preds_real = np.expm1(preds)
            rmse = np.sqrt(mean_squared_error(y_va_real, preds_real))
        else:
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            
        cv_rmse.append(rmse)
        
    return np.mean(cv_rmse)

def run_hpo_pipeline():
    print("=" * 70)
    print("  EY Challenge 2026 — Optuna Hyperparameter Optimization")
    print("=" * 70)
    
    train_df, features = load_and_preprocess_training()
    print(f"[1/3] Features loaded via ensemble pipeline ({len(features)} total).")
    
    best_params_dict = {}
    N_TRIALS = 30 # Set to 30 per model per target for reasonable time. Can increase for final sweeps.
    
    for target in TARGET_COLS:
        print(f"\n[2/3] Automating HPO for Target: {target}")
        
        X = train_df[features].copy()
        
        # Phase H: Target Transformation Reverted (Optimizing raw MSE, no Log1p!)
        use_log1p = False
        y = train_df[target].copy()
            
        groups = train_df['spatial_group']
        best_params_dict[target] = {}
        
        # XGBoost
        print("  -> Tuning XGBoost...")
        study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_xgb.optimize(lambda trial: optimize_xgb(trial, X, y, groups, use_log1p), n_trials=N_TRIALS)
        print(f"     Best RMSE: {study_xgb.best_value:.4f}")
        best_params_dict[target]['xgb'] = study_xgb.best_params
        
        # LightGBM
        print("  -> Tuning LightGBM...")
        study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_lgb.optimize(lambda trial: optimize_lgb(trial, X, y, groups, use_log1p), n_trials=N_TRIALS)
        print(f"     Best RMSE: {study_lgb.best_value:.4f}")
        best_params_dict[target]['lgb'] = study_lgb.best_params
        
        # CatBoost
        print("  -> Tuning CatBoost...")
        study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_cat.optimize(lambda trial: optimize_cat(trial, X, y, groups, use_log1p), n_trials=N_TRIALS)
        print(f"     Best RMSE: {study_cat.best_value:.4f}")
        best_params_dict[target]['cat'] = study_cat.best_params
        
    print("\n[3/3] Optuna Search Complete. Writing optimal parameters to disk...")
    # Save optimized params so ensemble_model can read them
    param_path = os.path.join(OUTPUT_DIR, 'best_optuna_params.joblib')
    joblib.dump(best_params_dict, param_path)
    print(f"Saved optimized hyperparameters to {param_path}")

if __name__ == '__main__':
    run_hpo_pipeline()

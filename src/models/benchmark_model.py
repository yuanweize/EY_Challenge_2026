#!/usr/bin/env python3
"""
EY Challenge 2026 — Benchmark Model Pipeline
Refactored from official Benchmark_Model_Notebook.ipynb into a clean, 
modular Python script for local development.

Usage:
    source .venv/bin/activate
    python3 src/benchmark_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'code', 'general')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output')

FEATURE_COLS = ['swir22', 'NDMI', 'MNDWI', 'pet']   # Benchmark uses only these 4
TARGET_COLS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus',
]
MERGE_KEYS = ['Latitude', 'Longitude', 'Sample Date']

TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ESTIMATORS = 100


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_training_data():
    """Load and merge the three training CSV files."""
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    ls = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))

    # Horizontal merge (concat on columns, drop duplicate cols)
    data = pd.concat([wq, ls, tc], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def load_validation_data():
    """Load validation features and submission template."""
    sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    ls_val = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
    tc_val = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))

    val_data = pd.DataFrame({
        'Longitude': ls_val['Longitude'].values,
        'Latitude': ls_val['Latitude'].values,
        'Sample Date': ls_val['Sample Date'].values,
        'nir': ls_val['nir'].values,
        'green': ls_val['green'].values,
        'swir16': ls_val['swir16'].values,
        'swir22': ls_val['swir22'].values,
        'NDMI': ls_val['NDMI'].values,
        'MNDWI': ls_val['MNDWI'].values,
        'pet': tc_val['pet'].values,
    })
    return sub, val_data


def impute_missing(df):
    """Fill NaN with column median (benchmark strategy)."""
    return df.fillna(df.median(numeric_only=True))


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
def train_single_target(X_train, y_train, X_test, y_test, target_name):
    """Train a RandomForest model for one target variable.
    
    Returns: (model, scaler, results_dict)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)

    # Evaluate
    r2_train = r2_score(y_train, model.predict(X_train_s))
    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train_s)))
    r2_test = r2_score(y_test, model.predict(X_test_s))
    rmse_test = np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))

    results = {
        'Parameter': target_name,
        'R2_Train': round(r2_train, 4),
        'RMSE_Train': round(rmse_train, 4),
        'R2_Test': round(r2_test, 4),
        'RMSE_Test': round(rmse_test, 4),
    }
    return model, scaler, results


def run_benchmark():
    """Execute the full benchmark pipeline and generate submission."""
    print("=" * 60)
    print("  EY Challenge 2026 — Benchmark Model Pipeline")
    print("=" * 60)

    # ---- Load & preprocess training data ----
    print("\n[1/5] Loading training data...")
    data = load_training_data()
    data = impute_missing(data)
    print(f"      Training data shape: {data.shape}")

    X = data[FEATURE_COLS]
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, data.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ---- Train models for each target ----
    print("\n[2/5] Training models...")
    models = {}
    scalers = {}
    all_results = []

    for target in TARGET_COLS:
        y = data[target]
        y_train = y.loc[idx_train]
        y_test = y.loc[idx_test]

        model, scaler, results = train_single_target(
            X_train, y_train, X_test, y_test, target
        )
        models[target] = model
        scalers[target] = scaler
        all_results.append(results)

        print(f"      {target}: R²_test={results['R2_Test']:.4f}, RMSE_test={results['RMSE_Test']:.4f}")

    # ---- Results summary ----
    print("\n[3/5] Results Summary:")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # ---- Predict on validation set ----
    print("\n[4/5] Generating predictions on validation set...")
    sub_template, val_data = load_validation_data()
    val_data = impute_missing(val_data)
    X_val = val_data[FEATURE_COLS]

    submission = sub_template[['Latitude', 'Longitude', 'Sample Date']].copy()
    for target in TARGET_COLS:
        X_val_s = scalers[target].transform(X_val)
        submission[target] = models[target].predict(X_val_s)

    # ---- Save submission ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'submission_benchmark.csv')
    submission.to_csv(out_path, index=False)
    print(f"\n[5/5] Submission saved to: {out_path}")
    print(f"      Shape: {submission.shape}")
    print(f"\nFirst 5 predictions:")
    print(submission.head().to_string(index=False))

    return models, scalers, results_df, submission


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    models, scalers, results_df, submission = run_benchmark()

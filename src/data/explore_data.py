#!/usr/bin/env python3
"""
EY Challenge 2026 — Data Exploration Script
Analyzes training datasets: water quality, landsat features, terraclimate features.
Outputs statistics, missing values, correlations, and distribution info.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'code', 'general')

def load_datasets():
    """Load all training CSV files."""
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    ls = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
    val_ls = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_validation.csv'))
    val_tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_validation.csv'))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_template.csv'))
    return wq, ls, tc, val_ls, val_tc, sub

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def analyze_dataset(name, df):
    """Print comprehensive analysis of a dataframe."""
    section(f"{name} — Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Pct(%)': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0]
    if len(missing_df) > 0:
        print(f"\nMissing Values:")
        print(missing_df)
    else:
        print(f"\nNo missing values!")

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nDescriptive Statistics (numeric):")
        print(df[numeric_cols].describe().round(4))

def main():
    wq, ls, tc, val_ls, val_tc, sub = load_datasets()

    # ----- Water Quality Training -----
    analyze_dataset("Water Quality Training", wq)

    # Date range
    wq['Sample Date'] = pd.to_datetime(wq['Sample Date'], format='%d-%m-%Y')
    print(f"\nDate Range: {wq['Sample Date'].min()} to {wq['Sample Date'].max()}")
    print(f"Unique Locations: {wq.groupby(['Latitude','Longitude']).ngroups}")
    
    # Target variable correlations
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    print(f"\nTarget Variable Correlations:")
    print(wq[targets].corr().round(4))

    # ----- Landsat Features Training -----
    analyze_dataset("Landsat Features Training", ls)

    # ----- TerraClimate Features Training -----
    analyze_dataset("TerraClimate Features Training", tc)

    # ----- Validation Sets -----
    section("Validation Sets Overview")
    print(f"Landsat Validation: {val_ls.shape}")
    print(f"TerraClimate Validation: {val_tc.shape}")
    print(f"Submission Template: {sub.shape}")

    # ----- Merged Dataset Check -----
    section("Merge Feasibility Check")
    # Check if all three datasets can be merged on (Lat, Lon, Date)
    merge_keys = ['Latitude', 'Longitude', 'Sample Date']
    merged = wq.copy()
    merged['Sample Date'] = merged['Sample Date'].dt.strftime('%d-%m-%Y')
    
    merged_ls = merged.merge(ls, on=merge_keys, how='left', suffixes=('', '_ls'))
    print(f"After merging WQ + Landsat: {merged_ls.shape}")
    ls_missing = merged_ls[ls.columns.difference(merge_keys)].isnull().any(axis=1).sum()
    print(f"  Rows with ANY missing Landsat feature: {ls_missing} / {len(merged_ls)} ({ls_missing/len(merged_ls)*100:.1f}%)")

    merged_all = merged_ls.merge(tc, on=merge_keys, how='left', suffixes=('', '_tc'))
    print(f"After merging WQ + Landsat + TerraClimate: {merged_all.shape}")
    tc_cols_only = [c for c in tc.columns if c not in merge_keys]
    tc_missing = merged_all[tc_cols_only].isnull().any(axis=1).sum()
    print(f"  Rows with ANY missing TerraClimate feature: {tc_missing} / {len(merged_all)} ({tc_missing/len(merged_all)*100:.1f}%)")

    # Feature count summary
    section("Feature Summary for Modeling")
    feature_cols = [c for c in merged_all.columns if c not in merge_keys + targets]
    print(f"Total features after merge: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    print(f"Total training samples: {len(merged_all)}")
    print(f"Samples with complete features (no NaN): {merged_all.dropna().shape[0]}")

if __name__ == '__main__':
    main()

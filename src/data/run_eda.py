import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
EDA_OUT_DIR = os.path.join(PROJECT_ROOT, 'eda_plots')
CLEAN_DATA_OUT = os.path.join(PROJECT_ROOT, 'data', 'merged_training_data_clean.csv')

os.makedirs(EDA_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_DATA_OUT), exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
CLIMATE_FEATURES = ['pr', 'tmax', 'tmin', 'ro'] # TerraClimate
SATELLITE_FEATURES = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']

def load_and_merge_data():
    print("Loading constituent datasets...")
    # 1. Main Water Quality Labels
    wq_df = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    
    # 2. Satellite Features (Landsat)
    landsat_api_path = os.path.join(PROCESSED_DATA_DIR, 'landsat_api_training.csv')
    if os.path.exists(landsat_api_path):
        ls_df = pd.read_csv(landsat_api_path)
        # Using Index if available
        if 'Index' in ls_df.columns:
            ls_df = ls_df.set_index('Index')
        wq_df = wq_df.join(ls_df, how='inner')
    else:
        # Fallback to local baseline if api not generated yet
        ls_df = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
        wq_df = wq_df.merge(ls_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
        
    # 3. Climate Features (TerraClimate)
    tc_api_path = os.path.join(PROCESSED_DATA_DIR, 'terraclimate_api_training.csv')
    if os.path.exists(tc_api_path):
        tc_df = pd.read_csv(tc_api_path)
        if 'Index' in tc_df.columns:
            tc_df = tc_df.set_index('Index')
            
        # Select only needed columns to avoid duplicates
        cols_to_use = [c for c in tc_df.columns if c in CLIMATE_FEATURES or c not in wq_df.columns]
        wq_df = wq_df.join(tc_df[cols_to_use], how='inner')
    else:
        # Fallback to local baseline if api not generated yet
        tc_df = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))
        wq_df = wq_df.merge(tc_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')

    print(f"Merged Data Shape: {wq_df.shape}")
    return wq_df

def clean_data(df):
    print("Cleaning Data...")
    df = df.copy()
    
    # 1. Date formatting
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    
    # 2. Type casting to float64
    all_features = SATELLITE_FEATURES + CLIMATE_FEATURES
    for col in TARGETS + all_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
    # 3. Missing Value Imputation (Median)
    print("Checking NaNs:")
    for col in all_features:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f" - {col}: {missing} missing values. Imputing with median.")
                df[col] = df[col].fillna(df[col].median())
                
    return df

def generate_eda_plots(df):
    print(f"Generating EDA Plots to {EDA_OUT_DIR}...")
    
    # 1. Target Histograms
    plt.figure(figsize=(15, 5))
    for i, target in enumerate(TARGETS, 1):
        plt.subplot(1, 3, i)
        # Use log scale for x-axis if heavily skewed like DRP
        if target == 'Dissolved Reactive Phosphorus':
            sns.histplot(df[target], bins=50, kde=True, log_scale=(True, False))
            plt.title(f"{target} (Log Scale X)")
        else:
            sns.histplot(df[target], bins=50, kde=True)
            plt.title(target)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUT_DIR, 'targets_histogram.png'))
    plt.close()
    
    # 2. Target Boxplots
    plt.figure(figsize=(15, 5))
    for i, target in enumerate(TARGETS, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[target])
        if target == 'Dissolved Reactive Phosphorus':
            plt.yscale('log')
            plt.title(f"{target} (Log Scale Y)")
        else:
            plt.title(target)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUT_DIR, 'targets_boxplot.png'))
    plt.close()
    
    # 3. Climate Features Distribution
    available_climates = [c for c in CLIMATE_FEATURES if c in df.columns]
    if available_climates:
        plt.figure(figsize=(15, 5))
        for i, feat in enumerate(available_climates, 1):
            plt.subplot(1, len(available_climates), i)
            sns.histplot(df[feat], bins=30, kde=True)
            plt.title(f"Climate: {feat}")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUT_DIR, 'climate_features_histogram.png'))
        plt.close()

def main():
    print("=== Start Data Prep & EDA Pipeline ===")
    
    # Load and merge 
    df = load_and_merge_data()
    
    # Clean and impute
    df_clean = clean_data(df)
    
    # Export clean wide table
    df_clean.to_csv(CLEAN_DATA_OUT, index=False)
    print(f"Cleaned dataset exported to: {CLEAN_DATA_OUT}")
    
    # Generate Plots
    generate_eda_plots(df_clean)
    
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

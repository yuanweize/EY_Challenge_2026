#!/usr/bin/env python3
"""Batch extract TerraClimate features using monthly raster loading."""
import socket; socket.setdefaulttimeout(60)
import os, warnings
import numpy as np
import pandas as pd
import xarray as xr
import planetary_computer

warnings.filterwarnings("ignore")

PROJECT_ROOT = '/Users/yuanweize/我的文档/服务器/GITHUB/EY_Challenge_2026'
DATA_DIR = os.path.join(PROJECT_ROOT, 'resources', 'code', 'general')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

CLIMATE_VARS = ['ppt', 'tmax', 'tmin', 'q']
VAR_NAMES = {'ppt': 'pr', 'tmax': 'tmax', 'tmin': 'tmin', 'q': 'ro'}

def open_terraclimate():
    account = 'cpdataeuwest'
    container = 'cpdata'
    token = planetary_computer.sas.get_token(account, container)
    storage_options = {'account_name': account, 'sas_token': token.token}
    ds = xr.open_zarr(f'az://{container}/terraclimate.zarr',
                      storage_options=storage_options, consolidated=True)
    return ds

def extract_batch(ds, train_csv, val_csv, train_out, val_out):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    train_dates = pd.to_datetime(train_df['Sample Date'], dayfirst=True)
    val_dates = pd.to_datetime(val_df['Sample Date'], dayfirst=True)
    
    train_df['_month_key'] = train_dates.dt.to_period('M').astype(str)
    val_df['_month_key'] = val_dates.dt.to_period('M').astype(str)
    
    all_months = sorted(set(train_df['_month_key'].tolist() + val_df['_month_key'].tolist()))
    print(f"Need to load {len(all_months)} unique months")
    
    all_lats = np.concatenate([train_df['Latitude'].values, val_df['Latitude'].values])
    all_lons = np.concatenate([train_df['Longitude'].values, val_df['Longitude'].values])
    lat_min, lat_max = all_lats.min() - 1, all_lats.max() + 1
    lon_min, lon_max = all_lons.min() - 1, all_lons.max() + 1
    print(f"Bounding box: lat=[{lat_min:.1f}, {lat_max:.1f}], lon=[{lon_min:.1f}, {lon_max:.1f}]")
    
    for out_name in VAR_NAMES.values():
        train_df[out_name] = np.nan
        val_df[out_name] = np.nan
    
    for i, month_str in enumerate(all_months):
        print(f"[{i+1}/{len(all_months)}] Loading {month_str}...", end=' ', flush=True)
        t = pd.Timestamp(month_str + '-01')
        
        try:
            for var in CLIMATE_VARS:
                out_name = VAR_NAMES[var]
                raster = ds[var].sel(
                    time=t, method='nearest'
                ).sel(
                    lat=slice(lat_max, lat_min),
                    lon=slice(lon_min, lon_max)
                ).compute()
                
                mask = train_df['_month_key'] == month_str
                if mask.any():
                    lats = train_df.loc[mask, 'Latitude'].values
                    lons = train_df.loc[mask, 'Longitude'].values
                    vals = raster.sel(
                        lat=xr.DataArray(lats, dims='points'),
                        lon=xr.DataArray(lons, dims='points'),
                        method='nearest'
                    ).values
                    train_df.loc[mask, out_name] = vals
                
                mask_v = val_df['_month_key'] == month_str
                if mask_v.any():
                    lats = val_df.loc[mask_v, 'Latitude'].values
                    lons = val_df.loc[mask_v, 'Longitude'].values
                    vals = raster.sel(
                        lat=xr.DataArray(lats, dims='points'),
                        lon=xr.DataArray(lons, dims='points'),
                        method='nearest'
                    ).values
                    val_df.loc[mask_v, out_name] = vals
                    
            n_train = (train_df['_month_key'] == month_str).sum()
            n_val = (val_df['_month_key'] == month_str).sum()
            print(f"OK (train={n_train}, val={n_val})")
        except Exception as e:
            print(f"FAILED: {e}")
    
    out_cols = ['Latitude', 'Longitude', 'Sample Date'] + list(VAR_NAMES.values())
    train_df[out_cols].to_csv(train_out, index=False)
    val_df[out_cols].to_csv(val_out, index=False)
    
    for name, df in [('Training', train_df), ('Validation', val_df)]:
        valid = df[list(VAR_NAMES.values())].notna().all(axis=1).sum()
        print(f"{name}: {valid}/{len(df)} rows fully extracted")
        for v in VAR_NAMES.values():
            print(f"  {v}: min={df[v].min():.1f}, max={df[v].max():.1f}, NaN={df[v].isna().sum()}")

if __name__ == '__main__':
    print("Opening TerraClimate Zarr store...")
    ds = open_terraclimate()
    
    extract_batch(
        ds,
        train_csv=os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'),
        val_csv=os.path.join(DATA_DIR, 'landsat_features_validation.csv'),
        train_out=os.path.join(OUT_DIR, 'terraclimate_api_training.csv'),
        val_out=os.path.join(OUT_DIR, 'terraclimate_api_validation.csv')
    )
    print("\nDone!")

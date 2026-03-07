import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

import pystac_client
import planetary_computer as pc
import dask
import socket

warnings.filterwarnings("ignore")
# Silence dask warnings
dask.config.set(scheduler='single-threaded')

# CRITICAL FIX for GFW DNS/TCP hang:
# Force all underlying sockets to timeout after 10 seconds.
socket.setdefaulttimeout(10.0)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resources', 'code', 'general')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')

def get_unique_months(df):
    dates = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    return dates.dt.strftime('%Y-%m').dropna().unique()

def build_item_cache(unique_months):
    print(f"Building Planetary Computer STAC cache for {len(unique_months)} unique months...")
    cache = {}
    
    for ym in tqdm(unique_months):
        date_start = f"{ym}-01"
        date_end = pd.to_datetime(date_start) + pd.Timedelta(days=31)
        date_end_str = date_end.strftime('%Y-%m-01')
        
        success = False
        for attempt in range(5):
            try:
                # Open with strict 5 second timeout to avoid TLS hang
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=pc.sign_inplace,
                    timeout=5.0
                )
                
                search = catalog.search(
                    collections=["terraclimate"],
                    datetime=f"{date_start}/{date_end_str}"
                )
                items = list(search.items())
                if items:
                    cache[ym] = items[0]
                    success = True
                    break
                else:
                    search = catalog.search(
                        collections=["terraclimate"],
                        datetime=f"{int(ym[:4])-1}-01-01/{ym}-28"
                    )
                    items = list(search.items())
                    if items:
                        cache[ym] = items[0]
                        success = True
                        break
            except Exception as e:
                time.sleep(1) # Backoff for unstable connection
                
    return cache

def process_row(args):
    idx, lat, lon, date_str, item = args
    if not item:
        return idx, {}
        
    date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    bbox_size = 0.04
    bbox = [lon - bbox_size / 2, lat - bbox_size / 2, lon + bbox_size / 2, lat + bbox_size / 2]
    
    bands_of_interest = ["pr", "tmax", "tmin", "ro"]
    try:
        data = stac_load([item], bands=bands_of_interest, bbox=bbox).isel(time=0)
        
        res = {}
        for b in bands_of_interest:
            band_vals = data[b].values
            valid_vals = band_vals[~np.isnan(band_vals)]
            if len(valid_vals) > 0:
                res[b] = np.median(valid_vals)
            else:
                res[b] = np.nan
                
        sample_date_utc = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
        res['terraclimate_date'] = item.properties["datetime"]
        res['tc_days_offset'] = abs((pd.to_datetime(item.properties["datetime"]).tz_convert("UTC") - sample_date_utc).days)
        return idx, res
    except Exception:
        return idx, {}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = [
        ('water_quality_training_dataset.csv', 'terraclimate_api_training.csv', 'landsat_features_training.csv'),
        ('submission_template.csv', 'terraclimate_api_validation.csv', 'landsat_features_validation.csv')
    ]
    
    for _, out_name, val_source in datasets:
        print(f"\n--- Processing TerraClimate for {out_name} ---")
        out_file = os.path.join(OUTPUT_DIR, out_name)
        
        source_file = os.path.join(DATA_DIR, val_source)
        df = pd.read_csv(source_file)
        
        if os.path.exists(out_file):
            processed_df = pd.read_csv(out_file)
            processed_indices = set(processed_df['Index'].values)
        else:
            processed_indices = set()
            
        tasks = []
        unique_months = get_unique_months(df)
        item_cache = build_item_cache(unique_months)
        
        for i, row in df.iterrows():
            if i not in processed_indices:
                date_str = row['Sample Date']
                ym = pd.to_datetime(date_str, dayfirst=True, errors='coerce').strftime('%Y-%m')
                item = item_cache.get(ym)
                tasks.append((i, row['Latitude'], row['Longitude'], date_str, item))
                
        if not tasks:
            print(f"All rows already processed for {out_name}!")
            continue

        print(f"Total rows remaining to process: {len(tasks)}")
        
        batch_results = []
        batch_size = 500
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(process_row, t): t for t in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                idx, res = future.result()
                res['Index'] = idx
                batch_results.append(res)
                
                if len(batch_results) >= batch_size:
                    res_df = pd.DataFrame(batch_results).set_index('Index')
                    write_header = not os.path.exists(out_file)
                    res_df.to_csv(out_file, mode='a', header=write_header)
                    batch_results = []
                    
        if batch_results:
            res_df = pd.DataFrame(batch_results).set_index('Index')
            write_header = not os.path.exists(out_file)
            res_df.to_csv(out_file, mode='a', header=write_header)
            
    print("Done!")

if __name__ == '__main__':
    main()

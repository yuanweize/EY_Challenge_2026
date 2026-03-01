import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import traceback

import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'resources', 'code', 'general')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')

def process_row(args):
    idx, lat, lon, date_str, retries = args
    date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    
    # Basic bounding box (~100m)
    bbox_size = 0.00089831
    bbox = [lon - bbox_size / 2, lat - bbox_size / 2, lon + bbox_size / 2, lat + bbox_size / 2]
    
    # Try within +/- 30 days first
    window_days = 30
    date_start = (date - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    date_end = (date + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    
    for attempt in range(retries):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )
            
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=bbox,
                datetime=f"{date_start}/{date_end}"
            )
            items = list(search.items())
            break
        except Exception as e:
            if attempt == retries - 1:
                return idx, {}
            time.sleep(2 ** attempt)  # Exponential backoff
            
    if not items:
        # Fallback to +/- 60 days
        window_days = 60
        date_start = (date - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
        date_end = (date + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
        try:
            search = catalog.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=f"{date_start}/{date_end}")
            items = list(search.items())
        except:
            pass

    if not items:
        return idx, {}
        
    sample_date_utc = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
    items = sorted(items, key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc))
    
    bands_of_interest = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
    
    # Check top 3 closest images for clear pixels
    for item in items[:3]:
        try:
            signed_item = pc.sign(item)
            data = stac_load([signed_item], bands=bands_of_interest, bbox=bbox).isel(time=0)
            qa = data["qa_pixel"].values
            
            # Landsat Collection 2 QA_PIXEL bitmask for Clouds:
            # Bit 1 (2): Dilated Cloud
            # Bit 3 (8): Cloud
            # Bit 4 (16): Cloud Shadow
            bad_mask = np.bitwise_and(qa, 2) | np.bitwise_and(qa, 8) | np.bitwise_and(qa, 16)
            good_mask = (bad_mask == 0)
            
            if np.sum(good_mask) == 0:
                continue # Entire bbox is cloudy
                
            res = {}
            for b in bands_of_interest:
                if b == "qa_pixel": continue
                band_vals = data[b].values
                valid_vals = band_vals[good_mask & (band_vals > 0) & (band_vals < 65535)]
                if len(valid_vals) > 0:
                    res[b] = np.median(valid_vals)
                else:
                    res[b] = np.nan
                    
            # Check if majority of bands are present
            if pd.isna(res.get('green')):
                continue
                
            res['image_date'] = item.properties["datetime"]
            res['days_offset'] = abs((pd.to_datetime(item.properties["datetime"]).tz_convert("UTC") - sample_date_utc).days)
            return idx, res
            
        except Exception as e:
            continue
            
    return idx, {}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = [
        ('water_quality_training_dataset.csv', 'landsat_api_training.csv', 'landsat_features_training.csv'),
        ('submission_template.csv', 'landsat_api_validation.csv', 'landsat_features_validation.csv')
    ]
    
    for _, out_name, val_source in datasets:
        print(f"\n--- Processing for {out_name} ---")
        out_file = os.path.join(OUTPUT_DIR, out_name)
        
        # Load dataset
        # Wait, submission_template.csv only has IDs? No, landsat_features_validation.csv has the coords.
        # We can just read the coords from landsat_features_*.csv directly since they have Latitude, Longitude, Sample Date
        source_file = os.path.join(DATA_DIR, val_source)
        df = pd.read_csv(source_file)
        
        # Read already processed if resuming
        if os.path.exists(out_file):
            processed_df = pd.read_csv(out_file)
            processed_indices = set(processed_df['Index'].values)
        else:
            processed_indices = set()
            
        tasks = []
        for i, row in df.iterrows():
            if i not in processed_indices:
                tasks.append((i, row['Latitude'], row['Longitude'], row['Sample Date'], 3))
                
        if not tasks:
            print(f"All rows already processed for {out_name}!")
            continue

        print(f"Total rows remaining to process: {len(tasks)}")
        
        TEST_LIMIT = 20
        is_test_run = '--test' in sys.argv
        if is_test_run:
            print(f"TEST RUN: Only processing first {TEST_LIMIT} rows")
            tasks = tasks[:TEST_LIMIT]
        
        batch_results = []
        batch_size = 50
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
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

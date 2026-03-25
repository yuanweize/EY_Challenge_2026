import concurrent.futures
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
from odc.stac import stac_load
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "code", "general")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
SCENE_STATS = ["median", "p75", "std"]
DERIVED_STATS = ["NDVI", "NDWI", "MNDWI", "Turbidity", "clear_fraction", "days_offset"]
SUMMARY_STATS = ["mean", "median", "max"]
WINDOWS = [15, 30, 60]
BBOX_SCALES = {
    "fine": 0.0009,
    "wide": 0.0030,
}
MAX_SCENES_PER_WINDOW = 4


def expected_columns():
    cols = ["Index"]
    scene_keys = []
    for band in BANDS:
        if band == "qa_pixel":
            continue
        for stat in SCENE_STATS:
            scene_keys.append(f"{band}_{stat}")
    scene_keys.extend(DERIVED_STATS)

    for scale_name in BBOX_SCALES:
        for window in WINDOWS:
            prefix = f"{scale_name}_w{window}"
            cols.append(f"{prefix}_scene_count")
            for key in scene_keys:
                for summary in SUMMARY_STATS:
                    cols.append(f"{prefix}_{key}_{summary}")
    return cols


EXPECTED_COLUMNS = expected_columns()


def build_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )


def compute_scene_features(data, good_mask):
    row = {}
    for band in BANDS:
        if band == "qa_pixel":
            continue
        values = data[band].values
        valid = values[good_mask & (values > 0) & (values < 65535)]
        row[f"{band}_median"] = float(np.median(valid)) if len(valid) else np.nan
        row[f"{band}_p75"] = float(np.percentile(valid, 75)) if len(valid) else np.nan
        row[f"{band}_std"] = float(np.std(valid)) if len(valid) else np.nan

    row["NDVI"] = (row["nir08_median"] - row["red_median"]) / (row["nir08_median"] + row["red_median"] + 1e-8)
    row["NDWI"] = (row["green_median"] - row["nir08_median"]) / (row["green_median"] + row["nir08_median"] + 1e-8)
    row["MNDWI"] = (row["green_median"] - row["swir16_median"]) / (row["green_median"] + row["swir16_median"] + 1e-8)
    row["Turbidity"] = row["red_median"] / (row["blue_median"] + 1e-8)
    row["clear_fraction"] = float(np.mean(good_mask))
    return row


def summarize_rows(rows, prefix):
    result = {col: np.nan for col in EXPECTED_COLUMNS if col.startswith(prefix)}
    result[f"{prefix}_scene_count"] = len(rows)
    if not rows:
        return result

    df = pd.DataFrame(rows)
    for col in df.columns:
        result[f"{prefix}_{col}_mean"] = float(df[col].mean())
        result[f"{prefix}_{col}_median"] = float(df[col].median())
        result[f"{prefix}_{col}_max"] = float(df[col].max())
    return result


def fetch_item_features(item, bbox):
    signed_item = pc.sign(item)
    data = stac_load([signed_item], bands=BANDS, bbox=bbox).isel(time=0)
    qa = data["qa_pixel"].values
    bad_mask = np.bitwise_and(qa, 2) | np.bitwise_and(qa, 8) | np.bitwise_and(qa, 16)
    good_mask = bad_mask == 0
    if np.sum(good_mask) == 0:
        return None
    return compute_scene_features(data, good_mask)


def normalize_record(record):
    out = {col: np.nan for col in EXPECTED_COLUMNS}
    out.update(record)
    return out


def process_row(args):
    idx, lat, lon, date_str = args
    try:
        date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        sample_date_utc = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
        catalog = build_catalog()
        output = {"Index": idx}

        for scale_name, bbox_size in BBOX_SCALES.items():
            bbox = [lon - bbox_size / 2, lat - bbox_size / 2, lon + bbox_size / 2, lat + bbox_size / 2]
            for window in WINDOWS:
                date_start = (date - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
                date_end = (date + pd.Timedelta(days=window)).strftime("%Y-%m-%d")
                search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=bbox,
                    datetime=f"{date_start}/{date_end}",
                )
                items = list(search.items())
                items = sorted(
                    items,
                    key=lambda x: abs(pd.to_datetime(x.properties["datetime"]).tz_convert("UTC") - sample_date_utc),
                )

                rows = []
                for item in items[:MAX_SCENES_PER_WINDOW]:
                    try:
                        features = fetch_item_features(item, bbox)
                        if features is None:
                            continue
                        features["days_offset"] = abs(
                            (pd.to_datetime(item.properties["datetime"]).tz_convert("UTC") - sample_date_utc).days
                        )
                        rows.append(features)
                    except Exception:
                        continue

                prefix = f"{scale_name}_w{window}"
                output.update(summarize_rows(rows, prefix))

        return normalize_record(output)
    except Exception:
        return normalize_record({"Index": idx})


def process_dataset(source_name, output_name):
    source_file = os.path.join(DATA_DIR, source_name)
    out_file = os.path.join(OUTPUT_DIR, output_name)
    df = pd.read_csv(source_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    overwrite = "--overwrite" in sys.argv or "--test" in sys.argv
    if overwrite and os.path.exists(out_file):
        os.remove(out_file)

    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        if list(existing.columns) != EXPECTED_COLUMNS:
            os.remove(out_file)
            processed = set()
        else:
            processed = set(existing["Index"].tolist())
    else:
        processed = set()

    tasks = [
        (i, row["Latitude"], row["Longitude"], row["Sample Date"])
        for i, row in df.iterrows()
        if i not in processed
    ]

    if "--test" in sys.argv:
        tasks = tasks[:10]

    if not tasks:
        print("No rows left for", output_name)
        return

    batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_row, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            batch.append(future.result())
            if len(batch) >= 25:
                pd.DataFrame(batch, columns=EXPECTED_COLUMNS).to_csv(
                    out_file,
                    mode="a",
                    header=not os.path.exists(out_file),
                    index=False,
                )
                batch = []

    if batch:
        pd.DataFrame(batch, columns=EXPECTED_COLUMNS).to_csv(
            out_file,
            mode="a",
            header=not os.path.exists(out_file),
            index=False,
        )


def main():
    datasets = [
        ("landsat_features_training.csv", "landsat_context_training.csv"),
        ("landsat_features_validation.csv", "landsat_context_validation.csv"),
    ]
    start = time.time()
    for source_name, output_name in datasets:
        print("Processing", output_name)
        process_dataset(source_name, output_name)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

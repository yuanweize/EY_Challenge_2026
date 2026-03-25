import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.advanced_stacking_pipeline import (
    MERGE_KEYS,
    add_features,
    fit_target_chain,
    load_frames,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_advanced_target_mix")

COMMON_ALL = [
    "blue", "green", "red", "nir08", "swir16", "swir22",
    "pet", "ppt", "tmax", "tmin", "q",
    "nir", "green_old", "swir16_old", "swir22_old", "NDMI", "MNDWI",
    "month_sin", "month_cos", "quarter",
    "days_offset_filled", "days_offset_missing",
    "api_optical_complete", "old_optical_available", "api_preferred",
    "NDVI_new", "NDWI", "MNDWI_new", "SABI", "WRI", "NDTI",
    "turbidity_ratio", "green_red_ratio", "nir_swir_ratio", "swir_ratio",
    "ppt_pet_ratio", "temp_range", "runoff_ratio", "hydro_stress", "thermal_runoff",
    "pet_is_missing", "ppt_is_missing", "tmax_is_missing", "tmin_is_missing", "q_is_missing",
]
OLD_COLS = ["nir", "green_old", "swir16_old", "swir22_old", "NDMI", "MNDWI"]
BASE_FEATS = COMMON_ALL
NO_OLD_FEATS = [col for col in COMMON_ALL if col not in OLD_COLS]


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)

    ta = fit_target_chain(
        train_df,
        val_df,
        "Total Alkalinity",
        BASE_FEATS + ["Latitude", "Longitude"],
        filter_complete=False,
    )
    ta_oof_full = pd.Series(np.nan, index=train_df.index, name="ta_pred")
    ta_oof_full.loc[ta["train_mask"]] = ta["oof_pred"].values
    ta_val = ta["val_pred"].rename("ta_pred")

    ec = fit_target_chain(
        train_df,
        val_df,
        "Electrical Conductance",
        NO_OLD_FEATS,
        extra_train={"ta_pred": ta_oof_full},
        extra_val={"ta_pred": ta_val},
        filter_complete=True,
    )
    ec_oof_full = pd.Series(np.nan, index=train_df.index, name="ec_pred")
    ec_oof_full.loc[ec["train_mask"]] = ec["oof_pred"].values
    ec_val = ec["val_pred"].rename("ec_pred")

    drp = fit_target_chain(
        train_df,
        val_df,
        "Dissolved Reactive Phosphorus",
        BASE_FEATS + ["Latitude"],
        extra_train={"ta_pred": ta_oof_full, "ec_pred": ec_oof_full},
        extra_val={"ta_pred": ta_val, "ec_pred": ec_val},
        filter_complete=True,
    )

    metrics = pd.DataFrame([
        {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"]},
        {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"]},
        {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp["CV_R2"], "CV_RMSE": drp["CV_RMSE"], "rows_used": drp["rows_used"]},
    ])
    metrics.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta["val_pred"].values
    submission["Electrical Conductance"] = ec["val_pred"].values
    submission["Dissolved Reactive Phosphorus"] = drp["val_pred"].values
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_ensemble.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "feature_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("[Total Alkalinity]\n")
        fh.write("\n".join(BASE_FEATS + ["Latitude", "Longitude"]))
        fh.write("\n\n[Electrical Conductance]\n")
        fh.write("\n".join(NO_OLD_FEATS + ["ta_pred"]))
        fh.write("\n\n[Dissolved Reactive Phosphorus]\n")
        fh.write("\n".join(BASE_FEATS + ["Latitude", "ta_pred", "ec_pred"]))
        fh.write("\n")

    print(metrics.to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run()

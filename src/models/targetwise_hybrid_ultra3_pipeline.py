import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluate_local import estimate_public_score
from src.models.advanced_stacking_pipeline import MERGE_KEYS, add_features, fit_target_chain, load_frames
from src.models.targetwise_hybrid_pipeline import COMMON_ALL, NO_OLD, NO_QUALITY, fit_drp_specialist_oof

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_targetwise_hybrid_ultra3")

TA_BLEND_WEIGHT = 0.25
EC_BLEND_WEIGHT = 0.75
DRP_SPECIALIST_WEIGHT = 0.34
DRP_SPECIALIST_THRESHOLD = 144.0


def specialist_features():
    return [
        "blue", "green", "red", "nir08", "swir16", "swir22",
        "pet", "ppt", "tmax", "tmin", "q",
        "days_offset_filled", "days_offset_missing",
        "api_optical_complete", "api_preferred",
        "month_sin", "month_cos",
        "NDVI_new", "NDWI", "MNDWI_new", "SABI", "WRI", "NDTI", "Turbidity",
        "ppt_pet_ratio", "temp_range", "runoff_ratio", "Latitude",
    ]


def blend_series(a, b, weight):
    return (1.0 - weight) * a + weight * b


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    train_df["Turbidity"] = train_df["turbidity_ratio"]
    val_df["Turbidity"] = val_df["turbidity_ratio"]

    ta_no_quality = fit_target_chain(train_df, val_df, "Total Alkalinity", NO_QUALITY + ["Latitude", "Longitude"], filter_complete=False)
    ta_no_old = fit_target_chain(train_df, val_df, "Total Alkalinity", NO_OLD + ["Latitude", "Longitude"], filter_complete=False)
    ta_oof = blend_series(ta_no_quality["oof_pred"].values, ta_no_old["oof_pred"].values, TA_BLEND_WEIGHT)
    ta_val = blend_series(ta_no_quality["val_pred"].values, ta_no_old["val_pred"].values, TA_BLEND_WEIGHT)
    ta_y = train_df["Total Alkalinity"].reset_index(drop=True)
    ta_r2 = float(r2_score(ta_y, ta_oof))
    ta_rmse = float(np.sqrt(mean_squared_error(ta_y, ta_oof)))
    ta_oof_full = pd.Series(ta_oof, index=train_df.index, name="ta_pred")
    ta_val_series = pd.Series(ta_val, name="ta_pred")

    ec_no_quality = fit_target_chain(
        train_df,
        val_df,
        "Electrical Conductance",
        NO_QUALITY,
        extra_train={"ta_pred": ta_oof_full},
        extra_val={"ta_pred": ta_val_series},
        filter_complete=True,
    )
    ec_no_old = fit_target_chain(
        train_df,
        val_df,
        "Electrical Conductance",
        NO_OLD,
        extra_train={"ta_pred": ta_oof_full},
        extra_val={"ta_pred": ta_val_series},
        filter_complete=True,
    )
    ec_oof = blend_series(ec_no_quality["oof_pred"].values, ec_no_old["oof_pred"].values, EC_BLEND_WEIGHT)
    ec_val = blend_series(ec_no_quality["val_pred"].values, ec_no_old["val_pred"].values, EC_BLEND_WEIGHT)
    ec_mask = train_df["api_optical_complete"] == 1
    ec_y = train_df.loc[ec_mask, "Electrical Conductance"].reset_index(drop=True)
    ec_r2 = float(r2_score(ec_y, ec_oof))
    ec_rmse = float(np.sqrt(mean_squared_error(ec_y, ec_oof)))
    ec_oof_full = pd.Series(np.nan, index=train_df.index, name="ec_pred")
    ec_oof_full.loc[ec_mask] = ec_oof
    ec_val_series = pd.Series(ec_val, name="ec_pred")

    drp_chain_base = fit_target_chain(
        train_df,
        val_df,
        "Dissolved Reactive Phosphorus",
        COMMON_ALL + ["Latitude"],
        extra_train={"ta_pred": ta_oof_full, "ec_pred": ec_oof_full},
        extra_val={"ta_pred": ta_val_series, "ec_pred": ec_val_series},
        filter_complete=True,
    )

    drp_specialist = fit_drp_specialist_oof(train_df, val_df, specialist_features(), threshold=DRP_SPECIALIST_THRESHOLD)
    drp_oof = blend_series(drp_chain_base["oof_pred"].values, drp_specialist["oof_pred"].values, DRP_SPECIALIST_WEIGHT)
    drp_val = blend_series(drp_chain_base["val_pred"].values, drp_specialist["val_pred"].values, DRP_SPECIALIST_WEIGHT)

    drp_y = train_df.loc[ec_mask, "Dissolved Reactive Phosphorus"].reset_index(drop=True)
    drp_r2 = float(r2_score(drp_y, drp_oof))
    drp_rmse = float(np.sqrt(mean_squared_error(drp_y, drp_oof)))
    avg_spatial = float(np.mean([ta_r2, ec_r2, drp_r2]))
    target_scores = {
        "Total Alkalinity": ta_r2,
        "Electrical Conductance": ec_r2,
        "Dissolved Reactive Phosphorus": drp_r2,
    }
    estimated_public, _ = estimate_public_score(avg_spatial, 41, target_scores=target_scores)

    metrics = pd.DataFrame(
        [
            {"Parameter": "Total Alkalinity", "CV_R2": ta_r2, "CV_RMSE": ta_rmse, "rows_used": len(train_df), "feature_count": 40},
            {"Parameter": "Electrical Conductance", "CV_R2": ec_r2, "CV_RMSE": ec_rmse, "rows_used": int(ec_mask.sum()), "feature_count": 38},
            {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp_r2, "CV_RMSE": drp_rmse, "rows_used": int(ec_mask.sum()), "feature_count": 46},
        ]
    )
    metrics.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta_val
    submission["Electrical Conductance"] = ec_val
    submission["Dissolved Reactive Phosphorus"] = drp_val
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_ensemble.csv"), index=False)

    mix_config = pd.DataFrame(
        [
            {
                "ta_blend_weight": TA_BLEND_WEIGHT,
                "ec_blend_weight": EC_BLEND_WEIGHT,
                "drp_chain": "base_only",
                "drp_specialist_threshold": DRP_SPECIALIST_THRESHOLD,
                "drp_specialist_weight": DRP_SPECIALIST_WEIGHT,
                "avg_spatial": avg_spatial,
                "estimated_public": estimated_public,
            }
        ]
    )
    mix_config.to_csv(os.path.join(OUTPUT_DIR, "mix_config.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("TA blend = 0.75 * no_quality + 0.25 * no_old.\n")
        fh.write("EC blend = 0.25 * no_quality + 0.75 * no_old.\n")
        fh.write("DRP final = 0.66 * chain_base + 0.34 * specialist(threshold=144).\n")
        fh.write(f"Average spatial CV = {avg_spatial:.4f}; estimated public = {estimated_public:.4f}.\n")

    print(metrics.to_string(index=False))
    print(mix_config.to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run()

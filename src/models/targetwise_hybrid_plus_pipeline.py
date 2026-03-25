import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.evaluation.evaluate_local import estimate_public_score
from src.models.advanced_stacking_pipeline import MERGE_KEYS, add_features, fit_target_chain, load_frames
from src.models.targetwise_hybrid_pipeline import NO_OLD, NO_QUALITY, fit_drp_specialist_oof

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_targetwise_hybrid_plus")

SPECIALIST_THRESHOLDS = [80.0, 96.0, 112.0, 128.0, 144.0, 160.0]
SPECIALIST_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def build_specialist_features():
    return [
        "blue", "green", "red", "nir08", "swir16", "swir22",
        "pet", "ppt", "tmax", "tmin", "q",
        "days_offset_filled", "days_offset_missing",
        "api_optical_complete", "api_preferred",
        "month_sin", "month_cos",
        "NDVI_new", "NDWI", "MNDWI_new", "SABI", "WRI", "NDTI", "Turbidity",
        "ppt_pet_ratio", "temp_range", "runoff_ratio", "Latitude",
    ]


def scan_best_drp_mix(ta_r2, ec_r2, y_true, base_pred, specialist_runs):
    rows = []
    for variant, run in specialist_runs.items():
        for weight in SPECIALIST_WEIGHTS:
            pred = (1.0 - weight) * base_pred + weight * run["oof_pred"].values
            drp_r2 = float(r2_score(y_true, pred))
            target_scores = {
                "Total Alkalinity": ta_r2,
                "Electrical Conductance": ec_r2,
                "Dissolved Reactive Phosphorus": drp_r2,
            }
            avg_spatial = float(np.mean(list(target_scores.values())))
            est_public, _ = estimate_public_score(avg_spatial, 39, target_scores=target_scores)
            rows.append(
                {
                    "variant": variant,
                    "threshold": run["threshold"],
                    "weight": weight,
                    "drp_r2": drp_r2,
                    "avg_spatial": avg_spatial,
                    "estimated_public": est_public,
                    "specialist_only_r2": float(run["CV_R2"]),
                }
            )
    scan = pd.DataFrame(rows).sort_values(
        ["avg_spatial", "drp_r2", "estimated_public"],
        ascending=False,
    ).reset_index(drop=True)
    return scan, scan.iloc[0].to_dict()


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    train_df["Turbidity"] = train_df["turbidity_ratio"]
    val_df["Turbidity"] = val_df["turbidity_ratio"]

    ta_features = NO_QUALITY + ["Latitude", "Longitude"]
    ec_features = NO_QUALITY
    drp_features = NO_OLD + ["Latitude"]
    specialist_features = build_specialist_features()

    ta = fit_target_chain(train_df, val_df, "Total Alkalinity", ta_features, filter_complete=False)
    ta_oof = pd.Series(np.nan, index=train_df.index, name="ta_pred")
    ta_oof.loc[ta["train_mask"]] = ta["oof_pred"].values
    ta_val = ta["val_pred"].rename("ta_pred")

    ec = fit_target_chain(
        train_df,
        val_df,
        "Electrical Conductance",
        ec_features,
        extra_train={"ta_pred": ta_oof},
        extra_val={"ta_pred": ta_val},
        filter_complete=True,
    )
    ec_oof = pd.Series(np.nan, index=train_df.index, name="ec_pred")
    ec_oof.loc[ec["train_mask"]] = ec["oof_pred"].values
    ec_val = ec["val_pred"].rename("ec_pred")

    drp_chain = fit_target_chain(
        train_df,
        val_df,
        "Dissolved Reactive Phosphorus",
        drp_features,
        extra_train={"ta_pred": ta_oof, "ec_pred": ec_oof},
        extra_val={"ta_pred": ta_val, "ec_pred": ec_val},
        filter_complete=True,
    )

    specialist_runs = {}
    for threshold in SPECIALIST_THRESHOLDS:
        key = f"thr{int(threshold)}"
        run_info = fit_drp_specialist_oof(train_df, val_df, specialist_features, threshold=threshold)
        run_info["threshold"] = threshold
        specialist_runs[key] = run_info

    y = train_df.loc[train_df["api_optical_complete"] == 1, "Dissolved Reactive Phosphorus"].reset_index(drop=True)
    base_pred = drp_chain["oof_pred"].reset_index(drop=True).values
    scan, best = scan_best_drp_mix(float(ta["CV_R2"]), float(ec["CV_R2"]), y, base_pred, specialist_runs)
    scan.to_csv(os.path.join(OUTPUT_DIR, "drp_mix_scan.csv"), index=False)

    best_run = specialist_runs[best["variant"]]
    weight = float(best["weight"])

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta["val_pred"].values
    submission["Electrical Conductance"] = ec["val_pred"].values
    submission["Dissolved Reactive Phosphorus"] = (1.0 - weight) * drp_chain["val_pred"].values + weight * best_run["val_pred"].values
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_ensemble.csv"), index=False)

    metrics = pd.DataFrame(
        [
            {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"], "feature_count": len(ta_features)},
            {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"], "feature_count": len(ec_features)},
            {
                "Parameter": "Dissolved Reactive Phosphorus",
                "CV_R2": float(best["drp_r2"]),
                "CV_RMSE": float(np.sqrt(mean_squared_error(y, (1.0 - weight) * base_pred + weight * best_run["oof_pred"].values))),
                "rows_used": best_run["rows_used"],
                "feature_count": len(drp_features),
            },
        ]
    )
    metrics.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "feature_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("[Total Alkalinity]\n" + "\n".join(ta_features) + "\n\n")
        fh.write("[Electrical Conductance]\n" + "\n".join(ec_features) + "\n\n")
        fh.write("[Dissolved Reactive Phosphorus]\n" + "\n".join(drp_features) + "\n\n")
        fh.write("[DRP specialist]\n" + "\n".join(specialist_features) + "\n\n")
        fh.write(f"best_variant={best['variant']} threshold={best['threshold']:.0f} weight={best['weight']:.2f}\n")

    with open(os.path.join(OUTPUT_DIR, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("TA/EC from no_quality_flags target-chain stackers.\n")
        fh.write("DRP from no_old_optics target-chain plus expanded specialist threshold/weight search.\n")
        fh.write(
            f"Best DRP mix: {best['variant']} threshold={best['threshold']:.0f} "
            f"weight={best['weight']:.2f} avg_spatial={best['avg_spatial']:.4f} "
            f"estimated_public={best['estimated_public']:.4f}\n"
        )

    print(metrics.to_string(index=False))
    print(scan.head(10).to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run()

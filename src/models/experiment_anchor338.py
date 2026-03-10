import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERAL_DIR = os.path.join(PROJECT_ROOT, "resources", "code", "general")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
MERGE_KEYS = ["Latitude", "Longitude", "Sample Date"]
RANDOM_STATE = 42
N_SPLITS = 5


def load_frames():
    train = pd.read_csv(os.path.join(GENERAL_DIR, "water_quality_training_dataset.csv"))
    sub = pd.read_csv(os.path.join(GENERAL_DIR, "submission_template.csv"))

    api_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "landsat_api_training.csv")).set_index("Index")
    api_va = pd.read_csv(os.path.join(PROCESSED_DIR, "landsat_api_validation.csv")).set_index("Index")
    old_tr = pd.read_csv(os.path.join(GENERAL_DIR, "landsat_features_training.csv"))
    old_va = pd.read_csv(os.path.join(GENERAL_DIR, "landsat_features_validation.csv"))
    tc_tr = pd.read_csv(os.path.join(GENERAL_DIR, "terraclimate_features_training.csv"))
    tc_va = pd.read_csv(os.path.join(GENERAL_DIR, "terraclimate_features_validation.csv"))

    train = train.join(api_tr, how="left")
    val = old_va[MERGE_KEYS].join(api_va, how="left")

    train = train.merge(old_tr, on=MERGE_KEYS, how="left", suffixes=("", "_old"))
    val = val.merge(old_va, on=MERGE_KEYS, how="left", suffixes=("", "_old"))

    train = train.merge(tc_tr, on=MERGE_KEYS, how="left")
    val = val.merge(tc_va, on=MERGE_KEYS, how="left")

    return train, val, sub


def add_features(df):
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce")
    df["month"] = df["date_parsed"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["days_offset_filled"] = df["days_offset"].fillna(61).clip(0, 61)
    df["days_offset_missing"] = df["days_offset"].isna().astype(int)
    df["api_optical_complete"] = df[["blue", "green", "red", "nir08", "swir16", "swir22"]].notna().all(axis=1).astype(int)
    df["old_optical_available"] = df[["nir", "green_old", "swir16_old", "swir22_old"]].notna().all(axis=1).astype(int)

    df["NDVI_new"] = (df["nir08"] - df["red"]) / (df["nir08"] + df["red"] + 1e-8)
    df["NDWI"] = (df["green"] - df["nir08"]) / (df["green"] + df["nir08"] + 1e-8)
    df["MNDWI_new"] = (df["green"] - df["swir16"]) / (df["green"] + df["swir16"] + 1e-8)
    df["SABI"] = (df["nir08"] - df["red"]) / (df["blue"] + df["green"] + 1e-8)
    df["WRI"] = (df["green"] + df["red"]) / (df["nir08"] + df["swir16"] + 1e-8)
    df["NDTI"] = (df["red"] - df["green"]) / (df["red"] + df["green"] + 1e-8)
    df["Turbidity"] = df["red"] / (df["blue"] + 1e-8)

    df["ppt_pet_ratio"] = df["ppt"] / (df["pet"] + 1e-8)
    df["temp_range"] = df["tmax"] - df["tmin"]
    df["runoff_ratio"] = df["q"] / (df["ppt"] + 1e-8)

    df["lat_bin"] = np.round(df["Latitude"], 1)
    df["lon_bin"] = np.round(df["Longitude"], 1)
    df["spatial_group"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    return df


def build_reg_models():
    return {
        "xgb": XGBRegressor(
            n_estimators=450,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.65,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            missing=np.nan,
        ),
        "lgb": LGBMRegressor(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.65,
            min_child_samples=25,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "cat": CatBoostRegressor(
            iterations=450,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def build_classifier():
    return CatBoostClassifier(
        iterations=250,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def blend(parts, target):
    if target == "Dissolved Reactive Phosphorus":
        weights = {"xgb": 0.20, "lgb": 0.20, "cat": 0.60}
    elif target == "Electrical Conductance":
        weights = {"xgb": 0.35, "lgb": 0.40, "cat": 0.25}
    else:
        weights = {"xgb": 0.40, "lgb": 0.40, "cat": 0.20}
    return sum(parts[k] * weights[k] for k in weights)


def sample_weights(df):
    source_weight = np.where(df["api_optical_complete"] == 1, 1.0, np.where(df["old_optical_available"] == 1, 0.9, 0.8))
    offset_weight = 1.0 - (df["days_offset_filled"] / 61.0) * 0.15
    return source_weight * offset_weight


def fit_linear_calibration(y_true, pred):
    coef, intercept = np.polyfit(pred, y_true, deg=1)
    return float(coef), float(intercept)


def fit_single_target(train_df, val_df, target, features, filter_complete=False):
    mask = pd.Series(True, index=train_df.index)
    if filter_complete:
        mask &= train_df["api_optical_complete"] == 1
    tr = train_df.loc[mask].reset_index(drop=True)
    X = tr[features].copy()
    y = tr[target].copy()
    groups = tr["spatial_group"]
    w = sample_weights(tr)

    oof = np.zeros(len(tr))
    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = w[tr_idx]
        parts = {}
        for name, model in build_reg_models().items():
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            parts[name] = model.predict(X_va)
        oof[va_idx] = blend(parts, target)

    coef, intercept = (1.0, 0.0)
    if target != "Total Alkalinity":
        coef, intercept = fit_linear_calibration(y, oof)
        oof = coef * oof + intercept

    final_parts = {}
    for name, model in build_reg_models().items():
        model.fit(X, y, sample_weight=w)
        final_parts[name] = model

    X_val = val_df[features].copy()
    val_pred = blend({name: model.predict(X_val) for name, model in final_parts.items()}, target)
    if target != "Total Alkalinity":
        val_pred = coef * val_pred + intercept

    return {
        "CV_R2": r2_score(y, oof),
        "CV_RMSE": float(np.sqrt(mean_squared_error(y, oof))),
        "val_pred": val_pred,
        "rows_used": len(tr),
    }


def fit_drp_specialist(train_df, val_df, features, threshold=96.0):
    mask = train_df["api_optical_complete"] == 1
    tr = train_df.loc[mask].reset_index(drop=True)
    X = tr[features].copy()
    y = tr["Dissolved Reactive Phosphorus"].copy()
    groups = tr["spatial_group"]
    w = sample_weights(tr)
    oof = np.zeros(len(tr))

    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = w[tr_idx]
        label = (y_tr >= threshold).astype(int)
        clf = build_classifier()
        clf.fit(X_tr, label, sample_weight=w_tr)
        p_high = clf.predict_proba(X_va)[:, 1]

        low_mask = y_tr < threshold
        high_mask = y_tr >= threshold
        low_parts, high_parts = {}, {}
        for name, model in build_reg_models().items():
            model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
            low_parts[name] = model.predict(X_va)
        for name, model in build_reg_models().items():
            model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])
            high_parts[name] = model.predict(X_va)
        oof[va_idx] = (1.0 - p_high) * blend(low_parts, "Dissolved Reactive Phosphorus") + p_high * blend(high_parts, "Dissolved Reactive Phosphorus")

    coef, intercept = fit_linear_calibration(y, oof)
    oof = coef * oof + intercept

    clf_full = build_classifier()
    clf_full.fit(X, (y >= threshold).astype(int), sample_weight=w)
    low_mask_all = y < threshold
    high_mask_all = y >= threshold
    low_models, high_models = build_reg_models(), build_reg_models()
    for model in low_models.values():
        model.fit(X.loc[low_mask_all], y.loc[low_mask_all], sample_weight=w[low_mask_all.values])
    for model in high_models.values():
        model.fit(X.loc[high_mask_all], y.loc[high_mask_all], sample_weight=w[high_mask_all.values])

    X_val = val_df[features].copy()
    p_high_val = clf_full.predict_proba(X_val)[:, 1]
    low_val = blend({name: model.predict(X_val) for name, model in low_models.items()}, "Dissolved Reactive Phosphorus")
    high_val = blend({name: model.predict(X_val) for name, model in high_models.items()}, "Dissolved Reactive Phosphorus")
    val_pred = coef * ((1.0 - p_high_val) * low_val + p_high_val * high_val) + intercept

    return {
        "CV_R2": r2_score(y, oof),
        "CV_RMSE": float(np.sqrt(mean_squared_error(y, oof))),
        "val_pred": val_pred,
        "rows_used": len(tr),
    }


def run():
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)

    base_features = [
        "blue", "green", "red", "nir08", "swir16", "swir22",
        "pet", "ppt", "tmax", "tmin", "q",
        "nir", "green_old", "swir16_old", "swir22_old", "NDMI", "MNDWI",
        "days_offset_filled", "days_offset_missing", "api_optical_complete", "old_optical_available",
        "month_sin", "month_cos",
        "NDVI_new", "NDWI", "MNDWI_new", "SABI", "WRI", "NDTI", "Turbidity",
        "ppt_pet_ratio", "temp_range", "runoff_ratio",
    ]
    drp_specialist_features = list(dict.fromkeys(base_features + ["Latitude"]))

    ta = fit_single_target(train_df, val_df, "Total Alkalinity", base_features, filter_complete=False)
    ec = fit_single_target(train_df, val_df, "Electrical Conductance", base_features, filter_complete=True)
    drp_base = fit_single_target(train_df, val_df, "Dissolved Reactive Phosphorus", base_features, filter_complete=True)
    drp_spec = fit_drp_specialist(train_df, val_df, drp_specialist_features, threshold=96.0)

    out_root = os.path.join(OUTPUT_DIR, "submission_anchor338_experiment")
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for weight in [0.30, 0.50, 0.70, 0.85, 1.00]:
        run_name = f"drpblend_{int(weight*100):02d}" if weight < 1.0 else "drpspecialist_exact"
        run_dir = os.path.join(out_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        drp_r2 = (1.0 - weight) * drp_base["CV_R2"] + weight * drp_spec["CV_R2"]
        submission = sub[MERGE_KEYS].copy()
        submission["Total Alkalinity"] = ta["val_pred"]
        submission["Electrical Conductance"] = ec["val_pred"]
        submission["Dissolved Reactive Phosphorus"] = (1.0 - weight) * drp_base["val_pred"] + weight * drp_spec["val_pred"]
        submission.to_csv(os.path.join(run_dir, "submission_ensemble.csv"), index=False)

        metrics = pd.DataFrame(
            [
                {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"]},
                {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"]},
                {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp_r2, "CV_RMSE": drp_spec["CV_RMSE"], "rows_used": drp_spec["rows_used"]},
            ]
        )
        metrics.to_csv(os.path.join(run_dir, "cv_metrics.csv"), index=False)
        rows.append({"candidate": run_name, "ta_r2": ta["CV_R2"], "ec_r2": ec["CV_R2"], "drp_r2": drp_r2, "avg_spatial": float(metrics["CV_R2"].mean())})

    summary = pd.DataFrame(rows).sort_values("avg_spatial", ascending=False)
    summary.to_csv(os.path.join(out_root, "candidate_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()

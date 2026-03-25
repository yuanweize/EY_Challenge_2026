import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERAL_DIR = os.path.join(PROJECT_ROOT, "resources", "code", "general")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_advanced_stacking")

TARGET_COLS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]
MERGE_KEYS = ["Latitude", "Longitude", "Sample Date"]
N_SPLITS = 5
RANDOM_STATE = 42


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
    train = train.merge(old_tr, on=MERGE_KEYS, how="left", suffixes=("", "_old"))
    train = train.merge(tc_tr, on=MERGE_KEYS, how="left")

    val = old_va[MERGE_KEYS].join(api_va, how="left")
    val = val.merge(old_va, on=MERGE_KEYS, how="left", suffixes=("", "_old"))
    val = val.merge(tc_va, on=MERGE_KEYS, how="left")
    return train, val, sub


def add_features(df):
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["Sample Date"], dayfirst=True, errors="coerce")
    df["month"] = df["date_parsed"].dt.month
    df["quarter"] = df["date_parsed"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    optical_cols = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    old_cols = ["nir", "green_old", "swir16_old", "swir22_old"]
    climate_cols = ["pet", "ppt", "tmax", "tmin", "q"]

    df["days_offset_filled"] = df["days_offset"].fillna(61).clip(0, 61)
    df["days_offset_missing"] = df["days_offset"].isna().astype(int)
    df["api_optical_complete"] = df[optical_cols].notna().all(axis=1).astype(int)
    df["old_optical_available"] = df[old_cols].notna().all(axis=1).astype(int)
    df["api_preferred"] = ((df["api_optical_complete"] == 1) & (df["days_offset_filled"] <= 10)).astype(int)

    df["NDVI_new"] = (df["nir08"] - df["red"]) / (df["nir08"] + df["red"] + 1e-8)
    df["NDWI"] = (df["green"] - df["nir08"]) / (df["green"] + df["nir08"] + 1e-8)
    df["MNDWI_new"] = (df["green"] - df["swir16"]) / (df["green"] + df["swir16"] + 1e-8)
    df["SABI"] = (df["nir08"] - df["red"]) / (df["blue"] + df["green"] + 1e-8)
    df["WRI"] = (df["green"] + df["red"]) / (df["nir08"] + df["swir16"] + 1e-8)
    df["NDTI"] = (df["red"] - df["green"]) / (df["red"] + df["green"] + 1e-8)
    df["turbidity_ratio"] = df["red"] / (df["blue"] + 1e-8)
    df["green_red_ratio"] = df["green"] / (df["red"] + 1e-8)
    df["nir_swir_ratio"] = df["nir08"] / (df["swir16"] + 1e-8)
    df["swir_ratio"] = df["swir16"] / (df["swir22"] + 1e-8)

    df["ppt_pet_ratio"] = df["ppt"] / (df["pet"] + 1e-8)
    df["temp_range"] = df["tmax"] - df["tmin"]
    df["runoff_ratio"] = df["q"] / (df["ppt"] + 1e-8)
    df["hydro_stress"] = df["pet"] / (df["ppt"] + 1e-8)
    df["thermal_runoff"] = df["temp_range"] * df["q"]

    for col in climate_cols:
        df[f"{col}_is_missing"] = df[col].isna().astype(int)

    df["lat_bin"] = np.round(df["Latitude"], 1)
    df["lon_bin"] = np.round(df["Longitude"], 1)
    df["spatial_group"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    return df


def build_base_models():
    return {
        "xgb": XGBRegressor(
            n_estimators=700,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            missing=np.nan,
        ),
        "lgb": LGBMRegressor(
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=28,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "cat": CatBoostRegressor(
            iterations=700,
            learning_rate=0.025,
            depth=5,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def sample_weights(df):
    source_weight = np.where(df["api_optical_complete"] == 1, 1.0, np.where(df["old_optical_available"] == 1, 0.92, 0.85))
    offset_weight = 1.0 - (df["days_offset_filled"] / 61.0) * 0.18
    climate_weight = np.where(df["ppt"].notna(), 1.0, 0.95)
    return source_weight * offset_weight * climate_weight


def build_feature_sets():
    common = [
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
    return {
        "Total Alkalinity": common + ["Latitude", "Longitude"],
        "Electrical Conductance": common,
        "Dissolved Reactive Phosphorus": common + ["Latitude"],
    }


def fit_target_chain(train_df, val_df, target, feature_cols, extra_train=None, extra_val=None, filter_complete=False):
    train_mask = pd.Series(True, index=train_df.index)
    if filter_complete:
        train_mask &= train_df["api_optical_complete"] == 1

    tr = train_df.loc[train_mask].reset_index(drop=True)
    va = val_df.reset_index(drop=True).copy()
    X = tr[feature_cols].copy()
    X_val = va[feature_cols].copy()

    if extra_train is not None:
        for name, values in extra_train.items():
            X[name] = values.loc[train_mask].reset_index(drop=True)
    if extra_val is not None:
        for name, values in extra_val.items():
            X_val[name] = values.reset_index(drop=True)

    y = tr[target].copy()
    groups = tr["spatial_group"]
    weights = sample_weights(tr)
    gkf = GroupKFold(n_splits=N_SPLITS)

    base_keys = list(build_base_models().keys())
    oof_base = {name: np.zeros(len(tr)) for name in base_keys}
    val_fold_preds = {name: [] for name in base_keys}

    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = weights[tr_idx]

        for name, model in build_base_models().items():
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_base[name][va_idx] = model.predict(X_va)
            val_fold_preds[name].append(model.predict(X_val))

    stack_train = pd.DataFrame({f"{name}_pred": pred for name, pred in oof_base.items()})
    stack_val = pd.DataFrame({f"{name}_pred": np.mean(preds, axis=0) for name, preds in val_fold_preds.items()})

    passthrough_cols = ["days_offset_filled", "api_optical_complete", "old_optical_available", "Latitude", "Longitude", "month_sin", "month_cos"]
    for col in passthrough_cols:
        if col in X.columns:
            stack_train[col] = X[col].values
            stack_val[col] = X_val[col].values

    if extra_train is not None:
        for name, values in extra_train.items():
            stack_train[name] = values.loc[train_mask].reset_index(drop=True).values
    if extra_val is not None:
        for name, values in extra_val.items():
            stack_val[name] = values.reset_index(drop=True).values

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_train, y, sample_weight=weights)
    oof = meta_model.predict(stack_train)

    final_models = {}
    for name, model in build_base_models().items():
        model.fit(X, y, sample_weight=weights)
        final_models[name] = model

    final_stack_val = pd.DataFrame({f"{name}_pred": model.predict(X_val) for name, model in final_models.items()})
    for col in passthrough_cols:
        if col in X.columns:
            final_stack_val[col] = X_val[col].values
    if extra_val is not None:
        for name, values in extra_val.items():
            final_stack_val[name] = values.reset_index(drop=True).values
    val_pred = meta_model.predict(final_stack_val)

    return {
        "train_mask": train_mask,
        "oof_pred": pd.Series(oof, index=tr.index),
        "val_pred": pd.Series(val_pred, index=va.index),
        "CV_R2": r2_score(y, oof),
        "CV_RMSE": float(np.sqrt(mean_squared_error(y, oof))),
        "rows_used": len(tr),
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    feature_sets = build_feature_sets()

    ta = fit_target_chain(train_df, val_df, "Total Alkalinity", feature_sets["Total Alkalinity"], filter_complete=False)
    ta_oof_full = pd.Series(np.nan, index=train_df.index, name="ta_pred")
    ta_oof_full.loc[ta["train_mask"]] = ta["oof_pred"].values
    ta_val = ta["val_pred"].rename("ta_pred")

    ec = fit_target_chain(
        train_df,
        val_df,
        "Electrical Conductance",
        feature_sets["Electrical Conductance"],
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
        feature_sets["Dissolved Reactive Phosphorus"],
        extra_train={"ta_pred": ta_oof_full, "ec_pred": ec_oof_full},
        extra_val={"ta_pred": ta_val, "ec_pred": ec_val},
        filter_complete=True,
    )

    metrics = pd.DataFrame(
        [
            {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"]},
            {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"]},
            {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp["CV_R2"], "CV_RMSE": drp["CV_RMSE"], "rows_used": drp["rows_used"]},
        ]
    )
    metrics.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta["val_pred"].values
    submission["Electrical Conductance"] = ec["val_pred"].values
    submission["Dissolved Reactive Phosphorus"] = drp["val_pred"].values
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_ensemble.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "feature_summary.txt"), "w", encoding="utf-8") as fh:
        for target in TARGET_COLS:
            fh.write(f"[{target}]\n")
            fh.write("\n".join(feature_sets[target]))
            fh.write("\n\n")
        fh.write("[Target chain]\n")
        fh.write("Electrical Conductance <- ta_pred\n")
        fh.write("Dissolved Reactive Phosphorus <- ta_pred, ec_pred\n")

    print(metrics.to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run()

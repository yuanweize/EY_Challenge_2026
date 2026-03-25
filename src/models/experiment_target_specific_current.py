import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERAL_DIR = os.path.join(PROJECT_ROOT, "resources", "code", "general")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MERGE_KEYS = ["Latitude", "Longitude", "Sample Date"]
RANDOM_STATE = 42
N_SPLITS = 5


def load_frames():
    wq = pd.read_csv(os.path.join(GENERAL_DIR, "water_quality_training_dataset.csv"))
    sub = pd.read_csv(os.path.join(GENERAL_DIR, "submission_template.csv"))
    api_tr = pd.read_csv(os.path.join(PROCESSED_DIR, "landsat_api_training.csv")).set_index("Index")
    api_va = pd.read_csv(os.path.join(PROCESSED_DIR, "landsat_api_validation.csv")).set_index("Index")
    old_tr = pd.read_csv(os.path.join(GENERAL_DIR, "landsat_features_training.csv"))
    old_va = pd.read_csv(os.path.join(GENERAL_DIR, "landsat_features_validation.csv"))
    tc_tr = pd.read_csv(os.path.join(GENERAL_DIR, "terraclimate_features_training.csv"))
    tc_va = pd.read_csv(os.path.join(GENERAL_DIR, "terraclimate_features_validation.csv"))

    train = wq.join(api_tr, how="left")
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
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["days_offset_filled"] = df["days_offset"].fillna(61).clip(0, 61)
    df["days_offset_missing"] = df["days_offset"].isna().astype(int)
    df["api_complete"] = df[["blue", "green", "red", "nir08", "swir16", "swir22"]].notna().all(axis=1).astype(int)
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
    df["spatial_group"] = np.round(df["Latitude"], 1).astype(str) + "_" + np.round(df["Longitude"], 1).astype(str)
    return df


def reg_model():
    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
    )


def fit_linear_calibration(y_true, pred):
    coef, intercept = np.polyfit(pred, y_true, deg=1)
    return float(coef), float(intercept)


def fit_target(train_df, val_df, target, features, mask=None, calibrate=False):
    tr = train_df if mask is None else train_df.loc[mask].reset_index(drop=True)
    X = tr[features].copy()
    y = tr[target].copy()
    groups = tr["spatial_group"]
    oof = np.zeros(len(tr))
    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = reg_model()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx])
    coef, intercept = 1.0, 0.0
    if calibrate:
        coef, intercept = fit_linear_calibration(y, oof)
        oof = coef * oof + intercept
    model = reg_model()
    model.fit(X, y)
    val_pred = model.predict(val_df[features])
    if calibrate:
        val_pred = coef * val_pred + intercept
    return {
        "CV_R2": r2_score(y, oof),
        "CV_RMSE": float(np.sqrt(mean_squared_error(y, oof))),
        "val_pred": val_pred,
        "rows_used": len(tr),
    }


def fit_drp_two_stage(train_df, val_df, features, threshold=128.0):
    tr = train_df.loc[train_df["api_complete"] == 1].reset_index(drop=True)
    X = tr[features].copy()
    y = tr["Dissolved Reactive Phosphorus"].copy()
    groups = tr["spatial_group"]
    oof = np.zeros(len(tr))
    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        clf = CatBoostClassifier(
            iterations=250,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X_tr, (y_tr >= threshold).astype(int))
        p_high = clf.predict_proba(X_va)[:, 1]
        low_mask = y_tr < threshold
        high_mask = y_tr >= threshold
        low_model = reg_model()
        high_model = reg_model()
        low_model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask])
        high_model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask])
        oof[va_idx] = (1.0 - p_high) * low_model.predict(X_va) + p_high * high_model.predict(X_va)

    coef, intercept = fit_linear_calibration(y, oof)
    oof = coef * oof + intercept

    clf_full = CatBoostClassifier(
        iterations=250,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    clf_full.fit(X, (y >= threshold).astype(int))
    low_all = y < threshold
    high_all = y >= threshold
    low_model = reg_model()
    high_model = reg_model()
    low_model.fit(X.loc[low_all], y.loc[low_all])
    high_model.fit(X.loc[high_all], y.loc[high_all])

    X_val = val_df[features].copy()
    p_high_val = clf_full.predict_proba(X_val)[:, 1]
    val_pred = (1.0 - p_high_val) * low_model.predict(X_val) + p_high_val * high_model.predict(X_val)
    val_pred = coef * val_pred + intercept
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

    api12 = ["blue","green","red","nir08","swir16","swir22","pet","NDVI_new","NDWI","MNDWI_new","SABI","WRI"]
    climate = ["ppt","tmax","tmin","q","ppt_pet_ratio","temp_range","runoff_ratio"]
    quality = ["days_offset_filled","days_offset_missing","api_complete"]
    extra = ["month_sin","month_cos","NDTI","Turbidity"]

    ta_features = api12 + climate + extra + quality
    ec_features = api12 + climate + quality + extra
    drp_features = api12 + climate + quality + extra

    ta = fit_target(train_df, val_df, "Total Alkalinity", ta_features, mask=None, calibrate=False)
    ec = fit_target(train_df, val_df, "Electrical Conductance", ec_features, mask=train_df["api_complete"] == 1, calibrate=True)
    drp = fit_drp_two_stage(train_df, val_df, drp_features, threshold=128.0)

    out_root = os.path.join(OUTPUT_DIR, "submission_target_specific_20260310")
    os.makedirs(out_root, exist_ok=True)

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta["val_pred"]
    submission["Electrical Conductance"] = ec["val_pred"]
    submission["Dissolved Reactive Phosphorus"] = drp["val_pred"]
    submission.to_csv(os.path.join(out_root, "submission_ensemble.csv"), index=False)

    metrics = pd.DataFrame([
        {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"], "feature_count": len(ta_features)},
        {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"], "feature_count": len(ec_features)},
        {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp["CV_R2"], "CV_RMSE": drp["CV_RMSE"], "rows_used": drp["rows_used"], "feature_count": len(drp_features)},
    ])
    metrics.to_csv(os.path.join(out_root, "cv_metrics.csv"), index=False)
    with open(os.path.join(out_root, "feature_map.txt"), "w", encoding="utf-8") as fh:
        fh.write("[Total Alkalinity]\n" + "\n".join(ta_features) + "\n\n")
        fh.write("[Electrical Conductance]\n" + "\n".join(ec_features) + "\n\n")
        fh.write("[Dissolved Reactive Phosphorus]\n" + "\n".join(drp_features) + "\n")
    print(metrics.to_string(index=False))
    print("Saved to", out_root)


if __name__ == "__main__":
    run()

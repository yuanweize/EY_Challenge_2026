import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.models.advanced_stacking_pipeline import (
    MERGE_KEYS,
    OUTPUT_DIR as UNUSED_OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COLS,
    add_features,
    build_base_models,
    build_feature_sets,
    load_frames,
    sample_weights,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_advanced_strict_hybrid")
N_SPLITS = 5


def fit_strict_target(train_df, val_df, target, feature_cols, extra_train=None, extra_val=None, filter_complete=False):
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

    base_names = list(build_base_models().keys())
    oof_base = {name: np.zeros(len(tr)) for name in base_names}
    val_fold_preds = {name: [] for name in base_names}

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

    passthrough_cols = [
        "days_offset_filled",
        "api_optical_complete",
        "old_optical_available",
        "Latitude",
        "Longitude",
        "month_sin",
        "month_cos",
    ]
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

    meta_oof = np.zeros(len(tr))
    for tr_idx, va_idx in gkf.split(stack_train, y, groups):
        meta = Ridge(alpha=1.0)
        meta.fit(stack_train.iloc[tr_idx], y.iloc[tr_idx], sample_weight=weights[tr_idx])
        meta_oof[va_idx] = meta.predict(stack_train.iloc[va_idx])

    meta_full = Ridge(alpha=1.0)
    meta_full.fit(stack_train, y, sample_weight=weights)
    val_pred = meta_full.predict(stack_val)

    return {
        "train_mask": train_mask,
        "oof_pred": pd.Series(meta_oof, index=tr.index),
        "val_pred": pd.Series(val_pred, index=va.index),
        "CV_R2": r2_score(y, meta_oof),
        "CV_RMSE": float(np.sqrt(mean_squared_error(y, meta_oof))),
        "rows_used": len(tr),
    }


def fit_drp_specialist(train_df, val_df, feature_cols, extra_train=None, extra_val=None, threshold=96.0):
    train_mask = train_df["api_optical_complete"] == 1
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

    y = tr["Dissolved Reactive Phosphorus"].copy()
    groups = tr["spatial_group"]
    weights = sample_weights(tr)
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof = np.zeros(len(tr))
    val_parts = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = weights[tr_idx]
        label = (y_tr >= threshold).astype(int)

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
        clf.fit(X_tr, label, sample_weight=w_tr)
        p_high = clf.predict_proba(X_va)[:, 1]

        low_mask = y_tr < threshold
        high_mask = y_tr >= threshold
        low_preds = {}
        high_preds = {}
        for name, model in build_base_models().items():
            model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
            low_preds[name] = model.predict(X_va)
        for name, model in build_base_models().items():
            model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])
            high_preds[name] = model.predict(X_va)

        low_blend = 0.2 * low_preds["xgb"] + 0.2 * low_preds["lgb"] + 0.6 * low_preds["cat"]
        high_blend = 0.2 * high_preds["xgb"] + 0.2 * high_preds["lgb"] + 0.6 * high_preds["cat"]
        oof[va_idx] = (1.0 - p_high) * low_blend + p_high * high_blend

        clf_fold = CatBoostClassifier(
            iterations=250,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        clf_fold.fit(X_tr, label, sample_weight=w_tr)
        p_high_val = clf_fold.predict_proba(X_val)[:, 1]

        low_models = build_base_models()
        high_models = build_base_models()
        for model in low_models.values():
            model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
        for model in high_models.values():
            model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])

        low_val = 0.2 * low_models["xgb"].predict(X_val) + 0.2 * low_models["lgb"].predict(X_val) + 0.6 * low_models["cat"].predict(X_val)
        high_val = 0.2 * high_models["xgb"].predict(X_val) + 0.2 * high_models["lgb"].predict(X_val) + 0.6 * high_models["cat"].predict(X_val)
        val_parts.append((1.0 - p_high_val) * low_val + p_high_val * high_val)

    val_pred = np.mean(val_parts, axis=0)
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

    ta = fit_strict_target(train_df, val_df, "Total Alkalinity", feature_sets["Total Alkalinity"], filter_complete=False)
    ta_oof_full = pd.Series(np.nan, index=train_df.index, name="ta_pred")
    ta_oof_full.loc[ta["train_mask"]] = ta["oof_pred"].values
    ta_val = ta["val_pred"].rename("ta_pred")

    ec = fit_strict_target(
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

    drp_chain = fit_strict_target(
        train_df,
        val_df,
        "Dissolved Reactive Phosphorus",
        feature_sets["Dissolved Reactive Phosphorus"],
        extra_train={"ta_pred": ta_oof_full, "ec_pred": ec_oof_full},
        extra_val={"ta_pred": ta_val, "ec_pred": ec_val},
        filter_complete=True,
    )
    drp_specialist = fit_drp_specialist(
        train_df,
        val_df,
        feature_sets["Dissolved Reactive Phosphorus"],
        extra_train={"ta_pred": ta_oof_full, "ec_pred": ec_oof_full},
        extra_val={"ta_pred": ta_val, "ec_pred": ec_val},
        threshold=96.0,
    )

    specialist_weight = 0.70
    drp_train_y = train_df.loc[drp_specialist["train_mask"], "Dissolved Reactive Phosphorus"].reset_index(drop=True)
    drp_oof_blend = (1.0 - specialist_weight) * drp_chain["oof_pred"].values + specialist_weight * drp_specialist["oof_pred"].values
    drp_val_blend = (1.0 - specialist_weight) * drp_chain["val_pred"].values + specialist_weight * drp_specialist["val_pred"].values
    drp_r2 = r2_score(drp_train_y, drp_oof_blend)
    drp_rmse = float(np.sqrt(mean_squared_error(drp_train_y, drp_oof_blend)))

    metrics = pd.DataFrame(
        [
            {"Parameter": "Total Alkalinity", "CV_R2": ta["CV_R2"], "CV_RMSE": ta["CV_RMSE"], "rows_used": ta["rows_used"]},
            {"Parameter": "Electrical Conductance", "CV_R2": ec["CV_R2"], "CV_RMSE": ec["CV_RMSE"], "rows_used": ec["rows_used"]},
            {"Parameter": "Dissolved Reactive Phosphorus", "CV_R2": drp_r2, "CV_RMSE": drp_rmse, "rows_used": drp_specialist["rows_used"]},
        ]
    )
    metrics.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)

    submission = sub[MERGE_KEYS].copy()
    submission["Total Alkalinity"] = ta["val_pred"].values
    submission["Electrical Conductance"] = ec["val_pred"].values
    submission["Dissolved Reactive Phosphorus"] = drp_val_blend
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_ensemble.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("Strict two-level OOF stacking for TA/EC.\n")
        fh.write("DRP = 0.30 * strict chain + 0.70 * specialist(threshold=96).\n")

    print(metrics.to_string(index=False))
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    run()

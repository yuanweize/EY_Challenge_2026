import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.evaluation.evaluate_local import estimate_public_score
from src.models.advanced_stacking_pipeline import MERGE_KEYS, add_features, load_frames, fit_target_chain
from src.models.experiment_anchor338 import build_reg_models, blend, sample_weights

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "submission_targetwise_hybrid")
RANDOM_STATE = 42
N_SPLITS = 5

COMMON_ALL = [
    'blue','green','red','nir08','swir16','swir22','pet','ppt','tmax','tmin','q',
    'nir','green_old','swir16_old','swir22_old','NDMI','MNDWI',
    'month_sin','month_cos','quarter','days_offset_filled','days_offset_missing',
    'api_optical_complete','old_optical_available','api_preferred',
    'NDVI_new','NDWI','MNDWI_new','SABI','WRI','NDTI','turbidity_ratio','green_red_ratio','nir_swir_ratio','swir_ratio',
    'ppt_pet_ratio','temp_range','runoff_ratio','hydro_stress','thermal_runoff',
    'pet_is_missing','ppt_is_missing','tmax_is_missing','tmin_is_missing','q_is_missing'
]
NO_OLD = [c for c in COMMON_ALL if c not in {'nir','green_old','swir16_old','swir22_old','NDMI','MNDWI','old_optical_available'}]
NO_QUALITY = [c for c in COMMON_ALL if c not in {'days_offset_missing','old_optical_available','api_preferred','pet_is_missing','ppt_is_missing','tmax_is_missing','tmin_is_missing','q_is_missing'}]


def fit_drp_specialist_oof(train_df, val_df, features, threshold):
    mask = train_df['api_optical_complete'] == 1
    tr = train_df.loc[mask].reset_index(drop=True)
    va = val_df.reset_index(drop=True).copy()
    X = tr[features].copy()
    X_val = va[features].copy()
    y = tr['Dissolved Reactive Phosphorus'].copy()
    groups = tr['spatial_group']
    w = sample_weights(tr)
    oof = np.zeros(len(tr))
    val_parts = []

    for tr_idx, va_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        w_tr = w[tr_idx]
        label = (y_tr >= threshold).astype(int)
        clf = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=5, l2_leaf_reg=3.0, random_strength=1.0, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
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
        oof[va_idx] = (1.0 - p_high) * blend(low_parts, 'Dissolved Reactive Phosphorus') + p_high * blend(high_parts, 'Dissolved Reactive Phosphorus')

        clf_val = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=5, l2_leaf_reg=3.0, random_strength=1.0, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
        clf_val.fit(X_tr, label, sample_weight=w_tr)
        p_high_val = clf_val.predict_proba(X_val)[:, 1]
        low_models, high_models = build_reg_models(), build_reg_models()
        for model in low_models.values():
            model.fit(X_tr.loc[low_mask], y_tr.loc[low_mask], sample_weight=w_tr[low_mask.values])
        for model in high_models.values():
            model.fit(X_tr.loc[high_mask], y_tr.loc[high_mask], sample_weight=w_tr[high_mask.values])
        low_val = blend({name: model.predict(X_val) for name, model in low_models.items()}, 'Dissolved Reactive Phosphorus')
        high_val = blend({name: model.predict(X_val) for name, model in high_models.items()}, 'Dissolved Reactive Phosphorus')
        val_parts.append((1.0 - p_high_val) * low_val + p_high_val * high_val)

    return {
        'train_mask': mask,
        'oof_pred': pd.Series(oof, index=tr.index),
        'val_pred': pd.Series(np.mean(val_parts, axis=0), index=va.index),
        'CV_R2': float(r2_score(y, oof)),
        'CV_RMSE': float(np.sqrt(mean_squared_error(y, oof))),
        'rows_used': len(tr),
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df, val_df, sub = load_frames()
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    train_df['Turbidity'] = train_df['turbidity_ratio']
    val_df['Turbidity'] = val_df['turbidity_ratio']

    ta_features = NO_QUALITY + ['Latitude', 'Longitude']
    ec_features = NO_QUALITY
    drp_features = NO_OLD + ['Latitude']

    ta = fit_target_chain(train_df, val_df, 'Total Alkalinity', ta_features, filter_complete=False)
    ta_oof = pd.Series(np.nan, index=train_df.index, name='ta_pred')
    ta_oof.loc[ta['train_mask']] = ta['oof_pred'].values
    ta_val = ta['val_pred'].rename('ta_pred')

    ec = fit_target_chain(train_df, val_df, 'Electrical Conductance', ec_features, extra_train={'ta_pred': ta_oof}, extra_val={'ta_pred': ta_val}, filter_complete=True)
    ec_oof = pd.Series(np.nan, index=train_df.index, name='ec_pred')
    ec_oof.loc[ec['train_mask']] = ec['oof_pred'].values
    ec_val = ec['val_pred'].rename('ec_pred')

    drp_chain = fit_target_chain(train_df, val_df, 'Dissolved Reactive Phosphorus', drp_features, extra_train={'ta_pred': ta_oof, 'ec_pred': ec_oof}, extra_val={'ta_pred': ta_val, 'ec_pred': ec_val}, filter_complete=True)

    specialist_features = [
        'blue','green','red','nir08','swir16','swir22','pet','ppt','tmax','tmin','q',
        'days_offset_filled','days_offset_missing','api_optical_complete','api_preferred',
        'month_sin','month_cos','NDVI_new','NDWI','MNDWI_new','SABI','WRI','NDTI','Turbidity',
        'ppt_pet_ratio','temp_range','runoff_ratio','Latitude'
    ]
    specialist_runs = {
        'thr96': fit_drp_specialist_oof(train_df, val_df, specialist_features, threshold=96.0),
        'thr128': fit_drp_specialist_oof(train_df, val_df, specialist_features, threshold=128.0),
    }

    y = train_df.loc[train_df['api_optical_complete'] == 1, 'Dissolved Reactive Phosphorus'].reset_index(drop=True)
    base_pred = drp_chain['oof_pred'].reset_index(drop=True).values
    rows = []
    for variant, run in specialist_runs.items():
        for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            pred = (1.0 - w) * base_pred + w * run['oof_pred'].values
            drp_r2 = float(r2_score(y, pred))
            target_scores = {
                'Total Alkalinity': float(ta['CV_R2']),
                'Electrical Conductance': float(ec['CV_R2']),
                'Dissolved Reactive Phosphorus': drp_r2,
            }
            avg_spatial = float(np.mean(list(target_scores.values())))
            est_public, _ = estimate_public_score(avg_spatial, 39, target_scores=target_scores)
            rows.append({'variant': variant, 'weight': w, 'drp_r2': drp_r2, 'avg_spatial': avg_spatial, 'estimated_public': est_public})
    scan = pd.DataFrame(rows).sort_values(['estimated_public', 'avg_spatial', 'drp_r2'], ascending=False).reset_index(drop=True)
    scan.to_csv(os.path.join(OUTPUT_DIR, 'drp_mix_scan.csv'), index=False)
    best = scan.iloc[0].to_dict()
    best_run = specialist_runs[best['variant']]
    w = float(best['weight'])

    submission = sub[MERGE_KEYS].copy()
    submission['Total Alkalinity'] = ta['val_pred'].values
    submission['Electrical Conductance'] = ec['val_pred'].values
    submission['Dissolved Reactive Phosphorus'] = (1.0 - w) * drp_chain['val_pred'].values + w * best_run['val_pred'].values
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_ensemble.csv'), index=False)

    metrics = pd.DataFrame([
        {'Parameter': 'Total Alkalinity', 'CV_R2': ta['CV_R2'], 'CV_RMSE': ta['CV_RMSE'], 'rows_used': ta['rows_used'], 'feature_count': len(ta_features)},
        {'Parameter': 'Electrical Conductance', 'CV_R2': ec['CV_R2'], 'CV_RMSE': ec['CV_RMSE'], 'rows_used': ec['rows_used'], 'feature_count': len(ec_features)},
        {'Parameter': 'Dissolved Reactive Phosphorus', 'CV_R2': float(best['drp_r2']), 'CV_RMSE': float(np.sqrt(mean_squared_error(y, (1.0 - w) * base_pred + w * best_run['oof_pred'].values))), 'rows_used': best_run['rows_used'], 'feature_count': len(drp_features)},
    ])
    metrics.to_csv(os.path.join(OUTPUT_DIR, 'cv_metrics.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'feature_summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('[Total Alkalinity]\n' + '\n'.join(ta_features) + '\n\n')
        fh.write('[Electrical Conductance]\n' + '\n'.join(ec_features) + '\n\n')
        fh.write('[Dissolved Reactive Phosphorus]\n' + '\n'.join(drp_features) + '\n\n')
        fh.write('[DRP specialist]\n')
        fh.write(f"best_variant={best['variant']} weight={best['weight']:.2f}\n")

    with open(os.path.join(OUTPUT_DIR, 'notes.txt'), 'w', encoding='utf-8') as fh:
        fh.write('TA/EC from no_quality_flags target-chain stackers.\\n')
        fh.write('DRP from no_old_optics target-chain plus specialist blend.\\n')
        fh.write(f"Best DRP mix: {best['variant']} weight={best['weight']:.2f} estimated_public={best['estimated_public']:.4f}\\n")

    print(metrics.to_string(index=False))
    print(scan.head(10).to_string(index=False))
    print('Saved to', OUTPUT_DIR)


if __name__ == '__main__':
    run()

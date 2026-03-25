import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = os.path.join(PROJECT_ROOT, "resources", "code", "general")
KNOWN_PUBLIC_ANCHORS = [
    {
        "run_name": "submission_unified_20260310_095952",
        "avg_spatial": 0.25169463467791714,
        "feature_count": 29,
        "public_score": 0.2710,
        "target_scores": {
            "Total Alkalinity": 0.399449,
            "Electrical Conductance": 0.232796,
            "Dissolved Reactive Phosphorus": 0.122839,
        },
    },
    {
        "run_name": "submission_optuna_20260310_101408",
        "avg_spatial": 0.2885096071239351,
        "feature_count": 41,
        "public_score": 0.2800,
        "target_scores": {
            "Total Alkalinity": 0.411702,
            "Electrical Conductance": 0.280833,
            "Dissolved Reactive Phosphorus": 0.172994,
        },
    },
    {
        "run_name": "submission_optuna_20260310_102302",
        "avg_spatial": 0.29317834055933595,
        "feature_count": 45,
        "public_score": 0.2530,
        "target_scores": {
            "Total Alkalinity": 0.419273,
            "Electrical Conductance": 0.283938,
            "Dissolved Reactive Phosphorus": 0.176325,
        },
    },
    {
        "run_name": "submission_online_conservative_20260310_125226/anchor90_ecsmall10.csv",
        "avg_spatial": 0.26637987280664327,
        "feature_count": 41,
        "public_score": 0.2760,
        "target_scores": None,
    },
    {
        "run_name": "submission_domain_shift_20260310_144608/base41_domain_weighted_filter_missing_postcal",
        "avg_spatial": 0.29589033916592494,
        "feature_count": 41,
        "public_score": 0.3370,
        "target_scores": {
            "Total Alkalinity": 0.399605,
            "Electrical Conductance": 0.304478,
            "Dissolved Reactive Phosphorus": 0.183589,
        },
    },
    {
        "run_name": "submission_anchor337_tunedmissing_blends_20260310_151555/tunedmissing_exact.csv",
        "avg_spatial": 0.30687166266136566,
        "feature_count": 41,
        "public_score": 0.3370,
        "target_scores": {
            "Total Alkalinity": 0.404856,
            "Electrical Conductance": 0.326231,
            "Dissolved Reactive Phosphorus": 0.189528,
        },
    },
    {
        "run_name": "submission_drp_occ_aet_20260310_175331/anchor_drp_blend_70",
        "avg_spatial": 0.31131593845712986,
        "feature_count": 43,
        "public_score": 0.3380,
        "target_scores": {
            "Total Alkalinity": 0.399605,
            "Electrical Conductance": 0.304478,
            "Dissolved Reactive Phosphorus": 0.229865,
        },
    },
    {
        "run_name": "submission_advanced_stacking",
        "avg_spatial": 0.30462210511075704,
        "feature_count": 46,
        "public_score": 0.3459,
        "target_scores": {
            "Total Alkalinity": 0.408516,
            "Electrical Conductance": 0.327985,
            "Dissolved Reactive Phosphorus": 0.177365,
        },
    },
    {
        "run_name": "submission_advanced_stacking_candidates/localfav_adv_taec_drp_anchor15.csv",
        "avg_spatial": 0.30592711318184286,
        "feature_count": 46,
        "public_score": 0.3529,
        "target_scores": {
            "Total Alkalinity": 0.408516,
            "Electrical Conductance": 0.327985,
            "Dissolved Reactive Phosphorus": 0.181280,
        },
    },
    {
        "run_name": "submission_advanced_online_hybrid",
        "avg_spatial": 0.3058866415552108,
        "feature_count": 46,
        "public_score": 0.3489,
        "target_scores": {
            "Total Alkalinity": 0.408516,
            "Electrical Conductance": 0.327985,
            "Dissolved Reactive Phosphorus": 0.181159,
        },
    },
    {
        "run_name": "submission_targetwise_hybrid_plus",
        "avg_spatial": 0.3141898794206862,
        "feature_count": 38,
        "public_score": 0.3559,
        "target_scores": {
            "Total Alkalinity": 0.414095,
            "Electrical Conductance": 0.344245,
            "Dissolved Reactive Phosphorus": 0.184229,
        },
    },
    {
        "run_name": "submission_targetwise_hybrid_ultra",
        "avg_spatial": 0.3165556989738068,
        "feature_count": 41,
        "public_score": 0.3590,
        "target_scores": {
            "Total Alkalinity": 0.414483,
            "Electrical Conductance": 0.345191,
            "Dissolved Reactive Phosphorus": 0.189993,
        },
    },
    {
        "run_name": "submission_targetwise_hybrid_ultra4",
        "avg_spatial": 0.31801472918860466,
        "feature_count": 41,
        "public_score": 0.3599,
        "target_scores": {
            "Total Alkalinity": 0.414483,
            "Electrical Conductance": 0.345185,
            "Dissolved Reactive Phosphorus": 0.194376,
        },
    },
]


_KNOWN_SUBMISSION_CACHE = {}


def _anchor_submission_path(run_name: str):
    base = Path(PROJECT_ROOT) / "output"
    normalized = run_name.replace("\\", "/")
    if normalized.endswith(".csv"):
        return base / Path(normalized)
    return base / normalized / "submission_ensemble.csv"


def _load_submission_frame(path_like):
    path = Path(path_like)
    if not path.exists():
        return None
    key = str(path.resolve())
    if key not in _KNOWN_SUBMISSION_CACHE:
        _KNOWN_SUBMISSION_CACHE[key] = pd.read_csv(path)
    return _KNOWN_SUBMISSION_CACHE[key]


def _submission_distance(candidate_df, anchor_df):
    total = 0.0
    for target, weight in [("Total Alkalinity", 1.0), ("Electrical Conductance", 1.0), ("Dissolved Reactive Phosphorus", 2.0)]:
        cand = candidate_df[target]
        anch = anchor_df[target]
        scale = float(anch.std()) + 1e-6
        total += weight * float((cand - anch).abs().mean() / scale)
        total += 0.30 * weight * abs(float(cand.mean() - anch.mean())) / scale
        total += 0.30 * weight * abs(float(cand.std() - anch.std())) / scale
    return float(total)


def _estimate_from_submission_neighbors(submission_path):
    candidate_df = _load_submission_frame(submission_path)
    if candidate_df is None:
        return None, []

    rows = []
    for anchor in KNOWN_PUBLIC_ANCHORS:
        anchor_path = _anchor_submission_path(anchor["run_name"])
        anchor_df = _load_submission_frame(anchor_path)
        if anchor_df is None:
            continue
        try:
            if candidate_df[TARGET_COLS].equals(anchor_df[TARGET_COLS]):
                return float(anchor["public_score"]), [{**anchor, "submission_distance": 0.0}]
        except Exception:
            pass
        dist = _submission_distance(candidate_df, anchor_df)
        rows.append({**anchor, "submission_distance": dist})

    if not rows:
        return None, []

    weights = np.array([1.0 / (row["submission_distance"] + 1e-6) for row in rows], dtype=float)
    score = float(np.average([row["public_score"] for row in rows], weights=weights))
    return score, rows


def load_and_merge_data():
    wq = pd.read_csv(os.path.join(DATA_DIR, 'water_quality_training_dataset.csv'))
    ls_api_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'landsat_api_training.csv')
    ls_api = pd.read_csv(ls_api_path).set_index('Index')
    ls_old = pd.read_csv(os.path.join(DATA_DIR, 'landsat_features_training.csv'))
    tc = pd.read_csv(os.path.join(DATA_DIR, 'terraclimate_features_training.csv'))

    df = wq.join(ls_api, how='left')
    df = df.merge(tc, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    df = df.merge(ls_old, on=['Latitude', 'Longitude', 'Sample Date'], suffixes=('', '_old'), how='left')

    df['NDVI_new'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 1e-8)
    df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + 1e-8)
    df['MNDWI_new'] = (df['green'] - df['swir16']) / (df['green'] + df['swir16'] + 1e-8)
    df['SABI'] = (df['nir08'] - df['red']) / (df['blue'] + df['green'] + 1e-8)
    df['WRI'] = (df['green'] + df['red']) / (df['nir08'] + df['swir16'] + 1e-8)

    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    df['month'] = df['Sample Date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    lat_bin = np.round(df['Latitude'], 1)
    lon_bin = np.round(df['Longitude'], 1)
    df['spatial_group'] = lat_bin.astype(str) + '_' + lon_bin.astype(str)
    return df


def _build_model(model_cls: str):
    if model_cls == 'RF':
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        missing=np.nan,
    )


def _read_feature_count(run_dir: str):
    for name in ['cv_metrics.csv', 'metrics.csv']:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'feature_count' in df.columns:
            vals = pd.to_numeric(df['feature_count'], errors='coerce').dropna()
            if len(vals):
                return int(round(float(vals.mean())))

    feature_list = os.path.join(run_dir, 'feature_list.txt')
    if os.path.exists(feature_list):
        with open(feature_list, 'r', encoding='utf-8') as fh:
            return len([line.strip() for line in fh if line.strip()])

    feature_summary = os.path.join(run_dir, 'feature_summary.txt')
    if os.path.exists(feature_summary):
        with open(feature_summary, 'r', encoding='utf-8') as fh:
            lines = [line.strip() for line in fh if line.strip()]
        counts, current = [], 0
        active = False
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                if active and current:
                    counts.append(current)
                label = line[1:-1]
                active = label in TARGET_COLS
                current = 0
            elif active and '<-' not in line:
                current += 1
        if current:
            counts.append(current)
        if counts:
            return int(round(float(np.mean(counts))))

    feature_map = os.path.join(run_dir, 'feature_map.txt')
    if os.path.exists(feature_map):
        with open(feature_map, 'r', encoding='utf-8') as fh:
            lines = [line.strip() for line in fh if line.strip()]
        counts, current = [], 0
        active = False
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                if active and current:
                    counts.append(current)
                label = line[1:-1]
                active = label in TARGET_COLS
                current = 0
            elif active:
                current += 1
        if current:
            counts.append(current)
        if counts:
            return int(round(float(np.mean(counts))))
    return None


def _read_target_scores(run_dir: str):
    for name in ['cv_metrics.csv', 'metrics.csv']:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if {'Parameter', 'CV_R2'}.issubset(df.columns):
            return {row['Parameter']: float(row['CV_R2']) for _, row in df.iterrows()}
        if {'target', 'postcal_r2'}.issubset(df.columns):
            return {row['target']: float(row['postcal_r2']) for _, row in df.iterrows()}
        if {'target', 'base_r2'}.issubset(df.columns):
            return {row['target']: float(row['base_r2']) for _, row in df.iterrows()}
    return None


def _infer_anchor_feature_count(run_dir: str):
    normalized = run_dir.replace('\\', '/')
    for row in KNOWN_PUBLIC_ANCHORS:
        if normalized.endswith(row['run_name'].replace('\\', '/')):
            return row['feature_count']
    return None


def estimate_public_score(avg_spatial: float, feature_count: int, target_scores=None, submission_path=None):
    submission_estimate, submission_neighbors = (None, [])
    if submission_path:
        submission_estimate, submission_neighbors = _estimate_from_submission_neighbors(submission_path)

    for row in KNOWN_PUBLIC_ANCHORS:
        exact_profile = True
        if target_scores and row.get('target_scores'):
            for target in TARGET_COLS:
                if abs(target_scores.get(target, np.nan) - row['target_scores'].get(target, np.nan)) > 1e-4:
                    exact_profile = False
                    break
        if abs(avg_spatial - row['avg_spatial']) < 1e-4 and feature_count == row['feature_count'] and exact_profile:
            if submission_path:
                anchor_path = _anchor_submission_path(row['run_name'])
                cand_df = _load_submission_frame(submission_path)
                anchor_df = _load_submission_frame(anchor_path)
                if cand_df is not None and anchor_df is not None and cand_df[TARGET_COLS].equals(anchor_df[TARGET_COLS]):
                    return row['public_score'], KNOWN_PUBLIC_ANCHORS
            elif row['run_name'] == 'submission_advanced_online_hybrid':
                continue
            else:
                return row['public_score'], KNOWN_PUBLIC_ANCHORS

    avg_scale = 0.018
    feature_scale = 10.0
    target_scale = {
        'Total Alkalinity': 0.03,
        'Electrical Conductance': 0.03,
        'Dissolved Reactive Phosphorus': 0.025,
    }

    weights = []
    for row in KNOWN_PUBLIC_ANCHORS:
        dist_terms = [((avg_spatial - row['avg_spatial']) / avg_scale) ** 2]
        dist_terms.append(((feature_count - row['feature_count']) / feature_scale) ** 2)
        anchor_scores = row.get('target_scores')
        if target_scores and anchor_scores:
            for target in TARGET_COLS:
                dist_terms.append(((target_scores[target] - anchor_scores[target]) / target_scale[target]) ** 2)
        dist = float(np.sqrt(sum(dist_terms)))
        weights.append(1.0 / (dist + 1e-6))

    weights_arr = np.array(weights, dtype=float)
    if target_scores:
        for idx, row in enumerate(KNOWN_PUBLIC_ANCHORS):
            if row.get('target_scores') and row['public_score'] >= 0.34:
                weights_arr[idx] *= 1.15
    weighted_score = float(np.average([row['public_score'] for row in KNOWN_PUBLIC_ANCHORS], weights=weights_arr))
    max_anchor_features = max(row['feature_count'] for row in KNOWN_PUBLIC_ANCHORS)
    complexity_penalty = max(feature_count - max_anchor_features, 0) * 0.0015

    profile_bonus = 0.0
    if target_scores:
        drp = target_scores.get('Dissolved Reactive Phosphorus', 0.0)
        ec = target_scores.get('Electrical Conductance', 0.0)
        ta = target_scores.get('Total Alkalinity', 0.0)
        if drp >= 0.17:
            profile_bonus += 0.006
        if ec >= 0.32:
            profile_bonus += 0.006
        if ta >= 0.40:
            profile_bonus += 0.004
        if drp < 0.10:
            profile_bonus -= 0.010

    metric_estimate = float(np.clip(weighted_score - complexity_penalty + profile_bonus, 0.0, 1.0))
    if submission_estimate is None:
        return metric_estimate, KNOWN_PUBLIC_ANCHORS

    blended = 0.45 * metric_estimate + 0.55 * submission_estimate
    return float(np.clip(blended, 0.0, 1.0)), submission_neighbors or KNOWN_PUBLIC_ANCHORS


def evaluate_model(df, name, feature_cols, model_cls='RF', use_imputer=False):
    print('=' * 60)
    print(f'  Evaluating Model: {name}')
    print('=' * 60)
    print(f'Features ({len(feature_cols)}): {feature_cols}')

    X = df[feature_cols].copy()
    groups = df['spatial_group'].copy()
    spatial_r2_list = []
    random_r2_list = []
    target_scores = {}

    for target in TARGET_COLS:
        y = df[target].copy()

        gkf = GroupKFold(n_splits=5)
        oof_spatial = np.zeros(len(df))
        for tr_idx, va_idx in gkf.split(X, y, groups):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            if use_imputer:
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_va = imputer.transform(X_va)
            model = _build_model(model_cls)
            model.fit(X_tr, y_tr)
            oof_spatial[va_idx] = model.predict(X_va)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_random = np.zeros(len(df))
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            if use_imputer:
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_va = imputer.transform(X_va)
            model = _build_model(model_cls)
            model.fit(X_tr, y_tr)
            oof_random[va_idx] = model.predict(X_va)

        spatial_r2 = max(0.0, r2_score(y, oof_spatial))
        random_r2 = max(0.0, r2_score(y, oof_random))
        spatial_r2_list.append(spatial_r2)
        random_r2_list.append(random_r2)
        target_scores[target] = spatial_r2

        print(f'\n[{target}]')
        print(f'  -> Spatial CV R2 : {spatial_r2:.4f}')
        print(f'  -> Random CV R2  : {random_r2:.4f}')

    avg_spatial = float(np.mean(spatial_r2_list))
    avg_random = float(np.mean(random_r2_list))
    estimated_lb = (avg_spatial + avg_random) / 2.0
    calibrated_public, anchors = estimate_public_score(avg_spatial, len(feature_cols), target_scores=target_scores)

    print('-' * 60)
    print(f'Overall Spatial CV : {avg_spatial:.4f}')
    print(f'Overall Random CV  : {avg_random:.4f}')
    print(f'ESTIMATED LB SCORE : {estimated_lb:.4f}')
    print(f'Calibrated Public  : {calibrated_public:.4f}')
    print(f'Calibration Basis  : {len(anchors)} known runs | target-profile anchor blend')
    print('=' * 60 + '\n')
    return estimated_lb


def score_submission_dir(run_dir: str):
    target_scores = _read_target_scores(run_dir)
    if not target_scores:
        raise FileNotFoundError(f'Could not find usable target metrics under: {run_dir}')

    avg_spatial = float(np.mean([target_scores[target] for target in TARGET_COLS if target in target_scores]))
    feature_count = _read_feature_count(run_dir)
    if feature_count is None:
        feature_count = _infer_anchor_feature_count(run_dir)
    if feature_count is None:
        feature_count = -1

    submission_path = os.path.join(run_dir, 'submission_ensemble.csv')
    if not os.path.exists(submission_path):
        csvs = [name for name in os.listdir(run_dir) if name.lower().endswith('.csv') and name not in {'cv_metrics.csv', 'metrics.csv', 'drp_mix_scan.csv', 'candidate_summary.csv'}]
        if csvs:
            submission_path = os.path.join(run_dir, csvs[0])
        else:
            submission_path = None

    calibrated_public, anchors = estimate_public_score(avg_spatial, feature_count, target_scores=target_scores, submission_path=submission_path)

    print('=' * 60)
    print(f'  Submission Diagnostics: {os.path.basename(run_dir)}')
    print('=' * 60)
    for target in TARGET_COLS:
        if target in target_scores:
            print(f'{target:<28}: {target_scores[target]:.4f}')
    print(f'Average Spatial CV            : {avg_spatial:.4f}')
    print(f'Feature Count                 : {feature_count}')
    print(f'Calibrated Public             : {calibrated_public:.4f}')
    print('Known Anchors:')
    for row in anchors:
        print(f"  - {row['run_name']}: spatial={row['avg_spatial']:.4f}, features={row['feature_count']}, public={row['public_score']:.4f}")
    print('=' * 60 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EY Challenge Local Evaluator')
    parser.add_argument('--model', type=str, choices=['benchmark', 'optimized', 'all'], default=None)
    parser.add_argument('--submission-dir', type=str, default=None, help='Existing submission directory to score from saved cv_metrics.csv/metrics.csv')
    args = parser.parse_args()

    if args.submission_dir:
        score_submission_dir(os.path.join(PROJECT_ROOT, args.submission_dir) if not os.path.isabs(args.submission_dir) else args.submission_dir)
        raise SystemExit(0)

    if args.model is None:
        args.model = 'all'

    print('Loading data...')
    df = load_and_merge_data()
    if args.model in ['benchmark', 'all']:
        evaluate_model(
            df,
            name='1. Original Benchmark (RandomForest, Median Fill, 4 old feats)',
            feature_cols=['swir22_old', 'NDMI', 'MNDWI', 'pet'],
            model_cls='RF',
            use_imputer=True,
        )
    if args.model in ['optimized', 'all']:
        evaluate_model(
            df,
            name='2. Phase H Cleansed (XGBoost, Native NaN, 12 Pure Spectral Features)',
            feature_cols=['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'pet', 'NDVI_new', 'NDWI', 'MNDWI_new', 'SABI', 'WRI'],
            model_cls='XGB',
            use_imputer=False,
        )

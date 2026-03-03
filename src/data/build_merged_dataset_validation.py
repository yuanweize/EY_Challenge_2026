#!/usr/bin/env python3
"""
build_merged_dataset.py
=======================
EY Open Science Data Challenge 2026 — 数据工程管线

功能：
  1. 加载 water_quality_validation_dataset.csv、两份 Landsat 源（Official + API）、
     terraclimate_features_validation.csv（已有 pet，动态拉取 ppt/tmax/tmin/q）。
  2. 合并 Official Landsat 与 API Landsat：优先 API 数据（更纯净的云掩码），
     API 缺失时回退到 Official 数据，最终仅 ~2.6% 行双源均缺（保留 NaN）。
  3. TerraClimate：检查 CSV 是否缺少 ppt/tmax/tmin/q，若缺则通过 Planetary Computer
     Zarr 向量化批量拉取。
  4. 三表拼接 + 基础清洗：
     - 类型转换（float64 + datetime）
     - Landsat NaN **不做**中位数填充 → XGBoost/LightGBM/CatBoost 原生处理 NaN
       （开发日志 Phase 4-7 已证实中位数填充会制造虚假光谱信号，降低泛化能力）
  5. 输出 data/merged_training_data_clean.csv。

用法：
  python -m src.data.build_merged_dataset          # 正常运行
  python -m src.data.build_merged_dataset --skip-api  # 跳过 API，仅合并已有数据
"""

import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GENERAL_DIR = os.path.join(ROOT, "resources", "code", "general")
DATA_DIR = os.path.join(ROOT, "resources", "data")
OUTPUT_DIR = os.path.join(ROOT, "data")

WQ_FILE = os.path.join(DATA_DIR, "submission_template.csv")
LANDSAT_OFFICIAL = os.path.join(GENERAL_DIR, "landsat_features_validation.csv")
LANDSAT_API = os.path.join(ROOT, "data", "processed", "landsat_api_validation.csv")
TERRA_FILE = os.path.join(GENERAL_DIR, "terraclimate_features_validation.csv")
MERGED_OUT = os.path.join(OUTPUT_DIR, "merged_validation_data_clean.csv")

TERRA_VARS_NEEDED = ["ppt", "tmax", "tmin", "q"]  # q = runoff in TerraClimate


# ── 1. TerraClimate API 拉取（向量化高速版）─────────────────────────────────
def load_terraclimate_dataset():
    """从 Planetary Computer 打开 TerraClimate Zarr 数据集。"""
    import pystac_client
    import planetary_computer as pc
    import xarray as xr

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
    return ds


def fetch_terra_vars_vectorized(wq_df: pd.DataFrame, terra_df: pd.DataFrame, variables: list):
    """
    高速向量化方案：
    1. 一次性从 Zarr 中加载南非区域所有所需变量 (sel + where)。
    2. 用 xarray 的 sel(method='nearest') 做空间最近邻。
    3. 用 pandas 向量化做时间最近月匹配。
    """
    import xarray as xr

    print("  → 打开 TerraClimate Zarr …")
    ds = load_terraclimate_dataset()

    # 时间、空间裁剪（一次完成）
    # 注意：TerraClimate lat 坐标从正到负（降序），所以 slice 要大→小
    ds = ds[variables].sel(
        time=slice("2011-01-01", "2015-12-31"),
        lat=slice(-21.72, -35.18),
        lon=slice(14.97, 32.79),
    )
    print(f"  → 裁剪后 ds shape: { {k: ds[k].shape for k in variables} }")

    # 准备采样点日期
    dates = pd.to_datetime(wq_df["Sample Date"], dayfirst=True, errors="coerce")
    lats = wq_df["Latitude"].values
    lons = wq_df["Longitude"].values

    # 对每个采样点，找最近的网格月份
    # TerraClimate 是月度数据，每月1日。将采样日期 floor 到月初即可。
    month_starts = dates.dt.to_period("M").dt.to_timestamp()

    # 使用 xarray sel(method='nearest') 进行批量空间+时间查询
    # 需要构造 DataArray 坐标
    lat_da = xr.DataArray(lats, dims="sample")
    lon_da = xr.DataArray(lons, dims="sample")
    time_da = xr.DataArray(month_starts.values, dims="sample")

    print("  → 批量 sel(method='nearest') 查询中…")
    result = ds.sel(lat=lat_da, lon=lon_da, time=time_da, method="nearest")

    for var in variables:
        vals = result[var].values
        terra_df[var] = vals
        n_nan = np.isnan(vals).sum() if np.issubdtype(vals.dtype, np.floating) else 0
        print(f"    ✅ {var}: 提取完成 (NaN: {n_nan})")

    return terra_df


def fetch_missing_terra_vars(wq_df: pd.DataFrame, terra_df: pd.DataFrame) -> pd.DataFrame:
    """检查并拉取缺失的 TerraClimate 变量，合并到 terra_df 返回。"""
    missing = [v for v in TERRA_VARS_NEEDED if v not in terra_df.columns]
    if not missing:
        print("✅ TerraClimate CSV 已包含所有所需变量，无需调用 API。")
        return terra_df

    print(f"⚠️  缺少变量: {missing}，正在通过 Planetary Computer API 拉取…")
    terra_df = fetch_terra_vars_vectorized(wq_df, terra_df, missing)

    # 保存更新后的 terraclimate CSV，方便下次直接使用
    terra_df.to_csv(TERRA_FILE, index=False)
    print(f"✅ 已更新 {TERRA_FILE}，新增列: {missing}")
    return terra_df


# ── 2. Landsat 双源合并 (API-first, Official-fallback) ─────────────────────
def merge_landsat_sources(wq_df: pd.DataFrame, official_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并两个 Landsat 数据源，最大化覆盖率：
    - 优先使用 API 数据（Phase 4 纯净云掩码，精确时间匹配）
    - API 缺失时回退到 Official 数据
    - 双源均缺失的行保留 NaN（仅 ~2.6%）
    
    统一输出列名为 ensemble_model.py 所用的 API 列名：
    blue, green, red, nir08, swir16, swir22
    """
    # Official 列名映射到 API 列名
    OFFICIAL_TO_API = {
        "nir": "nir08",
        "green": "green",
        "swir16": "swir16",
        "swir22": "swir22",
    }
    BAND_COLS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    
    # 和 wq_df 行索引对齐
    api_aligned = api_df.set_index("Index").reindex(range(len(wq_df)))
    
    # 用 API 的数据初始化
    result = pd.DataFrame(index=range(len(wq_df)))
    for col in BAND_COLS:
        if col in api_aligned.columns:
            result[col] = api_aligned[col].values
        else:
            result[col] = np.nan
    
    # 对于 API 缺失的行，用 Official 数据回填
    api_nan_mask = result["green"].isnull()
    n_api_nan = api_nan_mask.sum()
    
    filled_count = 0
    for api_col in BAND_COLS:
        # 找到 Official 中对应的列
        off_col = None
        for ok, av in OFFICIAL_TO_API.items():
            if av == api_col:
                off_col = ok
                break
        if off_col is None:
            # blue, red 在 Official 中不存在
            continue
        if off_col not in official_df.columns:
            continue
        
        # 仅在 API 缺失 & Official 有值时回填
        fill_mask = api_nan_mask & official_df[off_col].notna()
        result.loc[fill_mask, api_col] = official_df.loc[fill_mask, off_col].values
    
    # 同时用 Official 的 NDMI/MNDWI 回填（如果 API 没有这些列）
    for idx_col in ["NDMI", "MNDWI"]:
        if idx_col in official_df.columns:
            result[idx_col] = np.nan
            # Official 有值的行
            off_valid = official_df[idx_col].notna()
            # API 有数据的行可以重新算，但 Official-only 行需要直接用 Official 的值
            result.loc[off_valid, idx_col] = official_df.loc[off_valid, idx_col].values
    
    final_nan = result["green"].isnull().sum()
    recovered = n_api_nan - final_nan
    print(f"  Landsat 双源合并: API={len(wq_df)-n_api_nan}, Official回填={recovered}, 不可恢复NaN={final_nan} ({final_nan/len(wq_df)*100:.1f}%)")
    
    return result


# ── 3. 全量合并 + 清洗 ────────────────────────────────────────────────────────
def merge_and_clean(wq_df: pd.DataFrame, landsat_merged: pd.DataFrame, terra_df: pd.DataFrame) -> pd.DataFrame:
    """
    拼接 水质标签 + Landsat(双源合并) + TerraClimate，做基础清洗。
    
    关键决策（来自开发日志 Phase 4-7 的经验教训）：
    - Landsat NaN 保留不填充：XGBoost/LightGBM/CatBoost 均原生支持 NaN 路由，
      中位数填充会制造虚假"典型"光谱信号，对云遮蔽行不代表真实地表。
    - 目标变量 NaN 保留不填充：避免引入标签偏差。
    - TerraClimate 完整无缺（API 向量化拉取保证）。
    """
    assert len(wq_df) == len(landsat_merged) == len(terra_df), (
        f"行数不一致: wq={len(wq_df)}, landsat={len(landsat_merged)}, terra={len(terra_df)}"
    )

    # 验证 terra key 列一致
    assert np.allclose(wq_df["Latitude"].values, terra_df["Latitude"].values, atol=1e-4), \
        "terra Latitude 不对齐"
    assert np.allclose(wq_df["Longitude"].values, terra_df["Longitude"].values, atol=1e-4), \
        "terra Longitude 不对齐"

    # --- 拼接 ---
    terra_feats = terra_df.drop(columns=["Latitude", "Longitude", "Sample Date"], errors="ignore")

    merged = pd.concat([
        wq_df.reset_index(drop=True),
        landsat_merged.reset_index(drop=True),
        terra_feats.reset_index(drop=True),
    ], axis=1)

    print(f"合并后 shape: {merged.shape}")

    # --- 类型转换 ---
    merged["Sample Date"] = pd.to_datetime(merged["Sample Date"], dayfirst=True, errors="coerce")

    target_cols = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    # Rename the long DRP column if needed
    for c in merged.columns:
        if "Dissolved" in c and "Phosphorus" in c and c != "Dissolved Reactive Phosphorus":
            merged.rename(columns={c: "Dissolved Reactive Phosphorus"}, inplace=True)

    numeric_cols = [c for c in merged.columns if c not in ["Sample Date"]]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # --- 缺失值报告（不做中位数填充！）---
    print("\n缺失值统计：")
    missing = merged.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  无缺失")

    # 报告各类缺失
    feature_cols = [c for c in merged.columns if c not in target_cols + ["Sample Date", "Latitude", "Longitude"]]
    feat_nan = sum(merged[c].isnull().any() for c in feature_cols)
    if feat_nan > 0:
        print(f"\n  ℹ️  {feat_nan} 个特征列含 NaN → 保留原值，由树模型原生 NaN 路由处理")
        print("     (开发日志 Phase 4-7 已证实中位数填充会制造虚假信号，降低泛化能力)")

    for c in target_cols:
        if c in merged.columns and merged[c].isnull().any():
            n_miss = merged[c].isnull().sum()
            print(f"  ⚠ 目标列 {c} 有 {n_miss} 个 NaN（保留不填充）")

    print(f"\n最终 shape: {merged.shape}")
    total_nan = merged[feature_cols].isnull().sum().sum()
    total_cells = len(merged) * len(feature_cols)
    print(f"特征 NaN: {total_nan}/{total_cells} ({total_nan/total_cells*100:.2f}%)")
    return merged


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true", help="跳过 TerraClimate API 拉取")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载
    print("📂 加载数据…")
    wq_df = pd.read_csv(WQ_FILE)
    landsat_official = pd.read_csv(LANDSAT_OFFICIAL)
    landsat_api = pd.read_csv(LANDSAT_API)
    terra_df = pd.read_csv(TERRA_FILE)

    print(f"  water_quality    : {wq_df.shape}")
    print(f"  landsat_official : {landsat_official.shape} (NaN: {landsat_official['nir'].isnull().sum()})")
    print(f"  landsat_api      : {landsat_api.shape} (NaN: {landsat_api['green'].isnull().sum()})")
    print(f"  terraclimate     : {terra_df.shape}")

    # 2. 检查 & 拉取缺失气候变量
    if not args.skip_api:
        terra_df = fetch_missing_terra_vars(wq_df, terra_df)
    else:
        print("⏭  跳过 API 拉取（--skip-api）")

    # 3. Landsat 双源合并
    print("\n📡 Landsat 双源合并 (API-first, Official-fallback)…")
    landsat_merged = merge_landsat_sources(wq_df, landsat_official, landsat_api)

    # 4. 全量合并 & 清洗
    print("\n🔧 合并 & 清洗…")
    merged = merge_and_clean(wq_df, landsat_merged, terra_df)

    # 4. 输出
    merged.to_csv(MERGED_OUT, index=False)
    print(f"\n💾 已保存: {MERGED_OUT}")
    print("✅ 数据工程管线完成。")


if __name__ == "__main__":
    main()

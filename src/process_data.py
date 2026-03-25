from pathlib import Path
import logging
from collections import OrderedDict

# for manipulating data
import numpy as np
import pandas as pd
import math
from typing import Callable
import copy
import re
from pandas.api.types import is_string_dtype, is_numeric_dtype

# for Machine Learning
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

# for visualization
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def log_transform_targets(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = np.log1p(df[c])
    return df

def clip_outliers(df, col, lower_q=0.01, upper_q=0.99):
    low = df[col].quantile(lower_q)
    up  = df[col].quantile(upper_q)
    df[col] = df[col].clip(low, up)
    return df

NO_IMPUTE_COLS = {"blue","green","red","nir08","swir16","swir22","NDMI","MNDWI"}

def process_df_keep_nan(df, y_fields=None, skip_flds=None):
    df = df.copy().reset_index(drop=True)
    if skip_flds is None: skip_flds = []
    else: skip_flds = list(skip_flds)

    if y_fields is None:
        y = None
    else:
        y_fields = list(y_fields)
        y = df[y_fields].values
        skip_flds.extend(y_fields)

    df = df.drop(columns=skip_flds)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col + "_na"] = df[col].isna().astype(int)
                # Landsat 类列：保留 NaN
                if col in NO_IMPUTE_COLS:
                    continue
                # 其他数值列：中位数填充
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Missing")

    df = pd.get_dummies(df, dummy_na=False)
    return df, y


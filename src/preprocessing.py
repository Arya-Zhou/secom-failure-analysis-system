"""预处理管线：删全空列、缺失值填充、标准化，对应高维稀疏且大量缺失的数据特点。"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def drop_all_nan_columns(X: pd.DataFrame):
    """删除全为 NaN 的列，返回 (清洗后的 X, 被删列名列表)。"""
    all_nan = X.columns[X.isnull().all()].tolist()
    if all_nan:
        logger.info("删除 %d 个全空特征: %s%s",
                    len(all_nan), all_nan[:5], " ..." if len(all_nan) > 5 else "")
    return X.drop(columns=all_nan), all_nan


def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """按配置构建填充与标准化管线；删全空列不在其中，需在 fit 前单独执行。"""
    steps = [("impute", SimpleImputer(strategy=cfg["preprocessing"]["impute_strategy"]))]
    if cfg["preprocessing"]["scale"]:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)

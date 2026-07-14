"""预处理管线 —— 对应数据特点：高维稀疏 + 大量缺失。

两种模式（config: preprocessing.fit_on_train_only）：
    false = notebook 忠实模式：填充/标准化在全量数据上 fit_transform，
            与 secom.ipynb 的 feature_selection_analysis 完全一致，
            用于回归对齐基线（原 notebook 存在轻微泄漏，如实保留并注明）。
    true  = 生产模式：Pipeline 只在训练集 fit、测试集 transform，防数据泄漏
            （不平衡+缺失场景的常见面试考点）。
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def drop_all_nan_columns(X: pd.DataFrame):
    """删除全为 NaN 的列，返回 (清洗后的 X, 被删列名列表)。

    对应 notebook: feature_selection_analysis 第 1 步。
    """
    all_nan = X.columns[X.isnull().all()].tolist()
    if all_nan:
        logger.info("删除 %d 个全空特征: %s%s",
                    len(all_nan), all_nan[:5], " ..." if len(all_nan) > 5 else "")
    return X.drop(columns=all_nan), all_nan


def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """按配置构建填充+标准化管线（不含删空列，那步在 fit 前单独做）。"""
    steps = [("impute", SimpleImputer(strategy=cfg["preprocessing"]["impute_strategy"]))]
    if cfg["preprocessing"]["scale"]:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def preprocess_full(X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """notebook 忠实模式：删空列后在全量数据上 impute + scale。

    对应 notebook: 中位数填充 -> (残留 NaN 兜底填 0) -> StandardScaler。
    返回列名/索引保持不变的 DataFrame（X_scaled）。
    """
    pipe = build_preprocess_pipeline(cfg)
    arr = pipe.fit_transform(X)
    X_scaled = pd.DataFrame(arr, columns=X.columns, index=X.index)
    if X_scaled.isnull().any().any():  # notebook 的兜底逻辑，正常不会触发
        logger.warning("填充后仍存在 NaN，以 0 兜底")
        X_scaled = X_scaled.fillna(0)
    logger.info("预处理完成(全量 fit): %s", X_scaled.shape)
    return X_scaled

"""数据加载与标签转换：读入 591 维空格分隔的匿名特征与带时间戳的标签，标签由 1/-1 转为 1/0。"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_secom(features_path: str, labels_path: str, timestamp_format: str):
    """加载 SECOM 特征、标签、时间戳。"""
    feature_names = [f"F{i + 1:03d}" for i in range(591)]

    X = pd.read_csv(
        features_path, header=None, sep=" ",
        na_values=["NaN", "nan", "NA"], names=feature_names,
    )

    labels_df = pd.read_csv(
        labels_path, header=None, sep=" ", names=["label", "timestamp"],
    )
    timestamps = pd.to_datetime(
        labels_df["timestamp"], format=timestamp_format, errors="coerce",
    )
    # 原始: 1=失败, -1=通过  ->  标准: 1=失败, 0=通过
    y = labels_df["label"].map({1: 1, -1: 0})

    logger.info(
        "数据加载: X=%s, 通过=%d, 失败=%d (失败率 %.2f%%)",
        X.shape, int((y == 0).sum()), int((y == 1).sum()),
        (y == 1).mean() * 100,
    )
    return X, y, timestamps

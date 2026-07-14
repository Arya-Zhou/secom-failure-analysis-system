"""多方法综合特征选择 —— 搬运自 secom.ipynb 的 feature_selection_analysis。

四方法各自对全部特征排名（1 = 最好），取平均排名后选前 N 个：
    f_test        SelectKBest 的 f_classif F 值，降序排名
    mutual_info   互信息得分，降序排名（注意：原 notebook 未设种子，
                  本实现传入全局 random_state 保证可复现；若因此与
                  基线特征集有出入，可用 config 的 override_features_path 锁定）
    rfe           基于 liblinear 逻辑回归的递归特征消除，ranking_ 直接作排名
                  （被选中的 N 个特征 ranking_ 均为 1，存在并列，属原逻辑）
    random_forest 随机森林 feature_importances_，降序排名

quick 模式跳过最耗时的 RFE（约数百次逻辑回归拟合），用于冒烟调试。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _rank_desc(scores: np.ndarray, index) -> pd.Series:
    """按得分降序给出 1 起始的排名；NaN 得分（如常数列）排到最后。"""
    s = pd.Series(scores, index=index)
    r = s.rank(ascending=False, method="first")
    return r.fillna(float(len(s)))


def select_features(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    random_state: int,
    quick: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    """返回 (最终选中的特征名列表, 各方法排名明细表)。

    排名明细表列: 各方法排名 / 平均排名 / 综合排名，与 notebook 的
    selection_results 结构对应，便于导出与人工核对。
    """
    fs_cfg = cfg["feature_selection"]
    n = min(fs_cfg["n_features_to_select"], X_scaled.shape[1])
    methods = list(fs_cfg["methods"])
    if quick and "rfe" in methods:
        methods.remove("rfe")
        logger.info("quick 模式: 跳过 RFE，使用其余 %d 种方法", len(methods))

    ranks: dict[str, pd.Series] = {}

    if "f_test" in methods:
        f_scores, _ = f_classif(X_scaled, y)
        ranks["F检验排名"] = _rank_desc(f_scores, X_scaled.columns)
        logger.info("F 检验完成")

    if "mutual_info" in methods:
        mi = mutual_info_classif(X_scaled, y, random_state=random_state)
        ranks["互信息排名"] = _rank_desc(mi, X_scaled.columns)
        logger.info("互信息完成")

    if "rfe" in methods:
        lr = LogisticRegression(
            class_weight="balanced", random_state=random_state,
            max_iter=1000, solver="liblinear",
        )
        rfe = RFE(estimator=lr, n_features_to_select=n)
        rfe.fit(X_scaled, y)
        ranks["RFE排名"] = pd.Series(
            rfe.ranking_.astype(float), index=X_scaled.columns,
        )
        logger.info("RFE 完成")

    if "random_forest" in methods:
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        rf.fit(X_scaled, y)
        ranks["随机森林排名"] = _rank_desc(rf.feature_importances_, X_scaled.columns)
        logger.info("随机森林重要性完成")

    if not ranks:
        raise ValueError("feature_selection.methods 为空，至少需要一种方法")

    detail = pd.DataFrame(ranks)
    detail["平均排名"] = detail.mean(axis=1)
    detail["综合排名"] = detail["平均排名"].rank()

    final = detail["平均排名"].sort_values(kind="stable").head(n).index.tolist()
    logger.info("特征选择: 方法=%s, %d -> %d 个", methods, X_scaled.shape[1], n)
    return final, detail


def load_feature_override(path: str | Path) -> list[str]:
    """加载已保存的特征列表文件（notebook 输出格式，行如 '1. F026'）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"override_features_path 不存在: {path}")
    features: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(F\d{3})", line)
        if m:
            features.append(m.group(1))
    if not features:
        raise ValueError(f"未能从 {path} 解析出任何特征名（期待 'N. F0xx' 格式）")
    logger.info("使用覆盖特征列表 %s: %d 个特征", path.name, len(features))
    return features

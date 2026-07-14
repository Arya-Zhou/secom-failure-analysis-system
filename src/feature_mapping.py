"""匿名特征 -> 业务含义映射层（扩展点）。

SECOM 是匿名数据，映射表现在全为 unknown。架构上预留此层：
接入真实产线数据时，只需填 feature_map.yaml（F026 -> 某工艺步骤/传感器），
整套分析流程直接复用。这把"不懂业务"的短板转成"系统可迁移"的长板。
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_feature_map(path: str | Path = "feature_map.yaml") -> dict[str, str]:
    """加载特征->业务含义映射；缺文件或缺键时回退 'unknown'。"""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def describe(feature: str, feature_map: dict[str, str]) -> str:
    """返回特征的业务描述，匿名数据下为 'unknown(匿名)'。"""
    return feature_map.get(feature, "unknown(匿名)")

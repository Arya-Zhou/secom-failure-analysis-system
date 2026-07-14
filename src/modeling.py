"""模型构建 —— 配置驱动的查表选择（扩展点所在）。

后期加异常检测/新模型时：只在 _MODEL_REGISTRY 注册一项，
config.model.active 换个名字即可，主流程 (main.py) 不改。
这就是"预留干净扩展点"而非"预留空接口"的做法。

模型超参搬运自 secom.ipynb 的 build_and_evaluate_models（保持一致以对齐基线）：
    逻辑回归  class_weight='balanced', max_iter=1000（默认 lbfgs 求解器）
    随机森林  n_estimators=100, class_weight='balanced', n_jobs=-1
    岭分类器  class_weight='balanced'
"""
from __future__ import annotations

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

logger = logging.getLogger(__name__)


def _make_logistic(random_state, class_weight):
    return LogisticRegression(
        class_weight=class_weight, random_state=random_state, max_iter=1000,
    )


def _make_random_forest(random_state, class_weight):
    return RandomForestClassifier(
        n_estimators=100, class_weight=class_weight,
        random_state=random_state, n_jobs=-1,
    )


def _make_ridge(random_state, class_weight):
    return RidgeClassifier(class_weight=class_weight, random_state=random_state)


# 查表：新增模型在此注册一行即可（扩展点）
_MODEL_REGISTRY = {
    "logistic": _make_logistic,
    "random_forest": _make_random_forest,
    "ridge": _make_ridge,
    # 后期扩展示例：
    # "isolation_forest": _make_isolation_forest,   # 无监督异常检测
}

# 展示名映射：与 notebook / baseline_metrics.json 的中文键对齐
MODEL_DISPLAY_NAMES = {
    "logistic": "逻辑回归",
    "random_forest": "随机森林",
    "ridge": "岭分类器",
}


def _class_weight(cfg: dict):
    """不平衡横切关注点：策略为 class_weight 时启用加权。"""
    return "balanced" if cfg["imbalance"]["strategy"] == "class_weight" else None


def get_model(cfg: dict, random_state: int, name: str | None = None):
    """按名字（默认 config.model.active）查表返回未训练的模型实例。"""
    name = name or cfg["model"]["active"]
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"未注册的模型: {name}，可选: {list(_MODEL_REGISTRY)}")
    cw = _class_weight(cfg)
    logger.info("构建模型: %s (class_weight=%s)", name, cw)
    return _MODEL_REGISTRY[name](random_state, cw)


def get_models(cfg: dict, random_state: int) -> dict[str, object]:
    """按 config.model.active 返回 {注册名: 模型实例}。

    active = "all" 时返回注册表全部模型（notebook 行为：三模型对比）。
    """
    active = cfg["model"]["active"]
    names = list(_MODEL_REGISTRY) if active == "all" else [active]
    return {name: get_model(cfg, random_state, name) for name in names}

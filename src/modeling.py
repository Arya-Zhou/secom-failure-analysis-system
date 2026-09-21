"""模型构建：按名字查 _MODEL_REGISTRY 表返回未训练的分类器实例，配置项为 model.active。"""
from __future__ import annotations

import logging

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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


def _make_elasticnet(random_state, class_weight):
    return LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        class_weight=class_weight, random_state=random_state, max_iter=10000,
    )


def _make_hist_gradient_boosting(random_state, class_weight):
    return HistGradientBoostingClassifier(
        class_weight=class_weight, random_state=random_state, early_stopping=False,
    )


def _make_balanced_random_forest(random_state, class_weight):
    return BalancedRandomForestClassifier(
        n_estimators=100, class_weight=class_weight, random_state=random_state,
        sampling_strategy="all", replacement=True, bootstrap=False, n_jobs=1,
    )


# 查表：新增模型在此注册一行即可
_MODEL_REGISTRY = {
    "logistic": _make_logistic,
    "random_forest": _make_random_forest,
    "ridge": _make_ridge,
    "elasticnet": _make_elasticnet,
    "hist_gradient_boosting": _make_hist_gradient_boosting,
    "balanced_random_forest": _make_balanced_random_forest,
}

# 展示名映射：与基线指标文件的中文键对齐
MODEL_DISPLAY_NAMES = {
    "logistic": "逻辑回归",
    "random_forest": "随机森林",
    "ridge": "岭分类器",
    "elasticnet": "ElasticNet逻辑回归",
    "hist_gradient_boosting": "HistGradientBoosting",
    "balanced_random_forest": "BalancedRandomForest",
}


# 主流程可用的不平衡训练策略。smote/undersample 仅在对比实验中实现，
# 主流程训练路径未接入，故显式拒绝而非静默退化成未加权裸模型。
MAIN_IMBALANCE_STRATEGIES = ("none", "class_weight")


def _class_weight(cfg: dict):
    """按 config 的 imbalance.strategy 决定是否启用类别加权。"""
    strategy = cfg["imbalance"]["strategy"]
    if strategy not in MAIN_IMBALANCE_STRATEGIES:
        raise ValueError(
            f"主流程 imbalance.strategy 仅支持 {list(MAIN_IMBALANCE_STRATEGIES)}，"
            f"收到 {strategy!r}。smote/undersample 仅在对比实验"
            "（imbalance.comparison）中支持，主流程不静默退化")
    return "balanced" if strategy == "class_weight" else None


def registry_names() -> list[str]:
    """按注册名遍历用：返回注册表中全部模型名。"""
    return list(_MODEL_REGISTRY)


def make_model(name: str, random_state: int, class_weight=None, parameters=None):
    """按注册名构建模型，class_weight 由调用方显式指定。"""
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"未注册的模型: {name}，可选: {list(_MODEL_REGISTRY)}")
    model = _MODEL_REGISTRY[name](random_state, class_weight)
    if parameters is not None:
        if {"random_state", "class_weight"}.intersection(parameters):
            raise ValueError("模型 parameters 不得覆盖统一 random_state / class_weight")
        model.set_params(**parameters)
    return model


def get_model(cfg: dict, random_state: int, name: str | None = None):
    """按名字（默认 config.model.active）查表返回未训练的模型实例。"""
    name = name or cfg["model"]["active"]
    cw = _class_weight(cfg)
    logger.info("构建模型: %s (class_weight=%s)", name, cw)
    parameters = cfg["model"].get("parameters", {}).get(name)
    return make_model(name, random_state, cw, parameters)


def get_models(cfg: dict, random_state: int) -> dict[str, object]:
    """按 config.model.active 返回 {注册名: 模型实例}。"""
    active = cfg["model"]["active"]
    names = (list(_MODEL_REGISTRY) if active == "all"
             else active if isinstance(active, list) else [active])
    if not names or len(set(names)) != len(names):
        raise ValueError("model.active 必须包含不重复的模型名")
    return {name: get_model(cfg, random_state, name) for name in names}

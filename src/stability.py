"""训练侧重采样协议、四方法选择、方向证据与预定分档判据。"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from .feature_selection import select_features
from .preprocessing import build_preprocess_pipeline, drop_all_nan_columns
from .validation import resample_splits

STABILITY_PROTOCOL_FILE = "stability_protocol.json"
SELECTION_METHODS = ("f_test", "mutual_info", "rfe", "random_forest")


def _probability(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须为有限数值")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0 or (positive and value == 0):
        raise ValueError(f"{name} 必须落在 {'(0, 1]' if positive else '[0, 1]'}")
    return value


def resolve_stability(cfg: dict) -> dict:
    """读取独立重采样协议与预先定义的判据，缺项或弱化训练侧隔离即失败。"""
    node = cfg.get("stability")
    if not isinstance(node, dict):
        raise ValueError("config 缺少 stability")
    if not isinstance(node.get("enabled"), bool):
        raise ValueError("stability.enabled 必须为布尔值")
    if node.get("input_scope") != "reference_train_only":
        raise ValueError("stability.input_scope 必须为 reference_train_only")
    spec = {"enabled": node["enabled"], "input_scope": node["input_scope"]}
    for name, minimum in (("n_resamples", 2), ("n_jobs", 1), ("seed", 0)):
        value = node.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"stability.{name} 必须为 >= {minimum} 的整数")
        spec[name] = int(value)
    if spec["seed"] > 2 ** 32 - 1:
        raise ValueError("stability.seed 超出随机种子的有效范围")
    spec["test_size"] = _probability(node.get("test_size"), "stability.test_size", positive=True)
    if spec["test_size"] == 1:
        raise ValueError("stability.test_size 必须小于 1")
    thresholds = node.get("frequency_thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("stability.frequency_thresholds 必须为非空列表")
    thresholds = [_probability(value, "stability.frequency_thresholds", positive=True)
                  for value in thresholds]
    if thresholds != sorted(set(thresholds)):
        raise ValueError("stability.frequency_thresholds 必须递增且不重复")
    primary = _probability(node.get("primary_frequency_threshold"),
                           "stability.primary_frequency_threshold", positive=True)
    if primary not in thresholds:
        raise ValueError("stability.primary_frequency_threshold 必须包含在敏感性档位中")
    spec.update(frequency_thresholds=thresholds, primary_frequency_threshold=primary)
    spec["correlation_group_threshold"] = _probability(
        node.get("correlation_group_threshold"), "stability.correlation_group_threshold",
        positive=True)
    for name, expected in (("correlation_statistic", "absolute_spearman"),
                           ("correlation_grouping", "connected_components")):
        if node.get(name) != expected:
            raise ValueError(f"stability.{name} 必须为 {expected}")
        spec[name] = expected
    rules = node.get("classification")
    if not isinstance(rules, dict):
        raise ValueError("stability.classification 必填")
    lower = _probability(rules.get("exploratory_min_frequency"),
                         "stability.classification.exploratory_min_frequency")
    direction = _probability(rules.get("high_min_direction_consistency"),
                             "stability.classification.high_min_direction_consistency",
                             positive=True)
    if lower >= min(thresholds):
        raise ValueError("探索性频率下界必须小于所有高稳定频率阈值")
    if rules.get("missing_direction") != "insufficient_evidence":
        raise ValueError("缺失方向证据必须归为 insufficient_evidence")
    spec["classification"] = {
        "exploratory_min_frequency": lower,
        "high_min_direction_consistency": direction,
        "missing_direction": "insufficient_evidence",
    }
    selection = cfg.get("feature_selection") or {}
    methods = selection.get("methods")
    if (not isinstance(methods, list) or not all(isinstance(method, str) for method in methods)
            or sorted(methods) != sorted(SELECTION_METHODS)):
        raise ValueError("stability 必须沿用四方法投票选择器")
    selected_count = selection.get("n_features_to_select")
    if isinstance(selected_count, bool) or not isinstance(selected_count, Integral) or selected_count < 1:
        raise ValueError("stability 的 n_features_to_select 必须为正整数")
    if selection.get("override_features_path") is not None:
        raise ValueError("stability 不允许 override 特征列表代替重跑选择器")
    if (cfg.get("run") or {}).get("quick", False):
        raise ValueError("stability 不接受 quick 模式")
    return spec


def resolve_stability_analysis(cfg: dict) -> dict:
    """方向探针与趋势采样口径须在真实跑批前显式定义。"""
    node = (cfg.get("stability") or {}).get("analysis")
    if not isinstance(node, dict):
        raise ValueError("stability.analysis 必填")
    if node.get("direction_method") != "ridge_coefficient":
        raise ValueError("stability.analysis.direction_method 必须为 ridge_coefficient")
    if node.get("direction_class_weight") != "balanced":
        raise ValueError("stability.analysis.direction_class_weight 必须为 balanced")
    alpha = node.get("direction_alpha")
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("stability.analysis.direction_alpha 必须为有限正数")
    step = node.get("convergence_step")
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 1:
        raise ValueError("stability.analysis.convergence_step 必须为正整数")
    return {
        "direction_method": node["direction_method"],
        "direction_alpha": float(alpha),
        "direction_class_weight": node["direction_class_weight"],
        "direction_fit_side": "inner_train_selected_features",
        "direction_denominator": "selected_splits_including_zero",
        "rank_correlation_scope": "common_nonconstant_features",
        "correlation_fit_side": "reference_train_imputed",
        "top_k": int(cfg["feature_selection"]["n_features_to_select"]),
        "convergence_step": int(step),
        "convergence_interpretation": "descriptive_only_not_a_stopping_rule",
    }


def _reference_partition(labels, reference_train, reference_test) -> tuple:
    reference_train, reference_test = pd.Index(reference_train), pd.Index(reference_test)
    for name, indices in (("labels", labels.index), ("reference_train", reference_train),
                          ("reference_test", reference_test)):
        if indices.empty or not indices.is_unique or indices.hasnans:
            raise ValueError(f"{name} 索引必须非空、唯一且无缺失")
    if len(reference_train.intersection(reference_test)):
        raise ValueError("reference 训练侧与测试侧索引重叠")
    combined = reference_train.append(reference_test)
    if len(combined) != len(labels) or len(combined.difference(labels.index)):
        raise ValueError("固定 reference 划分必须完整覆盖标签索引")
    return reference_train, reference_test


def assert_training_only(train_indices, reference_train, reference_test) -> None:
    """在拟合前拒绝 reference 测试行、未知行或重复行。"""
    train_indices = pd.Index(train_indices)
    if train_indices.empty or not train_indices.is_unique or train_indices.hasnans:
        raise ValueError("稳定性训练索引必须非空、唯一且无缺失")
    if len(train_indices.intersection(reference_test)):
        raise ValueError("稳定性训练索引包含 reference 测试侧样本")
    if len(train_indices.difference(reference_train)):
        raise ValueError("稳定性训练索引超出 reference 训练侧")


def stability_splits(labels, reference_train, reference_test, cfg: dict) -> list[tuple]:
    """只在已固定的 reference 训练索引内重采样，不重新计算外层划分。"""
    spec = resolve_stability(cfg)
    reference_train, reference_test = _reference_partition(labels, reference_train, reference_test)
    train_labels = labels.loc[reference_train]
    if train_labels.isna().any() or set(train_labels.unique()) != {0, 1}:
        raise ValueError("reference 训练侧必须同时含 0/1 两类且无缺失标签")
    splits = resample_splits(train_labels, spec["test_size"], spec["seed"], spec["n_resamples"])
    if len(splits) != spec["n_resamples"]:
        raise ValueError("稳定性划分数量与预定 n_resamples 不符")
    for train_indices, holdout_indices in splits:
        assert_training_only(train_indices, reference_train, reference_test)
        assert_training_only(holdout_indices, reference_train, reference_test)
        if len(train_indices.intersection(holdout_indices)):
            raise ValueError("稳定性内部训练侧与留出侧索引重叠")
        if len(train_indices) + len(holdout_indices) != len(reference_train):
            raise ValueError("稳定性内部划分未完整覆盖 reference 训练侧")
    return splits


def _prepare_stability_training(features, labels, train_indices,
                                reference_train, reference_test, cfg: dict) -> tuple:
    resolve_stability(cfg)
    reference_train, reference_test = _reference_partition(labels, reference_train, reference_test)
    assert_training_only(train_indices, reference_train, reference_test)
    if not features.index.equals(labels.index) or not features.columns.is_unique:
        raise ValueError("特征与标签索引必须一致，特征列名必须唯一")
    train_features, _dropped = drop_all_nan_columns(features.loc[train_indices])
    if train_features.empty:
        raise ValueError("稳定性训练侧没有可用特征")
    train_labels = labels.loc[train_indices]
    if train_labels.isna().any() or set(train_labels.unique()) != {0, 1}:
        raise ValueError("稳定性训练侧必须同时含 0/1 两类且无缺失标签")
    preprocessor = build_preprocess_pipeline(cfg)
    scaled = pd.DataFrame(preprocessor.fit_transform(train_features),
                          index=train_features.index, columns=train_features.columns)
    return scaled, train_labels


def select_stability_features(features, labels, train_indices,
                              reference_train, reference_test, cfg: dict) -> tuple:
    """单次重采样只在合法训练行上清洗、拟合预处理并执行完整四方法投票。"""
    spec = resolve_stability(cfg)
    scaled, train_labels = _prepare_stability_training(
        features, labels, train_indices, reference_train, reference_test, cfg)
    return select_features(scaled, train_labels, cfg, spec["seed"], quick=False)


def fit_stability_split(features, labels, train_indices, reference_train,
                        reference_test, cfg: dict, split_no: int) -> dict:
    """逐轮保存真实投票排名与同一训练侧上的加权岭系数，留出侧不参与拟合。"""
    started = perf_counter()
    spec, analysis = resolve_stability(cfg), resolve_stability_analysis(cfg)
    with threadpool_limits(limits=1):
        scaled, train_labels = _prepare_stability_training(
            features, labels, train_indices, reference_train, reference_test, cfg)
        selected, ranks = select_features(scaled, train_labels, cfg, spec["seed"], quick=False)
        selection_seconds = perf_counter() - started
        model = RidgeClassifier(alpha=analysis["direction_alpha"],
                                class_weight=analysis["direction_class_weight"],
                                random_state=spec["seed"])
        model.fit(scaled[selected], train_labels)
    coefficients = model.coef_.ravel()
    if not np.isfinite(coefficients).all() or not np.isfinite(ranks.to_numpy()).all():
        raise ValueError("稳定性排名或方向系数含非有限值")
    return {
        "split": split_no,
        "selected_features": selected,
        "ranks": ranks.to_dict(orient="index"),
        "nonconstant_features": scaled.columns[scaled.nunique() > 1].tolist(),
        "coefficients": dict(zip(selected, coefficients.tolist())),
        "selection_seconds": selection_seconds,
        "elapsed_seconds": perf_counter() - started,
    }


def selection_frequencies(selections: list[list[str]], feature_names, n_splits: int) -> dict:
    """按全部预定划分计频率，未入选特征记零，缺轮次或重复入选记录即失败。"""
    if n_splits < 1 or len(selections) != n_splits:
        raise ValueError("入选频率必须包含全部预定划分")
    feature_names = list(feature_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("频率特征全集必须非空且无重复")
    counts = dict.fromkeys(feature_names, 0)
    for selected in selections:
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("每轮入选特征必须非空且无重复")
        if set(selected).difference(counts):
            raise ValueError("入选记录包含未知特征")
        for feature in selected:
            counts[feature] += 1
    return {feature: count / n_splits for feature, count in counts.items()}


def classify_candidate(frequency: float, direction_consistency: float | None,
                       spec: dict, frequency_threshold: float | None = None) -> str:
    """三档互斥：缺方向或未超过探索下界记不足；频率和方向均达标才记高稳定。"""
    frequency = _probability(frequency, "frequency")
    if direction_consistency is not None:
        direction_consistency = _probability(direction_consistency, "direction_consistency")
    threshold = (spec["primary_frequency_threshold"] if frequency_threshold is None
                 else _probability(frequency_threshold, "frequency_threshold", positive=True))
    if threshold not in spec["frequency_thresholds"]:
        raise ValueError("分类频率阈值必须是预先定义的敏感性档位")
    rules = spec["classification"]
    if direction_consistency is None or frequency <= rules["exploratory_min_frequency"]:
        return "当前证据不足"
    if frequency >= threshold and direction_consistency >= rules["high_min_direction_consistency"]:
        return "高稳定候选"
    return "探索性候选"


def build_stability_protocol(labels, reference_train, reference_test, cfg: dict) -> dict:
    """冻结独立总体、原始行索引和判据；仅生成协议，不训练、不生成分析结果。"""
    spec = resolve_stability(cfg)
    reference_train, reference_test = _reference_partition(labels, reference_train, reference_test)
    splits = stability_splits(labels, reference_train, reference_test, cfg)
    return {
        "artifact_schema_version": 1,
        "status": "protocol_only",
        "analysis_executed": False,
        "protocol": {
            "population": "StratifiedShuffleSplit",
            "input_scope": spec["input_scope"],
            "n_population": len(reference_train),
            "n_splits": spec["n_resamples"],
            "test_size": spec["test_size"],
            "seed": spec["seed"],
            "selection_fit_side": "inner_train",
            "reference_test_used": False,
            "same_population_as_metric_resampling": False,
            "parallel_metric_artifact": "resample_metrics.json",
        },
        "reference_split": {
            "train_indices": reference_train.tolist(),
            "test_indices": reference_test.tolist(),
        },
        "criteria": spec,
        "selection_recipe": dict(cfg["feature_selection"]),
        "preprocessing": dict(cfg["preprocessing"]),
        "splits": [
            {"split": split_no, "n_train": len(train_indices), "n_holdout": len(holdout_indices),
             "train_indices": train_indices.tolist(), "holdout_indices": holdout_indices.tolist()}
            for split_no, (train_indices, holdout_indices) in enumerate(splits)
        ],
    }


def write_stability_protocol(labels, reference_train, reference_test,
                             cfg: dict, output_dir: str | Path) -> Path:
    """单独保存开跑前协议，不覆盖指标产物或已冻结的不同协议。"""
    payload = build_stability_protocol(labels, reference_train, reference_test, cfg)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(output_dir) / STABILITY_PROTOCOL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if protocol_definition(existing) != protocol_definition(payload):
            raise FileExistsError("已存在不同的稳定性协议；须显式处理旧协议，不能静默覆盖判据")
        return path
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return path


def protocol_definition(payload: dict) -> dict:
    """启用开关与并行度不改变抽样或判据，复跑时保留首次冻结文件的原字节。"""
    criteria = {key: value for key, value in payload["criteria"].items()
                if key not in ("enabled", "n_jobs")}
    return {**payload, "criteria": criteria}

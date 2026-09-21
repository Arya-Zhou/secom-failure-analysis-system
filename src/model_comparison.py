"""固定模型族的训练侧配对比较与选择，不搜索策略笛卡尔积。"""
from __future__ import annotations

import warnings
from itertools import combinations
from numbers import Integral

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import canonical_costs
from .evaluation import (
    PROBABILITY_SPACES, classification_metrics, expected_cost, recall_at_fp_budget,
    score_samples,
)
from .modeling import MODEL_DISPLAY_NAMES, make_model
from .validation import interval, paired_diff_interval

DELIVERY_CANDIDATES = ("logistic", "ridge", "elasticnet", "hist_gradient_boosting")
REFERENCE_BASELINES = ("random_forest", "balanced_random_forest")
FEATURE_CONTROL = "elasticnet_vote"
MODEL_NAMES = DELIVERY_CANDIDATES + REFERENCE_BASELINES + (FEATURE_CONTROL,)
COMPARISON_METRICS = (
    "ber", "recall", "precision", "f1", "auc", "pr_auc", "brier",
    "cost_per_wafer", "recall_at_fp_budget",
)


def resolve_model_comparison(cfg: dict) -> dict:
    """固定候选角色、数据边界和替换判据，在读取结果前拒绝模糊配置。"""
    node = cfg.get("model_comparison")
    if not isinstance(node, dict) or not isinstance(node.get("enabled"), bool):
        raise ValueError("model_comparison.enabled 必须为布尔值")
    for key, expected in (
        ("input_scope", "reference_train_only"), ("reference_model", "ridge"),
        ("interval", "percentile"), ("primary_metric", "cost_per_wafer"),
        ("replacement_rule", "paired_upper_below_zero"),
    ):
        if node.get(key) != expected:
            raise ValueError(f"model_comparison.{key} 必须为 {expected}")
    for key, expected in (("delivery_candidates", DELIVERY_CANDIDATES),
                          ("reference_baselines", REFERENCE_BASELINES)):
        if node.get(key) != list(expected):
            raise ValueError(f"model_comparison.{key} 必须为 {list(expected)}")
    workers = node.get("n_jobs")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("model_comparison.n_jobs 必须为正整数")
    if cfg["run"]["quick"]:
        raise ValueError("模型配对比较不支持 quick 数据或简化选择器")
    if cfg["random_state"] != cfg["stability"]["seed"]:
        raise ValueError("模型比较与稳定性记录必须使用同一随机种子")
    costs = canonical_costs(cfg)
    if not all(np.isfinite(value) and value > 0 for value in costs):
        raise ValueError("模型比较 costs 必须为有限正数")
    budget = cfg["evaluation"]["fp_budget"]
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("evaluation.fp_budget 必须为非负整数")
    return dict(node)


def model_recipe(cfg: dict, name: str) -> dict:
    if name not in MODEL_NAMES:
        raise ValueError(f"不在冻结模型比较清单中: {name}")
    registered = "elasticnet" if name == FEATURE_CONTROL else name
    class_weight = None if name == "balanced_random_forest" else "balanced"
    parameters = cfg["model"].get("parameters", {}).get(registered, {})
    model = make_model(registered, int(cfg["random_state"]), class_weight, parameters)
    if registered == "random_forest":
        model.set_params(n_jobs=1)
    if registered == "hist_gradient_boosting" and model.early_stopping is not False:
        raise ValueError("固定 HGB 比较要求 early_stopping=false")
    if registered == "elasticnet" and (model.penalty != "elasticnet" or model.solver != "saga"):
        raise ValueError("ElasticNet 比较要求 penalty=elasticnet / solver=saga")
    role = ("delivery_candidate" if name in DELIVERY_CANDIDATES
            else "reference_baseline" if name in REFERENCE_BASELINES
            else "feature_selection_control")
    return {
        "registered_model": registered, "role": role,
        "display_name": ("ElasticNet + 四方法投票" if name == FEATURE_CONTROL
                         else MODEL_DISPLAY_NAMES[registered]),
        "feature_mode": "embedded" if name == "elasticnet" else "vote",
        "parameters": model.get_params(deep=False), "decision": "predict",
        "threshold_search": False, "calibration_search": False,
    }


def build_comparison_estimator(cfg, name, train_features, vote_features):
    """只用本折训练列装配完整链路，缓存投票记录不参与预处理的拟合。"""
    recipe = model_recipe(cfg, name)
    available = train_features.columns[~train_features.isna().all()].tolist()
    selected = available if recipe["feature_mode"] == "embedded" else list(vote_features)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError("投票特征必须是本折训练侧非全空列的非空、不重复子集")
    imputer = SimpleImputer(
        strategy=cfg["preprocessing"]["impute_strategy"], keep_empty_features=True,
    ).set_output(transform="pandas")
    steps = [("impute", imputer)]
    if cfg["preprocessing"]["scale"]:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("select", ColumnTransformer(
        [("features", "passthrough", selected)], verbose_feature_names_out=False,
    ).set_output(transform="pandas")))
    parameters = dict(recipe["parameters"])
    seed, class_weight = parameters.pop("random_state"), parameters.pop("class_weight")
    model = make_model(recipe["registered_model"], seed, class_weight, parameters)
    steps.append(("model", model))
    return Pipeline(steps), selected


def metrics_from_predictions(labels, predictions, scores, score_space, costs, fp_budget):
    """从行级证据重算指标，拒绝截短预测、非有限分数和伪概率。"""
    truth = np.asarray(labels)
    predicted, scored = np.asarray(predictions), np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or predicted.shape != truth.shape or scored.shape != truth.shape
            or set(np.unique(truth)) != {0, 1} or not np.isin(predicted, [0, 1]).all()
            or not np.isfinite(scored).all()):
        raise ValueError("比较预测证据必须与含两类的验证行逐项对齐且分数有限")
    if score_space not in (*PROBABILITY_SPACES, "decision_margin"):
        raise ValueError("比较记录的 score_space 无效")
    probability = score_space in PROBABILITY_SPACES
    if probability and ((scored < 0).any() or (scored > 1).any()):
        raise ValueError("概率必须落在 [0, 1]")
    boundary = 0.5 if probability else 0.0
    if not np.array_equal(predicted, (scored > boundary).astype(int)):
        raise ValueError("比较预测与冻结的默认决策边界不一致")
    base = classification_metrics(truth, predicted, scored)
    cost, confusion = expected_cost(truth, predicted, costs["fn"], costs["fp"])
    return {
        "ber": base["BER"], "recall": base["召回率"], "precision": base["精确率"],
        "f1": base["F1分数"], "auc": base["AUC"],
        "pr_auc": float(average_precision_score(truth, scored)),
        "brier": float(brier_score_loss(truth, scored)) if probability else None,
        "expected_cost": cost, "cost_per_wafer": cost / len(truth),
        "recall_at_fp_budget": recall_at_fp_budget(truth, scored, fp_budget),
        "specificity": confusion["tn"] / (confusion["tn"] + confusion["fp"]),
        "confusion": confusion, "n": int(len(truth)), "n_positive": int(truth.sum()),
    }


def fit_comparison_split(cfg, train_features, train_labels, validation_features,
                         validation_labels, vote_features, *, keep_fitted=False):
    """各模型共享同一组训练/验证行；选特征、填充与模型均不接触验证标签。"""
    if (not train_features.index.equals(train_labels.index)
            or not validation_features.index.equals(validation_labels.index)
            or not train_features.index.is_unique or not validation_features.index.is_unique
            or set(train_features.index).intersection(validation_features.index)
            or list(train_features.columns) != list(validation_features.columns)):
        raise ValueError("比较训练/验证索引或列序不一致，或两侧发生重叠")
    fn_cost, fp_cost = canonical_costs(cfg)
    costs = {"fn": fn_cost, "fp": fp_cost}
    records, fitted = {}, {}
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            estimator, selected = build_comparison_estimator(cfg, name, train_features, vote_features)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(train_features, train_labels)
            predictions = estimator.predict(validation_features)
            scores, space = score_samples(estimator, validation_features)
            model = estimator.named_steps["model"]
            coefficients = (dict(zip(selected, model.coef_.ravel().tolist()))
                            if hasattr(model, "coef_") else None)
            active = ([feature for feature, value in coefficients.items() if value != 0]
                      if coefficients is not None else None)
            records[name] = {
                "predictions": predictions.tolist(), "scores": scores.tolist(), "score_space": space,
                "metrics": metrics_from_predictions(
                    validation_labels, predictions, scores, space, costs, cfg["evaluation"]["fp_budget"]),
                "selector_features": selected, "coefficients": coefficients, "active_features": active,
            }
            if keep_fitted:
                fitted[name] = estimator
    return records, fitted


def summarize_model_comparison(records: list[dict], cfg: dict) -> dict:
    """配对差值先于选择，参考基线和特征对照永不进入替换池。"""
    spec = resolve_model_comparison(cfg)
    if len(records) < 2 or [row["split"] for row in records] != list(range(len(records))):
        raise ValueError("比较必须具有至少两个完整且有序的独立划分记录")
    if any(set(row["models"]) != set(MODEL_NAMES) for row in records):
        raise ValueError("比较缺少冻结候选、参考基线或嵌入式特征对照")
    series = {
        name: {metric: [row["models"][name]["metrics"][metric] for row in records]
               for metric in COMPARISON_METRICS}
        for name in MODEL_NAMES
    }
    seed, method = int(cfg["random_state"]), spec["interval"]
    intervals = {name: {metric: interval(values, method, seed) for metric, values in metrics.items()}
                 for name, metrics in series.items()}
    paired = {}
    for left, right in combinations(MODEL_NAMES, 2):
        paired[f"{left} - {right}"] = {
            metric: paired_diff_interval(series[left][metric], series[right][metric], method, seed)
            for metric in COMPARISON_METRICS
        }
    reference = spec["reference_model"]
    vs_reference = {
        name: {metric: paired_diff_interval(series[name][metric], series[reference][metric], method, seed)
               for metric in COMPARISON_METRICS}
        for name in MODEL_NAMES if name != reference
    }
    allowed = [name for name in DELIVERY_CANDIDATES if name != reference
               and vs_reference[name][spec["primary_metric"]]["high"] < 0]
    selected = (min(allowed, key=lambda name: (intervals[name][spec["primary_metric"]]["mean"], name))
                if allowed else reference)
    return {
        "intervals": intervals, "paired_diff": paired, "vs_reference": vs_reference,
        "selection": {
            "basis": "reference_train_repeated_holdout_paired_cost", "reference_model": reference,
            "candidate_pool": list(DELIVERY_CANDIDATES), "eligible_replacements": allowed,
            "selected_model": selected, "uses_outer_test": False,
            "replacement_rule": spec["replacement_rule"], "primary_metric": spec["primary_metric"],
        },
        "feature_selection": {
            name: {"active_count": interval(
                [len(row["models"][name]["active_features"]) for row in records], method, seed),
                "selector_count": interval(
                    [len(row["models"][name]["selector_features"]) for row in records], method, seed)}
            for name in ("elasticnet", FEATURE_CONTROL)
        },
    }


def validate_comparison_records(records, sampling_records, features, labels, cfg):
    """逐行重算所有模型指标，并核对每轮特征确实来自同一训练划分。"""
    if len(records) != len(sampling_records):
        raise ValueError("模型比较划分记录不完整")
    fn_cost, fp_cost = canonical_costs(cfg)
    costs = {"fn": fn_cost, "fp": fp_cost}
    for record, sampling in zip(records, sampling_records):
        for key in ("split", "train_indices", "validation_indices"):
            expected_key = "holdout_indices" if key == "validation_indices" else key
            if record[key] != sampling[expected_key]:
                raise ValueError(f"模型比较 {key} 与冻结划分不一致")
        if set(record["models"]) != set(MODEL_NAMES):
            raise ValueError("模型比较的角色或候选记录不完整")
        train = features.loc[record["train_indices"]]
        available = train.columns[~train.isna().all()].tolist()
        for name, row in record["models"].items():
            selected = available if name == "elasticnet" else sampling["selected_features"]
            if row["selector_features"] != selected:
                raise ValueError("模型比较特征与该轮训练侧选择记录不一致")
            coefficients = row["coefficients"]
            if name in (*DELIVERY_CANDIDATES[:3], FEATURE_CONTROL):
                if (not isinstance(coefficients, dict) or list(coefficients) != selected
                        or not np.isfinite(list(coefficients.values())).all()
                        or row["active_features"] != [key for key, value in coefficients.items() if value != 0]):
                    raise ValueError("模型比较嵌入式特征或系数记录不一致")
            elif coefficients is not None or row["active_features"] is not None:
                raise ValueError("树模型不得冒称线性系数")
            space = "decision_margin" if name == "ridge" else "probability"
            if row["score_space"] != space:
                raise ValueError("模型比较分数空间与冻结模型不符")
            recomputed = metrics_from_predictions(
                labels.loc[record["validation_indices"]], row["predictions"], row["scores"],
                space, costs, cfg["evaluation"]["fp_budget"])
            if row["metrics"] != recomputed:
                raise ValueError("模型比较指标与行级预测重算不一致")

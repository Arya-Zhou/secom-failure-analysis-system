"""制程与良率的时间漂移分析：原始分布描述与严格早训晚测的固定模型。"""
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .config import canonical_costs, load_config
from .data_io import load_secom
from .evaluation import expected_cost, score_samples
from .feature_selection import select_features
from .modeling import get_model
from .preprocessing import build_preprocess_pipeline
from .validation import prior_drift_table, resolve_temporal_holdout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_TITLE = "制程与良率的时间漂移分析"
DRIFT_FILE = "drift_report.json"
DRIFT_MARKDOWN = "drift_report.md"
DRIFT_PLOT = "drift_overview.png"
logger = logging.getLogger(__name__)


def resolve_drift(cfg: dict) -> dict:
    node = cfg.get("drift", {})
    if not isinstance(node, dict):
        raise ValueError("drift 必须是配置映射")
    spec = {"enabled": False, "model": "ridge", "window_size": 200, "step_size": 100,
            "psi_bins": 10, "psi_pseudocount": 0.5, "ks_min_samples": 2,
            "top_features": 10, **node}
    if not isinstance(spec["enabled"], bool):
        raise ValueError("drift.enabled 必须为布尔值")
    if spec["model"] != "ridge":
        raise ValueError("drift.model 固定为 ridge；本分析不重新选择模型族")
    for key, minimum in (("window_size", 2), ("step_size", 1), ("psi_bins", 2),
                         ("ks_min_samples", 2), ("top_features", 1)):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"drift.{key} 必须是 >= {minimum} 的整数")
    if spec["step_size"] > spec["window_size"]:
        raise ValueError("drift.step_size 不得大于 window_size，以免滑窗漏掉样本")
    value = spec["psi_pseudocount"]
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError("drift.psi_pseudocount 必须为有限正数")
    return spec


def _ordered_inputs(features, labels, timestamps):
    if (not isinstance(features, pd.DataFrame) or not isinstance(labels, pd.Series)
            or features.empty or not features.index.equals(labels.index)
            or not labels.index.is_unique or not features.columns.is_unique
            or not pd.api.types.is_integer_dtype(labels.index.dtype)
            or not all(isinstance(name, str) for name in features.columns)):
        raise ValueError("漂移分析要求非空特征、唯一列名及对齐的唯一整数行索引")
    if not labels.isin([0, 1]).all():
        raise ValueError("漂移分析标签必须完整且为 0/1")
    stamps = pd.Series(timestamps)
    if (not stamps.index.is_unique or len(stamps) != len(labels)
            or not labels.index.isin(stamps.index).all()):
        raise ValueError("漂移分析 timestamp 与标签行索引不一致")
    stamps = stamps.loc[labels.index]
    if not pd.api.types.is_datetime64_any_dtype(stamps.dtype) or stamps.isna().any():
        raise ValueError("漂移分析需要完整 timestamp，且须使用数据加载器解析时间")
    if (any(name.casefold() == "timestamp" for name in features.columns)
            or any(not pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)):
        raise ValueError("漂移分析仅接受数值特征，timestamp 不得作为特征")
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("漂移分析不接受无限特征值；缺失值须为 NaN")
    return stamps, stamps.sort_values(kind="stable").index


def rolling_windows(indices, window_size: int, step_size: int) -> list[pd.Index]:
    indices = pd.Index(indices)
    if window_size < 1 or step_size < 1 or step_size > window_size:
        raise ValueError("滑窗需要 1 <= step_size <= window_size")
    if len(indices) == 0:
        return []
    width = min(window_size, len(indices))
    starts = list(range(0, len(indices) - width + 1, step_size))
    if starts[-1] != len(indices) - width:
        starts.append(len(indices) - width)
    return [indices[start:start + width] for start in starts]


def _window_summary(stamps, labels, indices, name) -> dict:
    failures = int(labels.loc[indices].sum())
    count = len(indices)
    return {"window": name, "n": count, "failures": failures, "passes": count - failures,
            "failure_rate": failures / count, "pass_rate": (count - failures) / count,
            "start": str(stamps.loc[indices[0]]), "end": str(stamps.loc[indices[-1]])}


def prior_analysis(timestamps, labels, n_bins, window_size, step_size) -> dict:
    stamps = pd.Series(timestamps).loc[labels.index]
    order = stamps.sort_values(kind="stable").index
    if n_bins < 2 or n_bins > len(order):
        raise ValueError("漂移等分窗口数必须在 [2, 样本数] 内")
    equal_count = prior_drift_table(stamps, labels, n_bins)
    month_keys = stamps.loc[order].dt.to_period("M")
    monthly = [_window_summary(stamps, labels, order[month_keys == month], str(month))
               for month in month_keys.unique()]
    rolling = [_window_summary(stamps, labels, indices, number)
               for number, indices in enumerate(rolling_windows(order, window_size, step_size), 1)]
    return {"equal_count": equal_count, "monthly": monthly, "rolling": rolling,
            "zero_failure_bins": [row["bin"] for row in equal_count["bins"] if not row["failures"]],
            "ratio_definition": "max_over_min_among_positive_failure_rates"}


def _psi_counts(values, edges, total):
    missing = total - len(values)
    if len(edges) == 0:
        return np.asarray([len(values), missing], dtype=int)
    if len(edges) == 1:
        return np.asarray([np.sum(values < edges[0]), np.sum(values == edges[0]),
                           np.sum(values > edges[0]), missing], dtype=int)
    middle, _edges = np.histogram(values, bins=edges)
    return np.concatenate(([np.sum(values < edges[0])], middle,
                           [np.sum(values > edges[-1]), missing])).astype(int)


def feature_distribution(reference, current, *, psi_bins=10, pseudocount=0.5,
                         ks_min_samples=2) -> dict:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if (reference.ndim != 1 or current.ndim != 1 or not reference.size or not current.size
            or np.isinf(reference).any() or np.isinf(current).any()):
        raise ValueError("PSI/KS 需要非空一维数值，且不接受无限值")
    if (psi_bins < 2 or ks_min_samples < 2 or not np.isfinite(pseudocount)
            or pseudocount <= 0):
        raise ValueError("PSI/KS 分箱、样本下限或伪计数无效")
    reference_values = reference[~np.isnan(reference)]
    current_values = current[~np.isnan(current)]
    edges = (np.unique(np.quantile(reference_values, np.linspace(0, 1, psi_bins + 1)))
             if reference_values.size else np.asarray([]))
    reference_counts = _psi_counts(reference_values, edges, reference.size)
    current_counts = _psi_counts(current_values, edges, current.size)
    reference_share = (reference_counts + pseudocount) / (reference.size + pseudocount * len(reference_counts))
    current_share = (current_counts + pseudocount) / (current.size + pseudocount * len(current_counts))
    enough = min(reference_values.size, current_values.size) >= ks_min_samples
    test = ks_2samp(reference_values, current_values, method="asymp") if enough else None
    basis = ("missingness_only" if edges.size == 0 else "constant_with_tails_and_missing"
             if edges.size == 1 else "reference_quantiles_with_tails_and_missing")
    return {
        "psi": float(np.sum((current_share - reference_share) * np.log(current_share / reference_share))),
        "psi_basis": basis, "reference_edges": edges.tolist(),
        "reference_counts": reference_counts.tolist(), "current_counts": current_counts.tolist(),
        "reference_observed": int(reference_values.size), "current_observed": int(current_values.size),
        "reference_missing_rate": float(np.isnan(reference).mean()),
        "current_missing_rate": float(np.isnan(current).mean()),
        "missing_rate_delta": float(np.isnan(current).mean() - np.isnan(reference).mean()),
        "ks_statistic": float(test.statistic) if test is not None else None,
        "ks_p_value": float(test.pvalue) if test is not None else None,
        "ks_q_value": None, "ks_status": "ok" if enough else "insufficient_observations",
        "has_ties": bool(len(np.unique(np.concatenate((reference_values, current_values))))
                         < reference_values.size + current_values.size),
    }


def benjamini_hochberg(values) -> list[float]:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("BH 输入必须为 [0, 1] 内有限 p 值")
    if not len(probabilities):
        return []
    order = np.argsort(probabilities, kind="stable")
    adjusted = probabilities[order] * len(probabilities) / np.arange(1, len(probabilities) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def _feature_analysis(features, labels, stamps, partitions, spec) -> dict:
    windows, valid_tests = [], []
    for number, indices in enumerate(partitions[1:], 2):
        rows = {}
        for name in features.columns:
            row = feature_distribution(
                features.loc[partitions[0], name], features.loc[indices, name],
                psi_bins=spec["psi_bins"], pseudocount=spec["psi_pseudocount"],
                ks_min_samples=spec["ks_min_samples"])
            rows[name] = row
            if row["ks_p_value"] is not None:
                valid_tests.append(row)
        windows.append({**_window_summary(stamps, labels, indices, number), "features": rows})
    for row, adjusted in zip(valid_tests, benjamini_hochberg([row["ks_p_value"] for row in valid_tests])):
        row["ks_q_value"] = adjusted
    return {"reference": _window_summary(stamps, labels, partitions[0], 1),
            "feature_count": features.shape[1], "windows": windows,
            "ks_valid_tests": len(valid_tests),
            "multiple_testing": "BH_across_all_valid_feature_window_pairs",
            "ks_method": "asymp", "operational_alarm_threshold": None}


def window_performance(labels, predictions, scores, fn_cost, fp_cost) -> dict:
    truth, predicted, scored = (np.asarray(values) for values in (labels, predictions, scores))
    if (truth.ndim != 1 or not truth.size or truth.shape != predicted.shape or truth.shape != scored.shape
            or not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()
            or not np.isfinite(scored).all()):
        raise ValueError("时间窗预测必须与 0/1 标签逐行对齐且分数有限")
    cost, confusion = expected_cost(truth, predicted, fn_cost, fp_cost)
    positive = int(truth.sum())
    negative = int(truth.size - positive)
    recall = confusion["tp"] / positive if positive else None
    specificity = confusion["tn"] / negative if negative else None
    return {"n": int(truth.size), "positives": positive, "negatives": negative,
            "confusion": confusion, "recall": recall, "specificity": specificity,
            "ber": (2 - recall - specificity) / 2 if positive and negative else None,
            "auc": float(roc_auc_score(truth, scored)) if positive and negative else None,
            "expected_cost": cost, "cost_per_wafer": cost / truth.size,
            "status": "ok" if positive and negative else "single_class",
            "undefined_reason": None if positive and negative else "window_lacks_a_class"}


def _performance_analysis(cfg, features, labels, stamps, partitions, spec) -> dict:
    nominal_train = partitions[0]
    future = pd.Index(np.concatenate(partitions[1:]))
    train = nominal_train[stamps.loc[nominal_train] < stamps.loc[future[0]]]
    excluded = nominal_train[~nominal_train.isin(train)]
    if set(labels.loc[train].unique()) != {0, 1}:
        raise ValueError("早期训练窗在剔除边界同时间戳后必须包含两类；不得借用未来标签")
    available = features.columns[~features.loc[train].isna().all()].tolist()
    if not available:
        raise ValueError("早期训练窗没有可用特征；不得借用未来数值填充")
    seed = int(cfg["random_state"])
    with threadpool_limits(limits=1):
        preprocessing = build_preprocess_pipeline(cfg)
        training_values = pd.DataFrame(preprocessing.fit_transform(features.loc[train, available]),
                                       columns=available, index=train)
        selected, _detail = select_features(training_values, labels.loc[train], cfg, seed)
        model = get_model(cfg, seed, name=spec["model"])
        model.fit(training_values[selected], labels.loc[train])
        future_values = pd.DataFrame(preprocessing.transform(features.loc[future, available]),
                                     columns=available, index=future)
        predictions = pd.Series(model.predict(future_values[selected]), index=future)
        scores, score_space = score_samples(model, future_values[selected])
        scores = pd.Series(scores, index=future)
    fn_cost, fp_cost = canonical_costs(cfg)

    def summarize(indices, name):
        return {**_window_summary(stamps, labels, indices, name),
                "metrics": window_performance(labels.loc[indices], predictions.loc[indices],
                                              scores.loc[indices], fn_cost, fp_cost)}

    scaler = preprocessing.named_steps.get("scale")
    return {
        "model": spec["model"], "score_space": score_space,
        "nominal_train_indices": nominal_train.tolist(), "train_indices": train.tolist(),
        "excluded_boundary_indices": excluded.tolist(), "evaluation_indices": future.tolist(),
        "training": _window_summary(stamps, labels, train, 1),
        "available_features": available, "selected_features": selected,
        "dropped_training_empty_features": [name for name in features.columns if name not in available],
        "fitted_state": {"imputer_statistics": preprocessing.named_steps["impute"].statistics_.tolist(),
                         "scale_mean": scaler.mean_.tolist() if scaler is not None else None,
                         "scale_scale": scaler.scale_.tolist() if scaler is not None else None,
                         "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist()},
        "predictions": predictions.tolist(), "scores": scores.tolist(),
        "overall": summarize(future, "all_future"),
        "equal_count": [summarize(indices, number) for number, indices in enumerate(partitions[1:], 2)],
        "rolling": [summarize(indices, number) for number, indices in enumerate(
            rolling_windows(future, spec["window_size"], spec["step_size"]), 1)],
        "training_window_evaluated": False, "future_used_for_selection": False,
    }


def _analysis_plan(cfg, spec) -> dict:
    if cfg.get("run", {}).get("quick", False):
        raise ValueError("漂移分析不接受 quick 子采样，只分析完整时间序列")
    if cfg["feature_selection"].get("override_features_path") is not None:
        raise ValueError("漂移分析禁止 override_features_path，必须仅在早期训练窗选特征")
    if cfg["imbalance"]["strategy"] != "class_weight":
        raise ValueError("漂移分析固定使用 class_weight，不搜索不平衡策略")
    fn_cost, fp_cost = canonical_costs(cfg)
    if not all(np.isfinite(value) and value > 0 for value in (fn_cost, fp_cost)):
        raise ValueError("漂移分析 costs 必须为有限正数")
    return {
        "settings": {key: value for key, value in spec.items() if key != "enabled"},
        "n_bins": resolve_temporal_holdout(cfg)["prior_drift_bins"],
        "equal_count_partition": "stable_timestamp_sort_then_numpy_array_split",
        "sliding_windows": "sample_count_with_tail_anchored_last_window",
        "reference_window": "first_equal_count_window_descriptive_not_qualified_healthy_baseline",
        "performance_protocol": "fit_once_on_first_window_then_predict_only_future",
        "timestamp_ties": "purge_training_rows_tied_with_first_evaluation_timestamp",
        "timestamp_is_feature": False, "threshold_search": False, "model_selection": False,
        "decision": "predict_default_boundary", "random_state": int(cfg["random_state"]),
        "preprocessing": dict(cfg["preprocessing"]), "feature_selection": dict(cfg["feature_selection"]),
        "model_parameters": get_model(cfg, int(cfg["random_state"]), name=spec["model"]).get_params(),
        "costs": {"fn": fn_cost, "fp": fp_cost},
        "inference": "descriptive_only_no_independent_window_or_causal_claim",
    }


def data_fingerprint(features, labels, stamps) -> dict:
    digest = hashlib.sha256("\n".join(features.columns).encode("utf-8"))
    for values in (features.astype(float), labels.astype(int), stamps):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    sources = ("drift.py", "drift_reporting.py", "validation.py", "data_io.py", "evaluation.py",
               "feature_selection.py", "preprocessing.py", "modeling.py")
    return {"data_sha256": digest.hexdigest(),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in sources},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__}}


def build_drift_report(cfg, features, labels, timestamps) -> dict:
    spec = resolve_drift(cfg)
    plan = _analysis_plan(cfg, spec)
    stamps, order = _ordered_inputs(features, labels, timestamps)
    if not 2 <= plan["n_bins"] <= len(order):
        raise ValueError("漂移等分窗口数必须在 [2, 样本数] 内")
    partitions = [pd.Index(indices) for indices in np.array_split(order.to_numpy(), plan["n_bins"])]
    fingerprint = data_fingerprint(features, labels, stamps)
    performance = _performance_analysis(cfg, features, labels, stamps, partitions, spec)
    result = {"status": "ok", "title": DRIFT_TITLE, "artifact_schema_version": 1,
              "analysis_plan": plan, "fingerprint": fingerprint,
              "prior_drift": prior_analysis(stamps, labels, plan["n_bins"],
                                            spec["window_size"], spec["step_size"]),
              "feature_drift": _feature_analysis(features, labels, stamps, partitions, spec),
              "performance": performance}
    if data_fingerprint(features, labels, stamps) != fingerprint:
        raise ValueError("漂移分析运行期间数据或源码发生变化，拒绝混合产物")
    return result


def main() -> None:
    from .drift_reporting import run_drift_analysis, validate_drift_artifacts

    parser = argparse.ArgumentParser(description=DRIFT_TITLE)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--validate-only", action="store_true", help="从原始数据重算并核对全部漂移产物")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    features, labels, stamps = load_secom(
        str(PROJECT_ROOT / cfg["data"]["features_path"]),
        str(PROJECT_ROOT / cfg["data"]["labels_path"]), cfg["data"]["timestamp_format"])
    action = validate_drift_artifacts if args.validate_only else run_drift_analysis
    result = action(cfg, features, labels, stamps, PROJECT_ROOT / cfg["output"]["results_dir"])
    print(f"{DRIFT_TITLE}：{result['status']}，详见 {cfg['output']['results_dir']}/{DRIFT_MARKDOWN}")


if __name__ == "__main__":
    main()

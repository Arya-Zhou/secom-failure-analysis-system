"""折外校准层：校准器进唯一决策路径，成本阈值按训练侧完全折外的校准概率选。"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .evaluation import SCORE_SPACE_CALIBRATED, score_samples

logger = logging.getLogger(__name__)

CALIBRATION_METHODS = ("sigmoid", "isotonic")

# 分数空间的名字由 evaluation 唯一定义（它是 score_samples 的产出），此处只是转出给调用方。
CALIBRATED_SCORE_SPACE = SCORE_SPACE_CALIBRATED

# 单调性判定的浮点余量：sigmoid/isotonic 在数学上非递减，逐点求值仍有 ~1e-8 的抖动。
MONOTONIC_TOLERANCE = 1e-6

# AUC 判定余量：只在校准映射未并出新并列时才要求 AUC 逐位不变，理由见 structural_problems。
AUC_TOLERANCE = 1e-9


def resolve_calibration(cfg: dict) -> dict:
    """读 imbalance.calibration：主口径方法、同表对照方法、校准器内层 CV 折数。"""
    node = ((cfg.get("imbalance") or {}).get("calibration") or {})
    method = node.get("method")
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"imbalance.calibration.method 仅支持 {list(CALIBRATION_METHODS)}，"
            f"收到 {method!r}：校准方法必填、无默认值，它决定部署产物的分数空间语义")
    contrast = node.get("contrast_method")
    if contrast is not None:
        if contrast not in CALIBRATION_METHODS:
            raise ValueError(
                f"imbalance.calibration.contrast_method 仅支持 "
                f"{list(CALIBRATION_METHODS)}，收到 {contrast!r}")
        if contrast == method:
            raise ValueError(
                "imbalance.calibration.contrast_method 与 method 相同：对照行会与主口径"
                "逐位一致，读者会把复现当成对照。要么换一个方法，要么设为 null 关掉对照")
    cv_folds = int(node.get("cv_folds", 5))
    if cv_folds < 2:
        raise ValueError("imbalance.calibration.cv_folds 必须 >= 2")
    return {"method": method, "contrast_method": contrast, "cv_folds": cv_folds}


def build_calibrated_estimator(pipeline, method: str, cv_folds: int, seed: int):
    """校准整条链路以隔离内层 CV；ensemble=False 保持唯一基础链路。
    全训练集拟合后的同集 predict_proba 不是 OOF，部署阈值须另取完全折外分数。"""
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"未知校准方法: {method!r}，可选: {list(CALIBRATION_METHODS)}")
    return CalibratedClassifierCV(
        estimator=pipeline, method=method,
        cv=StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed)),
        ensemble=False,
    )


def is_calibrated(est) -> bool:
    """该 estimator 是否带校准层。判类型不判属性：属性能被 clone 丢掉，类型不会。"""
    return isinstance(est, CalibratedClassifierCV)


def base_pipeline(est):
    """取出被校准器包住的那条链路；未包校准器时返回它自己。"""
    if not is_calibrated(est):
        return est
    fitted = getattr(est, "calibrated_classifiers_", None)
    if not fitted:
        return est.estimator
    if len(fitted) != 1:
        raise ValueError(
            f"校准器内含 {len(fitted)} 条基础链路（ensemble=True）："
            "各条链路选出的特征不同，'所选特征'与'基础链路'都不再唯一，"
            "本项目的产物字段与三层校验按唯一链路设计，拒绝使用")
    return fitted[0].estimator


def raw_scores(est, X) -> tuple[np.ndarray, str]:
    """校准前的基础链路分数。**只作故障分诊，不作判据**——校验对象必须是最终输出。"""
    scores, space = score_samples(base_pipeline(est), X)
    return np.asarray(scores, dtype=float), space


def cv_protocol_name(est) -> str | None:
    """校准器内层 CV 的口径字符串，进产物摘要（换折数/换种子即换了一条 recipe）。"""
    if not is_calibrated(est):
        return None
    cv = est.get_params().get("cv")
    if isinstance(cv, StratifiedKFold):
        return (f"StratifiedKFold(n_splits={cv.n_splits}, shuffle={cv.shuffle}, "
                f"random_state={cv.random_state})")
    return str(cv)


def positive_class(est):
    """被当作正类分数那一列对应的类别标签，取不到记 None。"""
    classes = getattr(est, "classes_", None)
    if classes is None or len(classes) != 2:
        return None
    return int(classes[1])


def calibration_block(est) -> dict:
    """按已拟合对象**重算**校准语义字段，不采信任何记录值。"""
    return {
        "enabled": is_calibrated(est),
        "method": est.get_params().get("method") if is_calibrated(est) else None,
        "cv_protocol": cv_protocol_name(est),
        "positive_class": positive_class(est),
    }


def decision_graph(est, threshold) -> list[str]:
    """按已拟合对象重算推理图。重采样步骤不进图：它只在 fit 时起作用，predict 时被跳过。"""
    base = base_pipeline(est)
    graph = [name for name, step in getattr(base, "steps", [("model", base)])
             if not hasattr(step, "fit_resample")]
    if is_calibrated(est):
        graph.append("calibrator")
    graph.append("threshold" if threshold is not None else "argmax")
    return graph


def probe_coverage(scores, threshold) -> dict:
    """记录阈值两侧行数与不同输出分数数；不同输出数不代表校准曲线的区间覆盖数。"""
    arr = np.asarray(scores, dtype=float)
    below = int((arr < threshold).sum()) if threshold is not None else None
    above = int((arr >= threshold).sum()) if threshold is not None else None
    return {
        "n_rows": int(arr.size),
        "score_min": float(arr.min()) if arr.size else None,
        "score_max": float(arr.max()) if arr.size else None,
        "distinct_scores": int(np.unique(arr).size),
        "below_threshold": below,
        "at_or_above_threshold": above,
        "straddles_threshold": (None if threshold is None
                                else bool(below and above)),
    }


def assert_probe_straddles_threshold(coverage: dict, label: str) -> None:
    """探针必须覆盖阈值两侧，否则阈值篡改检测是哑的（改阈值不改变任何一行的预测）。"""
    if coverage.get("straddles_threshold") is False:
        raise ValueError(
            f"{label}: 探针 {coverage['n_rows']} 行全部落在阈值同一侧"
            f"（下方 {coverage['below_threshold']} / 上方 "
            f"{coverage['at_or_above_threshold']}，分数范围 "
            f"[{coverage['score_min']}, {coverage['score_max']}]）——"
            "此时改写阈值不会改变任何一行的最终预测，行为指纹对阈值篡改失效")


def _monotonicity(raw, calibrated) -> dict:
    """校准是分数的单调非递减映射：按校准前分数排序后校准后概率不得下降。
    isotonic 在节点间线性插值，其常数平台可能并出新并列，使 AUC 改变。"""
    raw_arr = np.asarray(raw, dtype=float)
    cal_arr = np.asarray(calibrated, dtype=float)
    order = np.argsort(raw_arr, kind="stable")
    diffs = np.diff(cal_arr[order])
    worst = float(diffs.min()) if diffs.size else 0.0
    n_raw = int(np.unique(raw_arr).size)
    n_cal = int(np.unique(cal_arr).size)
    return {"monotonic_non_decreasing": bool(worst >= -MONOTONIC_TOLERANCE),
            "max_decrease": float(-worst if worst < 0 else 0.0),
            "tolerance": MONOTONIC_TOLERANCE,
            "n_distinct_raw": n_raw, "n_distinct_calibrated": n_cal,
            "ties_introduced": bool(n_cal < n_raw)}


def calibration_diagnostics(est, X_train, y_train, oof_prob) -> dict:
    """正确性独立于篡改检测：映射单调性与值域在训练集上检查。
    相关方向与 Brier 使用完全折外概率，不使用测试集。"""
    y = np.asarray(y_train).astype(float)
    prob = np.asarray(est.predict_proba(X_train)[:, 1], dtype=float)
    raw, raw_space = raw_scores(est, X_train)
    oof = np.asarray(oof_prob, dtype=float)

    prior = float(y.mean())
    brier = float(np.mean((oof - y) ** 2))
    brier_prior = float(np.mean((prior - y) ** 2))
    corr = (float(np.corrcoef(oof, y)[0, 1])
            if np.std(oof) > 0 and np.std(y) > 0 else None)
    finite = bool(np.isfinite(prob).all() and np.isfinite(oof).all())

    diag = {
        "raw_score_space": raw_space,
        "n_train": int(y.size),
        "prior": prior,
        **_monotonicity(raw, prob),
        "auc_raw": float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else None,
        "auc_calibrated": (float(roc_auc_score(y, prob))
                           if len(np.unique(y)) == 2 else None),
        "oof_corr_with_label": corr,
        "oof_brier": brier,
        "brier_constant_prior": brier_prior,
        "oof_brier_improvement": brier_prior - brier,
        "prob_min": float(prob.min()), "prob_max": float(prob.max()),
        "oof_prob_min": float(oof.min()), "oof_prob_max": float(oof.max()),
        "shape_ok": bool(prob.shape == (len(y),) and oof.shape == (len(y),)),
        "finite": finite,
        "in_unit_interval": bool(prob.min() >= 0.0 and prob.max() <= 1.0
                                 and oof.min() >= 0.0 and oof.max() <= 1.0),
    }
    diag["auc_delta"] = (None if diag["auc_raw"] is None or diag["auc_calibrated"] is None
                         else diag["auc_calibrated"] - diag["auc_raw"])
    diag["structural_problems"] = structural_problems(diag)
    diag["quality_problems"] = quality_problems(diag)
    diag["passed"] = not (diag["structural_problems"] or diag["quality_problems"])
    return diag


def structural_problems(diag: dict) -> list[str]:
    """接错线才会犯的错：单调性反向 = 正类索引反了；越界/NaN/形状 = 装配错误。"""
    problems = []
    if not diag["monotonic_non_decreasing"]:
        problems.append(
            f"校准后概率不是校准前分数的单调非递减映射（最大下降 "
            f"{diag['max_decrease']:.3e}）：正类索引很可能反了")
    delta = diag.get("auc_delta")
    # 「校准不改变 AUC」只在映射严格单调时成立：AUC 给并列记 0.5 分，isotonic 把一对本来
    # 判反的样本并成同一概率，那一对反而由 0 分变 0.5 分，AUC 会升。故只在未并出新并列时断言。
    if (delta is not None and not diag["ties_introduced"]
            and abs(delta) > AUC_TOLERANCE):
        problems.append(
            f"校准映射未并出新并列（不同分数 {diag['n_distinct_raw']} → "
            f"{diag['n_distinct_calibrated']}）却让 AUC 变了 {delta:+.3e}："
            "校准后概率不是校准前分数的函数")
    if not diag["in_unit_interval"]:
        problems.append(
            f"校准后概率越出 [0,1]：训练集 [{diag['prob_min']:.4f}, "
            f"{diag['prob_max']:.4f}]，折外 [{diag['oof_prob_min']:.4f}, "
            f"{diag['oof_prob_max']:.4f}]")
    if not diag["finite"]:
        problems.append("校准后概率含 NaN 或 Inf")
    if not diag["shape_ok"]:
        problems.append("校准后概率形状与样本数不符")
    return problems


def quality_problems(diag: dict) -> list[str]:
    """数据说了算的那两条：装配全对、模型没用，照样不该被标成可部署工作点。"""
    problems = []
    corr = diag["oof_corr_with_label"]
    if corr is None or corr <= 0:
        problems.append(f"折外校准概率与真实标签的相关方向不为正（corr={corr}）")
    if diag["oof_brier_improvement"] <= 0:
        problems.append(
            f"折外 Brier {diag['oof_brier']:.6f} 不优于常数预测先验 "
            f"{diag['brier_constant_prior']:.6f}（改善 "
            f"{diag['oof_brier_improvement']:+.6f}）")
    return problems


def cost_grid(ratios, fp_cost: float, canonical_fn: float) -> list[dict]:
    """成本档位表：ratio = FN/FP，fp 固定为 canonical 的那一档，canonical 必须在表内。"""
    grid = [{"ratio": float(r), "fn": float(r) * float(fp_cost), "fp": float(fp_cost)}
            for r in ratios]
    canonical_ratio = float(canonical_fn) / float(fp_cost)
    if not any(abs(g["ratio"] - canonical_ratio) < 1e-12 for g in grid):
        raise ValueError(
            f"imbalance.cost_sensitivity 档位 {[g['ratio'] for g in grid]} 未包含 "
            f"canonical 成本比 {canonical_ratio:g}：主表与正式 pkl 绑的那一档"
            "必须在敏感性表里出现，否则读者无法把两张表对上")
    for g in grid:
        g["is_canonical"] = bool(abs(g["ratio"] - canonical_ratio) < 1e-12)
    return grid

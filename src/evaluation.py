"""评估指标：不平衡场景以 BER 与召回为主，而非准确率。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# 分数空间的封闭集合。校准后概率与模型自报概率语义不同，故不合并成一个名字。
SCORE_SPACE_PROBABILITY = "probability"
SCORE_SPACE_CALIBRATED = "calibrated_probability"
SCORE_SPACE_MARGIN = "decision_margin"
SCORE_SPACES = (SCORE_SPACE_PROBABILITY, SCORE_SPACE_CALIBRATED, SCORE_SPACE_MARGIN)

# 可当概率读（能算 Brier）的那些空间；decision_margin 不在内，不拿分数冒充概率。
PROBABILITY_SPACES = (SCORE_SPACE_PROBABILITY, SCORE_SPACE_CALIBRATED)


def balanced_error_rate(y_true, y_pred) -> float:
    """平衡错误率 BER = (FPR + FNR) / 2。"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] != 2:
        return 1.0
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return (fpr + fnr) / 2


def score_samples(model, X) -> tuple[np.ndarray, str]:
    """统一取分：有 predict_proba 用正类概率，否则用 decision_function（如岭分类器）。
    带校准层的 estimator 单列一个空间——它的概率经折外拟合的校准器映射过，与自报概率不同义。"""
    if hasattr(model, "predict_proba"):
        space = (SCORE_SPACE_CALIBRATED if isinstance(model, CalibratedClassifierCV)
                 else SCORE_SPACE_PROBABILITY)
        return model.predict_proba(X)[:, 1], space
    return np.ravel(model.decision_function(X)), SCORE_SPACE_MARGIN


def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """按给定阈值出预测：score >= threshold 判为失败(1)。"""
    scores, _ = score_samples(model, X)
    return (scores >= threshold).astype(int)


def expected_cost(y_true, y_pred, fn_cost: float, fp_cost: float) -> tuple[float, dict]:
    """期望代价与混淆矩阵计数。返回 (代价, {tn, fp, fn, tp})。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    cost = fn * fn_cost + fp * fp_cost
    return float(cost), {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """单数据集的分类指标（准确率、精确率、召回率、F1、BER、AUC），供策略对比复用。"""
    auc = None
    if y_score is not None and len(np.unique(np.asarray(y_true))) == 2:
        auc = float(roc_auc_score(y_true, y_score))
    return {
        "准确率": float(accuracy_score(y_true, y_pred)),
        "精确率": float(precision_score(y_true, y_pred, zero_division=0)),
        "召回率": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1分数": float(f1_score(y_true, y_pred, zero_division=0)),
        "BER": float(balanced_error_rate(y_true, y_pred)),
        "AUC": auc,
    }


def _auc_score(model, X, y) -> float | None:
    """AUC：优先 predict_proba，无该方法时用 decision_function（如岭分类器）。"""
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)
        return float(roc_auc_score(y, y_score))
    except Exception:  # noqa: BLE001：AUC 不可算时（如单一类别）记 None，不牵连其余指标
        return None


def evaluate_train_test(model, X_train, y_train, X_test, y_test,
                        threshold: float | None = None) -> dict:
    """已训练模型的训练/测试集评估，返回键与基线指标文件对齐。"""
    if threshold is None:
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)
    else:
        y_pred_train = predict_with_threshold(model, X_train, threshold)
        y_pred = predict_with_threshold(model, X_test, threshold)
    return {
        "训练集BER": float(balanced_error_rate(y_train, y_pred_train)),
        "测试集BER": float(balanced_error_rate(y_test, y_pred)),
        "准确率": float(accuracy_score(y_test, y_pred)),
        "精确率": float(precision_score(y_test, y_pred, zero_division=0)),
        "召回率": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1分数": float(f1_score(y_test, y_pred, zero_division=0)),
        "AUC": _auc_score(model, X_test, y_test),
    }


def recall_at_fp_budget(y_true, y_score, budget: int) -> float:
    """FP 预算内的最大召回：工艺侧每批只能复查 budget 片时，这条链路能抓到多少失效。"""
    y_arr = np.asarray(y_true).astype(int)
    n_pos = int((y_arr == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    ranked = y_arr[order]
    fp_cum, tp_cum = np.cumsum(ranked == 0), np.cumsum(ranked == 1)
    allowed = np.flatnonzero(fp_cum <= int(budget))
    return float(tp_cum[allowed[-1]] / n_pos) if allowed.size else 0.0


def extended_test_metrics(model, X_test, y_test, fn_cost: float, fp_cost: float,
                          fp_budget: int) -> dict:
    """测试集的补充指标（只增键不改 evaluate_train_test 的既有键，回归锚点不受影响）。
    Brier 需要概率，岭分类器只有 decision_function，记 None 而不用分数冒充概率。"""
    y_pred = np.asarray(model.predict(X_test))
    cost, cm = expected_cost(y_test, y_pred, fn_cost, fp_cost)
    scores, space = score_samples(model, X_test)
    brier = (float(np.mean((scores - np.asarray(y_test, dtype=float)) ** 2))
             if space in PROBABILITY_SPACES else None)
    return {
        "期望代价": cost,
        "测试集混淆": cm,
        "PR_AUC": float(average_precision_score(y_test, scores)),
        "Brier分数": brier,
        "Recall@FP预算": recall_at_fp_budget(y_test, scores, fp_budget),
        "FP预算": int(fp_budget),
        "分数空间": space,
    }


def cross_val_oof_metrics(
    model, X, y, n_splits: int, random_state: int,
    fn_cost: float, fp_cost: float,
) -> dict:
    """训练侧分层 CV 折外指标（逐折 clone→训练折 fit→验证折 predict）；调用方只许传训练集。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_bers = []
    for tr_idx, va_idx in skf.split(X, y):
        est = clone(model)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.asarray(est.predict(X.iloc[va_idx]))
        oof_pred[va_idx] = pred
        fold_bers.append(balanced_error_rate(y.iloc[va_idx], pred))
    cost, cm = expected_cost(y, oof_pred, fn_cost, fp_cost)
    return {
        "CV_BER均值": float(np.mean(fold_bers)),
        "CV_BER标准差": float(np.std(fold_bers)),
        "CV折外混淆": cm,
        "CV折外期望代价": cost,
    }


def resolve_cv_folds(requested: int, y) -> int:
    """折数按最小类样本数收敛：quick 模式小样本下 10 折会超过正样本数而报错。"""
    y_arr = np.asarray(y)
    n_min = int(min((y_arr == 0).sum(), (y_arr == 1).sum()))
    if n_min < 2:
        raise ValueError(f"训练集最小类只有 {n_min} 个样本，无法做分层交叉验证")
    folds = max(2, min(int(requested), n_min))
    if folds != int(requested):
        logger.warning(
            "CV 折数由 %d 收敛为 %d（训练集最小类仅 %d 个样本）",
            int(requested), folds, n_min)
    return folds


def compare_with_baseline(
    metrics: dict, baseline_path: str | Path, tolerance: float,
) -> tuple[bool, list[str]]:
    """将当前指标与基线文件记录的指标逐项比对，差值在容差内视为一致。"""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return False, [f"基线文件不存在: {baseline_path}"]
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    all_ok, lines = True, []
    for model_name, base_metrics in baseline.items():
        # 下划线前缀是说明性字段（_note / _selection），不是模型
        if model_name.startswith("_") or not isinstance(base_metrics, dict):
            continue
        if model_name not in metrics:
            all_ok = False
            lines.append(f"[缺失] {model_name}: 本次运行未包含该模型")
            continue
        for key, base_val in base_metrics.items():
            if base_val is None:
                continue
            new_val = metrics[model_name].get(key)
            if new_val is None:
                all_ok = False
                lines.append(f"[缺失] {model_name}.{key}: 本次无该指标")
                continue
            diff = abs(new_val - base_val)
            ok = diff < tolerance
            all_ok = all_ok and ok
            lines.append(
                f"[{'PASS' if ok else 'FAIL'}] {model_name}.{key}: "
                f"新={new_val:.4f} 基线={base_val:.4f} 差={diff:.4f}"
            )
    return all_ok, lines

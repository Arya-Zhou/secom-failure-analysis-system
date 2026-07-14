"""评估 —— 不平衡场景以 BER/召回为主，而非准确率。

BER = (FPR + FNR) / 2，对两类等权，避免被多数类(通过)主导。
搬运自 secom.ipynb 的 balanced_error_rate 与 build_and_evaluate_models
的指标计算部分；输出键与 baseline_metrics.json 对齐（中文键）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def balanced_error_rate(y_true, y_pred) -> float:
    """平衡错误率 BER = (FPR + FNR) / 2。"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] != 2:
        return 1.0
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return (fpr + fnr) / 2


def _auc_score(model, X, y) -> float | None:
    """AUC：优先 predict_proba，否则 decision_function（如岭分类器）。"""
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)
        return float(roc_auc_score(y, y_score))
    except Exception:  # noqa: BLE001 —— 与 notebook 的 try/except 行为一致
        return None


def evaluate_train_test(model, X_train, y_train, X_test, y_test) -> dict:
    """已训练模型的训练/测试集评估，键与 baseline_metrics.json 对齐。"""
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    return {
        "训练集BER": float(balanced_error_rate(y_train, y_pred_train)),
        "测试集BER": float(balanced_error_rate(y_test, y_pred)),
        "准确率": float(accuracy_score(y_test, y_pred)),
        "精确率": float(precision_score(y_test, y_pred, zero_division=0)),
        "召回率": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1分数": float(f1_score(y_test, y_pred, zero_division=0)),
        "AUC": _auc_score(model, X_test, y_test),
    }


def cross_val_ber(model, X, y, n_splits: int, random_state: int) -> dict:
    """10 折分层交叉验证 BER（notebook: StratifiedKFold + make_scorer）。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(balanced_error_rate, greater_is_better=False)
    scores = -cross_val_score(model, X, y, cv=skf, scoring=scorer)
    return {"CV平均BER": float(scores.mean()), "CV_BER标准差": float(scores.std())}


def compare_with_baseline(
    metrics: dict, baseline_path: str | Path, tolerance: float,
) -> tuple[bool, list[str]]:
    """当前指标 vs notebook 基线，容差内视为一致。

    返回 (是否全部通过, 明细行列表)。基线中的 None（如无 AUC）跳过。
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return False, [f"基线文件不存在: {baseline_path}"]
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    all_ok, lines = True, []
    for model_name, base_metrics in baseline.items():
        if not isinstance(base_metrics, dict):  # 跳过 _note 等说明字段
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

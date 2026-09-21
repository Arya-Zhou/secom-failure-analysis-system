"""策略工作点预测适配器：只走受控工作点入口，不直接取用内部 pipeline。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .demo_data import (
    CHAIN_NOTE, CONFIG_PATH, FROZEN_SAMPLES, PROJECT_ROOT, SAMPLE_NOTE,
    frozen_row, frozen_sample, load_raw_features, operating_point_paths,
    predict_inputs_ready, predict_mode_enabled, prepare_operating_point_features,
)
from .imbalance import (
    apply_operating_point, load_operating_point, operating_point_input_names,
    score_operating_point, verify_operating_point,
)

FN_COST = 10.0
FP_COST = 1.0


def load_ridge_operating_point(root: Path = PROJECT_ROOT):
    """加载 Ridge 策略工作点，强制核对外部清单并做产物自检。"""
    pkl, manifest = operating_point_paths(root)
    op = load_operating_point(pkl, manifest=manifest, require_manifest=True)
    verify_operating_point(op)
    return op


def expected_feature_names(op: dict) -> list[str]:
    """工作点拟合时的输入列名。"""
    return operating_point_input_names(op)


def aligned_features(X: pd.DataFrame, op: dict) -> pd.DataFrame:
    """删除全空列并与工作点列名对齐。"""
    return prepare_operating_point_features(X, expected_feature_names(op))


def predict_frame(op: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """对已对齐特征出预测与分数。"""
    preds = apply_operating_point(op, X)
    scores, space = score_operating_point(op, X)
    return np.asarray(preds), np.asarray(scores, dtype=float), space


def wafer_cost(y_true: int, y_pred: int) -> float:
    """单片相对代价：FN×10 + FP×1。"""
    if int(y_true) == 1 and int(y_pred) == 0:
        return FN_COST
    if int(y_true) == 0 and int(y_pred) == 1:
        return FP_COST
    return 0.0


def verify_frozen_predictions(op: dict, X: pd.DataFrame, y: pd.Series) -> None:
    """冻结四象限样本必须与预存期望逐位一致。"""
    prepared = aligned_features(X, op)
    for row in FROZEN_SAMPLES:
        wafer_id = row["wafer_id"]
        actual_true = int(y.loc[wafer_id])
        if actual_true != row["y_true"]:
            raise ValueError(f"晶圆 {wafer_id} 真实标签变为 {actual_true}，与冻结清单不符")
        preds, scores, _space = predict_frame(op, frozen_row(prepared, wafer_id))
        if int(preds[0]) != row["y_pred"]:
            raise ValueError(
                f"晶圆 {wafer_id} 工作点预测 {int(preds[0])}，期望 {row['y_pred']}"
            )
        if not np.isclose(float(scores[0]), row["score"], atol=1e-9, rtol=0):
            raise ValueError(
                f"晶圆 {wafer_id} 工作点分数 {float(scores[0])}，期望 {row['score']}"
            )


def predict_frozen_wafer(op: dict, X: pd.DataFrame, y: pd.Series, wafer_id: int) -> dict:
    """对冻结样本做策略工作点预测。"""
    sample = frozen_sample(wafer_id)
    prepared = aligned_features(X, op)
    preds, scores, space = predict_frame(op, frozen_row(prepared, wafer_id))
    y_true = int(y.loc[wafer_id])
    y_pred = int(preds[0])
    score = float(scores[0])
    if y_true != sample["y_true"] or y_pred != sample["y_pred"]:
        raise ValueError(f"晶圆 {wafer_id} 的预测结果与冻结期望不一致")
    if not np.isclose(score, sample["score"], atol=1e-9, rtol=0):
        raise ValueError(f"晶圆 {wafer_id} 的分数与冻结期望不一致")
    label = {0: "正常", 1: "失效"}
    return {
        "chain_note": CHAIN_NOTE,
        "sample_note": SAMPLE_NOTE,
        "case": sample["case"],
        "wafer_id": wafer_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_text": label[y_true],
        "y_pred_text": label[y_pred],
        "score": score,
        "score_text": f"{score:.4f}",
        "score_space": space,
        "threshold": op.get("threshold"),
        "cost": wafer_cost(y_true, y_pred),
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
    }


def predict_status_markdown(status: dict | None = None) -> str:
    """本机预测未启用时的条件说明。"""
    status = predict_inputs_ready() if status is None else status
    lines = [
        "本机预测模式未启用。需要同时具备原始数据、工作点产物和外部清单：",
        f"- 特征：{status['features_path']}（{'有' if status['features'] else '缺'}）",
        f"- 标签：{status['labels_path']}（{'有' if status['labels'] else '缺'}）",
        f"- 工作点：{status['operating_point_path']}（{'有' if status['operating_point'] else '缺'}）",
        f"- 清单：{status['manifest_path']}（{'有' if status['manifest'] else '缺'}）",
        "报告模式仍可浏览总览、候选、时间风险和主分析案例。",
    ]
    return "\n".join(lines)


def frozen_display_result(wafer_id: int) -> dict:
    """不加载工作点时，用冻结清单填充与预测页相同的字段。"""
    sample = frozen_sample(wafer_id)
    label = {0: "正常", 1: "失效"}
    y_true = int(sample["y_true"])
    y_pred = int(sample["y_pred"])
    score = float(sample["score"])
    return {
        "chain_note": CHAIN_NOTE,
        "sample_note": SAMPLE_NOTE,
        "case": sample["case"],
        "wafer_id": wafer_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_text": label[y_true],
        "y_pred_text": label[y_pred],
        "score": score,
        "score_text": f"{score:.4f}",
        "score_space": "decision_margin",
        "threshold": None,
        "cost": wafer_cost(y_true, y_pred),
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
    }


def format_prediction(result: dict) -> str:
    """工作点预测页文案。"""
    threshold = result["threshold"]
    threshold_text = "产物默认决策边界" if threshold is None else str(threshold)
    return "\n".join([
        result["chain_note"],
        result["sample_note"],
        "",
        "**样本**",
        f"- 象限：{result['case']}",
        f"- 晶圆：{result['wafer_id']}",
        "",
        "**判定**",
        f"- 真实标签：{result['y_true_text']}（{result['y_true']}）",
        f"- 工作点判定：{result['y_pred_text']}（{result['y_pred']}）",
        "",
        "**分数**",
        f"- 决策分数：{result['score_text']}（{result['score_space']}）",
        f"- 阈值：{threshold_text}",
        "",
        "**相对代价**",
        f"- 规则：FN×{result['fn_cost']:.0f} + FP×{result['fp_cost']:.0f}",
        f"- 本片：{result['cost']:.0f}",
    ])


def startup_predict_bundle(root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH):
    """本机预测模式启动：加载一次工作点并核对冻结样本。"""
    if not predict_mode_enabled(predict_inputs_ready(root, config_path)):
        return None
    op = load_ridge_operating_point(root)
    X, y = load_raw_features(config_path)
    verify_frozen_predictions(op, X, y)
    return {"op": op, "X": X, "y": y}

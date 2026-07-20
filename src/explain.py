"""可解释性 / 失效根因分析。

使用 SHAP 提供两层解释：
  1. explain_global：全局特征重要性（测试集聚合，平均 |SHAP|）
  2. explain_wafer：单晶圆失效归因（样本级 SHAP 分解 + 文本报告 + 贡献图）

解释器按模型自动选择（_make_explainer）：
  - 线性模型（有 coef_）  -> LinearExplainer，解析解，毫秒级
  - 树集成（estimators_） -> TreeExplainer，多项式时间精确解
  - 其他                  -> KernelExplainer 兜底（采样近似，慢，背景需降采样）

两个必须区分的概念（报告与 JSON 中均显式标注，不可混用）：
  - 展示分数（_model_score）：给人看的模型判定强度（概率优先，其次决策分数）；
  - 解释空间（output_space）：SHAP 值实际分解所在的输出空间——linear 为
    decision margin（对逻辑回归即 log-odds），tree 为正类概率，kernel 与
    传入的 predict_fn 严格一致。自洽校验只能在解释空间内做
    （baseline + sum(SHAP) ≈ _explained_output），拿概率去对 margin 是错的。

注意：SHAP 背景数据（background）必须是能代表总体分布的样本集（如训练集抽样），
绝不能用待解释样本自身——那样期望基线=样本输出，所有 SHAP 值恒为 0。

图表内文字一律用英文：运行环境（WSL/CI）通常无中文字体，中文会渲染成方框；
中文表述放在 .txt / .json 产物中（不依赖字体）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（WSL/CI）下生成图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# KernelExplainer 兜底时的背景/解释样本上限（防止运行时间失控）
_KERNEL_BACKGROUND_MAX = 50
_KERNEL_EXPLAIN_MAX = 100

# 自洽校验容差：解析解（linear/tree）应精确；kernel 为采样近似，放宽但仍须接近
_CONSISTENCY_TOL = {"linear": 1e-6, "tree": 1e-6, "kernel": 1e-2}

_CASE_DESC = {
    "TP": "真实失效，模型命中",
    "FN": "真实失效，模型漏检",
    "FP": "真实正常，模型误报",
    "TN": "真实正常，模型正常",
}


def _make_explainer(model, background: pd.DataFrame, seed: int = 0):
    """按模型类型选择 SHAP 解释器。

    返回 (explainer, method, output_space)：
      method in {"linear", "tree", "kernel"}；
      output_space 为 SHAP 值分解所在的输出空间标注，自洽校验必须以它为准。
    """
    if hasattr(model, "estimators_"):  # 树集成（RandomForest 等）
        try:
            return shap.TreeExplainer(model), "tree", "positive-class probability"
        except Exception as e:
            logger.warning("TreeExplainer 初始化失败，降级 KernelExplainer: %s", e)
    elif hasattr(model, "coef_"):  # 线性模型：精确解析解
        space = ("log-odds (decision margin)" if hasattr(model, "predict_proba")
                 else "decision margin")
        return shap.LinearExplainer(model, background), "linear", space

    # 兜底：模型无关的采样近似。背景降采样，否则运行时间失控。
    bg = shap.sample(background, min(_KERNEL_BACKGROUND_MAX, len(background)),
                     random_state=seed)

    def predict_fn(x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        if hasattr(model, "decision_function"):
            return np.ravel(model.decision_function(x))
        return model.predict(x).astype(float)

    if hasattr(model, "predict_proba"):
        space = "positive-class probability"
    elif hasattr(model, "decision_function"):
        space = "decision margin"
    else:
        space = "predicted class (0/1)"
    return shap.KernelExplainer(predict_fn, bg), "kernel", space


def _explained_output(model, x_2d, method: str) -> float:
    """SHAP 分解所在空间的模型原始输出——自洽校验的唯一正确比较对象。

    与 _make_explainer 的空间约定严格一致：
      linear -> decision_function（逻辑回归即 log-odds）
      tree   -> 正类概率
      kernel -> 与 predict_fn 的分支完全相同
    """
    if method == "tree":
        return float(model.predict_proba(x_2d)[0, 1])
    if method == "linear":
        return float(np.ravel(model.decision_function(x_2d))[0])
    # kernel：镜像 _make_explainer 中 predict_fn 的分支
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_2d)[0, 1])
    if hasattr(model, "decision_function"):
        return float(np.ravel(model.decision_function(x_2d))[0])
    return float(model.predict(x_2d)[0])


def _positive_class_values(shap_values) -> np.ndarray:
    """统一 SHAP 输出格式为二维 (n_samples, n_features)，二分类取正类。"""
    if isinstance(shap_values, list):  # 老接口：[负类, 正类]
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # 新接口：(n, features, classes)
        shap_values = shap_values[:, :, 1]
    return shap_values


def _model_score(model, x_2d) -> tuple[float, str]:
    """返回 (展示分数, 语义标签)。概率与决策分数严格区分，不冒充。"""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_2d)[0, 1]), "失效概率"
    if hasattr(model, "decision_function"):
        return float(np.ravel(model.decision_function(x_2d))[0]), "决策分数(>0 判失效)"
    return float(model.predict(x_2d)[0]), "预测类别"


def pick_case_positions(y_true, y_pred, scores) -> dict:
    """TP/FN 案例极值选样（确定性，不依赖随机）。

    TP = 失效分数最高的命中（模型最有把握抓对的），
    FN = 失效分数最高的漏检（最接近判定阈值、"差一点就抓到"，衔接阈值移动叙事）。

    返回 {"TP": 位置索引或 None, "FN": 位置索引或 None}；某类不存在时为 None，
    由调用方记录 warning（跳过就写跳过，不与"通过"混淆）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    out = {}
    for case, mask in (("TP", (y_true == 1) & (y_pred == 1)),
                       ("FN", (y_true == 1) & (y_pred == 0))):
        pos_arr = np.flatnonzero(mask)
        out[case] = int(pos_arr[np.argmax(scores[pos_arr])]) if len(pos_arr) else None
    return out


def _validate_columns(feature_names: list[str], n_cols: int, cols_a, cols_b) -> None:
    """输入维度与列序校验：错列序会导致错误归因，必须显式失败。"""
    if len(feature_names) != n_cols:
        raise ValueError(
            f"feature_names 数量({len(feature_names)})与特征列数({n_cols})不一致")
    if list(cols_a) != list(cols_b):
        raise ValueError("待解释数据与 background 的特征列不一致（数量或顺序）")


def explain_global(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    output_dir: Path,
    background: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict:
    """全局特征影响分析（平均 |SHAP| 排名）。

    Args:
        model: 已训练的分类器
        X: 待解释特征矩阵（DataFrame，保留特征名以避免 sklearn 警告）
        feature_names: 特征名列表
        model_name: 模型展示名（用于文件名/日志）
        output_dir: 输出目录
        background: SHAP 背景数据（训练集抽样）；None 则退化用 X 本身
        seed: 随机种子（KernelExplainer 降采样用）

    生成文件：
        - shap_summary_bar_<model_name>.png : Top-15 特征重要性柱状图
        - shap_values_<model_name>.json     : 全部特征 SHAP 汇总 + 解释元数据

    返回 dict：{"shap_method", "output_space", "importance": {feature: mean_abs_shap 降序}}，
    供调用方（pipeline manifest）回填解释元数据。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if background is None:
        background = X
    if len(background) == 0:
        raise ValueError("背景数据为空（explain.background_size 必须大于 0）")
    _validate_columns(feature_names, X.shape[1], background.columns, X.columns)

    explainer, method, space = _make_explainer(model, background, seed)
    if method == "kernel" and len(X) > _KERNEL_EXPLAIN_MAX:
        logger.info("[SHAP] kernel 方法较慢，解释样本 %d -> %d", len(X), _KERNEL_EXPLAIN_MAX)
        X = X.sample(_KERNEL_EXPLAIN_MAX, random_state=seed)

    logger.info("[SHAP] 全局解释 | 方法=%s | 解释空间=%s | 样本=%d | 模型=%s",
                method, space, len(X), model_name)
    shap_values = _positive_class_values(explainer.shap_values(X))

    mean_abs = np.abs(shap_values).mean(axis=0)
    std_abs = np.abs(shap_values).std(axis=0)
    order = np.argsort(-mean_abs)

    for i in order[:5]:
        logger.info("  Top特征 %s: mean|SHAP|=%.4f", feature_names[i], mean_abs[i])

    # 1. Top-15 柱状图（英文标签：环境无中文字体时避免方框）
    top = order[: min(15, len(order))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), mean_abs[top], color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feature_names[i] for i in top])
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"Global Feature Importance (SHAP, method={method})")
    fig.tight_layout()
    png_path = output_dir / f"shap_summary_bar_{model_name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[SHAP] 全局重要性图已保存: %s", png_path)

    # 2. JSON 汇总（含解释元数据：任何旧产物可独立判断来源与能否复用）
    summary = {
        "model": model_name,
        "shap_method": method,
        "output_space": space,
        "random_seed": int(seed),
        "background_size": int(len(background)),
        "n_samples_explained": int(shap_values.shape[0]),
        "features": {
            feature_names[i]: {
                "mean_abs_shap": float(mean_abs[i]),
                "std_abs_shap": float(std_abs[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(order)
        },
    }
    json_path = output_dir / f"shap_values_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("[SHAP] 全局汇总已保存: %s", json_path)

    return {
        "shap_method": method,
        "output_space": space,
        "importance": {feature_names[i]: float(mean_abs[i]) for i in order},
    }


def explain_wafer(
    model,
    x_row: pd.Series,
    y_true: int,
    feature_names: list[str],
    model_name: str,
    output_dir: Path,
    background: pd.DataFrame,
    wafer_id,
    case: str | None = None,
    top_n: int = 5,
    seed: int = 0,
) -> dict:
    """单晶圆失效归因（样本级 SHAP 分解）。

    Args:
        model: 已训练的分类器
        x_row: 单个样本（pd.Series，保留特征名）
        y_true: 真实标签（0=正常, 1=失效）
        feature_names: 特征名列表
        model_name: 模型展示名
        output_dir: 输出目录
        background: SHAP 背景数据（训练集抽样，不可用样本自身）
        wafer_id: 晶圆稳定标识（数据集原始行号），用于文件名与追溯
        case: 案例类型标注（TP/FN/FP/TN），提供时写入报告并作为文件名后缀
        top_n: 报告中列出的拉动/抑制特征数（文本 Top-5；贡献图固定 Top-10）
        seed: 随机种子

    生成文件（case 提供时文件名追加 _<case> 后缀）：
        - shap_explanation_wafer_<id>[_case].txt  : 文本归因报告
        - shap_contribution_wafer_<id>[_case].png : Top-10 签名贡献条形图
          （红=拉动失效，蓝=抑制；非严格意义的 SHAP waterfall，故不用该名）

    返回 dict：预测、基线、自洽校验结果、各特征 SHAP 值。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(background) == 0:
        raise ValueError("背景数据为空（explain.background_size 必须大于 0）")
    _validate_columns(feature_names, len(x_row), x_row.index, background.columns)

    x_2d = x_row.to_frame().T  # 保持 DataFrame 形态（含特征名）
    explainer, method, space = _make_explainer(model, background, seed)
    sv = _positive_class_values(explainer.shap_values(x_2d))[0]

    baseline = explainer.expected_value
    if isinstance(baseline, (list, np.ndarray)):
        baseline = float(np.ravel(baseline)[-1])
    else:
        baseline = float(baseline)

    score, score_label = _model_score(model, x_2d)
    pred_class = int(model.predict(x_2d)[0])

    # ---- 自洽校验：只在解释空间内比较；偏差如实输出，超容差显式标注 ----
    explained_out = _explained_output(model, x_2d, method)
    recon = baseline + float(sv.sum())
    deviation = abs(recon - explained_out)
    tol = _CONSISTENCY_TOL[method] * max(1.0, abs(explained_out))
    consistency_ok = deviation <= tol
    if not consistency_ok:
        logger.warning(
            "[SHAP] 自洽校验超容差: 晶圆=%s 模型=%s 方法=%s recon=%.6f 期望=%.6f 偏差=%.2e",
            wafer_id, model_name, method, recon, explained_out, deviation,
        )

    contrib = sorted(
        zip(feature_names, x_row.to_numpy(), sv), key=lambda t: abs(t[2]), reverse=True
    )
    pushing = [(n, v, s) for n, v, s in contrib if s > 0][:top_n]
    protecting = [(n, v, s) for n, v, s in contrib if s < 0][:top_n]

    # ---- 文本报告（中文放 txt，不依赖字体）----
    case_line = f" | 案例类型: {case}（{_CASE_DESC.get(case, '')}）" if case else ""
    lines = [
        "单晶圆失效归因报告",
        "=" * 62,
        f"晶圆编号(数据集行号): {wafer_id}{case_line}",
        f"模型: {model_name} | SHAP 方法: {method}",
        f"真实状态: {'失效' if y_true == 1 else '正常'}",
        f"模型判定: {'失效' if pred_class == 1 else '正常'} | 展示分数[{score_label}]: {score:.4f}",
        "",
        f"SHAP 解释空间: {space}（自洽校验在该空间进行，勿与展示分数直接比较）",
        f"自洽校验: 基线({baseline:.4f}) + SHAP总和({sv.sum():.4f}) = {recon:.4f}",
        f"          解释空间模型输出 = {explained_out:.4f} | 偏差 = {deviation:.2e}"
        f" [{'OK' if consistency_ok else '超容差，该解释存疑'}]",
        "",
        f"拉动失效风险 Top{len(pushing)}（SHAP > 0）:",
    ]
    lines += [
        f"  {i}. {n}: +{s:.4f} (标准化后取值={v:.3f})"
        for i, (n, v, s) in enumerate(pushing, 1)
    ] or ["  (无)"]
    lines += ["", f"抑制失效风险 Top{len(protecting)}（SHAP < 0）:"]
    lines += [
        f"  {i}. {n}: {s:.4f} (标准化后取值={v:.3f})"
        for i, (n, v, s) in enumerate(protecting, 1)
    ] or ["  (无)"]
    lines += [
        "",
        "注: 特征为匿名传感器参数，以上为统计关联候选（非确证根因），",
        "    接真实产线数据后需结合工艺知识与实验验证。",
        "=" * 62,
    ]
    suffix = f"_{case}" if case else ""
    txt_path = output_dir / f"shap_explanation_wafer_{wafer_id}{suffix}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[SHAP] 单晶圆报告已保存: %s", txt_path)

    # ---- Top-10 签名贡献图 ----
    top10 = contrib[:10][::-1]  # barh 自下而上
    names = [t[0] for t in top10]
    vals = [t[2] for t in top10]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (red: push to fail, blue: protect)")
    title_case = f", case={case}" if case else ""
    ax.set_title(f"Wafer #{wafer_id} SHAP Contribution (method={method}{title_case})")
    fig.tight_layout()
    png_path = output_dir / f"shap_contribution_wafer_{wafer_id}{suffix}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[SHAP] 单晶圆贡献图已保存: %s", png_path)

    return {
        "wafer_id": wafer_id,
        "case": case,
        "shap_method": method,
        "output_space": space,
        "y_true": int(y_true),
        "y_pred": pred_class,
        "score": score,
        "score_label": score_label,
        "baseline": baseline,
        "explained_output": explained_out,
        "reconstruction": recon,
        "consistency_ok": bool(consistency_ok),
        "shap_values": {n: float(s) for n, _, s in contrib},
    }

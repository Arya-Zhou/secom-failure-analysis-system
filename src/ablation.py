"""2x2 消融实验：特征选择方式（四方法投票 / SHAP 引导）× 不平衡处理（无 / class_weight）；
输入原始训练/测试划分，模块内自建严格链路（只在训练集拟合）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import canonical_costs
from .evaluation import evaluate_train_test, expected_cost
from .explain import shap_feature_ranking
from .feature_selection import select_features
from .modeling import MODEL_DISPLAY_NAMES, make_model, registry_names
from .preprocessing import build_preprocess_pipeline

logger = logging.getLogger(__name__)

# 消融两个因子的取值。selector 决定 40 个特征怎么选，weighting 决定训练时是否类别加权。
SELECTORS = ("vote", "shap")
WEIGHTINGS = ("none", "class_weight")

SELECTOR_LABELS = {"vote": "四方法投票", "shap": "SHAP 引导"}
WEIGHTING_LABELS = {"none": "无", "class_weight": "class_weight"}
_CELL_SHORT = {("vote", "none"): "Vote", ("vote", "class_weight"): "Vote+CW",
               ("shap", "none"): "SHAP", ("shap", "class_weight"): "SHAP+CW"}


def _class_weight_of(weighting: str):
    return "balanced" if weighting == "class_weight" else None


def cell_key(selector: str, weighting: str) -> str:
    """消融格的稳定键名，json 与 md 共用。"""
    return f"{selector}__{weighting}"


def _evaluate_cell(model, X_tr, y_tr, X_te, y_te, fn_cost, fp_cost) -> dict:
    """训练并评估一个消融格，返回指标 + 混淆矩阵 + 期望代价。"""
    model.fit(X_tr, y_tr)
    metrics = evaluate_train_test(model, X_tr, y_tr, X_te, y_te)
    cost, cm = expected_cost(y_te, model.predict(X_te), fn_cost, fp_cost)
    return {**metrics, "confusion": cm, "expected_cost": cost}


def _train_recall(model, X, y) -> float | None:
    """探针模型在训练集上的召回，用于判读 SHAP 排名是否出自一个近乎恒定输出的模型。"""
    mask = np.asarray(y) == 1
    if not mask.any():
        return None
    return float((np.asarray(model.predict(X))[mask] == 1).mean())


def _shap_features(cfg, model_name, weighting, X_tr, y_tr, k, seed) -> tuple[list[str], dict]:
    """SHAP 引导特征选择：全维探针 -> 训练集平均 |SHAP| 排名 -> 取前 k 个（测试集不参与）。

    探针的 class_weight 跟随本格，使同一格内"选特征"与"训练"口径一致。
    """
    probe = make_model(model_name, seed, _class_weight_of(weighting))
    probe.fit(X_tr, y_tr)
    bg_size = int((cfg.get("explain") or {}).get("background_size", 100))
    if bg_size <= 0:
        raise ValueError("explain.background_size 必须大于 0")
    background = X_tr.sample(min(bg_size, len(X_tr)), random_state=seed)
    feats, method = shap_feature_ranking(
        probe, X_tr, list(X_tr.columns), background, seed=seed, top_k=k)
    probe_info = {
        "probe_model": model_name,
        "probe_class_weight": _class_weight_of(weighting),
        "probe_n_features": int(X_tr.shape[1]),
        "probe_train_recall": _train_recall(probe, X_tr, y_tr),
        "shap_method": method,
        "background_size": int(len(background)),
    }
    return feats, probe_info


def _plot_grid(models_result: dict, out_dir: Path) -> str:
    """两幅子图（测试集 BER / 召回率），x 轴为四个消融格，同一模型一组柱。"""
    cells = [(s, w) for s in SELECTORS for w in WEIGHTINGS]
    labels = [_CELL_SHORT[c] for c in cells]
    names = list(models_result)
    x = np.arange(len(cells))
    width = 0.8 / max(len(names), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in (
        (axes[0], "测试集BER", "Test BER (lower is better)"),
        (axes[1], "召回率", "Recall on failed wafers (higher is better)"),
    ):
        for i, name in enumerate(names):
            vals = [models_result[name]["cells"][cell_key(s, w)][key] for s, w in cells]
            # 图内文字用英文：注册名本就是英文，展示名（中文）依赖字体，运行环境未必有
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("feature selection / class weighting")
        ax.grid(axis="y", alpha=0.3)
    # BER=0.5 是无技能基准（全判正常即为此值），画出来才看得出哪些格根本没抓到失效。
    # 用文字标注而非图例：柱子占满绘图区，图例框会盖住数据。
    axes[0].axhline(0.5, color="gray", ls="--", lw=1.0)
    axes[0].set_ylim(0, 0.6)
    axes[0].text(len(cells) - 0.55, 0.508, "no-skill (BER=0.5)",
                 ha="right", va="bottom", fontsize=8, color="gray")
    axes[0].set_ylabel("BER")
    axes[1].set_ylabel("Recall")
    axes[1].legend(title="model", fontsize=8)
    fig.suptitle("Ablation: feature selection (Vote vs SHAP) x class weighting")
    fig.tight_layout()
    path = out_dir / "ablation_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[消融] 网格图已保存: %s", path)
    return path.name


def _write_markdown(result: dict, out_dir: Path) -> str:
    """把 json 结果渲染成对外可读的对比表与结论段。"""
    n_test = result["n_test"]
    n_fail = result["n_test_failed"]
    costs = result["costs"]
    lines = [
        "# 消融实验：特征选择方式 × 不平衡处理",
        "",
        f"- 统一口径：80/20 分层划分（seed={result['random_state']}），"
        f"测试集 {n_test} 片（失败 {n_fail} 片）；每格特征数 {result['n_features']}",
        "- 严格链路：填充/标准化只在训练集拟合，两种特征选择也只看训练集，"
        "测试集仅做最终评估",
        "- SHAP 引导：先在全维训练集上训一个探针模型，按训练集平均 |SHAP| 取前 N 个特征后重训；"
        "探针的 class_weight 与本格一致",
        f"- 成本假设（与策略对比同口径）：漏检 FN={costs['fn']:g}，误报 FP={costs['fp']:g}",
        "",
    ]
    for name, res in result["models"].items():
        display = MODEL_DISPLAY_NAMES.get(name, name)
        lines += [
            f"## {display}",
            "",
            "| 特征选择 | 类别加权 | Accuracy | Recall | Precision | F1 | AUC | BER | FN | FP | 期望代价 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in SELECTORS:
            for w in WEIGHTINGS:
                c = res["cells"][cell_key(s, w)]
                auc = f"{c['AUC']:.3f}" if c["AUC"] is not None else "N/A"
                lines.append(
                    f"| {SELECTOR_LABELS[s]} | {WEIGHTING_LABELS[w]} | "
                    f"{c['准确率']:.3f} | {c['召回率']:.3f} | {c['精确率']:.3f} | "
                    f"{c['F1分数']:.3f} | {auc} | {c['测试集BER']:.3f} | "
                    f"{c['confusion']['fn']} | {c['confusion']['fp']} | {c['expected_cost']:g} |"
                )
        eff = res["effects"]
        lines += [
            "",
            f"- 加类别加权的效果（四方法投票下）：BER {eff['weighting_on_vote']['BER变化']:+.3f}、"
            f"召回 {eff['weighting_on_vote']['召回变化']:+.3f}、"
            f"漏检 {eff['weighting_on_vote']['漏检变化']:+d} 片",
            f"- 换 SHAP 引导选特征的效果（不加权时）：BER {eff['selector_on_none']['BER变化']:+.3f}、"
            f"召回 {eff['selector_on_none']['召回变化']:+.3f}、"
            f"漏检 {eff['selector_on_none']['漏检变化']:+d} 片",
            f"- 换 SHAP 引导选特征的效果（加权时）：BER {eff['selector_on_weighted']['BER变化']:+.3f}、"
            f"召回 {eff['selector_on_weighted']['召回变化']:+.3f}、"
            f"漏检 {eff['selector_on_weighted']['漏检变化']:+d} 片",
            f"- 两种特征选择的重合：不加权 {res['feature_overlap']['none']} 个、"
            f"加权 {res['feature_overlap']['class_weight']} 个（共 {result['n_features']} 个）",
            "",
        ]

    lines += [
        "## 边界",
        "",
        "- 测试集只有 21 片失败晶圆，单格指标的抽样波动不可忽略，"
        "结论以方向与幅度为准，不宜按小数点后第三位排名。",
        "- SHAP 引导特征选择的排名依赖探针模型；探针本身若在该配置下近乎恒定输出，"
        "排名的信息量随之下降，本表已记录探针在训练集上的召回供判读。",
        "- 特征匿名，所有入选特征均为统计关联候选，非工艺因果。",
        "",
    ]
    path = out_dir / "ablation_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[消融] 对比表已保存: %s", path)
    return path.name


def _delta(a: dict, b: dict) -> dict:
    """b 相对 a 的变化（BER / 召回 / 漏检片数）。"""
    return {
        "BER变化": float(b["测试集BER"] - a["测试集BER"]),
        "召回变化": float(b["召回率"] - a["召回率"]),
        "漏检变化": int(b["confusion"]["fn"] - a["confusion"]["fn"]),
    }


def run_ablation(
    cfg: dict,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    seed: int,
    quick: bool = False,
) -> dict:
    """跑完整 2x2 消融并落盘；返回结果字典（status: ok / disabled）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ab_cfg = cfg.get("ablation") or {}
    if not ab_cfg.get("enabled", True):
        result = {"status": "disabled", "reason": "config ablation.enabled=false"}
        (out_dir / "ablation_comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("ablation.enabled=false：跳过消融实验（已记录）")
        return result

    logger.info("===== 消融实验（特征选择 × 不平衡处理）=====")
    names = (registry_names() if ab_cfg.get("models", "all") == "all"
             else list(ab_cfg["models"]))
    fn_cost, fp_cost = canonical_costs(cfg)
    k = min(int(cfg["feature_selection"]["n_features_to_select"]), X_train_raw.shape[1])

    # 严格链路：预处理只在训练集拟合，测试集只 transform
    pipe = build_preprocess_pipeline(cfg)
    X_tr = pd.DataFrame(pipe.fit_transform(X_train_raw),
                        columns=X_train_raw.columns, index=X_train_raw.index)
    X_te = pd.DataFrame(pipe.transform(X_test_raw),
                        columns=X_test_raw.columns, index=X_test_raw.index)

    vote_features, _detail = select_features(X_tr, y_train, cfg, seed, quick)
    logger.info("[消融] 四方法投票特征集: %d 个", len(vote_features))

    models_result: dict[str, dict] = {}
    for name in names:
        display = MODEL_DISPLAY_NAMES.get(name, name)
        cells: dict[str, dict] = {}
        features_used: dict[str, list[str]] = {}
        probes: dict[str, dict] = {}
        for weighting in WEIGHTINGS:
            cw = _class_weight_of(weighting)
            for selector in SELECTORS:
                if selector == "vote":
                    feats = list(vote_features)
                else:
                    feats, probe_info = _shap_features(
                        cfg, name, weighting, X_tr, y_train, k, seed)
                    probes[weighting] = probe_info
                cell = _evaluate_cell(make_model(name, seed, cw),
                                      X_tr[feats], y_train, X_te[feats], y_test,
                                      fn_cost, fp_cost)
                cell["n_features"] = len(feats)
                cells[cell_key(selector, weighting)] = cell
                features_used[cell_key(selector, weighting)] = feats
                logger.info("[消融] %s | %s + %s: BER=%.3f 召回=%.3f FN=%d",
                            display, SELECTOR_LABELS[selector],
                            WEIGHTING_LABELS[weighting], cell["测试集BER"],
                            cell["召回率"], cell["confusion"]["fn"])

        overlap = {
            w: len(set(features_used[cell_key("vote", w)])
                   & set(features_used[cell_key("shap", w)]))
            for w in WEIGHTINGS
        }
        models_result[name] = {
            "display_name": display,
            "cells": cells,
            "shap_probe": probes,
            "feature_overlap": overlap,
            "features": features_used,
            "effects": {
                "weighting_on_vote": _delta(cells[cell_key("vote", "none")],
                                            cells[cell_key("vote", "class_weight")]),
                "selector_on_none": _delta(cells[cell_key("vote", "none")],
                                           cells[cell_key("shap", "none")]),
                "selector_on_weighted": _delta(cells[cell_key("vote", "class_weight")],
                                               cells[cell_key("shap", "class_weight")]),
            },
        }

    result = {
        "status": "ok",
        "random_state": int(seed),
        "n_features": k,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_failed": int((y_test == 1).sum()),
        "costs": {"fn": fn_cost, "fp": fp_cost},
        "selectors": list(SELECTORS),
        "weightings": list(WEIGHTINGS),
        "features_vote": list(vote_features),
        "models": models_result,
    }

    result["artifacts"] = {
        "grid_png": _plot_grid(models_result, out_dir),
        "markdown": _write_markdown(result, out_dir),
    }
    (out_dir / "ablation_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[消融] 结果已保存: %s", out_dir / "ablation_comparison.json")
    return result

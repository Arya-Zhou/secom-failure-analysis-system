"""全流程编排 —— main.py 与回归测试共用的唯一入口。

流程（notebook 忠实模式，对齐 secom.ipynb）：
    加载 -> 删全空列 -> 全量填充+标准化 -> 四方法特征选择(591->40)
    -> 80/20 分层划分(seed) -> 训练注册表模型 -> 训练/测试集指标
    -> 保存产物(outputs/) -> 与 baseline_metrics.json 容差比对

生产模式（preprocessing.fit_on_train_only: true）：
    先划分，填充/标准化与特征选择只在训练集上 fit（防泄漏），
    指标会与 notebook 基线略有差异，属预期。
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_io import load_secom
from .evaluation import compare_with_baseline, cross_val_ber, evaluate_train_test
from .explain import explain_global, explain_wafer, pick_case_positions
from .feature_selection import load_feature_override, select_features
from .modeling import MODEL_DISPLAY_NAMES, get_models
from .preprocessing import (
    build_preprocess_pipeline, drop_all_nan_columns, preprocess_full,
)

logger = logging.getLogger(__name__)

# 本重构目录的根：config 中的相对路径一律相对它解析，
# 保证从任意 cwd（如仓库根跑 pytest）运行结果一致。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _quick_subsample(X, y, timestamps, size: int, seed: int):
    """quick 模式的分层子采样：保持失败率，样本降到 size。"""
    if size >= len(y):
        return X, y, timestamps
    idx, _ = train_test_split(
        y.index, train_size=size, random_state=seed, stratify=y,
    )
    logger.info("quick 模式: 分层子采样 %d -> %d", len(y), len(idx))
    return X.loc[idx], y.loc[idx], timestamps.loc[idx]


def _run_explain_stage(
    cfg: dict, model, model_name: str, X_train, X_test, y_test,
    features: list, out_dir, seed: int, run_tag: str,
) -> dict:
    """SHAP 可解释性阶段：全局解释 + TP/FN 案例 + manifest 落盘。

    manifest（outputs/shap_manifest.json）三态记录每个案例：
      generated（附产物路径与自洽结果）/ skipped（附原因）/ failed（附错误）。
    finally 保证异常时 manifest 仍落盘，供 verify.sh 阶段 5.5 条件验收——
    required=false 的降级运行中若执行失败，主流程不中断，但 manifest.status
    保持 "failed"，完整验证仍会红灯（不允许静默降级成"全绿"）。
    required=true（缺省）时执行失败向上抛出，使 main.py 非零退出。
    """
    explain_cfg = cfg.get("explain") or {}
    manifest_path = Path(out_dir) / "shap_manifest.json"

    if not explain_cfg.get("enabled", True):
        manifest = {"enabled": False, "reason": "config explain.enabled=false"}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("config explain.enabled=false：跳过 SHAP（已记录 manifest）")
        return manifest

    logger.info("===== 可解释性分析（SHAP）=====")
    manifest: dict = {
        "enabled": True, "status": "failed",  # 成功结束时改写为 ok
        "run_tag": run_tag, "model": model_name,
        "random_seed": int(seed), "global": None, "cases": {},
    }
    try:
        bg_size = int(explain_cfg.get("background_size", 100))
        if bg_size <= 0:
            raise ValueError("explain.background_size 必须大于 0")
        # 背景 = 训练集抽样（代表总体分布；不可用待解释样本自身，
        # 否则基线=样本输出，SHAP 值恒为 0）
        background = X_train.sample(min(bg_size, len(X_train)), random_state=seed)
        manifest["background_size"] = int(len(background))

        g_res = explain_global(model, X_test, features, model_name, out_dir,
                               background=background, seed=seed)
        # 回填解释元数据到 manifest 顶层：解释方法在创建解释器时才确定，
        # 需由 explain_global 返回后写入，供审计与条件验收
        manifest["shap_method"] = g_res["shap_method"]
        manifest["output_space"] = g_res["output_space"]
        manifest["global"] = {
            "png": f"shap_summary_bar_{model_name}.png",
            "json": f"shap_values_{model_name}.json",
            "n_samples_explained": int(len(X_test)),
        }

        # 单晶圆案例（确定性极值选样，见 pick_case_positions docstring）：
        # TP=最有把握的命中，FN=最接近阈值的漏检（衔接后续阈值移动叙事）
        y_pred_arr = np.asarray(model.predict(X_test))
        if hasattr(model, "decision_function"):
            score_arr = np.ravel(model.decision_function(X_test))
        else:
            score_arr = model.predict_proba(X_test)[:, 1]
        picks = pick_case_positions(y_test.to_numpy(), y_pred_arr, score_arr)
        for case, pos in picks.items():
            if pos is None:
                reason = f"测试集中无 {case} 样本"
                manifest["cases"][case] = {"status": "skipped", "reason": reason}
                logger.warning("%s，本次跳过该报告（非 SHAP 失败）", reason)
                continue
            try:
                res = explain_wafer(
                    model, X_test.iloc[pos], int(y_test.iloc[pos]),
                    features, model_name, out_dir,
                    background=background, wafer_id=int(y_test.index[pos]),
                    case=case, seed=seed,
                )
                manifest["cases"][case] = {
                    "status": "generated",
                    "wafer_id": int(res["wafer_id"]),
                    "report": f"shap_explanation_wafer_{res['wafer_id']}_{case}.txt",
                    "plot": f"shap_contribution_wafer_{res['wafer_id']}_{case}.png",
                    "output_space": res["output_space"],
                    "consistency_ok": bool(res["consistency_ok"]),
                    "deviation": abs(res["reconstruction"] - res["explained_output"]),
                }
            except Exception as e:
                manifest["cases"][case] = {"status": "failed", "error": str(e)}
                raise
        manifest["status"] = "ok"
    except Exception as exc:
        manifest["error"] = str(exc)  # 顶层失败原因入 manifest：全局阶段失败也可审计
        logger.exception("可解释性分析失败")
        if explain_cfg.get("required", True):
            raise RuntimeError(
                "SHAP 可解释性分析失败（explain.required=true，中止流程；"
                "如需允许降级请设 explain.required=false 或 enabled=false）"
            )
        logger.warning(
            "explain.required=false：主流程继续，但 manifest.status=failed，"
            "完整验证（verify.sh full 阶段 5.5）将不通过")
    finally:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        logger.info("[SHAP] manifest 已保存: %s", manifest_path)
    return manifest


def run_pipeline(cfg: dict, quick: bool | None = None) -> dict:
    """按配置跑通全流程，返回结果字典。

    返回:
        {
          "metrics": {展示名: {训练集BER/测试集BER/准确率/.../AUC}},
          "best_model": 展示名,
          "selected_features": [...],
          "baseline_ok": bool | None,      # None = 未比对(quick 模式)
          "baseline_report": [明细行],
          "output_dir": 产物目录,
        }
    """
    seed = int(cfg["random_state"])
    quick = bool(cfg["run"]["quick"]) if quick is None else quick

    # ---- 1. 加载 ----
    X, y, timestamps = load_secom(
        str(_resolve(cfg["data"]["features_path"])),
        str(_resolve(cfg["data"]["labels_path"])),
        cfg["data"]["timestamp_format"],
    )
    if quick:
        X, y, timestamps = _quick_subsample(
            X, y, timestamps, int(cfg["run"]["quick_sample_size"]), seed,
        )

    # ---- 2. 删全空列 ----
    X, dropped = drop_all_nan_columns(X)

    fit_on_train_only = bool(cfg["preprocessing"].get("fit_on_train_only", False))
    override_path = cfg["feature_selection"].get("override_features_path")

    # ---- 3/4/5. 预处理 + 特征选择 + 划分 ----
    if fit_on_train_only:
        # 生产模式：先划分，一切 fit 只发生在训练集
        X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
            X, y, test_size=cfg["split"]["test_size"],
            random_state=seed, stratify=y if cfg["split"]["stratify"] else None,
        )
        pipe = build_preprocess_pipeline(cfg)
        X_train_all = pd.DataFrame(
            pipe.fit_transform(X_tr_raw), columns=X.columns, index=X_tr_raw.index,
        )
        X_test_all = pd.DataFrame(
            pipe.transform(X_te_raw), columns=X.columns, index=X_te_raw.index,
        )
        if override_path:
            features = load_feature_override(_resolve(override_path))
        else:
            features, _detail = select_features(X_train_all, y_train, cfg, seed, quick)
        X_train, X_test = X_train_all[features], X_test_all[features]
    else:
        # notebook 忠实模式：全量 fit（对齐基线；泄漏问题见模块 docstring）
        X_scaled = preprocess_full(X, cfg)
        if override_path:
            features = load_feature_override(_resolve(override_path))
        else:
            features, _detail = select_features(X_scaled, y, cfg, seed, quick)
        X_sel = X_scaled[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=cfg["split"]["test_size"],
            random_state=seed, stratify=y if cfg["split"]["stratify"] else None,
        )

    logger.info(
        "数据划分: 训练=%d (失败 %d) / 测试=%d (失败 %d) / 特征=%d",
        len(y_train), int((y_train == 1).sum()),
        len(y_test), int((y_test == 1).sum()), len(features),
    )

    # ---- 6/7. 训练与评估 ----
    metrics: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for reg_name, model in get_models(cfg, seed).items():
        display = MODEL_DISPLAY_NAMES.get(reg_name, reg_name)
        model.fit(X_train, y_train)
        metrics[display] = evaluate_train_test(model, X_train, y_train, X_test, y_test)
        if cfg["evaluation"].get("run_cv") and not quick:
            X_all = pd.concat([X_train, X_test]).sort_index()
            y_all = pd.concat([y_train, y_test]).sort_index()
            metrics[display].update(
                cross_val_ber(model, X_all, y_all, cfg["model"]["cv_folds"], seed)
            )
        fitted[display] = model
        logger.info(
            "%s: 测试BER=%.3f 召回=%.3f AUC=%s",
            display, metrics[display]["测试集BER"], metrics[display]["召回率"],
            f"{metrics[display]['AUC']:.3f}" if metrics[display]["AUC"] else "N/A",
        )

    best_model = min(metrics, key=lambda k: metrics[k]["测试集BER"])
    logger.info("最佳模型(按测试集 BER): %s", best_model)

    # ---- 8. 保存产物 ----
    out_dir = _resolve(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "quick" if quick else "full"

    with open(out_dir / f"metrics_{tag}_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    with open(out_dir / f"selected_features_{tag}_{ts}.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{i}. {feat}\n" for i, feat in enumerate(features, 1))
    with open(out_dir / f"best_model_{best_model}_{tag}_{ts}.pkl", "wb") as f:
        pickle.dump(fitted[best_model], f)
    logger.info("产物已保存到 %s (时间戳 %s)", out_dir, ts)

    # ---- 8.5. 可解释性分析（SHAP，仅 full 模式；详见 _run_explain_stage）----
    explain_manifest: dict | None = None
    if quick:
        logger.info("quick 模式：跳过 SHAP 可解释性")
    else:
        explain_manifest = _run_explain_stage(
            cfg, fitted[best_model], best_model, X_train, X_test, y_test,
            list(features), out_dir, seed, ts,
        )

    # ---- 9. 基线比对（quick 模式不比对：子采样必然偏离基线）----
    baseline_ok: bool | None = None
    baseline_report: list[str] = []
    if not quick:
        baseline_ok, baseline_report = compare_with_baseline(
            metrics,
            _resolve(cfg["reproducibility"]["baseline_path"]),
            float(cfg["reproducibility"]["tolerance"]),
        )

    return {
        "metrics": metrics,
        "best_model": best_model,
        "selected_features": features,
        "dropped_columns": dropped,
        "baseline_ok": baseline_ok,
        "baseline_report": baseline_report,
        "explain": explain_manifest,
        "output_dir": str(out_dir),
    }

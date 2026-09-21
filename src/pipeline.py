"""全流程编排：main.py 与回归测试共用的唯一入口。"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from .ablation import run_ablation
from .config import canonical_costs
from .data_io import load_secom
from .drift import resolve_drift
from .drift_reporting import run_drift_analysis
from .evaluation import (
    compare_with_baseline, cross_val_oof_metrics, evaluate_train_test,
    extended_test_metrics, resolve_cv_folds,
)
from .explain import explain_global, explain_wafer, pick_case_positions
from .feature_selection import load_feature_override, select_features
from .imbalance import run_imbalance_comparison
from .modeling import MODEL_DISPLAY_NAMES, get_models
from .model_comparison import resolve_model_comparison
from .model_comparison_analysis import run_model_comparison
from .preprocessing import build_preprocess_pipeline, drop_all_nan_columns
from .stability import resolve_stability, resolve_stability_analysis
from .stability_analysis import run_stability_analysis
from .validation import (
    ARTIFACT_SPLIT_INDEX, interval, paired_diff_interval, point_vs_interval,
    prior_drift_table, reference_percentile, resample_splits, resolve_artifact_split,
    resolve_resample, resolve_temporal_holdout, temporal_holdout_split,
)

logger = logging.getLogger(__name__)

# 仓库根目录：config 中的相对路径一律相对它解析，
# 保证从任意 cwd（如仓库根跑 pytest）运行结果一致。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 落盘的选择依据标识。与 imbalance.py 的 SELECTION_BASIS 同值：两处选择用的是同一把尺子。
SELECTION_BASIS = "train_cv_expected_cost"

# 选择信息在 metrics_<tag>_<ts>.json 里的键；下划线前缀与基线文件的 _note 同一约定，
# 表示"这不是一个模型"，逐模型遍历的消费方据此跳过。
SELECTION_KEY = "_selection"

# 进区间的测试集指标。只列测试集口径：区间刻画的是"这条 recipe 换一次抽样"的性能分布。
INTERVAL_METRICS = (
    "测试集BER", "召回率", "精确率", "F1分数", "AUC",
    "PR_AUC", "Brier分数", "期望代价", "Recall@FP预算",
)

# README 引用的头条指标。reference 百分位的硬闸门只加在这三项上：
# 逐格 27 个 cell 都设闸门必然误报——成员的分位数按构造可以取到 100（它自己就在分布里）。
HEADLINE_METRICS = ("测试集BER", "召回率", "期望代价")

# 同源断言比对的键。取标量指标全集：混淆等嵌套字段另行逐字段比。
SAME_SOURCE_KEYS = ("训练集BER", "测试集BER", "准确率", "精确率", "召回率", "F1分数",
                    "AUC", "PR_AUC", "Brier分数", "期望代价", "Recall@FP预算",
                    "CV_BER均值", "CV折外期望代价")

# 重采样折级指标的落盘文件名。
RESAMPLE_FILE = "resample_metrics.json"

# 时间序协议指标的落盘文件名。两协议并列落盘、互不覆盖。
TEMPORAL_FILE = "temporal_metrics.json"

# 头条指标"变差"的方向：+1 表示数值越大越差（BER / 期望代价），-1 表示越小越差（召回）。
# 升级规则要判的是"时间序是否显著恶化"，光看"落在区间外"分不出恶化还是变好。
METRIC_WORSE_SIGN = {"测试集BER": 1, "召回率": -1, "期望代价": 1}


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
    """SHAP 阶段：全局解释、TP/FN 案例与 manifest 落盘，返回 manifest。"""
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
        # 背景取训练集抽样：须代表总体分布，用待解释样本自身会使 SHAP 值恒为 0
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

        # 单晶圆案例（确定性极值选样，见 pick_case_positions）：
        # TP 取最有把握的命中，FN 取最接近阈值的漏检
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


def _select_reference_model(metrics: dict[str, dict], folds: int,
                            fn_cost: float, fp_cost: float) -> dict:
    """按训练侧 CV 折外期望代价选 reference 模型；另算 CV BER 排序，两者不一致要报出来。"""
    # 平手时 min 取字典序首个 = 模型注册表顺序，与运行次数无关
    by_cost = sorted(metrics, key=lambda k: (metrics[k]["CV折外期望代价"], k))
    by_ber = sorted(metrics, key=lambda k: (metrics[k]["CV_BER均值"], k))
    return {
        "selection_basis": SELECTION_BASIS,
        "reference_model": by_cost[0],
        "cv_folds": int(folds),
        "costs": {"fn": fn_cost, "fp": fp_cost},
        "ranking_by_cv_cost": by_cost,
        "ranking_by_cv_ber": by_ber,
        "criteria_agree": by_cost == by_ber,
    }


def _fit_and_evaluate_split(
    cfg: dict, X, y, train_idx, test_idx, seed: int, quick: bool,
    fn_cost: float, fp_cost: float, fp_budget: int,
) -> dict:
    """一条划分上的完整 recipe：预处理 → 特征选择 → 训练 → 训练侧 CV 选模型 → 测试集评估。
    reference（划分 #0）与每个重采样划分共用它，不存在第二条代码路径（§7.1 同源要求）。"""
    X_tr_raw, X_te_raw = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    pipe = build_preprocess_pipeline(cfg)
    X_train_all = pd.DataFrame(
        pipe.fit_transform(X_tr_raw), columns=X.columns, index=X_tr_raw.index,
    )
    X_test_all = pd.DataFrame(
        pipe.transform(X_te_raw), columns=X.columns, index=X_te_raw.index,
    )
    override_path = cfg["feature_selection"].get("override_features_path")
    if override_path:
        features = load_feature_override(_resolve(override_path))
    else:
        features, _detail = select_features(X_train_all, y_train, cfg, seed, quick)
    X_train, X_test = X_train_all[features], X_test_all[features]

    logger.info(
        "数据划分: 训练=%d (失败 %d) / 测试=%d (失败 %d) / 特征=%d",
        len(y_train), int((y_train == 1).sum()),
        len(y_test), int((y_test == 1).sum()), len(features),
    )

    folds = resolve_cv_folds(cfg["model"]["cv_folds"], y_train)
    metrics: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for reg_name, model in get_models(cfg, seed).items():
        display = MODEL_DISPLAY_NAMES.get(reg_name, reg_name)
        model.fit(X_train, y_train)
        metrics[display] = evaluate_train_test(model, X_train, y_train, X_test, y_test)
        metrics[display].update(extended_test_metrics(
            model, X_test, y_test, fn_cost, fp_cost, fp_budget))
        metrics[display].update(cross_val_oof_metrics(
            model, X_train, y_train, folds, seed, fn_cost, fp_cost))
        fitted[display] = model
        logger.info(
            "%s: 训练侧CV代价=%g (BER %.3f) | 测试BER=%.3f 召回=%.3f 代价=%g AUC=%s",
            display, metrics[display]["CV折外期望代价"], metrics[display]["CV_BER均值"],
            metrics[display]["测试集BER"], metrics[display]["召回率"],
            metrics[display]["期望代价"],
            f"{metrics[display]['AUC']:.3f}" if metrics[display]["AUC"] else "N/A",
        )

    selection = _select_reference_model(metrics, folds, fn_cost, fp_cost)
    return {
        "metrics": metrics, "fitted": fitted, "features": list(features),
        "selection": selection, "folds": folds,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    }


def _split_payload(result: dict, split_no: int) -> dict:
    """折级结果里可落盘 / 可跨进程回传的那部分（不含已拟合模型与数据）。"""
    return {
        "split": int(split_no),
        "reference_model": result["selection"]["reference_model"],
        "n_test": int(len(result["y_test"])),
        "n_test_positive": int((result["y_test"] == 1).sum()),
        "models": result["metrics"],
    }


def _same_source_check(ref_metrics: dict, split0_metrics: dict) -> dict:
    """同源断言：重采样循环里重算的划分 #0 必须与 reference 产物的指标逐位相同（容差 0）。
    不成立即说明产物生成与折级评估走的不是同一条路径（§4.4.1）。"""
    diffs: list[str] = []
    max_abs = 0.0
    for name, ref in ref_metrics.items():
        got = split0_metrics.get(name)
        if got is None:
            diffs.append(f"{name}: 划分 #0 缺该模型")
            continue
        for key in SAME_SOURCE_KEYS:
            a, b = ref.get(key), got.get(key)
            if a is None or b is None:
                if a is not b:
                    diffs.append(f"{name}.{key}: reference={a!r} 划分#0={b!r}")
                continue
            d = abs(float(a) - float(b))
            max_abs = max(max_abs, d)
            if d != 0.0:
                diffs.append(f"{name}.{key}: reference={a!r} 划分#0={b!r} 差={d!r}")
        if ref.get("测试集混淆") != got.get("测试集混淆"):
            diffs.append(f"{name}.测试集混淆: {ref.get('测试集混淆')} vs {got.get('测试集混淆')}")
    return {"ok": not diffs, "max_abs_diff": max_abs,
            "compared_keys": list(SAME_SOURCE_KEYS) + ["测试集混淆"],
            "mismatches": diffs}


def _resample_stage(
    cfg: dict, X, y, ref: dict, spec: dict, rs_cfg: dict, seed: int, quick: bool,
    fn_cost: float, fp_cost: float, fp_budget: int, out_dir: Path,
) -> dict:
    """重采样总体：跑全部 R 个划分出区间与配对差值；划分 #0 重算并与 reference 做同源断言。"""
    splits = resample_splits(y, spec["test_size"], spec["seed"], rs_cfg["n_splits"])
    logger.info("===== 重采样总体：%d 个划分（n_jobs=%d）=====",
                len(splits), rs_cfg["n_jobs"])

    def one(i: int) -> dict:
        tr, te = splits[i]
        logger.info("[重采样] 划分 %d/%d", i, len(splits) - 1)
        return _split_payload(
            _fit_and_evaluate_split(cfg, X, y, tr, te, seed, quick,
                                    fn_cost, fp_cost, fp_budget), i)

    # 划分 #0 留在主进程串行重算：同源断言的容差是 0，而 loky 子进程会把 BLAS 线程数压到 1，
    # 归约次序随之改变——实测逻辑回归概率差 4.5e-13，足以让逐位断言误报（详见 D11 边界）。
    rest = [i for i in range(len(splits)) if i != ARTIFACT_SPLIT_INDEX]
    payloads = [one(ARTIFACT_SPLIT_INDEX)]
    if rs_cfg["n_jobs"] and rs_cfg["n_jobs"] != 1:
        payloads += Parallel(n_jobs=rs_cfg["n_jobs"])(delayed(one)(i) for i in rest)
    else:
        payloads += [one(i) for i in rest]
    payloads = sorted(payloads, key=lambda p: p["split"])

    same_source = _same_source_check(
        ref["metrics"], payloads[ARTIFACT_SPLIT_INDEX]["models"])
    if not same_source["ok"]:
        raise RuntimeError(
            "同源断言失败：划分 #0 的折级指标与 reference 产物的测试指标不一致，"
            "说明产物生成与折级评估不同源。\n  " + "\n  ".join(same_source["mismatches"]))
    logger.info("同源断言通过：划分 #0 与 reference 产物逐位相同（最大差值 %r）",
                same_source["max_abs_diff"])

    method, models = rs_cfg["interval"], list(ref["metrics"])
    series = {m: {k: [p["models"][m][k] for p in payloads] for k in INTERVAL_METRICS}
              for m in models}
    intervals = {m: {k: interval(series[m][k], method, seed) for k in INTERVAL_METRICS}
                 for m in models}
    percentiles = {
        m: {k: reference_percentile(ref["metrics"][m][k], series[m][k])
            for k in INTERVAL_METRICS}
        for m in models
    }
    paired: dict[str, dict] = {}
    if rs_cfg["paired_diff"]:
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                paired[f"{a} - {b}"] = {
                    k: paired_diff_interval(series[a][k], series[b][k], method, seed)
                    for k in INTERVAL_METRICS}

    votes: dict[str, int] = {}
    for p in payloads:
        votes[p["reference_model"]] = votes.get(p["reference_model"], 0) + 1

    result = {
        "status": "ok",
        "protocol": {
            "population": "StratifiedShuffleSplit",
            "test_size": spec["test_size"], "seed": spec["seed"],
            "n_splits": rs_cfg["n_splits"], "interval": method,
            "quick": bool(quick),
            "artifact_split_index": ARTIFACT_SPLIT_INDEX,
            "selection_basis": SELECTION_BASIS,
        },
        "same_source_assertion": same_source,
        "reference_model": ref["selection"]["reference_model"],
        "reference_point": {m: {k: ref["metrics"][m][k] for k in INTERVAL_METRICS}
                            for m in models},
        "reference_percentile": percentiles,
        "intervals": intervals,
        "paired_diff": paired,
        "selection_frequency": votes,
        "metrics_by_split": payloads,
    }
    (out_dir / RESAMPLE_FILE).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[重采样] 折级指标与区间已保存: %s", out_dir / RESAMPLE_FILE)
    return result


def temporal_escalation(cells: dict, ref_model: str) -> dict:
    """主流程 reference 的头条指标落在随机区间更差侧时升级风险。
    记录实际判定模型，避免与时间序协议另选的 reference 混淆。"""
    reasons = []
    for metric, sign in METRIC_WORSE_SIGN.items():
        cell = (cells.get(ref_model) or {}).get(metric)
        if cell is None or cell["within_interval"]:
            continue
        bound = cell["interval"][1] if sign > 0 else cell["interval"][0]
        if (cell["value"] - bound) * sign > 0:
            reasons.append(
                f"{ref_model}.{metric}={cell['value']:.4f} 落在随机协议区间 "
                f"[{cell['interval'][0]:.4f}, {cell['interval'][1]:.4f}] 的更差一侧")
    return {"required": bool(reasons), "evaluated_on": ref_model, "reasons": reasons}


def _load_random_intervals(
    resample_result: dict | None, out_dir: Path, spec: dict, rs_cfg: dict,
    ref_metrics: dict, quick: bool,
) -> tuple[dict | None, str]:
    """取随机协议的区间作对照。本次运行跑了重采样就用内存结果；否则读落盘产物，
    但**先核协议与陈旧性**——拿上一版代码留下的区间去和本轮时间序比，两侧就不是同一条 recipe。"""
    if resample_result and resample_result.get("status") == "ok":
        return resample_result, "in_run"
    path = out_dir / RESAMPLE_FILE
    if not path.exists():
        return None, f"缺少 {RESAMPLE_FILE}（validation.resample.enabled=false 且无历史产物）"
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001：读不动就如实降级，不让对照拖垮主流程
        return None, f"{RESAMPLE_FILE} 解析失败: {exc}"
    proto = res.get("protocol") or {}
    if res.get("status") != "ok":
        return None, f"{RESAMPLE_FILE} 的 status={res.get('status')!r}"
    if bool(proto.get("quick")) != bool(quick):
        return None, f"{RESAMPLE_FILE} 出自 quick={proto.get('quick')!r} 的运行，与本轮不同"
    for key, want in (("test_size", spec["test_size"]), ("seed", spec["seed"]),
                      ("n_splits", rs_cfg["n_splits"])):
        if proto.get(key) != want:
            return None, (f"{RESAMPLE_FILE} 的协议 {key}={proto.get(key)!r} "
                          f"与本轮配置 {want!r} 不同")
    # 陈旧性：产物记的 reference 点估计必须与本轮主流程逐位相符（1e-9 是跨运行容差，见 D11）
    for model, point in (res.get("reference_point") or {}).items():
        cur = ref_metrics.get(model)
        if cur is None:
            return None, f"{RESAMPLE_FILE} 含模型 {model}，本轮主流程没有：产物已陈旧"
        for metric, val in point.items():
            now = cur.get(metric)
            if val is None or now is None:
                if val is not now:
                    return None, f"{RESAMPLE_FILE} 的 {model}.{metric} 与本轮不符（None 侧）"
                continue
            if abs(float(val) - float(now)) > 1e-9:
                return None, (f"{RESAMPLE_FILE} 的 {model}.{metric}={val!r} 与本轮 "
                              f"{now!r} 不符：产物出自旧代码或旧配置，须重跑")
    return res, "artifact"


def _temporal_stage(
    cfg: dict, X, y, timestamps, ref: dict, th_cfg: dict, seed: int, quick: bool,
    fn_cost: float, fp_cost: float, fp_budget: int, out_dir: Path,
    resample_result: dict | None, spec: dict, rs_cfg: dict,
) -> dict:
    """时间序协议：按 timestamp 前 80% 训练 / 后 20% 测试，跑与随机协议同一条 recipe。
    两个协议是两个总体，并列落盘不混——随机协议衡量方法本身，时间序协议贴合部署顺序。"""
    logger.info("===== 时间序留出协议（test_fraction=%.2f, gap=%d）=====",
                th_cfg["test_fraction"], th_cfg["gap"])
    train_idx, test_idx = temporal_holdout_split(
        timestamps, y, th_cfg["test_fraction"], th_cfg["gap"])
    res = _fit_and_evaluate_split(cfg, X, y, train_idx, test_idx, seed, quick,
                                  fn_cost, fp_cost, fp_budget)
    ts = pd.Series(timestamps)
    drift = prior_drift_table(timestamps, y, th_cfg["prior_drift_bins"])

    random_res, source = _load_random_intervals(
        resample_result, out_dir, spec, rs_cfg, ref["metrics"], quick)
    comparison: dict | None = None
    # 无对照时也把"判的是哪个模型"记下来，字段形状与有对照时一致
    escalation = {"required": False,
                  "evaluated_on": ref["selection"]["reference_model"], "reasons": []}
    if random_res is None:
        logger.warning("时间序对照：无可用的随机协议区间（%s），本轮只落时间序单点", source)
    else:
        comparison = {"interval_source": source, "models": {}}
        series = {p["split"]: p["models"] for p in random_res["metrics_by_split"]}
        for model, m in res["metrics"].items():
            ivs = (random_res.get("intervals") or {}).get(model) or {}
            folds = [series[k][model] for k in sorted(series)]
            cells = {}
            for metric in INTERVAL_METRICS:
                cell = point_vs_interval(m.get(metric), ivs.get(metric))
                if cell is not None:
                    cell["percentile_in_random"] = reference_percentile(
                        m.get(metric), [f.get(metric) for f in folds])
                cells[metric] = cell
            comparison["models"][model] = cells
        # 升级规则见 temporal_escalation（判据预先写死，不看结果再定）
        escalation = temporal_escalation(
            comparison["models"], ref["selection"]["reference_model"])
        comparison["escalation"] = escalation
        if escalation["required"]:
            logger.warning("时间序协议显著恶化，按升级规则须在 README 作为主要风险结论：%s",
                           "；".join(escalation["reasons"]))

    result = {
        "status": "ok",
        "protocol": {
            "name": "temporal_holdout",
            "test_fraction": th_cfg["test_fraction"], "gap": th_cfg["gap"],
            "quick": bool(quick), "selection_basis": SELECTION_BASIS,
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "n_train_positive": int((y.loc[train_idx] == 1).sum()),
            "n_test_positive": int((y.loc[test_idx] == 1).sum()),
            "train_span": [str(ts.loc[train_idx].min()), str(ts.loc[train_idx].max())],
            "test_span": [str(ts.loc[test_idx].min()), str(ts.loc[test_idx].max())],
        },
        "prior_drift": drift,
        "reference_model": res["selection"]["reference_model"],
        "selection": res["selection"],
        "models": res["metrics"],
        "random_protocol_comparison": comparison,
        "comparison_unavailable_reason": None if comparison else source,
        "escalation": escalation,
    }
    (out_dir / TEMPORAL_FILE).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[时间序] 指标与先验漂移已保存: %s", out_dir / TEMPORAL_FILE)
    return result


def run_pipeline(cfg: dict, quick: bool | None = None) -> dict:
    """按配置跑通全流程。"""
    seed = int(cfg["random_state"])
    quick = bool(cfg["run"]["quick"]) if quick is None else quick
    split_spec = resolve_artifact_split(cfg)
    rs_cfg = resolve_resample(cfg)
    th_cfg = resolve_temporal_holdout(cfg)
    drift_cfg = resolve_drift(cfg)
    stability_node = cfg.get("stability")
    if stability_node is not None and (
            not isinstance(stability_node, dict) or not isinstance(stability_node.get("enabled"), bool)):
        raise ValueError("stability.enabled 必须为布尔值")
    stability_enabled = bool(stability_node and stability_node["enabled"])
    stability_cfg = {**cfg, "run": {**cfg["run"], "quick": bool(quick)}}
    if stability_enabled and not quick:
        resolve_stability(stability_cfg)
        resolve_stability_analysis(stability_cfg)
    comparison_node = cfg.get("model_comparison")
    if comparison_node is not None and (
            not isinstance(comparison_node, dict) or not isinstance(comparison_node.get("enabled"), bool)):
        raise ValueError("model_comparison.enabled 必须为布尔值")
    comparison_enabled = bool(comparison_node and comparison_node["enabled"])
    if comparison_enabled and not quick:
        resolve_model_comparison(stability_cfg)

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
    stability_input = X
    X, dropped = drop_all_nan_columns(X)

    # ---- 3~7. reference 划分 = 重采样总体的第 0 个划分（与 train_test_split 逐位等价）；
    # 预处理/特征选择/模型选择全在该划分的训练侧完成，测试集只在最终评估出现一次。
    fn_cost, fp_cost = canonical_costs(cfg)
    fp_budget = int(cfg["evaluation"]["fp_budget"])
    train_idx, test_idx = resample_splits(
        y, split_spec["test_size"], split_spec["seed"],
        rs_cfg["n_splits"] if rs_cfg["enabled"] else 1,
    )[ARTIFACT_SPLIT_INDEX]
    ref = _fit_and_evaluate_split(cfg, X, y, train_idx, test_idx, seed, quick,
                                  fn_cost, fp_cost, fp_budget)
    metrics, fitted, features = ref["metrics"], ref["fitted"], ref["features"]
    X_train, X_test, y_test = ref["X_train"], ref["X_test"], ref["y_test"]
    y_train = ref["y_train"]
    selection = ref["selection"]
    reference_model = selection["reference_model"]
    logger.info(
        "reference 模型(按训练侧 CV %d 折折外期望代价): %s | 两判据排序%s",
        ref["folds"], reference_model,
        "一致" if selection["criteria_agree"] else "不一致（见产物）",
    )

    # ---- 8. 保存产物 ----
    out_dir = _resolve(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "quick" if quick else "full"

    with open(out_dir / f"metrics_{tag}_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, SELECTION_KEY: selection}, f,
                  ensure_ascii=False, indent=4)
    with open(out_dir / f"selected_features_{tag}_{ts}.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{i}. {feat}\n" for i, feat in enumerate(features, 1))
    with open(out_dir / f"reference_model_{reference_model}_{tag}_{ts}.pkl", "wb") as f:
        pickle.dump(fitted[reference_model], f)
    logger.info("产物已保存到 %s (时间戳 %s)", out_dir, ts)

    # ---- 8.5. 可解释性分析（SHAP，仅 full 模式；详见 _run_explain_stage）----
    explain_manifest: dict | None = None
    if quick:
        logger.info("quick 模式：跳过 SHAP 可解释性")
    else:
        explain_manifest = _run_explain_stage(
            cfg, fitted[reference_model], reference_model, X_train, X_test, y_test,
            list(features), out_dir, seed, ts,
        )

    # ---- 8.6. 不平衡策略对比与消融实验（仅 full 模式）：两者都传删全空列后的原始特征划分，
    # 各自在模块内自建只对训练侧拟合的严格链路，与主流程模式解耦；失败向上抛，状态落盘可审计。
    imbalance_result: dict | None = None
    ablation_result: dict | None = None
    if quick:
        logger.info("quick 模式：跳过不平衡策略对比与消融实验")
    else:
        imbalance_result = run_imbalance_comparison(
            cfg, X.loc[y_train.index], y_train, X.loc[y_test.index], y_test,
            out_dir, seed,
        )
        # 消融同样自建严格链路，与主流程模式无关，故传原始特征划分而非上面的 X_train/X_test
        ablation_result = run_ablation(
            cfg, X.loc[y_train.index], y_train, X.loc[y_test.index], y_test,
            out_dir, seed, quick,
        )

    # ---- 8.7. 重采样总体（默认关，见 config validation.resample）：出区间与配对差值，
    # 并对划分 #0 做同源断言。产物已在上面落盘，此处只是同一 recipe 换 R-1 次抽样重跑。
    resample_result: dict | None = None
    if rs_cfg["enabled"]:
        resample_result = _resample_stage(
            cfg, X, y, ref, split_spec, rs_cfg, seed, quick,
            fn_cost, fp_cost, fp_budget, out_dir,
        )
    else:
        logger.info("validation.resample.enabled=false：跳过重采样，只出 reference 单点数字")

    # ---- 8.8. 时间序留出协议（默认开）：与随机协议并列的另一个总体，跑同一条 recipe，
    # 回答"按部署顺序划分还剩多少性能"（随机划分让模型见到晚于测试集的晶圆）。
    temporal_result: dict | None = None
    if th_cfg["enabled"]:
        temporal_result = _temporal_stage(
            cfg, X, y, timestamps, ref, th_cfg, seed, quick,
            fn_cost, fp_cost, fp_budget, out_dir, resample_result, split_spec, rs_cfg,
        )
    else:
        logger.info("validation.temporal_holdout.enabled=false：跳过时间序对照")

    drift_result = None
    if drift_cfg["enabled"] and not quick:
        drift_result = run_drift_analysis(stability_cfg, stability_input, y, timestamps, out_dir)
    else:
        logger.info("drift.enabled=false 或 quick 模式：不执行时间漂移分析")

    stability_result: dict | None = None
    if stability_enabled and not quick:
        stability_result = run_stability_analysis(
            stability_cfg, stability_input, y, train_idx, test_idx, out_dir, features)
    else:
        logger.info("stability.enabled=false 或 quick 模式：不执行稳定性跑批")

    comparison_result = None
    if comparison_enabled and not quick:
        comparison_result = run_model_comparison(
            stability_cfg, stability_input, y, train_idx, test_idx, out_dir)
    else:
        logger.info("model_comparison.enabled=false 或 quick 模式：不执行模型比较跑批")

    # ---- 9. 基线比对（quick 模式不比对：子采样必然偏离基线）----
    baseline_ok: bool | None = None
    baseline_report: list[str] = []
    if not quick:
        repro = cfg["reproducibility"]
        baseline_path = repro.get("baseline_path")
        if not baseline_path:
            raise ValueError("config reproducibility 缺少 baseline_path")
        baseline_ok, baseline_report = compare_with_baseline(
            metrics, _resolve(baseline_path), float(repro["tolerance"]),
        )

    return {
        "metrics": metrics,
        "selection": selection,
        "reference_model": reference_model,
        "selected_features": features,
        "dropped_columns": dropped,
        "baseline_ok": baseline_ok,
        "baseline_report": baseline_report,
        "explain": explain_manifest,
        "imbalance": imbalance_result,
        "ablation": ablation_result,
        "resample": resample_result,
        "temporal": temporal_result,
        "drift": drift_result,
        "stability": stability_result,
        "model_comparison": comparison_result,
        "split": {**split_spec, "index": ARTIFACT_SPLIT_INDEX,
                  "n_train": int(len(y_train)), "n_test": int(len(y_test))},
        "output_dir": str(out_dir),
    }

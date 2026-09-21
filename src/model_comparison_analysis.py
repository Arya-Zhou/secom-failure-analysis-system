"""复用已审计的训练侧选择记录，落盘模型比较、选择锁与唯一交付物。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path
from time import perf_counter, time

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

from .config import canonical_costs, load_config
from .data_io import load_secom
from .explain import explain_global, explain_wafer, pick_case_positions
from .imbalance import load_operating_point, save_fitted_operating_point, verify_operating_point_evaluation
from .model_comparison import (
    DELIVERY_CANDIDATES, FEATURE_CONTROL, MODEL_NAMES, REFERENCE_BASELINES,
    fit_comparison_split, model_recipe, resolve_model_comparison,
    summarize_model_comparison, validate_comparison_records,
)
from .stability_analysis import STABILITY_DETAIL_FILE, validate_stability_artifacts
from .validation import INTERVAL_HIGH_PCT, INTERVAL_LOW_PCT, artifact_split, resolve_artifact_split

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DETAIL_FILE = "model_comparison_detail.json"
COMPARISON_FILE = "model_comparison.json"
COMPARISON_MARKDOWN = "model_comparison.md"
COMPARISON_PLAN_FILE = "model_comparison_plan.json"
COMPARISON_SELECTION_FILE = "model_comparison_selection.json"
COMPARISON_MODEL_FILE = "model_comparison_operating_point.pkl"


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _data_hash(features, labels, indices):
    digest = hashlib.sha256(json.dumps(list(features.columns), ensure_ascii=False).encode("utf-8"))
    for values in (features.loc[indices].astype(float), labels.loc[indices].astype(float)):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _fingerprint(features, labels, reference_train, reference_test):
    sources = ("model_comparison.py", "model_comparison_analysis.py", "modeling.py",
               "evaluation.py", "imbalance.py", "calibration.py", "explain.py")
    return {
        "training_data_sha256": _data_hash(features, labels, reference_train),
        "outer_test_data_sha256": _data_hash(features, labels, reference_test),
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in sources},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "imblearn": imblearn.__version__, "joblib": joblib.__version__},
    }


def _analysis_plan(cfg, stability, output_dir):
    spec = resolve_model_comparison(cfg)
    fn_cost, fp_cost = canonical_costs(cfg)
    return {
        "artifact_schema_version": 1,
        "comparison": {key: value for key, value in spec.items() if key not in ("enabled", "n_jobs")},
        "recipes": {name: model_recipe(cfg, name) for name in MODEL_NAMES},
        "costs": {"fn": fn_cost, "fp": fp_cost}, "fp_budget": cfg["evaluation"]["fp_budget"],
        "preprocessing": dict(cfg["preprocessing"]),
        "stability_detail_sha256": _sha256(Path(output_dir) / STABILITY_DETAIL_FILE),
        "sampling_protocol_sha256": stability["sampling_protocol_sha256"],
        "reference_split": stability["sampling_protocol"]["reference_split"],
        "artifact_split": resolve_artifact_split(cfg),
        "reference_features": stability["reference_features"],
        "protocol": {
            "population": "reference_train_repeated_holdout", "input_scope": "reference_train_only",
            "n_population": stability["protocol"]["n_population"],
            "n_splits": stability["protocol"]["n_splits"],
            "test_size": stability["protocol"]["test_size"], "seed": stability["protocol"]["seed"],
            "outer_test_used_for_selection": False, "same_population_as_metric_resampling": False,
            "feature_selection_source": "audited_per_split_four_method_vote",
            "interval_interpretation": "overlapping_split_percentiles_not_mean_confidence_intervals",
            "interval_percentiles": [INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT],
        },
        "explain": {key: cfg["explain"][key] for key in ("enabled", "required", "background_size")},
    }


def external_benchmark(result):
    selected = result["summary"]["selection"]["selected_model"]
    metrics = result["outer_holdout"]["models"][selected]["metrics"]
    test_fraction = result["analysis_plan"]["artifact_split"]["test_size"]
    return {
        "paper": "Noh et al. 2024, Sensors", "doi": "10.3390/s24175461",
        "external": {"model": "LR + SMOTE", "train_test_ratio": "7:3", "n_test": 471,
                     "n_test_positive": 31, "recall": 0.5806, "specificity": 0.8318,
                     "derived_ber": round(((1 - 0.5806) + (1 - 0.8318)) / 2, 4)},
        "local": {"model": selected, "train_test_ratio": f"{100 * (1 - test_fraction):g}:{100 * test_fraction:g}", "n_test": metrics["n"],
                  "n_test_positive": metrics["n_positive"], "ber": metrics["ber"],
                  "recall": metrics["recall"], "specificity": metrics["specificity"]},
        "paired_comparison_available": False,
        "limitations": [
            "本项目模型只按训练侧折外配对代价选定；使用本次外层实测值，不沿用曾按测试 BER 挑出的旧值。",
            "双方不是同一划分；测试失效数少且缺少论文行级预测，不能构造跨研究配对区间或宣称优越性。训练侧区间不等于外层测试置信区间。",
            "BER 0.2938 是我方由论文 Sensitivity=0.5806 / Specificity=0.8318 推导，论文并未将其报告为最佳 BER。",
        ],
    }


def _report_payload(result):
    selected = result["summary"]["selection"]["selected_model"]
    models = {}
    for name, recipe in result["analysis_plan"]["recipes"].items():
        models[name] = {
            "role": recipe["role"], "display_name": recipe["display_name"],
            "feature_mode": recipe["feature_mode"],
            "training_intervals": result["summary"]["intervals"][name],
            "outer_holdout": result["outer_holdout"]["models"][name]["metrics"],
            "artifacts": result["deployment"] if name == selected else {},
        }
    return {
        "artifact_schema_version": 1, "status": "ok", "protocol": result["analysis_plan"]["protocol"],
        "analysis_plan_sha256": result["analysis_plan_sha256"], "fingerprint": result["fingerprint"],
        "models": models, "summary": result["summary"], "external_benchmark": external_benchmark(result),
        "explanation": result["explanation"], "execution": result["execution"],
        "evidence_file": COMPARISON_DETAIL_FILE,
    }


def _format_interval(value):
    if value is None:
        return "N/A"
    return f"{value['point']:.4f} [{value['low']:.4f}, {value['high']:.4f}]"


def render_model_comparison(report):
    selection, summary = report["summary"]["selection"], report["summary"]
    selected = selection["selected_model"]
    reference = selection["reference_model"]
    protocol = report["protocol"]
    lines = [
        "# 模型选型与配对比较", "",
        f"训练侧总体: 固定 reference 训练集 {protocol['n_population']} 行内 {protocol['n_splits']} 次重复分层留出。",
        "各轮复用已审计的四方法投票记录，重新拟合本轮预处理和分类器；不使用外层测试集选模型。",
        f"冻结基准: {reference} + class_weight + 四方法投票 + 默认决策边界；不是折内 F 检验策略表的工作点。",
        f"区间均为同一批重叠划分的 {protocol['interval_percentiles'][0]:g}%–{protocol['interval_percentiles'][1]:g}% 分位区间，不是独立样本均值置信区间。",
        f"差值 = 行模型 − {reference}；BER / 每片代价越小越好，召回越大越好。判定读配对差值是否跨零，不读两组独立区间是否重叠。",
        f"选择: **{selected}**。只允许每片代价配对区间上界 < 0 的交付候选替换 {reference}；不按测试结果排名。", "",
    ]
    for title, names in (("交付候选", DELIVERY_CANDIDATES), ("参考基线（不参与选择、不产模型交付物）", REFERENCE_BASELINES),
                         ("特征选择对照（不参与选择）", (FEATURE_CONTROL,))):
        lines.extend([f"## {title}", "", "| 模型 | 训练内 BER | 训练内召回 | 每片代价 | 配对 ΔBER | 配对 Δ召回 | 配对 Δ每片代价 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for name in names:
            metrics = summary["intervals"][name]
            delta = summary["vs_reference"].get(name)
            differences = ([_format_interval(delta[metric]) for metric in ("ber", "recall", "cost_per_wafer")]
                           if delta is not None else ["基准", "基准", "基准"])
            cells = [report["models"][name]["display_name"], *[
                _format_interval(metrics[metric]) for metric in ("ber", "recall", "cost_per_wafer")], *differences]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    embedded = summary["paired_diff"][f"elasticnet - {FEATURE_CONTROL}"]
    lines.extend([
        "同超参 ElasticNet 的嵌入式全列 − 四方法预筛配对差值: "
        f"BER {_format_interval(embedded['ber'])}，召回 {_format_interval(embedded['recall'])}，"
        f"每片代价 {_format_interval(embedded['cost_per_wafer'])}。",
        "非零系数数目（不是工艺因果结论）: " + "；".join(
            f"{name}: {_format_interval(summary['feature_selection'][name]['active_count'])}"
            for name in ("elasticnet", FEATURE_CONTROL)), "", "## 判读", "",
    ])
    if selected == selection["reference_model"]:
        lines.append("未有交付候选通过预定的配对代价改善门槛，保留线性基准。在此样本量下模型族不是主要矛盾，是本协议下的操作结论，不是模型等效证明。")
    else:
        lines.append(f"{selected} 通过训练侧配对代价门槛；这是预定候选内的选择证据，不是跨研究优越性或无偏的选后性能估计。")
    lines.extend(["全部两两配对差值见 JSON；参考基线即使更好也不混入交付选择。", "", "## 外层留出与外部对标", "",
                  "训练侧已锁定模型后才评估外层测试集。下列为不同研究的点估计，不能配对、不能据此排名。", "",
                  "| 来源 | 模型 | 测试片数 / 失效数 | BER | Recall | Specificity |",
                  "| --- | --- | --- | --- | --- | --- |"])
    benchmark = report["external_benchmark"]
    local, external = benchmark["local"], benchmark["external"]
    lines.append(f"| 本项目 | {selected} | {local['n_test']} / {local['n_test_positive']} | {local['ber']:.4f} | {local['recall']:.4f} | {local['specificity']:.4f} |")
    lines.append(f"| Noh 2024 | LR + SMOTE | 471 / 31 | {external['derived_ber']:.4f}（推导） | 0.5806 | 0.8318 |")
    lines.append("")
    lines.extend(f"{number}. {text}" for number, text in enumerate(benchmark["limitations"], 1))
    lines.extend(["", f"论文 DOI: {benchmark['doi']}。", "",
                  "唯一交付物包含整条已拟合链路，必须通过 load_operating_point 显式指定本报告 JSON 清单，再用 apply_operating_point 预测。",
                  "它不覆盖旧三模型回归产物；SHAP 解释空间中 HGB 是原始 log-odds，不能误读为概率加和。", ""])
    return "\n".join(lines)


def _save_explanation(cfg, estimator, name, features, labels, reference_train, reference_test, record, output_dir):
    if not cfg["explain"]["enabled"]:
        return {"enabled": False, "artifacts": {}}
    selected = record["selector_features"]
    train = pd.DataFrame(estimator[:-1].transform(features.loc[reference_train]), columns=selected, index=reference_train)
    test = pd.DataFrame(estimator[:-1].transform(features.loc[reference_test]), columns=selected, index=reference_test)
    size = int(cfg["explain"]["background_size"])
    if size < 1:
        raise ValueError("explain.background_size 必须为正整数")
    seed = int(cfg["random_state"])
    background = train.sample(min(size, len(train)), random_state=seed)
    destination = Path(output_dir) / "model_comparison_explain"
    model = estimator.named_steps["model"]
    global_result = explain_global(model, test, selected, name, destination, background=background, seed=seed)
    files = [destination / f"shap_summary_bar_{name}.png", destination / f"shap_values_{name}.json"]
    cases = {}
    for case, position in pick_case_positions(labels.loc[reference_test], record["predictions"], record["scores"]).items():
        if position is None:
            continue
        wafer_id = int(reference_test[position])
        cases[case] = explain_wafer(model, test.iloc[position], int(labels.loc[wafer_id]), selected,
                                    name, destination, background, wafer_id, case=case, seed=seed)
        files.extend([destination / f"shap_explanation_wafer_{wafer_id}_{case}.txt",
                      destination / f"shap_contribution_wafer_{wafer_id}_{case}.png"])
        if not cases[case]["consistency_ok"]:
            raise ValueError("模型比较交付物的 SHAP 加和自洽失败")
    return {"enabled": True, "model": name, "global": global_result, "cases": cases,
            "artifacts": {str(path.relative_to(output_dir)): _sha256(path)
                          for path in files}}


def run_model_comparison(cfg, features, labels, reference_train, reference_test, output_dir):
    started, wall_started = perf_counter(), time()
    out_dir = Path(output_dir)
    spec = resolve_model_comparison(cfg)
    stability = validate_stability_artifacts(cfg, features, labels, reference_train, reference_test, out_dir)
    plan = _analysis_plan(cfg, stability, out_dir)
    plan_path = out_dir / COMPARISON_PLAN_FILE
    if plan_path.exists() and _read_json(plan_path) != plan:
        raise FileExistsError("模型比较的冻结计划已不同；请使用独立产物目录，不得看结果后覆盖判据")
    if not plan_path.exists():
        _write_json(plan_path, plan)
    fingerprint = _fingerprint(features, labels, reference_train, reference_test)
    for filename in (COMPARISON_DETAIL_FILE, COMPARISON_FILE):
        _write_json(out_dir / filename, {"status": "running"})
    training_features, training_labels = features.loc[reference_train], labels.loc[reference_train]

    def one(sampling):
        train_indices, validation_indices = sampling["train_indices"], sampling["holdout_indices"]
        models, _fitted = fit_comparison_split(
            cfg, training_features.loc[train_indices], training_labels.loc[train_indices],
            training_features.loc[validation_indices], training_labels.loc[validation_indices],
            sampling["selected_features"])
        logger.info("模型比较完成训练内划分 %d/%d", sampling["split"] + 1, len(stability["records"]))
        return {"split": sampling["split"], "train_indices": train_indices,
                "validation_indices": validation_indices, "models": models}

    logger.info("模型比较: %d 次训练内划分，%d 个固定配方，n_jobs=%d", len(stability["records"]), len(MODEL_NAMES), spec["n_jobs"])
    if spec["n_jobs"] == 1:
        records = [one(sampling) for sampling in stability["records"]]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            records = Parallel(n_jobs=spec["n_jobs"], verbose=10)(delayed(one)(sampling) for sampling in stability["records"])
    validate_comparison_records(records, stability["records"], features, labels, cfg)
    summary = summarize_model_comparison(records, cfg)
    lock = {"analysis_plan_sha256": _sha256(plan_path), "training_data_sha256": fingerprint["training_data_sha256"],
            "selection": summary["selection"]}
    _write_json(out_dir / COMPARISON_SELECTION_FILE, lock)
    selected = summary["selection"]["selected_model"]
    logger.info("训练侧选择已锁定: %s；现在执行外层测试评估", selected)
    outer_models, fitted = fit_comparison_split(
        cfg, training_features, training_labels, features.loc[reference_test], labels.loc[reference_test],
        stability["reference_features"], keep_fitted=True)
    with threadpool_limits(limits=1):
        filename, digest = save_fitted_operating_point(
            fitted[selected], selected, outer_models[selected]["selector_features"],
            outer_models[selected]["metrics"]["confusion"], plan["costs"], out_dir, COMPARISON_MODEL_FILE)
        explanation = _save_explanation(cfg, fitted[selected], selected, features, labels,
                                        reference_train, reference_test, outer_models[selected], out_dir)
    if fingerprint != _fingerprint(features, labels, reference_train, reference_test):
        raise RuntimeError("模型比较期间数据或实现发生变化，不得保存成功产物")
    result = {
        "artifact_schema_version": 1, "status": "ok", "analysis_plan": plan,
        "analysis_plan_sha256": _sha256(plan_path), "fingerprint": fingerprint,
        "records": records, "summary": summary, "selection_lock": lock,
        "outer_holdout": {"split": 0, "train_indices": reference_train.tolist(),
                          "validation_indices": reference_test.tolist(), "models": outer_models},
        "deployment": {"operating_point": filename, "operating_point_sha256": digest},
        "explanation": explanation,
        "execution": {"n_jobs": spec["n_jobs"], "threads_per_worker": 1,
                      "reused_selection_records": len(records), "feature_selector_refits": 0,
                      "elapsed_seconds": perf_counter() - started, "wall_seconds": time() - wall_started},
    }
    report = _report_payload(result)
    _write_json(out_dir / COMPARISON_DETAIL_FILE, result)
    _write_json(out_dir / COMPARISON_FILE, report)
    (out_dir / COMPARISON_MARKDOWN).write_text(render_model_comparison(report), encoding="utf-8")
    validate_model_comparison_artifacts(cfg, features, labels, reference_train, reference_test, out_dir)
    logger.info("模型比较产物重算及交付复现通过；耗时 %.1f s", result["execution"]["wall_seconds"])
    return result


def validate_model_comparison_artifacts(cfg, features, labels, reference_train, reference_test, output_dir):
    out_dir = Path(output_dir)
    stability = validate_stability_artifacts(cfg, features, labels, reference_train, reference_test, out_dir)
    result = _read_json(out_dir / COMPARISON_DETAIL_FILE)
    if result.get("status") != "ok" or result.get("artifact_schema_version") != 1:
        raise ValueError("模型比较未成功完成")
    plan = _analysis_plan(cfg, stability, out_dir)
    if result["analysis_plan"] != plan or _read_json(out_dir / COMPARISON_PLAN_FILE) != plan:
        raise ValueError("模型比较冻结协议、参数或选择来源已陈旧")
    if result["analysis_plan_sha256"] != _sha256(out_dir / COMPARISON_PLAN_FILE):
        raise ValueError("模型比较计划摘要不一致")
    if result["fingerprint"] != _fingerprint(features, labels, reference_train, reference_test):
        raise ValueError("模型比较数据、代码或环境来源已陈旧")
    validate_comparison_records(result["records"], stability["records"], features, labels, cfg)
    if result["summary"] != summarize_model_comparison(result["records"], cfg):
        raise ValueError("模型比较配对区间或选择结论重算不一致")
    expected_lock = {"analysis_plan_sha256": result["analysis_plan_sha256"],
                     "training_data_sha256": result["fingerprint"]["training_data_sha256"],
                     "selection": result["summary"]["selection"]}
    if result["selection_lock"] != expected_lock or _read_json(out_dir / COMPARISON_SELECTION_FILE) != expected_lock:
        raise ValueError("模型比较训练侧选择锁不一致")
    outer = result["outer_holdout"]
    validate_comparison_records([outer], [{"split": 0, "train_indices": reference_train.tolist(),
                                          "holdout_indices": reference_test.tolist(),
                                          "selected_features": stability["reference_features"]}], features, labels, cfg)
    for key in ("elapsed_seconds", "wall_seconds"):
        if not np.isfinite(result["execution"][key]) or result["execution"][key] < 0:
            raise ValueError("模型比较耗时记录无效")
    if (result["execution"]["reused_selection_records"] != len(result["records"])
            or result["execution"]["feature_selector_refits"] != 0):
        raise ValueError("模型比较选择器复用记录不一致")
    report = _report_payload(result)
    if _read_json(out_dir / COMPARISON_FILE) != report:
        raise ValueError("模型比较报告的角色、指标或交付清单与详情不一致")
    if (out_dir / COMPARISON_MARKDOWN).read_text(encoding="utf-8") != render_model_comparison(report):
        raise ValueError("模型比较报告文本与详情不一致")
    selected = result["summary"]["selection"]["selected_model"]
    if selected not in DELIVERY_CANDIDATES or result["deployment"]["operating_point"] != COMPARISON_MODEL_FILE:
        raise ValueError("模型比较交付物不是唯一的合法候选")
    with threadpool_limits(limits=1):
        operating_point = load_operating_point(out_dir / COMPARISON_MODEL_FILE,
                                               manifest=out_dir / COMPARISON_FILE, require_manifest=True)
        if (operating_point["model"] != selected or operating_point["costs"] != plan["costs"]
                or operating_point["threshold"] is not None or operating_point["strategy"] != "class_weight"):
            raise ValueError("模型比较部署模型或成本与选择锁不一致")
        row = outer["models"][selected]
        verify_operating_point_evaluation(operating_point, features.loc[reference_test],
                                          row["predictions"], row["scores"], row["score_space"])
    if (operating_point["selected_features"] != row["selector_features"]
            or operating_point["test_confusion"] != row["metrics"]["confusion"]):
        raise ValueError("模型比较交付物未逐位复现外层预测")
    explanation = result["explanation"]
    if explanation["enabled"] != cfg["explain"]["enabled"]:
        raise ValueError("模型比较解释开关与配置不一致")
    if explanation["enabled"]:
        if explanation["model"] != selected or not explanation["artifacts"]:
            raise ValueError("模型比较解释对象不正确或缺少产物")
        for filename, digest in explanation["artifacts"].items():
            if _sha256(out_dir / filename) != digest:
                raise ValueError("模型比较解释产物摘要不一致")
    return result


def main():
    parser = argparse.ArgumentParser(description="固定训练侧模型比较；依赖同目录已完成的稳定性产物")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--check", action="store_true", help="只重算和核验已有产物，不重训")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    features, labels, _timestamps = load_secom(
        str(PROJECT_ROOT / cfg["data"]["features_path"]), str(PROJECT_ROOT / cfg["data"]["labels_path"]),
        cfg["data"]["timestamp_format"])
    split = resolve_artifact_split(cfg)
    reference_train, reference_test = artifact_split(labels, split["test_size"], split["seed"])
    action = validate_model_comparison_artifacts if args.check else run_model_comparison
    result = action(cfg, features, labels, reference_train, reference_test, PROJECT_ROOT / cfg["output"]["results_dir"])
    print(json.dumps(result["summary"]["selection"], ensure_ascii=False))
    print("OK: 模型角色、训练侧选择、配对差值、外层预测与唯一交付物核验通过")


if __name__ == "__main__":
    main()

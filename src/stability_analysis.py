"""稳定性跑批、可审计产物与无需重训的证据重算校验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

from .config import load_config
from .data_io import load_secom
from .feature_selection import load_feature_override
from .preprocessing import build_preprocess_pipeline, drop_all_nan_columns
from .stability import (
    STABILITY_PROTOCOL_FILE, build_stability_protocol, fit_stability_split,
    protocol_definition, resolve_stability, resolve_stability_analysis,
    select_stability_features, write_stability_protocol,
)
from .stability_metrics import TIERS, correlation_groups, summarize_stability
from .validation import artifact_split, resolve_artifact_split

logger = logging.getLogger(__name__)
STABILITY_DETAIL_FILE = "stability_detail.json"
STABILITY_PLAN_FILE = "stability_analysis_plan.json"
CANDIDATES_FILE = "parameter_candidates.json"
CANDIDATES_TEXT_FILE = "parameter_candidates.txt"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(features, labels, reference_train) -> dict:
    digest = hashlib.sha256(json.dumps(list(features.columns), ensure_ascii=False).encode("utf-8"))
    for values in (features.loc[reference_train].astype(float), labels.loc[reference_train].astype(float)):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    sources = ("stability.py", "stability_metrics.py", "stability_analysis.py",
               "feature_selection.py", "preprocessing.py", "validation.py")
    return {
        "reference_training_data_sha256": digest.hexdigest(),
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in sources},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "joblib": joblib.__version__},
    }


def _analysis_plan(cfg: dict, protocol_path: Path) -> dict:
    return {"artifact_schema_version": 1, "sampling_protocol_sha256": _sha256(protocol_path),
            "analysis": resolve_stability_analysis(cfg)}


def _groups_from_training(features, reference_train, cfg, spec) -> list[dict]:
    train_features, _dropped = drop_all_nan_columns(features.loc[reference_train])
    preprocessor = build_preprocess_pipeline(cfg)
    imputed = pd.DataFrame(preprocessor.fit_transform(train_features),
                           index=reference_train, columns=train_features.columns)
    return correlation_groups(imputed.reindex(columns=features.columns), spec["correlation_group_threshold"])


def _candidate_payload(result: dict) -> dict:
    keys = ("artifact_schema_version", "status", "analysis_executed", "protocol",
            "sampling_protocol_sha256", "analysis_plan", "fingerprint", "reference_features")
    summary_keys = ("counts_by_tier", "frequency_threshold_sensitivity",
                    "outside_reference_high_frequency", "correlation_groups", "candidates")
    return {**{key: result[key] for key in keys}, "evidence_file": STABILITY_DETAIL_FILE,
            **{key: result["summary"][key] for key in summary_keys}}


def render_parameter_candidates(result: dict) -> str:
    """候选清单显式区分三档、条件方向、相关组与抽样总体。"""
    protocol, summary = result["protocol"], result["summary"]
    criteria = result["sampling_protocol"]["criteria"]
    lines = [
        "参数候选清单（统计关联，不是已确认工艺根因）", "",
        f"总体：固定 reference 训练侧 {protocol['n_population']} 行内的 {protocol['n_splits']} 次独立重采样；"
        "不使用外层测试集，不与性能指标重采样混为同一总体。",
        f"抽样协议 SHA-256：{result['sampling_protocol_sha256']}",
        f"主频率阈值 {criteria['primary_frequency_threshold']}；"
        f"高稳定还须方向一致率 >= {criteria['classification']['high_min_direction_consistency']}。",
        "方向：每轮入选特征上的加权岭系数；+ 表示更高的模型失效分数，- 表示更低；"
        "一致率分母为入选轮次（含零系数），不是工艺因果方向。",
        "相关组：reference 训练侧拟合填充后 abs(Spearman) 达阈值连边的连通分量；"
        "连通不代表组内任意两项都达阈值。组频率按至少一项入选计，不累加成员频率。",
        "证据不足不代表没有工艺价值；低频候选可保留为后续实验线索。", "",
        "频率阈值敏感性（全部预定档位）：",
    ]
    for threshold, entry in summary["frequency_threshold_sensitivity"].items():
        counts = " / ".join(f"{tier} {entry['counts_by_tier'][tier]}" for tier in TIERS)
        lines.append(f"  {threshold}: {counts}")
    outside = summary["outside_reference_high_frequency"]
    lines.extend(["", "reference 清单之外的主档高频候选：" + (", ".join(outside) or "无"),
                  "（高频不自动等于高稳定，仍须检查方向一致率。）"])
    for tier in TIERS:
        lines.extend(["", f"{tier}（{summary['counts_by_tier'][tier]} 项）：",
                      "特征 | 入选次数/总次数 | 频率 | 方向 | 一致率 | 有方向次数 | 相关组 | reference内"])
        for row in summary["candidates"]:
            if row["tier"] != tier:
                continue
            evidence = row["direction"]
            direction = {1: "+", -1: "-", 0: "混合", None: "缺失"}[evidence["majority_sign"]]
            consistency = "缺失" if evidence["consistency"] is None else f"{evidence['consistency']:.3f}"
            lines.append(f"{row['feature']} | {row['selected_count']}/{protocol['n_splits']} | "
                         f"{row['selection_frequency']:.3f} | {direction} | {consistency} | "
                         f"{evidence['evidence_count']} | {row['correlation_group']} | "
                         f"{'是' if row['in_reference'] else '否'}")
    lines.extend(["", "含多项的相关组（按至少一项入选统计组频率）："])
    for group in summary["correlation_groups"]:
        if len(group["members"]) > 1:
            lines.append(f"{group['group']} | {group['selection_frequency']:.3f} | {', '.join(group['members'])}")
    lines.extend(["", "抽样一致性（两两分布是描述统计，不是独立重复实验的置信区间）："])
    for key in ("spearman", "top_k_jaccard", "reference_top_k_jaccard"):
        stats = summary["pairwise"][key]
        lines.append(f"{key}: median={stats['median']}, p05={stats['p05']}, p95={stats['p95']}, "
                     f"n={stats['n']}, missing={stats['missing']}")
    lines.extend(["", "累计重采样趋势（不据此宣称最佳次数或已经收敛）："])
    for checkpoint in summary["convergence"]["checkpoints"]:
        lines.append(f"R={checkpoint['n_resamples']}: 最大频率变化={checkpoint['max_frequency_change']}; "
                     f"高稳定新增={checkpoint['high_stability_added']}; "
                     f"高稳定移出={checkpoint['high_stability_removed']}")
    return "\n".join(lines) + "\n"


def run_stability_analysis(cfg, features, labels, reference_train, reference_test,
                           output_dir, reference_features=None) -> dict:
    """显式调用执行完整选择器；日常流水线由 stability.enabled 控制是否调度。"""
    started = perf_counter()
    spec, analysis = resolve_stability(cfg), resolve_stability_analysis(cfg)
    out_dir = Path(output_dir)
    protocol_path = write_stability_protocol(labels, reference_train, reference_test, cfg, out_dir)
    frozen = _read_json(protocol_path)
    plan = _analysis_plan(cfg, protocol_path)
    plan_path = out_dir / STABILITY_PLAN_FILE
    if plan_path.exists() and _read_json(plan_path) != plan:
        raise FileExistsError("已冻结的稳定性分析口径不同；请使用独立输出目录，不能静默覆盖判据")
    if not plan_path.exists():
        _write_json(plan_path, plan)
    _write_json(out_dir / STABILITY_DETAIL_FILE, {"status": "running", "analysis_executed": False})
    fingerprint = _fingerprint(features, labels, reference_train)
    reference_started = perf_counter()
    if reference_features is None:
        with threadpool_limits(limits=1):
            reference_features, _ranks = select_stability_features(
                features, labels, reference_train, reference_train, reference_test, cfg)
    reference_seconds = perf_counter() - reference_started
    groups = _groups_from_training(features, reference_train, cfg, spec)
    logger.info("稳定性分析：reference 训练侧 %d 行，%d 次重采样，n_jobs=%d",
                len(reference_train), spec["n_resamples"], spec["n_jobs"])

    def one(split):
        record = fit_stability_split(features, labels, pd.Index(split["train_indices"]),
                                     reference_train, reference_test, cfg, split["split"])
        return {**split, **record}

    if spec["n_jobs"] == 1:
        records = [one(split) for split in frozen["splits"]]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            records = Parallel(n_jobs=spec["n_jobs"], verbose=10)(
                delayed(one)(split) for split in frozen["splits"])
    summary = summarize_stability(records, list(features.columns), list(reference_features),
                                  groups, spec, analysis)
    if _fingerprint(features, labels, reference_train) != fingerprint:
        raise RuntimeError("稳定性跑批期间训练输入或实现发生变化；不得保存为成功产物")
    result = {
        "artifact_schema_version": 1, "status": "ok", "analysis_executed": True,
        "protocol": frozen["protocol"], "sampling_protocol": frozen,
        "sampling_protocol_sha256": _sha256(protocol_path), "analysis_plan": plan,
        "fingerprint": fingerprint,
        "feature_names": list(features.columns), "reference_features": list(reference_features),
        "records": records, "summary": summary,
        "execution": {"n_jobs": spec["n_jobs"], "blas_threads_per_worker": 1,
                      "reference_selection_seconds": reference_seconds,
                      "elapsed_seconds": perf_counter() - started},
    }
    _write_json(out_dir / STABILITY_DETAIL_FILE, result)
    _write_json(out_dir / CANDIDATES_FILE, _candidate_payload(result))
    (out_dir / CANDIDATES_TEXT_FILE).write_text(render_parameter_candidates(result), encoding="utf-8")
    logger.info("稳定性分析完成：%s；总耗时 %.1f s", summary["counts_by_tier"], result["execution"]["elapsed_seconds"])
    return result


def validate_stability_artifacts(cfg, features, labels, reference_train, reference_test,
                                 output_dir, reference_features=None) -> dict:
    """核对训练来源与冻结协议，从逐轮证据重算清单、分布、相关组及趋势。"""
    out_dir = Path(output_dir)
    spec, analysis = resolve_stability(cfg), resolve_stability_analysis(cfg)
    result = _read_json(out_dir / STABILITY_DETAIL_FILE)
    if result.get("artifact_schema_version") != 1 or result.get("status") != "ok" or result.get("analysis_executed") is not True:
        raise ValueError("稳定性分析未成功完成")
    protocol_path = out_dir / STABILITY_PROTOCOL_FILE
    frozen = _read_json(protocol_path)
    expected = build_stability_protocol(labels, reference_train, reference_test, cfg)
    if protocol_definition(frozen) != protocol_definition(expected):
        raise ValueError("稳定性产物的抽样协议或判据已陈旧")
    if result["sampling_protocol"] != frozen or result["protocol"] != frozen["protocol"]:
        raise ValueError("稳定性详情未携带原独立抽样协议")
    plan = _analysis_plan(cfg, protocol_path)
    if (result["sampling_protocol_sha256"] != _sha256(protocol_path)
            or result["analysis_plan"] != plan or _read_json(out_dir / STABILITY_PLAN_FILE) != plan):
        raise ValueError("稳定性冻结协议摘要或分析口径不一致")
    if result["fingerprint"] != _fingerprint(features, labels, reference_train):
        raise ValueError("稳定性训练数据、实现或运行依赖已陈旧；须重跑")
    if result["feature_names"] != list(features.columns):
        raise ValueError("稳定性特征全集不一致")
    if reference_features is not None and result["reference_features"] != list(reference_features):
        raise ValueError("稳定性 reference 特征清单与当前主流程不一致")
    if len(result["records"]) != len(frozen["splits"]):
        raise ValueError("稳定性逐轮证据数量不完整")
    execution = result["execution"]
    for key in ("reference_selection_seconds", "elapsed_seconds"):
        if not np.isfinite(execution[key]) or execution[key] < 0:
            raise ValueError("稳定性总耗时记录无效")
    if execution["reference_selection_seconds"] > execution["elapsed_seconds"]:
        raise ValueError("reference 选择耗时超过总耗时")
    for record, split in zip(result["records"], frozen["splits"]):
        if {key: record[key] for key in split} != split:
            raise ValueError("稳定性逐轮证据的训练索引与冻结协议不同")
        for key in ("selection_seconds", "elapsed_seconds"):
            if not np.isfinite(record[key]) or record[key] < 0:
                raise ValueError("稳定性耗时记录无效")
        if record["selection_seconds"] > record["elapsed_seconds"]:
            raise ValueError("单轮选择耗时超过单轮总耗时")
    groups = _groups_from_training(features, reference_train, cfg, spec)
    rebuilt = summarize_stability(result["records"], result["feature_names"],
                                  result["reference_features"], groups, spec, analysis)
    if result["summary"] != rebuilt:
        raise ValueError("稳定性摘要与逐轮证据重算不一致")
    if _read_json(out_dir / CANDIDATES_FILE) != _candidate_payload(result):
        raise ValueError("参数候选 JSON 与稳定性详情不一致")
    if (out_dir / CANDIDATES_TEXT_FILE).read_text(encoding="utf-8") != render_parameter_candidates(result):
        raise ValueError("参数候选文本与稳定性详情不一致")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="独立执行 reference 训练侧稳定性分析")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--check", action="store_true", help="只核验已有证据，不重训")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    features, labels, _timestamps = load_secom(
        str(PROJECT_ROOT / cfg["data"]["features_path"]),
        str(PROJECT_ROOT / cfg["data"]["labels_path"]), cfg["data"]["timestamp_format"])
    split = resolve_artifact_split(cfg)
    reference_train, reference_test = artifact_split(labels, split["test_size"], split["seed"])
    out_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    if args.check:
        reference_files = sorted(out_dir.glob("selected_features_full_*.txt"))
        reference_features = load_feature_override(reference_files[-1]) if reference_files else None
        result = validate_stability_artifacts(cfg, features, labels, reference_train, reference_test,
                                              out_dir, reference_features)
        print("OK: 稳定性协议、训练来源、逐轮证据及候选清单重算通过")
    else:
        result = run_stability_analysis(cfg, features, labels, reference_train, reference_test, out_dir)
    print(json.dumps(result["summary"]["counts_by_tier"], ensure_ascii=False))


if __name__ == "__main__":
    main()

"""漂移分析图文落盘及从原始数据重算的产物验收。"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .drift import DRIFT_FILE, DRIFT_MARKDOWN, DRIFT_PLOT, DRIFT_TITLE, build_drift_report

logger = logging.getLogger(__name__)


def _number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _ranked_features(windows, limit, *, missingness_only=False):
    names = [name for name, row in windows[0]["features"].items()
             if (row["psi_basis"] == "missingness_only") == missingness_only]
    return sorted(names, key=lambda name: (
        -max(window["features"][name]["psi"] for window in windows), name))[:limit]


def _missingness_groups(windows):
    groups = {}
    for name in _ranked_features(windows, None, missingness_only=True):
        rates = tuple(window["features"][name]["current_missing_rate"] for window in windows)
        groups.setdefault(rates, []).append(name)
    return list(groups.values())


def render_drift_report(result: dict) -> str:
    prior = result["prior_drift"]
    equal_count = prior["equal_count"]
    plan = result["analysis_plan"]
    performance = result["performance"]
    settings = plan["settings"]
    lines = [f"# {DRIFT_TITLE}", "", "离线描述性分析，不是在线监控或工艺归因。",
             "SECOM 样本通过率仅作良率代理，不代表全厂良率。", "",
             f"![Temporal drift overview]({DRIFT_PLOT})", "", "## 先验与样本通过率", "",
             f"时间跨度：{equal_count['span'][0]} — {equal_count['span'][1]}；"
             f"{equal_count['n']} 片，{equal_count['failures']} 片失效，"
             f"失败率 {equal_count['overall_failure_rate']:.2%}。",
             "按时间稳定排序后用 numpy.array_split 等样本数分箱，并非等日历时长。", "",
             "| 窗口 | 起止时间 | n | 失效 | 失败率 | 样本通过率 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in equal_count["bins"]:
        lines.append(f"| {row['bin']} | {row['start']} — {row['end']} | {row['n']} | "
                     f"{row['failures']} | {row['failure_rate']:.2%} | {1 - row['failure_rate']:.2%} |")
    lines += ["", f"非零失败率最大/最小比 {_number(equal_count['max_over_min_ratio'], 2)}；"
              f"首窗占全部失效 {_number(equal_count['first_bin_share_of_failures'] * 100 if equal_count['first_bin_share_of_failures'] is not None else None, 2)}%。"
              f"零失效窗口：{prior['zero_failure_bins']}（不纳入该比值，不能解读为无限倍已被估计）。",
              "", "| 月份 | n | 失效 | 失败率 |", "| --- | ---: | ---: | ---: |"]
    for row in prior["monthly"]:
        lines.append(f"| {row['window']} | {row['n']} | {row['failures']} | {row['failure_rate']:.2%} |")
    lines += ["", f"另报 {len(prior['rolling'])} 个样本滑窗：窗宽 {settings['window_size']}、"
              f"步长 {settings['step_size']}，末窗对齐最后一行；不足一个窗时使用全部可用行。"
              "各窗起止时间、n、失败数与通过率均在 JSON 中。", "", "## 原始特征分布", "",
              f"覆盖全部 {result['feature_drift']['feature_count']} 个原始匿名特征；"
              "后续窗口分别对照第一窗。第一窗只是描述性参照，不是工程确认的健康基准。",
              f"PSI：参照窗 {settings['psi_bins']} 等分位区间、重复分位点合并；上下越界及缺失单独成桶，"
              f"每桶加 {settings['psi_pseudocount']} 伪计数后归一化。常量参照分为低于/等于/高于/缺失；"
              "全缺失参照仅能比较缺失率，不能估计未观测数值的漂移。",
              f"KS：各侧至少 {settings['ks_min_samples']} 个非缺失值；不足时记 N/A。"
              f"全部 {result['feature_drift']['ks_valid_tests']} 个有效特征×窗口比较一起做 BH 校正。",
              "KS 的 p/q 值仍受连续分布、独立观测假设约束；量化重复值、时间相关与特征相关使它们只能探索性阅读。"
              "不设 PSI/KS 报警阈值，不把高 PSI、低 p/q 或缺失率变化当成工艺因果证据。", "",
              "有观测参照与全缺失参照分开展示，避免仅缺失率变化遮住其他分布变化；"
              "所有排序与合并只用于展示，不用于选特征或模型。", "",
              "### 有观测参照：分位或常量基准", "",
              f"每窗展示本类 PSI 最高的 {settings['top_features']} 项；PSI 仍含尾部和缺失桶，"
              "不能把它解释为排除了缺失影响的纯数值漂移。", "",
              "| 窗口 | 特征 | PSI 依据 | PSI | KS D | nominal p | BH q | 缺失率变化 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    basis_labels = {"missingness_only": "仅缺失率", "constant_with_tails_and_missing": "常量/尾部/缺失",
                    "reference_quantiles_with_tails_and_missing": "分位/尾部/缺失"}
    windows = result["feature_drift"]["windows"]
    for window in windows:
        for name in _ranked_features([window], settings["top_features"]):
            row = window["features"][name]
            lines.append(f"| {window['window']} | {name} | {basis_labels[row['psi_basis']]} | {_number(row['psi'])} | "
                         f"{_number(row['ks_statistic'])} | {_number(row['ks_p_value'], 6)} | "
                         f"{_number(row['ks_q_value'], 6)} | {row['missing_rate_delta']:+.2%} |")
    if not _ranked_features(windows, 1):
        lines += ["", "无有观测参照特征。"]
    missing_groups = _missingness_groups(windows)
    lines += ["", "### 全缺失参照：仅缺失率", "",
              f"共 {sum(map(len, missing_groups))} 列，按各评估窗完全相同的缺失率轨迹合并为 "
              f"{len(missing_groups)} 组；按跨窗最大 PSI 展示前 {settings['top_features']} 组。"
              "代表列和组内列数仅作去重展示，JSON 保留全部原始列与统计值。", "",
              "| 窗口 | 代表列 | PSI 依据 | 同轨迹列数 | PSI | 参照缺失率 | 当前缺失率 | 缺失率变化 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for window in windows:
        for names in missing_groups[:settings["top_features"]]:
            row = window["features"][names[0]]
            lines.append(f"| {window['window']} | {names[0]} | 仅缺失率 | {len(names)} | "
                         f"{_number(row['psi'])} | {row['reference_missing_rate']:.2%} | "
                         f"{row['current_missing_rate']:.2%} | {row['missing_rate_delta']:+.2%} |")
    if not missing_groups:
        lines += ["", "无全缺失参照特征。"]
    lines += ["", "## 固定早期模型的未来性能", "",
              f"固定 {performance['model']} + class_weight，默认决策边界；不校准、不调阈值、不重选模型。"
              f"仅第一窗候选 {len(performance['nominal_train_indices'])} 行用于训练；"
              f"剔除 {len(performance['excluded_boundary_indices'])} 行边界同时间戳后，实际训练 "
              f"{len(performance['train_indices'])} 行（{performance['training']['failures']} 片失效）。",
              f"训练时段 {performance['training']['start']} — {performance['training']['end']}；"
              "填充、缩放和特征选择只拟合这些行，全部评估行的 timestamp 严格更晚。",
              f"选出 {len(performance['selected_features'])} 个特征，一次拟合后在后续 "
              f"{len(performance['evaluation_indices'])} 行预测。分数空间为 {performance['score_space']}，不是概率。",
              "首窗不报训练内性能；不复用可能见过未来晶圆的随机划分交付模型。"
              "这是更小早期训练窗的另一协议，不能和前 80% 训练的时间序对照或随机协议直接配对。", "",
              "| 评估窗 | n | 失效 | TN / FP / FN / TP | AUC | BER | 召回 | 特异度 | 每片代价 |",
              "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in [*performance["equal_count"], performance["overall"]]:
        metrics = row["metrics"]
        counts = metrics["confusion"]
        lines.append(f"| {row['window']} | {row['n']} | {row['failures']} | "
                     f"{counts['tn']} / {counts['fp']} / {counts['fn']} / {counts['tp']} | "
                     f"{_number(metrics['auc'])} | {_number(metrics['ber'])} | {_number(metrics['recall'])} | "
                     f"{_number(metrics['specificity'])} | {_number(metrics['cost_per_wafer'])} |")
    lines += ["", "随机排序的 AUC 基线为 0.5，标签独立预测的 BER 基线为 0.5；"
              "AUC 越高、BER 越低越好。这些是点估计，不是与随机基线的显著性检验。"]
    low_auc = [row for row in performance["equal_count"]
               if row["metrics"]["auc"] is not None and row["metrics"]["auc"] < 0.5]
    if low_auc:
        readings = "；".join(f"窗口 {row['window']} AUC={_number(row['metrics']['auc'])}、"
                             f"BER={_number(row['metrics']['ber'])}" for row in low_auc)
        lines.append(f"其中 {readings}：AUC 点估计低于随机排序基线，不能把后续窗口的改善解读为模型已可用。"
                     "早期训练样本少、特征多只是背景，不能据此把低分归因于训练规模或漂移。")
    lines += ["", f"成本假设 FN={plan['costs']['fn']:g} / FP={plan['costs']['fp']:g}；"
              f"JSON 另含 {len(performance['rolling'])} 个仅未来样本的滑窗、逐行预测、分数及训练状态。",
              "缺正类时召回不可定义，缺负类时特异度不可定义；任一类别缺失时 BER/AUC 为 null。"
              "小窗口只有少量失效，单片即可显著改变召回；曲线不是性能区间，也不保证单调下降。", "",
              "## 边界与复核", "",
              "- 窗口重叠，不是独立重复实验；不把 KS/BH 当成控制限，不构造跨协议配对区间。",
              "- 先验、特征与性能共同变化不构成归因；缺少 lot / 设备 / 腔室 / recipe 标识。",
              "- 原始数据、配置、源码指纹与逐行预测保存在 JSON；验收会重新拟合早期链路并重算图文。",
              f"- 数据 SHA-256：`{result['fingerprint']['data_sha256']}`。", ""]
    return "\n".join(lines)


def plot_drift(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(13, 15), constrained_layout=True)
    prior = result["prior_drift"]
    for rows, label, alpha in ((prior["rolling"], "Rolling sample windows", 0.6),
                               (prior["equal_count"]["bins"], "Equal-count windows", 1.0)):
        axes[0].plot(pd.to_datetime([row["end"] for row in rows]),
                     [100 * row["failure_rate"] for row in rows], marker="o", label=label, alpha=alpha)
    axes[0].set(title="Observed failure rate over time", ylabel="Failure rate (%)")
    axes[0].legend()
    performance = result["performance"]
    for key, label in (("auc", "AUC"), ("ber", "BER"), ("recall", "Recall"), ("specificity", "Specificity")):
        rows = performance["equal_count"]
        axes[1].plot(pd.to_datetime([row["end"] for row in rows]),
                     [row["metrics"][key] for row in rows], marker="o", label=label)
    rolling = performance["rolling"]
    axes[1].plot(pd.to_datetime([row["end"] for row in rolling]),
                 [row["metrics"]["recall"] for row in rolling], alpha=0.35, linestyle="--",
                 label="Rolling recall (overlapping)")
    axes[1].set(title="Fixed early-trained Ridge: future windows only", ylabel="Metric", ylim=(-0.03, 1.03))
    axes[1].axhline(0.5, color="gray", linestyle=":", label="AUC/BER chance baseline")
    axes[1].legend(ncol=3)
    windows = result["feature_drift"]["windows"]
    top_features = result["analysis_plan"]["settings"]["top_features"]
    names = _ranked_features(windows, top_features)
    axes[2].set(title="Observed-reference PSI (includes tails and missingness; not alarm thresholds)",
                xticks=range(len(windows)), xticklabels=[f"Window {window['window']}" for window in windows],
                yticks=range(len(names)), yticklabels=names)
    if names:
        values = np.asarray([[window["features"][name]["psi"] for window in windows] for name in names])
        image = axes[2].imshow(values, aspect="auto", cmap="viridis")
        figure.colorbar(image, ax=axes[2], label="PSI")
    else:
        axes[2].text(0.5, 0.5, "No observed-reference features", ha="center", transform=axes[2].transAxes)
    missing_groups = _missingness_groups(windows)
    positions = [1, *[window["window"] for window in windows]]
    for names in missing_groups[:top_features]:
        rows = [window["features"][names[0]] for window in windows]
        rates = [rows[0]["reference_missing_rate"], *[row["current_missing_rate"] for row in rows]]
        axes[3].plot(positions, 100 * np.asarray(rates), marker="o", label=f"{names[0]} ({len(names)} features)")
    axes[3].set(title=f"Missingness only: {sum(map(len, missing_groups))} features / "
                       f"{len(missing_groups)} distinct trajectories (showing up to {top_features})",
                ylabel="Missing rate (%)", ylim=(-3, 103), xticks=positions,
                xticklabels=["Window 1 (reference)", *[f"Window {window['window']}" for window in windows]])
    if missing_groups:
        axes[3].legend(ncol=2)
    else:
        axes[3].text(0.5, 0.5, "No fully-missing reference features", ha="center", transform=axes[3].transAxes)
    try:
        figure.savefig(path, dpi=140, metadata={"Software": "SECOM drift analysis"})
    finally:
        plt.close(figure)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def run_drift_analysis(cfg, features, labels, timestamps, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / DRIFT_FILE, {"status": "running", "title": DRIFT_TITLE})
    try:
        result = build_drift_report(cfg, features, labels, timestamps)
        plot_drift(result, output / DRIFT_PLOT)
        result["plot_sha256"] = hashlib.sha256((output / DRIFT_PLOT).read_bytes()).hexdigest()
        (output / DRIFT_MARKDOWN).write_text(render_drift_report(result), encoding="utf-8")
        _write_json(output / DRIFT_FILE, result)
    except Exception as exc:
        _write_json(output / DRIFT_FILE, {"status": "failed", "title": DRIFT_TITLE, "error": str(exc)})
        raise
    logger.info("%s已保存: %s", DRIFT_TITLE, output / DRIFT_FILE)
    return result


def validate_drift_artifacts(cfg, features, labels, timestamps, output_dir):
    output = Path(output_dir)
    for name in (DRIFT_FILE, DRIFT_MARKDOWN, DRIFT_PLOT):
        if not (output / name).is_file():
            raise ValueError(f"漂移产物缺失: {name}")
    result = json.loads((output / DRIFT_FILE).read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise ValueError("漂移产物 status 不是 ok")
    expected = build_drift_report(cfg, features, labels, timestamps)
    with TemporaryDirectory(prefix="secom-drift-") as temporary:
        image_path = Path(temporary) / DRIFT_PLOT
        plot_drift(expected, image_path)
        expected["plot_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if json.dumps(result, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("漂移 JSON 与当前数据、协议及早期链路重算不一致")
    if (output / DRIFT_MARKDOWN).read_text(encoding="utf-8") != render_drift_report(expected):
        raise ValueError("漂移 Markdown 与重算结果不一致")
    if hashlib.sha256((output / DRIFT_PLOT).read_bytes()).hexdigest() != expected["plot_sha256"]:
        raise ValueError("漂移图像与重算结果不一致")
    return result

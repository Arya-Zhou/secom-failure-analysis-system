"""演示视图模型：只从已跟踪快照取数，不加载工作点产物。"""
from __future__ import annotations

import re
from pathlib import Path

from .demo_data import (
    DEMO_DIR, GLOBAL_CHOICE, GLOBAL_SHAP_NOTE, SHAP_NOTE,
    missing_report_files, read_json, read_text,
)

TEMPORAL_PANEL_SPECS = (
    {
        "choice": "失效比例随时间",
        "lines": (
            "**失效比例随时间**",
            "- 横轴是时间，纵轴是窗口失败率",
            "- 蓝线：滚动样本窗",
            "- 橙线：等样本数分箱",
            "- 两条曲线一起看失败率如何随时间变化",
        ),
    },
    {
        "choice": "窗口性能",
        "lines": (
            "**固定早期岭分类器的后续窗口表现**",
            "- 只在最早窗口训练，只在后续窗口评估",
            "- 同时给出 AUC、BER、召回和特异度",
            "- 灰色虚线是 AUC/BER 的随机基线 0.5",
            "- 淡虚线是重叠滚动窗的召回，波动会更大",
        ),
    },
    {
        "choice": "特征 PSI",
        "lines": (
            "**观察窗相对参考窗的 PSI**",
            "- 颜色越亮，该特征相对第一窗的分布偏移越大",
            "- 窗口按等样本数划分，便于对照后期偏移集中在哪些参数",
            "- 用于观察匿名传感器读数是否随时间变样",
        ),
    },
    {
        "choice": "缺失率轨迹",
        "lines": (
            "**缺失率轨迹**",
            "- 把缺失变化模式相同的特征合成一组",
            "- 绿线：F591 组保持接近全缺失",
            "- 蓝线：F110 组缺失率随时间下降",
            "- 橙线：F086 组后期仍保持较高缺失",
        ),
    },
)


def _r3(value) -> float:
    """与私有大纲一致的三位小数舍入。"""
    return float(f"{float(value):.3f}")


def _r2(value) -> float:
    """失败率用两位小数。"""
    return float(f"{float(value):.2f}")


def parse_overview_counts(drift_md: str) -> dict:
    """从漂移报告读取样本规模。"""
    match = re.search(r"(\d+)\s*片，\s*(\d+)\s*片失效，失败率\s*([0-9.]+)%", drift_md)
    if not match:
        raise ValueError("drift_report.md 未找到样本规模句")
    return {
        "n_wafers": int(match.group(1)),
        "n_fail": int(match.group(2)),
        "fail_rate_pct": _r2(match.group(3)),
    }


def parse_resample_interval(payload: dict) -> dict:
    """读取岭分类器重复划分区间。"""
    ridge = payload["intervals"]["岭分类器"]
    ber, rec = ridge["测试集BER"], ridge["召回率"]
    return {
        "n_splits": int(payload["protocol"]["n_splits"]),
        "ber": _r3(ber["point"]),
        "ber_low": _r3(ber["low"]),
        "ber_high": _r3(ber["high"]),
        "recall": _r3(rec["point"]),
        "recall_low": _r3(rec["low"]),
        "recall_high": _r3(rec["high"]),
    }


def parse_temporal_ridge(payload: dict) -> dict:
    """读取同名岭分类器的时间序对照。"""
    protocol = payload["protocol"]
    ridge = payload["models"]["岭分类器"]
    return {
        "n_train": int(protocol["n_train"]),
        "n_test": int(protocol["n_test"]),
        "n_test_fail": int(protocol["n_test_positive"]),
        "ber": _r3(ridge["测试集BER"]),
        "recall": _r3(ridge["召回率"]),
    }


def parse_candidate_counts(text: str) -> dict:
    """读取主频率阈值 0.7 的三档计数与高稳定名单。"""
    match = re.search(
        r"0\.7:\s*高稳定候选\s*(\d+)\s*/\s*探索性候选\s*(\d+)\s*/\s*当前证据不足\s*(\d+)",
        text,
    )
    if not match:
        raise ValueError("parameter_candidates.txt 未找到 0.7 档计数")
    names = []
    marker = "高稳定候选（"
    if marker not in text:
        raise ValueError("parameter_candidates.txt 未找到高稳定候选表")
    block = text.split(marker, 1)[1].split("探索性候选", 1)[0]
    for line in block.splitlines():
        if line.startswith("F") and "|" in line:
            names.append(line.split("|", 1)[0].strip())
    return {
        "high": int(match.group(1)),
        "exploratory": int(match.group(2)),
        "insufficient": int(match.group(3)),
        "high_names": names,
    }


def parse_ridge_cost_example(text: str) -> dict:
    """从策略对比表读取 Ridge 基线与 class_weight 代价例子。"""
    section = text.split("## 岭分类器", 1)[1]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 11 or cells[0] in {"策略", "---"}:
            continue
        if cells[0] in {"原始基线", "class_weight"}:
            rows[cells[0]] = {
                "fn": int(cells[8]),
                "fp": int(cells[9]),
                "cost": int(float(cells[10])),
            }
        if "原始基线" in rows and "class_weight" in rows:
            break
    if set(rows) != {"原始基线", "class_weight"}:
        raise ValueError("imbalance_comparison.md 未找到岭分类器基线与 class_weight 行")
    base, weighted = rows["原始基线"], rows["class_weight"]
    drop = (base["cost"] - weighted["cost"]) / base["cost"]
    return {
        "fn_from": base["fn"],
        "fn_to": weighted["fn"],
        "fp_from": base["fp"],
        "fp_to": weighted["fp"],
        "cost_from": base["cost"],
        "cost_to": weighted["cost"],
        "drop_pct": int(round(drop * 100)),
    }


def overview_view(demo_dir: Path = DEMO_DIR) -> dict:
    """总览页数字。"""
    counts = parse_overview_counts(read_text("drift_report.md", demo_dir))
    interval = parse_resample_interval(read_json("resample_metrics.json", demo_dir))
    temporal = parse_temporal_ridge(read_json("temporal_metrics.json", demo_dir))
    cost = parse_ridge_cost_example(read_text("imbalance_comparison.md", demo_dir))
    return {
        **counts,
        **{f"interval_{k}": v for k, v in interval.items()},
        **{f"temporal_{k}": v for k, v in temporal.items()},
        **{f"cost_{k}": v for k, v in cost.items()},
    }


def candidates_view(demo_dir: Path = DEMO_DIR) -> dict:
    """候选清单页：高稳定优先，其余只给计数。"""
    counts = parse_candidate_counts(read_text("parameter_candidates.txt", demo_dir))
    return {
        **counts,
        "caveat": "稳定性三档用于判断重采样结论是否站得住。",
    }


def shap_case_view(demo_dir: Path = DEMO_DIR) -> dict:
    """主分析 SHAP 案例路径与口径。"""
    fn_image = str(demo_dir / "shap_fn_wafer.png")
    tp_image = str(demo_dir / "shap_tp_wafer.png")
    global_image = str(demo_dir / "shap_global_ridge.png")
    return {
        "note": SHAP_NOTE,
        "fn_image": fn_image,
        "tp_image": tp_image,
        "global_image": global_image,
        "by_case": {"FN": fn_image, "TP": tp_image},
        "fn_report": read_text("shap_fn_report.txt", demo_dir),
    }


def image_for_sample(choice: str, shap_case: dict) -> str | None:
    """样本下拉对应的图片；FP/TN 无局部 SHAP。"""
    if choice == GLOBAL_CHOICE:
        return shap_case["global_image"]
    case = choice.split(" ", 1)[0]
    return shap_case.get("by_case", {}).get(case)


def sample_image_note(choice: str) -> str:
    """图片上方说明：有图解释读法，无图说明空白是预期。"""
    if choice == GLOBAL_CHOICE:
        return "下图是主分析 reference 链的全局 SHAP，按平均绝对贡献排序。"
    case = choice.split(" ", 1)[0]
    if case == "FN":
        return "下图是该漏检样本的局部 SHAP。红色推向失效判定，蓝色抑制失效判定。"
    if case == "TP":
        return "下图是该命中样本的局部 SHAP。红色推向失效判定，蓝色抑制失效判定。"
    if case == "FP":
        return "FP 是误报的正常片。主分析只为 FN、TP 预存局部 SHAP，这里保持空白。"
    if case == "TN":
        return "TN 是正确放行的正常片。主分析只为 FN、TP 预存局部 SHAP，这里保持空白。"
    raise ValueError(f"未知样本选项：{choice}")


def global_shap_markdown() -> str:
    """全局 SHAP 文案。"""
    return "\n".join([
        GLOBAL_SHAP_NOTE,
        "",
        "**全局特征贡献**",
        "- 按平均绝对 SHAP 排序",
        "- 来自主分析 reference 链的预计算图",
    ])


def temporal_view(demo_dir: Path = DEMO_DIR) -> dict:
    """时间风险页。"""
    interval = parse_resample_interval(read_json("resample_metrics.json", demo_dir))
    temporal = parse_temporal_ridge(read_json("temporal_metrics.json", demo_dir))
    return {
        "image": str(demo_dir / "drift_overview.png"),
        "random_ber": interval["ber"],
        "random_ber_low": interval["ber_low"],
        "random_ber_high": interval["ber_high"],
        "random_recall": interval["recall"],
        "random_recall_low": interval["recall_low"],
        "random_recall_high": interval["recall_high"],
        "temporal_ber": temporal["ber"],
        "temporal_recall": temporal["recall"],
        "temporal_n_test": temporal["n_test"],
        "temporal_n_test_fail": temporal["n_test_fail"],
    }


def overview_markdown(view: dict) -> str:
    """总览页文案。"""
    return "\n".join([
        "**数据**",
        f"- 晶圆 {view['n_wafers']} 片",
        f"- 失效 {view['n_fail']} 片",
        f"- 失效比例约 {view['fail_rate_pct']}%",
        "",
        "**随机划分**",
        (
            f"- 岭分类器 BER {view['interval_ber']:.3f} "
            f"[{view['interval_ber_low']:.3f}, {view['interval_ber_high']:.3f}]"
        ),
        (
            f"- 召回 {view['interval_recall']:.3f} "
            f"[{view['interval_recall_low']:.3f}, {view['interval_recall_high']:.3f}]"
        ),
        f"- {view['interval_n_splits']} 次重采样的中位数与 5%–95% 经验分位",
        "",
        "**时间划分**",
        f"- 同名岭分类器召回 {view['temporal_recall']:.3f}",
        f"- BER {view['temporal_ber']:.3f}",
        f"- 测试 {view['temporal_n_test']} 片 / {view['temporal_n_test_fail']} 片失效",
        "",
        "**成本例子**",
        f"- 类别加权后漏检 {view['cost_fn_from']}→{view['cost_fn_to']}",
        f"- 误报 {view['cost_fp_from']}→{view['cost_fp_to']}",
        (
            f"- 相对代价 {view['cost_cost_from']}→{view['cost_cost_to']}，"
            f"约降 {view['cost_drop_pct']}%"
        ),
    ])


def candidates_markdown(view: dict) -> str:
    """候选清单文案。"""
    names = [f"- {name}" for name in view["high_names"]]
    return "\n".join([
        f"**高稳定候选 {view['high']} 项**",
        *names,
        "",
        "**其他档位**",
        f"- 探索性 {view['exploratory']} 项",
        f"- 当前证据不足 {view['insufficient']} 项",
        "",
        view["caveat"],
    ])


def temporal_markdown(view: dict) -> str:
    """时间风险文案。"""
    return "\n".join([
        "**划分对照**",
        "",
        (
            f"| 评估方式 | BER | 失效召回率 |\n| --- | --- | --- |\n"
            f"| 20 次随机分层划分 | {view['random_ber']:.3f} "
            f"[{view['random_ber_low']:.3f}, {view['random_ber_high']:.3f}] | "
            f"{view['random_recall']:.3f} [{view['random_recall_low']:.3f}, {view['random_recall_high']:.3f}] |\n"
            f"| 按时间先后划分 | {view['temporal_ber']:.3f} | {view['temporal_recall']:.3f} |"
        ),
        "",
        "**时间序测试段**",
        f"- {view['temporal_n_test']} 片",
        f"- {view['temporal_n_test_fail']} 片失效",
    ])


def temporal_panel_choices() -> list[str]:
    """时间风险图下拉项。"""
    return [spec["choice"] for spec in TEMPORAL_PANEL_SPECS]


def temporal_panel_markdown(choice: str) -> str:
    """单张漂移图说明。"""
    for spec in TEMPORAL_PANEL_SPECS:
        if spec["choice"] == choice:
            return "\n".join(spec["lines"])
    allowed = "、".join(temporal_panel_choices())
    raise ValueError(f"未知时间风险图：{choice}，可选：{allowed}")


def split_drift_overview_panels(image_path: str | Path) -> list[str]:
    """把四联图按等高切成四张 PNG，便于页内切换。"""
    from PIL import Image
    source = Path(image_path)
    cache_dir = Path("/tmp/secom_demo_panels")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{source.stat().st_mtime_ns}_{source.stat().st_size}"
    paths = [cache_dir / f"drift_panel_{stamp}_{index}.png" for index in range(4)]
    if all(path.exists() for path in paths):
        return [str(path) for path in paths]
    image = Image.open(source).convert("RGB")
    width, height = image.size
    saved = []
    for index, path in enumerate(paths):
        top = int(round(index * height / 4))
        bottom = int(round((index + 1) * height / 4))
        image.crop((0, top, width, bottom)).save(path)
        saved.append(str(path))
    return saved


def temporal_panel_image(view: dict, choice: str) -> str:
    """返回与下拉项对应的单张漂移图路径。"""
    choices = temporal_panel_choices()
    if choice not in choices:
        allowed = "、".join(choices)
        raise ValueError(f"未知时间风险图：{choice}，可选：{allowed}")
    panels = split_drift_overview_panels(view["image"])
    return panels[choices.index(choice)]


def report_status(demo_dir: Path = DEMO_DIR) -> dict:
    """报告模式是否可启动。"""
    missing = missing_report_files(demo_dir)
    return {"ready": not missing, "missing": missing}

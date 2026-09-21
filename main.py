"""统一入口：读配置、跑全流程、打印指标摘要与基线比对结果。"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config, load_secrets
from src.pipeline import run_pipeline


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SECOM 失效分析系统")
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    p.add_argument("--quick", action="store_true", help="快速模式：小样本、跳过耗时步骤")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["run"]["quick"] = True
    setup_logging(cfg["output"]["log_level"])
    log = logging.getLogger("main")

    load_secrets()  # 基础版无外部 API，可能全 None，属于正常表现
    log.info(
        "配置加载完成 | quick=%s | seed=%s | 模型=%s",
        cfg["run"]["quick"], cfg["random_state"], cfg["model"]["active"],
    )

    result = run_pipeline(cfg)

    # ---- 结果摘要 ----
    sel = result["selection"]
    print("\n" + "=" * 62)
    print(f"模型指标摘要（reference 模型按训练侧 CV {sel['cv_folds']} 折折外期望代价选出）")
    print("=" * 62)
    for name, m in result["metrics"].items():
        auc = f"{m['AUC']:.3f}" if m["AUC"] is not None else "N/A"
        cm = m["CV折外混淆"]
        print(
            f"  {name}\n"
            f"    [训练侧CV] 代价={m['CV折外期望代价']:g} "
            f"(FN={cm['fn']} FP={cm['fp']}) BER={m['CV_BER均值']:.3f}±{m['CV_BER标准差']:.3f}\n"
            f"    [测试集]   BER={m['测试集BER']:.3f} 召回={m['召回率']:.3f} "
            f"F1={m['F1分数']:.3f} AUC={auc} 准确率={m['准确率']:.3f}"
        )
    print(f"\nreference 模型: {sel['reference_model']}（依据 {sel['selection_basis']}）")
    if not sel["criteria_agree"]:
        # 两把尺子给出相反次序时不静默按代价了事：这件事本身是结论
        print(f"  ⚠ 两判据排序不一致：按代价 {sel['ranking_by_cv_cost']} / "
              f"按 CV BER {sel['ranking_by_cv_ber']}（以期望代价为准，差异已落盘）")
    print(f"产物目录: {result['output_dir']}")

    # ---- 重采样区间摘要（validation.resample.enabled=true 时才有）----
    rs = result.get("resample")
    if rs and rs.get("status") == "ok":
        p = rs["protocol"]
        print("\n" + "=" * 62)
        print(f"重采样区间（{p['n_splits']} 个划分，{p['interval']} 口径，"
              f"详见 {result['output_dir']}/resample_metrics.json）")
        print("=" * 62)
        print(f"  区间刻画的是这条 recipe 换一次 80/20 抽样的性能分布；"
              f"上面的单点数字是划分 #{p['artifact_split_index']} 这一次抽样。")
        for name, ivs in rs["intervals"].items():
            cells = []
            for metric in ("测试集BER", "召回率", "期望代价"):
                iv, pct = ivs[metric], rs["reference_percentile"][name][metric]
                if iv is None:
                    continue
                cells.append(f"{metric}={iv['point']:.3f} "
                             f"[{iv['low']:.3f}, {iv['high']:.3f}] (ref 第 {pct:.0f} 百分位)")
            print(f"  {name}: " + " | ".join(cells))
        for pair, ivs in rs["paired_diff"].items():
            iv = ivs["测试集BER"]
            if iv is None:
                continue
            verdict = "区间不含 0" if iv["excludes_zero"] else "区间含 0，差异不显著"
            print(f"  配对差值 BER({pair}): {iv['point']:+.3f} "
                  f"[{iv['low']:+.3f}, {iv['high']:+.3f}] —— {verdict}")
        votes = rs["selection_frequency"]
        print(f"  各划分选出的 reference 模型分布: {votes}")

    stability_result = result.get("stability")
    if stability_result and stability_result.get("status") == "ok":
        summary = stability_result["summary"]
        print("\n训练侧参数候选稳定性（独立于性能指标重采样）：")
        print(f"  三档数量：{summary['counts_by_tier']}")
        print(f"  reference 之外的主档高频候选：{summary['outside_reference_high_frequency']}")
        print(f"  详见 {result['output_dir']}/parameter_candidates.txt")

    # ---- 时间序协议摘要（validation.temporal_holdout.enabled=true 时才有）----
    tp = result.get("temporal")
    if tp and tp.get("status") == "ok":
        p, dr = tp["protocol"], tp["prior_drift"]
        print("\n" + "=" * 62)
        print(f"时间序留出对照（前 {1 - p['test_fraction']:.0%} 训练 / 后 "
              f"{p['test_fraction']:.0%} 测试，详见 {result['output_dir']}/temporal_metrics.json）")
        print("=" * 62)
        print(f"  训练 {p['n_train']} 片(失败 {p['n_train_positive']}) / "
              f"测试 {p['n_test']} 片(失败 {p['n_test_positive']})；"
              f"测试段 {p['test_span'][0]} ~ {p['test_span'][1]}")
        rates = " → ".join(f"{b['failure_rate']:.2%}" for b in dr["bins"])
        print(f"  先验漂移（时间{dr['n_bins']}等分失败率）: {rates}"
              f" | 最大/最小 {dr['max_over_min_ratio']:.1f} 倍")
        cmp_ = tp.get("random_protocol_comparison")
        for name, m in tp["models"].items():
            line = (f"  {name}: BER={m['测试集BER']:.3f} 召回={m['召回率']:.3f} "
                    f"代价={m['期望代价']:g}")
            cell = ((cmp_ or {}).get("models", {}).get(name) or {}).get("测试集BER")
            if cell:
                line += (f" | 随机协议区间 [{cell['interval'][0]:.3f}, "
                         f"{cell['interval'][1]:.3f}]"
                         f"{'内' if cell['within_interval'] else '外'}")
            print(line)
        print(f"  时间序 reference 模型: {tp['reference_model']}")
        if cmp_ is None:
            print(f"  ⚠ 未与随机协议区间对照：{tp['comparison_unavailable_reason']}")
        elif tp["escalation"]["required"]:
            # 升级规则预先写死：恶化到区间之外就不是"一个对照"，是主要风险结论
            print("  ⚠ 升级规则触发，时间序结果须在 README 作为主要风险结论：")
            for reason in tp["escalation"]["reasons"]:
                print(f"      - {reason}")

    # ---- 不平衡策略对比摘要 ----
    imb = result.get("imbalance")
    if imb and imb.get("status") == "ok":
        print("\n" + "=" * 62)
        print("不平衡策略对比（漏检风险控制，详见 outputs/imbalance_comparison.md）")
        print("=" * 62)
        for display, m in imb["models"].items():
            best = m["best_by_cv_cost"]
            r = m["strategies"][best]
            d = r.get("delta_vs_baseline")
            delta = (
                f" | 测试集 vs 基线: 召回{d['召回率提升']:+.3f} 漏检{-d['漏检减少']:+d} "
                f"误报{d['误报增加']:+d} 代价{d['期望代价变化']:+g}"
            ) if d else ""
            print(
                f"  {display}: 最优工作点={best}（按训练侧CV代价 "
                f"{r['cv']['CV期望代价']:g} 在池 {m['selection_pool']} 内选出） "
                f"测试集独立验证: "
                f"FN={r['confusion']['fn']} FP={r['confusion']['fp']} "
                f"代价={r['expected_cost']:g}{delta}"
            )
            for strat, sr in m["strategies"].items():
                # 被挡在池外的原因必须打出来：静默排除与"它本来就没赢"读起来一样
                if sr.get("selection_ineligible_reason"):
                    print(f"      · {strat} 不进选择池：{sr['selection_ineligible_reason']}")
            c = m.get("calibration_contrast")
            if c:
                print(
                    f"      · 校准方法对照 {c['primary_method']} vs "
                    f"{c['contrast_method']}：CV 代价 {c['cv_cost_primary']:g} vs "
                    f"{c['cv_cost_contrast']:g}；对照若入池"
                    f"{'会' if c['would_change_selection'] else '不会'}改变选择")

    # ---- 成本敏感性摘要 ----
    cs = (imb or {}).get("cost_sensitivity") if imb else None
    if cs:
        print("\n" + "=" * 62)
        print(f"成本敏感性（档位 FN:FP = {cs['ratios']}，详见 outputs/{cs['markdown']}）")
        print("=" * 62)
        for display, changed in cs["changes_operating_point"].items():
            print(f"  {display}: 换成本比{'会' if changed else '不会'}选出不同工作点")

    # ---- 消融实验摘要 ----
    ab = result.get("ablation")
    if ab and ab.get("status") == "ok":
        print("\n" + "=" * 62)
        print("消融实验（特征选择 × 不平衡处理，详见 outputs/ablation_comparison.md）")
        print("=" * 62)
        for name, res in ab["models"].items():
            e = res["effects"]
            print(
                f"  {res['display_name']}: 加 class_weight BER"
                f"{e['weighting_on_vote']['BER变化']:+.3f}/漏检"
                f"{e['weighting_on_vote']['漏检变化']:+d} | 换 SHAP 选特征(不加权) BER"
                f"{e['selector_on_none']['BER变化']:+.3f}/漏检"
                f"{e['selector_on_none']['漏检变化']:+d} | 换 SHAP 选特征(加权) BER"
                f"{e['selector_on_weighted']['BER变化']:+.3f}/漏检"
                f"{e['selector_on_weighted']['漏检变化']:+d}"
            )

    drift = result.get("drift")
    if drift and drift.get("status") == "ok":
        training = drift["performance"]["training"]
        print(f"\n制程与良率的时间漂移分析已完成，固定模型仅用最早 {training['n']} 行训练。")
        print(f"详见 {result['output_dir']}/drift_report.md；描述性分析，不是在线监控或工艺归因。")

    comparison = result.get("model_comparison")
    if comparison and comparison.get("status") == "ok":
        selected = comparison["summary"]["selection"]["selected_model"]
        print(f"\n固定模型族配对比较已完成，训练侧选择: {selected}")
        print(f"详见 {result['output_dir']}/model_comparison.md；与旧三模型回归口径分开。")

    # ---- 基线比对 ----
    if result["baseline_ok"] is None:
        print("\n[quick 模式] 子采样运行，不与基线比对。")
        return 0

    print("\n" + "=" * 62)
    print("与基线指标比对")
    print("=" * 62)
    for line in result["baseline_report"]:
        print("  " + line)
    if result["baseline_ok"]:
        print("\n✓ 全部指标落在容差内，当前流程与基线一致。")
        return 0
    print(
        "\n✗ 存在超出容差的指标。常见原因：\n"
        "  1) 互信息估计的随机性导致特征集有出入。可在 config.yaml 设\n"
        "     feature_selection.override_features_path，指向此前运行生成的\n"
        "     outputs/selected_features_*.txt 锁定特征；\n"
        "  2) sklearn/numpy 版本差异；3) 预处理或划分顺序被改动。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

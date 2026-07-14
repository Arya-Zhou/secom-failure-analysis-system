"""统一入口 —— clone 后一条命令跑通全流程。

    python main.py                # 完整流程（含与 notebook 基线的容差比对）
    python main.py --quick        # 小样本/跳过 RFE，用于调试和现场演示
    python main.py --config other.yaml
"""
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

    load_secrets()  # 基础版无外部 API，可能全 None，正常
    log.info(
        "配置加载完成 | quick=%s | seed=%s | 模型=%s",
        cfg["run"]["quick"], cfg["random_state"], cfg["model"]["active"],
    )

    result = run_pipeline(cfg)

    # ---- 结果摘要 ----
    print("\n" + "=" * 62)
    print("模型指标摘要")
    print("=" * 62)
    for name, m in result["metrics"].items():
        auc = f"{m['AUC']:.3f}" if m["AUC"] is not None else "N/A"
        print(
            f"  {name}: 测试BER={m['测试集BER']:.3f} 召回={m['召回率']:.3f} "
            f"F1={m['F1分数']:.3f} AUC={auc} 准确率={m['准确率']:.3f}"
        )
    print(f"\n最佳模型(按测试集 BER): {result['best_model']}")
    print(f"产物目录: {result['output_dir']}")

    # ---- 基线比对 ----
    if result["baseline_ok"] is None:
        print("\n[quick 模式] 子采样运行，不与基线比对。")
        return 0

    print("\n" + "=" * 62)
    print("与 notebook 基线比对 (baseline_metrics.json)")
    print("=" * 62)
    for line in result["baseline_report"]:
        print("  " + line)
    if result["baseline_ok"]:
        print("\n✓ 全部指标落在容差内，重构后行为与 notebook 一致。")
        return 0
    print(
        "\n✗ 存在超出容差的指标。常见原因：\n"
        "  1) 原 notebook 互信息未设种子导致特征集有出入 ——\n"
        "     可在 config.yaml 设 feature_selection.override_features_path\n"
        "     指向 ../secom_results/secom_选择的特征_20260610_172752.txt 锁定特征；\n"
        "  2) sklearn/numpy 版本差异；3) 搬运逻辑偏差（检查预处理与划分顺序）。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

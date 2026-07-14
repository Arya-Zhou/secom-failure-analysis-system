"""可复现性回归测试 —— 兜住"重构后行为不变"。

做法（务实版）：不追求逐位相等，用容差比对。
    重构后跑出的指标 vs baseline_metrics.json 中 notebook 的基线，
    |new - old| < tolerance(config, 默认 0.01) 即视为一致。
超出容差 -> 大概率是数据泄漏、填充/划分顺序变了，正是要抓的 bug。

运行:  cd secom_refactor_20260710 && pytest tests/ -v
说明:  全流程含 RFE（数百次逻辑回归拟合），单次约 1~3 分钟；
       module 级 fixture 保证整个测试会话只跑一次流程。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src.config import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

BASELINE = ROOT / "baseline_metrics.json"
MODEL_NAMES = ["逻辑回归", "随机森林", "岭分类器"]


def load_baseline() -> dict:
    with open(BASELINE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline_result():
    """整个测试模块只跑一次全流程（quick=False，与基线同口径）。"""
    cfg = load_config(ROOT / "config.yaml")
    data_path = (ROOT / cfg["data"]["features_path"]).resolve()
    if not data_path.exists():
        pytest.skip(f"数据文件不存在: {data_path}（见 README 数据准备步骤）")
    return run_pipeline(cfg, quick=False)


@pytest.fixture(scope="module")
def tolerance() -> float:
    cfg = load_config(ROOT / "config.yaml")
    return float(cfg["reproducibility"]["tolerance"])


def test_baseline_file_exists():
    assert BASELINE.exists(), "缺少 baseline_metrics.json（notebook 基线）"


def test_selected_feature_count(pipeline_result):
    cfg = load_config(ROOT / "config.yaml")
    expected = cfg["feature_selection"]["n_features_to_select"]
    assert len(pipeline_result["selected_features"]) == expected


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metrics_match_baseline(pipeline_result, tolerance, model_name):
    """重构后各模型指标应落在 notebook 基线的容差内。"""
    baseline = load_baseline()[model_name]
    actual = pipeline_result["metrics"]
    assert model_name in actual, f"本次运行缺少模型: {model_name}"
    for metric, base_val in baseline.items():
        if base_val is None:
            continue
        new_val = actual[model_name].get(metric)
        assert new_val is not None, f"{model_name}.{metric}: 本次无该指标"
        assert abs(new_val - base_val) < tolerance, (
            f"{model_name}.{metric}: 新={new_val:.4f} 基线={base_val:.4f} "
            f"超出容差 {tolerance}（排查提示见 main.py 输出末尾）"
        )

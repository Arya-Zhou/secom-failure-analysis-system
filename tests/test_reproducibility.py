"""可复现性回归测试：跑一次全流程，指标与基线文件逐项比对，差值小于容差即一致。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src.config import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

MODEL_NAMES = ["逻辑回归", "随机森林", "岭分类器"]

# 严格链路（唯一链路）的锚点。文件名在这里写死而非取自 config，改配置不应让回归保护跟着走空。
BASELINE_FILE = "baseline_metrics_strict.json"


def load_baseline() -> dict:
    with open(ROOT / BASELINE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tolerance() -> float:
    cfg = load_config(ROOT / "config.yaml")
    return float(cfg["reproducibility"]["tolerance"])


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory) -> dict:
    """跑一次全流程（quick=False）；产物写临时目录、SHAP 与 imbalance/ablation 关闭，
    否则会把仓库 outputs/ 的正式产物覆盖成降级状态。"""
    base = load_config(ROOT / "config.yaml")
    data_path = (ROOT / base["data"]["features_path"]).resolve()
    if not data_path.exists():
        pytest.skip(f"数据文件不存在: {data_path}（见 README 数据准备步骤）")

    cfg = copy.deepcopy(base)
    cfg["explain"]["enabled"] = False
    cfg["imbalance"]["comparison"]["enabled"] = False
    cfg["ablation"]["enabled"] = False
    cfg["output"]["results_dir"] = str(tmp_path_factory.mktemp("out_strict"))
    return run_pipeline(cfg, quick=False)


def test_results_dir_is_isolated(pipeline_result):
    """守住上面那条：本模块的产物不得落进仓库 outputs/，否则会覆盖正式产物。"""
    assert Path(pipeline_result["output_dir"]).resolve() != (ROOT / "outputs").resolve(), (
        "回归测试把产物写进了仓库 outputs/")


def test_baseline_file_exists():
    assert (ROOT / BASELINE_FILE).exists(), f"缺少基线指标文件: {BASELINE_FILE}"


def test_config_points_at_the_only_baseline():
    """config 的锚点必须就是本模块写死的那一份，否则 main.py 与测试比的是两份文件。"""
    cfg = load_config(ROOT / "config.yaml")
    assert cfg["reproducibility"]["baseline_path"] == BASELINE_FILE


def test_selected_feature_count(pipeline_result):
    cfg = load_config(ROOT / "config.yaml")
    expected = cfg["feature_selection"]["n_features_to_select"]
    assert len(pipeline_result["selected_features"]) == expected


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_metrics_match_baseline(pipeline_result, tolerance, model_name):
    """各模型指标应落在基线的容差内。"""
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


def test_baseline_comparison_passes(pipeline_result):
    """比对不通过说明链路行为已漂移。"""
    assert pipeline_result["baseline_ok"] is True, (
        "与基线不一致:\n" + "\n".join(pipeline_result["baseline_report"])
    )

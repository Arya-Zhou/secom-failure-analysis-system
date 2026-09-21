"""演示界面专项测试：不启动浏览器，导入阶段不依赖 Gradio。"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.demo_data import (
    BANNER, DEMO_DIR, FROZEN_SAMPLES, GLOBAL_CHOICE, PROJECT_ROOT, QUADRANT_GLOSSARY, SAMPLE_NOTE, missing_report_files,
    prepare_operating_point_features, frozen_sample,
)
from src.demo_predict import (
    aligned_features, format_prediction, frozen_display_result, load_ridge_operating_point,
    predict_frame, predict_frozen_wafer, predict_inputs_ready, predict_mode_enabled,
    startup_predict_bundle, verify_frozen_predictions, wafer_cost,
)
from src.demo_views import (
    candidates_markdown, candidates_view, image_for_sample, overview_markdown, overview_view, sample_image_note,
    parse_candidate_counts, parse_overview_counts,
    parse_resample_interval, parse_ridge_cost_example, parse_temporal_ridge,
    report_status, shap_case_view, temporal_markdown, temporal_panel_choices,
    temporal_panel_image, temporal_view,
)
from src.imbalance import apply_operating_point, load_operating_point

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "outputs" / "imbalance_operating_point_ridge.pkl"
MANIFEST = ROOT / "outputs" / "imbalance_comparison.json"
SRC_APP = ROOT / "src" / "demo_app.py"
NEED_PREDICT = pytest.mark.skipif(
    not predict_mode_enabled(),
    reason="本机预测模式需要原始数据、Ridge 工作点 pkl 与清单",
)


def test_import_demo_modules_without_gradio():
    """普通测试收集不得因为缺 Gradio 失败。"""
    import src.demo_app as demo_app
    import src.demo_data  # noqa: F401
    import src.demo_predict  # noqa: F401
    import src.demo_views  # noqa: F401
    source = SRC_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "gradio" not in imported
    assert "share=False" in source
    assert 'server_name="127.0.0.1"' in source
    assert "analytics_enabled=False" in source
    assert "GRADIO_ANALYTICS_ENABLED" in source
    assert 'if __name__ == "__main__"' in source
    assert "demo.launch" in source.split("def main")[1]
    assert demo_app.INSTALL_HINT.startswith("未安装 Gradio")
    assert "requirements-demo.txt" in demo_app.STACK_HINT


def test_demo_requirements_pin_gradio_stack():
    """演示依赖必须锁住与 Gradio 4.44.1 兼容的 FastAPI/Starlette。"""
    text = (ROOT / "requirements-demo.txt").read_text(encoding="utf-8")
    assert "gradio==4.44.1" in text
    assert "huggingface_hub==0.25.2" in text
    assert "fastapi==0.112.4" in text
    assert "starlette==0.38.6" in text


def test_starlette_compat_and_localhost_proxy(monkeypatch):
    """过新 Starlette 必须拒绝；本机地址必须绕过 HTTP_PROXY。"""
    from src.demo_app import (
        disable_gradio_analytics, ensure_localhost_not_proxied, starlette_incompatible,
    )
    assert starlette_incompatible("0.38.6") is False
    assert starlette_incompatible("0.44.0") is False
    assert starlette_incompatible("0.45.0") is True
    assert starlette_incompatible("1.6.0") is True
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("no_proxy", "")
    ensure_localhost_not_proxied()
    assert "127.0.0.1" in os.environ["NO_PROXY"].split(",")
    assert "localhost" in os.environ["no_proxy"].split(",")
    disable_gradio_analytics()
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_report_mode_builds_views_without_pkl(tmp_path):
    """缺 pkl 时仍可从已跟踪快照构建报告视图。"""
    assert PKL.name not in {path.name for path in DEMO_DIR.iterdir()}
    assert report_status()["ready"] is True
    overview = overview_view()
    candidates = candidates_view()
    temporal = temporal_view()
    shap_case = shap_case_view()
    assert overview["n_wafers"] == 1567
    assert overview["n_fail"] == 104
    assert overview["fail_rate_pct"] == 6.64
    assert overview["interval_ber"] == 0.353
    assert overview["interval_ber_low"] == 0.245
    assert overview["interval_ber_high"] == 0.432
    assert overview["interval_recall"] == 0.571
    assert overview["interval_recall_low"] == 0.381
    assert overview["interval_recall_high"] == 0.767
    assert overview["temporal_recall"] == 0.235
    assert overview["temporal_ber"] == 0.440
    assert overview["cost_fn_from"] == 21
    assert overview["cost_fn_to"] == 7
    assert overview["cost_fp_from"] == 0
    assert overview["cost_fp_to"] == 69
    assert overview["cost_cost_from"] == 210
    assert overview["cost_cost_to"] == 139
    overview_text = overview_markdown(overview)
    candidates_text = candidates_markdown(candidates)
    temporal_text = temporal_markdown(temporal)
    assert "34%" in overview_text
    assert "9 项" in candidates_text
    assert "0.235" in temporal_text
    assert "0.440" in temporal_text
    assert "0.440" in overview_text
    assert "- 晶圆 1567 片" in overview_text
    assert "- F060" in candidates_text
    for text in (overview_text, candidates_text, temporal_text):
        assert BANNER not in text
        assert "不是" not in text
        assert "并非" not in text
    assert candidates["high"] == 9
    assert candidates["exploratory"] == 153
    assert candidates["insufficient"] == 429
    assert len(candidates["high_names"]) == 9
    assert temporal["temporal_n_test"] == 313
    assert Path(shap_case["fn_image"]).exists()
    missing = missing_report_files(tmp_path)
    assert "resample_metrics.json" in missing
    assert report_status(tmp_path)["ready"] is False


def test_view_parsers_match_source_fields():
    """页面数字必须从源字段读取，而不是手写。"""
    from src.demo_data import read_json, read_text
    counts = parse_overview_counts(read_text("drift_report.md"))
    interval = parse_resample_interval(read_json("resample_metrics.json"))
    temporal = parse_temporal_ridge(read_json("temporal_metrics.json"))
    cost = parse_ridge_cost_example(read_text("imbalance_comparison.md"))
    cand = parse_candidate_counts(read_text("parameter_candidates.txt"))
    assert counts["n_wafers"] == 1567
    assert interval["ber"] == 0.353
    assert temporal["recall"] == 0.235
    assert cost["fn_to"] == 7
    assert cand["high"] == 9


def test_prepare_features_drops_empty_column():
    """原始 591 列含全空 F591 时，预处理后应与 590 列工作点对齐。"""
    columns = [f"F{i:03d}" for i in range(1, 592)]
    frame = pd.DataFrame(np.zeros((2, 591)), columns=columns)
    frame["F591"] = np.nan
    expected = [f"F{i:03d}" for i in range(1, 591)]
    prepared = prepare_operating_point_features(frame, expected)
    assert list(prepared.columns) == expected
    with pytest.raises(ValueError, match="特征列与工作点不一致"):
        prepare_operating_point_features(frame, expected + ["F591"])


def test_predict_mode_missing_artifacts(tmp_path, monkeypatch):
    """本机预测模式在缺产物时不得静默启用。"""
    monkeypatch.setattr("src.demo_data.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.demo_predict.PROJECT_ROOT", tmp_path)
    status = predict_inputs_ready(tmp_path)
    assert status["operating_point"] is False
    assert predict_mode_enabled(status) is False
    assert startup_predict_bundle(tmp_path) is None


@NEED_PREDICT
def test_raw_591_columns_rejected_by_operating_point():
    """未删除 F591 的原始列直接预测必须失败。"""
    from src.demo_data import load_raw_features
    op = load_ridge_operating_point()
    X, _y = load_raw_features()
    assert list(X.columns)[-1] == "F591"
    with pytest.raises(ValueError, match="F591"):
        apply_operating_point(op, X.loc[[518]])


@NEED_PREDICT
def test_wrong_load_path_rejected():
    """自行反序列化不得调用 apply_operating_point。"""
    import pickle
    with PKL.open("rb") as handle:
        raw = pickle.load(handle)
    from src.demo_data import load_raw_features
    op = load_ridge_operating_point()
    X, _y = load_raw_features()
    prepared = aligned_features(X, op)
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        apply_operating_point(raw, prepared.loc[[518]])


@NEED_PREDICT
def test_require_manifest_none_rejected():
    """require_manifest=True 且 manifest=None 必须拒绝。"""
    with pytest.raises(ValueError, match="require_manifest"):
        load_operating_point(PKL, manifest=None, require_manifest=True)


@NEED_PREDICT
def test_frozen_quadrants_match_operating_point():
    """四象限冻结样本的预测与分数必须与预存期望一致。"""
    bundle = startup_predict_bundle()
    assert bundle is not None
    verify_frozen_predictions(bundle["op"], bundle["X"], bundle["y"])
    cases = {row["case"] for row in FROZEN_SAMPLES}
    assert cases == {"TN", "FP", "FN", "TP"}
    for row in FROZEN_SAMPLES:
        result = predict_frozen_wafer(bundle["op"], bundle["X"], bundle["y"], row["wafer_id"])
        assert result["y_pred"] == row["y_pred"]
        assert result["y_true"] == row["y_true"]
        assert result["score"] == pytest.approx(row["score"], abs=1e-9)
        text = format_prediction(result)
        assert "独立成本策略链" in text
        assert "主分析四方法投票" not in text
        assert "不是" not in text
        assert str(row["wafer_id"]) in text
    assert frozen_sample(518)["case"] == "FN"
    assert wafer_cost(1, 0) == 10
    assert wafer_cost(0, 1) == 1


@NEED_PREDICT
def test_aligned_predict_matches_loader():
    """预处理后的 590 列可以送入工作点。"""
    bundle = startup_predict_bundle()
    prepared = aligned_features(bundle["X"], bundle["op"])
    assert "F591" not in prepared.columns
    assert prepared.shape[1] == 590
    preds, scores, space = predict_frame(bundle["op"], prepared.loc[[518]])
    assert int(preds[0]) == 0
    assert space == "decision_margin"
    assert float(scores[0]) == pytest.approx(-0.19777629614980508, abs=1e-9)



def test_sample_image_follows_case():
    """样本切换只显示对应 SHAP 图。"""
    shap = shap_case_view()
    assert image_for_sample("FN 518", shap).endswith("shap_fn_wafer.png")
    assert image_for_sample("TP 96", shap).endswith("shap_tp_wafer.png")
    assert image_for_sample("FP 689", shap) is None
    assert image_for_sample("TN 1433", shap) is None
    assert image_for_sample(GLOBAL_CHOICE, shap).endswith("shap_global_ridge.png")
    frozen = frozen_display_result(518)
    assert frozen["case"] == "FN"
    assert "不是" not in format_prediction(frozen)
    assert "用于复核演示" not in SAMPLE_NOTE
    assert "用于复核演示" not in format_prediction(frozen)
    assert "演示样本" not in SRC_APP.read_text(encoding="utf-8")
    for token in ("**T**", "**F**", "**P**", "**N**", "**TP**", "**FN**", "**FP**", "**TN**"):
        assert token in QUADRANT_GLOSSARY
    assert "空白" in sample_image_note("FP 689")
    assert "空白" in sample_image_note("TN 1433")
    assert "下图" in sample_image_note("FN 518")
    assert "下图" in sample_image_note("TP 96")
    assert "不是" not in sample_image_note("FP 689")


def test_temporal_panels_split_four_charts():
    """时间风险图按四张切换，不整页叠放。"""
    from PIL import Image
    view = temporal_view()
    choices = temporal_panel_choices()
    assert choices == ["失效比例随时间", "窗口性能", "特征 PSI", "缺失率轨迹"]
    first = Image.open(temporal_panel_image(view, choices[0]))
    last = Image.open(temporal_panel_image(view, choices[-1]))
    assert first.size[0] == last.size[0]
    assert first.size[1] == last.size[1]
    assert first.size[1] * 4 == 2100


def test_demo_api_info_accepts_image_outputs():
    """首页 API 文档生成不得因为 Image 输出崩溃。"""
    pytest.importorskip("gradio")
    from src.demo_app import build_interface
    demo = build_interface(None)
    info = demo.get_api_info()
    assert "named_endpoints" in info

"""时间漂移的数值口径、早训晚测隔离、入口和产物故障注入测试。"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_io import load_secom
from src.drift import (
    DRIFT_FILE, DRIFT_MARKDOWN, DRIFT_PLOT, DRIFT_TITLE, benjamini_hochberg,
    build_drift_report, feature_distribution, prior_analysis, resolve_drift,
    rolling_windows, window_performance,
)
from src.drift_reporting import plot_drift, render_drift_report, run_drift_analysis, validate_drift_artifacts
from src.pipeline import run_pipeline
from src.validation import artifact_split, prior_drift_table


@pytest.fixture(scope="module")
def sample():
    cfg = load_config(ROOT / "config.yaml")
    cfg["drift"].update(window_size=30, step_size=15, top_features=3)
    cfg["feature_selection"]["n_features_to_select"] = 3
    generator = np.random.RandomState(42)
    labels = pd.Series(np.tile([0, 0, 0, 1, 0], 32), index=np.arange(2000, 2160))
    features = pd.DataFrame(generator.normal(size=(160, 6)), index=labels.index,
                            columns=[f"F{number:03d}" for number in range(1, 7)])
    features["F001"] += 3 * labels
    features["F007"] = np.nan
    features.loc[labels.index[32:], "F007"] = 10.0
    features["F008"] = 1.0
    features["F009"] = generator.normal(size=len(labels))
    features.loc[labels.index[::7], "F009"] = np.nan
    features["F010"] = np.nan
    timestamps = pd.Series(pd.date_range("2008-07-19", periods=len(labels), freq="h"), index=labels.index)
    timestamps.iloc[31] = timestamps.iloc[32]
    return cfg, features, labels, timestamps.sample(frac=1.0, random_state=17)


@pytest.fixture(scope="module")
def fitted(sample, tmp_path_factory):
    cfg, features, labels, timestamps = sample
    output = tmp_path_factory.mktemp("drift")
    result = run_drift_analysis(cfg, features, labels, timestamps, output)
    return output, result


@pytest.mark.parametrize("key,value", [
    ("enabled", "false"), ("model", "random_forest"), ("window_size", 1),
    ("window_size", True), ("window_size", 2.5), ("step_size", 0), ("step_size", 201),
    ("psi_bins", 1), ("psi_bins", True), ("psi_pseudocount", 0),
    ("psi_pseudocount", True), ("psi_pseudocount", float("nan")),
    ("psi_pseudocount", float("inf")), ("ks_min_samples", 1), ("top_features", 0),
])
def test_invalid_config_is_rejected(key, value):
    with pytest.raises(ValueError, match="drift"):
        resolve_drift({"drift": {key: value}})


def test_defaults_and_non_mapping_config():
    assert resolve_drift({})["enabled"] is False
    assert resolve_drift(load_config(ROOT / "config.yaml"))["enabled"] is True
    with pytest.raises(ValueError, match="drift"):
        resolve_drift({"drift": None})


def test_psi_counts_missing_bin_tails_and_pseudocount_have_known_values():
    row = feature_distribution([0, 1, np.nan, np.nan], [0, 0, 1, 1], psi_bins=2)
    assert row["reference_edges"] == [0.0, 0.5, 1.0]
    assert row["reference_counts"] == [0, 1, 1, 0, 2]
    assert row["current_counts"] == [0, 2, 2, 0, 0]
    expected = 2 * ((2.5 - 1.5) / 6.5) * np.log(2.5 / 1.5) + ((0.5 - 2.5) / 6.5) * np.log(0.5 / 2.5)
    assert row["psi"] == pytest.approx(expected)
    assert row["missing_rate_delta"] == -0.5
    assert row["ks_statistic"] == 0.0


@pytest.mark.parametrize("value", [-5.0, 5.0])
def test_constant_reference_detects_both_directions(value):
    row = feature_distribution(np.zeros(20), np.full(20, value))
    assert row["psi_basis"] == "constant_with_tails_and_missing"
    assert row["psi"] > 0
    assert row["ks_statistic"] == 1.0
    assert sum(row["reference_counts"]) == sum(row["current_counts"]) == 20


@pytest.mark.parametrize("values", [[1.0] * 10, [np.nan] * 10, list(range(10))])
def test_identical_distributions_have_zero_psi(values):
    row = feature_distribution(values, values)
    assert row["psi"] == 0.0
    assert row["ks_statistic"] in (0.0, None)


def test_reference_edges_do_not_depend_on_current_values():
    regular = feature_distribution(np.arange(20), np.arange(20))
    shifted = feature_distribution(np.arange(20), np.arange(20) * 1000)
    assert regular["reference_edges"] == shifted["reference_edges"]
    assert regular["reference_counts"] == shifted["reference_counts"]
    assert shifted["current_counts"][-2] == 19


def test_unobserved_reference_is_missingness_only_and_ks_requires_samples():
    row = feature_distribution([np.nan] * 5, [1, 2, 3, 4, 5])
    assert row["psi_basis"] == "missingness_only" and row["psi"] > 0
    assert row["reference_edges"] == [] and row["ks_p_value"] is None
    row = feature_distribution([0, 1, 2], [5, np.nan, np.nan])
    assert row["ks_status"] == "insufficient_observations"
    assert row["ks_statistic"] is None and row["ks_q_value"] is None


@pytest.mark.parametrize("reference,current", [([], [1]), ([1], []), ([np.inf], [1]), ([[1]], [1])])
def test_invalid_distributions_are_rejected(reference, current):
    with pytest.raises(ValueError, match="PSI/KS"):
        feature_distribution(reference, current)


def test_bh_is_monotonic_in_rank_and_restores_original_order():
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.8]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.8])
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0, 1]) == [0, 1]
    for values in ([np.nan], [np.inf], [-0.1], [1.1]):
        with pytest.raises(ValueError, match="BH"):
            benjamini_hochberg(values)


def test_sliding_windows_cover_tail_without_duplicates():
    windows = rolling_windows(pd.Index(range(11)), 4, 3)
    assert [indices.tolist() for indices in windows] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [7, 8, 9, 10]]
    assert rolling_windows(range(3), 5, 2)[0].tolist() == [0, 1, 2]
    assert rolling_windows([], 5, 2) == []
    assert len(rolling_windows(range(10), 4, 3)) == 3
    with pytest.raises(ValueError, match="滑窗"):
        rolling_windows(range(11), 3, 4)


def test_binary_window_metrics_match_hand_calculation():
    result = window_performance([0, 0, 1, 1], [0, 1, 0, 1], [-1, 0.4, -0.2, 0.8], 10, 1)
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["ber"] == result["recall"] == result["specificity"] == 0.5
    assert result["auc"] == 0.75 and result["cost_per_wafer"] == 2.75


@pytest.mark.parametrize("truth,predicted,missing,defined,cost", [
    ([0, 0, 0], [0, 1, 0], "recall", "specificity", 1 / 3),
    ([1, 1, 1], [1, 0, 1], "specificity", "recall", 10 / 3),
])
def test_single_class_windows_do_not_invent_ber_or_recall(truth, predicted, missing, defined, cost):
    result = window_performance(truth, predicted, predicted, 10, 1)
    assert result["status"] == "single_class"
    assert result["ber"] is None and result["auc"] is None and result[missing] is None
    assert result[defined] == pytest.approx(2 / 3)
    assert result["cost_per_wafer"] == pytest.approx(cost)


def test_prior_uses_shared_stable_timestamp_order_and_records_zero_rate_bins(sample):
    cfg, _features, labels, timestamps = sample
    result = prior_analysis(timestamps, labels, 5, 30, 15)
    assert result["equal_count"] == prior_drift_table(timestamps, labels, 5)
    assert sum(row["failures"] for row in result["monthly"]) == labels.sum()
    assert result["rolling"][-1]["end"] == str(timestamps.max())
    labels = labels.copy()
    labels.iloc[32:64] = 0
    assert 2 in prior_analysis(timestamps, labels, 5, 30, 15)["zero_failure_bins"]
    with pytest.raises(ValueError, match="等分窗口数"):
        prior_analysis(timestamps, labels, len(labels) + 1, 30, 15)


def test_real_prior_reproduces_documented_failure_counts_and_months():
    cfg = load_config(ROOT / "config.yaml")
    if not (ROOT / cfg["data"]["features_path"]).exists():
        pytest.skip("SECOM 原始数据未安装")
    _features, labels, timestamps = load_secom(
        ROOT / cfg["data"]["features_path"], ROOT / cfg["data"]["labels_path"], cfg["data"]["timestamp_format"])
    result = prior_analysis(timestamps, labels, 5, 200, 100)
    equal_count = result["equal_count"]
    assert equal_count["n"] == 1567 and equal_count["failures"] == 104
    assert [row["n"] for row in equal_count["bins"]] == [314, 314, 313, 313, 313]
    assert [row["failures"] for row in equal_count["bins"]] == [44, 21, 11, 11, 17]
    assert equal_count["max_over_min_ratio"] == pytest.approx(4, abs=0.02)
    assert equal_count["first_bin_share_of_failures"] == 44 / 104
    assert [row["n"] for row in result["monthly"]] == [63, 555, 590, 359]
    assert [row["failures"] for row in result["monthly"]] == [14, 51, 17, 22]


def test_raw_universe_bh_family_and_strict_boundary_are_auditable(sample, fitted):
    _cfg, features, labels, timestamps = sample
    _output, result = fitted
    performance = result["performance"]
    assert result["feature_drift"]["feature_count"] == features.shape[1]
    assert performance["nominal_train_indices"] == labels.index[:32].tolist()
    assert performance["train_indices"] == labels.index[:31].tolist()
    assert performance["excluded_boundary_indices"] == [labels.index[31]]
    assert timestamps.loc[performance["train_indices"]].max() < timestamps.loc[performance["evaluation_indices"]].min()
    assert set(performance["train_indices"]).isdisjoint(performance["evaluation_indices"])
    assert set(performance["dropped_training_empty_features"]) == {"F007", "F010"}
    assert not set(performance["selected_features"]).intersection({"F007", "F010"})
    assert not performance["training_window_evaluated"] and not performance["future_used_for_selection"]
    assert [row["window"] for row in performance["equal_count"]] == [2, 3, 4, 5]
    assert sum(row["n"] for row in performance["equal_count"]) == len(performance["evaluation_indices"])
    pairs = [row for window in result["feature_drift"]["windows"] for row in window["features"].values()
             if row["ks_p_value"] is not None]
    assert result["feature_drift"]["ks_valid_tests"] == len(pairs)
    assert [row["ks_q_value"] for row in pairs] == benjamini_hochberg([row["ks_p_value"] for row in pairs])


def test_future_labels_cannot_change_fitted_state_features_or_predictions(sample, fitted):
    cfg, features, labels, timestamps = sample
    _output, original = fitted
    changed_labels = labels.copy()
    changed_labels.iloc[31:] = 1 - changed_labels.iloc[31:]
    changed = build_drift_report(cfg, features, changed_labels, timestamps)
    for key in ("selected_features", "fitted_state", "train_indices", "predictions", "scores"):
        assert changed["performance"][key] == original["performance"][key]
    assert changed["performance"]["overall"]["metrics"] != original["performance"]["overall"]["metrics"]


def test_future_features_do_not_enter_imputation_selection_or_model_fit(sample, fitted, monkeypatch):
    cfg, features, labels, timestamps = sample
    changed_features = features.copy()
    changed_features.iloc[31:, :6] += 1e6
    observed = []
    imputer_fit, model_fit = SimpleImputer.fit, RidgeClassifier.fit

    def trace_imputer(estimator, values, *args, **kwargs):
        observed.append(values.index.tolist())
        return imputer_fit(estimator, values, *args, **kwargs)

    def trace_model(estimator, values, *args, **kwargs):
        observed.append(values.index.tolist())
        return model_fit(estimator, values, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", trace_imputer)
    monkeypatch.setattr(RidgeClassifier, "fit", trace_model)
    changed = build_drift_report(cfg, changed_features, labels, timestamps)
    assert observed == [labels.index[:31].tolist()] * 2
    for key in ("selected_features", "fitted_state", "available_features"):
        assert changed["performance"][key] == fitted[1]["performance"][key]
    assert changed["performance"]["scores"] != fitted[1]["performance"]["scores"]


@pytest.mark.parametrize("fault", ["timestamp_missing", "timestamp_index", "duplicate_index", "label",
                                   "feature_infinite", "timestamp_feature", "empty_training", "one_class_training"])
def test_bad_inputs_fail_without_future_fallback(sample, fault):
    cfg, features, labels, timestamps = copy.deepcopy(sample)
    if fault == "timestamp_missing":
        timestamps.iloc[0] = pd.NaT
    elif fault == "timestamp_index":
        timestamps = timestamps.iloc[1:]
    elif fault == "duplicate_index":
        features.index = [2000] * len(features)
        labels.index = features.index
    elif fault == "label":
        labels.iloc[0] = np.nan
    elif fault == "feature_infinite":
        features.iloc[0, 0] = np.inf
    elif fault == "timestamp_feature":
        features["timestamp"] = np.arange(len(features))
    elif fault == "empty_training":
        features.iloc[:32] = np.nan
    else:
        labels.iloc[:32] = 0
    with pytest.raises(ValueError):
        build_drift_report(cfg, features, labels, timestamps)


@pytest.mark.parametrize("fault", ["quick", "override", "strategy", "cost", "windows"])
def test_analysis_rejects_incompatible_recipes(sample, fault):
    cfg, features, labels, timestamps = copy.deepcopy(sample)
    if fault == "quick":
        cfg["run"]["quick"] = True
    elif fault == "override":
        cfg["feature_selection"]["override_features_path"] = "future_features.txt"
    elif fault == "strategy":
        cfg["imbalance"]["strategy"] = "none"
    elif fault == "cost":
        cfg["costs"]["fn"] = float("inf")
    else:
        cfg["validation"]["temporal_holdout"]["prior_drift_bins"] = len(labels) + 1
    with pytest.raises(ValueError, match="漂移"):
        build_drift_report(cfg, features, labels, timestamps)


def test_artifacts_recompute_exactly_and_are_finite_utf8(sample, fitted):
    output, result = fitted
    assert validate_drift_artifacts(*sample, output) == result
    assert (output / DRIFT_MARKDOWN).read_text(encoding="utf-8") == render_drift_report(result)
    assert result["title"] == DRIFT_TITLE
    encoded = (output / DRIFT_FILE).read_bytes()
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert b"NaN" not in encoded and b"Infinity" not in encoded


def test_report_labels_missingness_only_psi_separately(fitted):
    markdown = render_drift_report(fitted[1])
    assert "PSI 依据" in markdown
    assert "| F007 | 仅缺失率 |" in markdown


@pytest.mark.parametrize("auc,display,below_baseline", [(0.25, "0.2500", True),
                                                       (0.75, "0.7500", False), (None, "N/A", False)])
def test_performance_report_shows_auc_and_descriptive_chance_baseline(fitted, auc, display, below_baseline):
    result = copy.deepcopy(fitted[1])
    for row in result["performance"]["equal_count"]:
        row["metrics"]["auc"] = 0.75
    result["performance"]["equal_count"][0]["metrics"]["auc"] = auc
    original = copy.deepcopy(result)
    markdown = render_drift_report(result)
    table = markdown.split("| 评估窗 |", 1)[1]
    assert "| AUC | BER |" in table
    first_row = next(line for line in table.splitlines() if line.startswith("| 2 |"))
    assert first_row.split("|")[5].strip() == display
    overall_row = next(line for line in table.splitlines() if line.startswith("| all_future |"))
    assert overall_row.split("|")[5].strip() == f"{result['performance']['overall']['metrics']['auc']:.4f}"
    assert "AUC 基线为 0.5" in markdown and "不是与随机基线的显著性检验" in markdown
    assert ("AUC 点估计低于随机排序基线" in markdown) == below_baseline
    assert result == original


@pytest.fixture
def ranked_display(fitted):
    result = copy.deepcopy(fitted[1])
    result["analysis_plan"]["settings"]["top_features"] = 1
    result["feature_drift"]["feature_count"] += 1
    for window in result["feature_drift"]["windows"]:
        for row in window["features"].values():
            row["psi"] = 1.0
        window["features"]["F002"]["psi"] = 10.0
        window["features"]["F007"]["psi"] = 100.0
        window["features"]["F011"] = copy.deepcopy(window["features"]["F007"])
    return result


def test_report_separates_observed_rankings_and_groups_identical_missingness(ranked_display):
    markdown = render_drift_report(ranked_display)
    observed, missing = markdown.split("### 全缺失参照：仅缺失率", 1)
    assert "| F002 |" in observed and "| F007 |" not in observed
    assert "| F007 | 仅缺失率 | 2 | 100.0000 |" in missing
    assert "| F011 |" not in missing
    assert "3 列" in missing and "2 组" in missing
    assert "PSI 仍含尾部和缺失桶" in observed


def test_missingness_groups_keep_distinct_earlier_windows(ranked_display):
    ranked_display["analysis_plan"]["settings"]["top_features"] = 3
    first_row = ranked_display["feature_drift"]["windows"][0]["features"]["F011"]
    first_row["current_missing_rate"] = 0.5
    first_row["missing_rate_delta"] = -0.5
    markdown = render_drift_report(ranked_display).split("### 全缺失参照：仅缺失率", 1)[1]
    assert "3 列" in markdown and "3 组" in markdown
    assert "| F007 | 仅缺失率 | 1 |" in markdown
    assert "| F011 | 仅缺失率 | 1 |" in markdown


def test_plot_separates_observed_psi_and_deduplicates_missingness(ranked_display, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    original = copy.deepcopy(ranked_display)
    plot_drift(ranked_display, tmp_path / DRIFT_PLOT)
    axes = figures[0].axes
    assert [label.get_text() for label in axes[2].get_yticklabels()] == ["F002"]
    np.testing.assert_array_equal(axes[2].images[0].get_array(), [[10.0] * 4])
    assert len(axes[3].lines) == 1 and axes[3].lines[0].get_label() == "F007 (2 features)"
    np.testing.assert_array_equal(axes[3].lines[0].get_ydata(), [100.0, 0.0, 0.0, 0.0, 0.0])
    performance_lines = {line.get_label(): line for line in axes[1].lines}
    np.testing.assert_array_equal(performance_lines["AUC"].get_ydata(),
                                  [row["metrics"]["auc"] for row in ranked_display["performance"]["equal_count"]])
    np.testing.assert_array_equal(performance_lines["AUC/BER chance baseline"].get_ydata(), [0.5, 0.5])
    assert ranked_display == original


@pytest.mark.parametrize("missingness_only", [False, True])
def test_plot_and_report_handle_absent_reference_group(fitted, tmp_path, monkeypatch, missingness_only):
    from matplotlib.figure import Figure

    result = copy.deepcopy(fitted[1])
    for window in result["feature_drift"]["windows"]:
        window["features"] = {name: row for name, row in window["features"].items()
                              if (row["psi_basis"] == "missingness_only") == missingness_only}
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    plot_drift(result, tmp_path / DRIFT_PLOT)
    markdown = render_drift_report(result)
    if missingness_only:
        assert not figures[0].axes[2].images
        assert "No observed-reference features" in [text.get_text() for text in figures[0].axes[2].texts]
        assert "无有观测参照特征" in markdown
    else:
        assert not figures[0].axes[3].lines
        assert "No fully-missing reference features" in [text.get_text() for text in figures[0].axes[3].texts]
        assert "无全缺失参照特征" in markdown


@pytest.mark.parametrize("fault", ["prior", "psi", "q", "predictions", "fitted_state", "training_index",
                                   "protocol", "source", "markdown", "image", "image_and_hash"])
def test_artifact_tampering_is_rejected(sample, fitted, tmp_path, fault):
    output, _result = fitted
    shutil.copytree(output, tmp_path, dirs_exist_ok=True)
    report_path = tmp_path / DRIFT_FILE
    result = json.loads(report_path.read_text(encoding="utf-8"))
    if fault == "prior":
        result["prior_drift"]["equal_count"]["failures"] += 1
    elif fault in ("psi", "q"):
        key = "psi" if fault == "psi" else "ks_q_value"
        result["feature_drift"]["windows"][0]["features"]["F001"][key] = 123.0
    elif fault == "predictions":
        result["performance"]["predictions"][0] ^= 1
    elif fault == "fitted_state":
        result["performance"]["fitted_state"]["imputer_statistics"][0] += 1
    elif fault == "training_index":
        result["performance"]["train_indices"][0] = result["performance"]["evaluation_indices"][0]
    elif fault == "protocol":
        result["analysis_plan"]["timestamp_is_feature"] = True
    elif fault == "source":
        result["fingerprint"]["source_sha256"]["drift.py"] = "0" * 64
    elif fault == "markdown":
        (tmp_path / DRIFT_MARKDOWN).write_text("伪造结论", encoding="utf-8")
    else:
        (tmp_path / DRIFT_PLOT).write_bytes(b"altered image")
        if fault == "image_and_hash":
            result["plot_sha256"] = hashlib.sha256(b"altered image").hexdigest()
    report_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="漂移"):
        validate_drift_artifacts(*sample, tmp_path)


@pytest.mark.parametrize("filename", [DRIFT_FILE, DRIFT_MARKDOWN, DRIFT_PLOT])
def test_incomplete_artifact_set_cannot_pass(sample, fitted, tmp_path, filename):
    shutil.copytree(fitted[0], tmp_path, dirs_exist_ok=True)
    (tmp_path / filename).unlink()
    with pytest.raises(ValueError, match="缺失"):
        validate_drift_artifacts(*sample, tmp_path)


def test_failure_overwrites_old_success_status(sample, fitted, tmp_path, monkeypatch):
    shutil.copytree(fitted[0], tmp_path, dirs_exist_ok=True)

    def fail(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr("src.drift_reporting.build_drift_report", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_drift_analysis(*sample, tmp_path)
    assert json.loads((tmp_path / DRIFT_FILE).read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(ValueError, match="status"):
        validate_drift_artifacts(*sample, tmp_path)


@pytest.mark.parametrize("enabled,quick", [(True, False), (False, False), (True, True)])
def test_pipeline_dispatch_uses_raw_data_and_honors_quick_override(sample, tmp_path, monkeypatch, enabled, quick):
    cfg, features, labels, timestamps = copy.deepcopy(sample)
    cfg["drift"]["enabled"] = enabled
    cfg["run"].update(quick=not quick, quick_sample_size=len(labels) + 1)
    cfg["validation"]["temporal_holdout"]["enabled"] = False
    cfg["output"]["results_dir"] = str(tmp_path)
    train, test = artifact_split(labels, 0.2, 42)
    reference = {"metrics": {}, "fitted": {"岭分类器": None}, "features": ["F001"],
                 "selection": {"reference_model": "岭分类器", "criteria_agree": True}, "folds": 2,
                 "X_train": features.loc[train, ["F001"]], "X_test": features.loc[test, ["F001"]],
                 "y_train": labels.loc[train], "y_test": labels.loc[test]}
    monkeypatch.setattr("src.pipeline.load_secom", lambda *args: (features, labels, timestamps))
    monkeypatch.setattr("src.pipeline._fit_and_evaluate_split", lambda *args: reference)
    monkeypatch.setattr("src.pipeline._run_explain_stage", lambda *args: None)
    monkeypatch.setattr("src.pipeline.run_imbalance_comparison", lambda *args: None)
    monkeypatch.setattr("src.pipeline.run_ablation", lambda *args: None)
    monkeypatch.setattr("src.pipeline.compare_with_baseline", lambda *args: (True, []))
    calls = []

    def capture(config, raw_features, raw_labels, raw_timestamps, output_dir):
        pd.testing.assert_frame_equal(raw_features, features)
        pd.testing.assert_series_equal(raw_labels, labels)
        pd.testing.assert_series_equal(raw_timestamps, timestamps)
        assert config["run"]["quick"] is False
        assert output_dir == tmp_path
        calls.append(True)
        return {"status": "ok"}

    monkeypatch.setattr("src.pipeline.run_drift_analysis", capture)
    result = run_pipeline(cfg, quick=quick)
    assert len(calls) == int(enabled and not quick)
    assert result["drift"] == ({"status": "ok"} if calls else None)

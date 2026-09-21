"""完整稳定性分析的行为、复现、训练侧隔离及产物故障注入测试。"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_pipeline
from src.stability import (
    fit_stability_split, resolve_stability, resolve_stability_analysis, write_stability_protocol,
)
from src.stability_analysis import (
    CANDIDATES_FILE, CANDIDATES_TEXT_FILE, STABILITY_DETAIL_FILE, STABILITY_PLAN_FILE,
    run_stability_analysis, validate_stability_artifacts,
)
from src.stability_metrics import (
    METHOD_RANKS, correlation_groups, direction_evidence, pairwise_stability, summarize_stability,
)
from src.validation import artifact_split


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    cfg = load_config(ROOT / "config.yaml")
    cfg["stability"].update(n_resamples=4, n_jobs=1)
    cfg["stability"]["analysis"]["convergence_step"] = 2
    cfg["feature_selection"]["n_features_to_select"] = 3
    generator = np.random.RandomState(42)
    labels = pd.Series(np.tile([0, 0, 0, 0, 1], 40), index=np.arange(2000, 2200))
    features = pd.DataFrame(generator.normal(size=(200, 6)), index=labels.index,
                            columns=[f"F{number:03d}" for number in range(1, 7)])
    features["F001"] = 8 * labels + generator.normal(scale=0.2, size=len(labels))
    features["F002"] = -features["F001"] + generator.normal(scale=0.1, size=len(labels))
    features["F007"], features["F008"] = np.nan, 1.0
    reference_train, reference_test = artifact_split(labels, 0.2, 42)
    out_dir = tmp_path_factory.mktemp("stability")
    result = run_stability_analysis(cfg, features, labels, reference_train, reference_test, out_dir)
    return cfg, features, labels, reference_train, reference_test, out_dir, result


@pytest.mark.parametrize("key,value", [
    ("direction_method", "shap_mean_sign"), ("direction_class_weight", None),
    ("direction_alpha", 0), ("direction_alpha", float("nan")),
    ("direction_alpha", float("inf")), ("direction_alpha", True),
    ("convergence_step", 0), ("convergence_step", 1.5), ("convergence_step", True),
])
def test_invalid_analysis_recipe_fails_before_fitting(key, value):
    cfg = load_config(ROOT / "config.yaml")
    cfg["stability"]["analysis"][key] = value
    with pytest.raises(ValueError, match="stability.analysis"):
        resolve_stability_analysis(cfg)


@pytest.mark.parametrize("key", ["direction_method", "direction_alpha", "direction_class_weight", "convergence_step"])
def test_analysis_recipe_fields_are_required(key):
    cfg = load_config(ROOT / "config.yaml")
    del cfg["stability"]["analysis"][key]
    with pytest.raises(ValueError, match="stability.analysis"):
        resolve_stability_analysis(cfg)


def test_missing_analysis_recipe_is_rejected():
    cfg = load_config(ROOT / "config.yaml")
    del cfg["stability"]["analysis"]
    with pytest.raises(ValueError, match="stability.analysis"):
        resolve_stability_analysis(cfg)


@pytest.mark.parametrize("values,positive,negative,zero,consistency,sign", [
    ([], 0, 0, 0, None, None), ([0.0, 0.0], 0, 0, 2, None, None),
    ([1, 0, 0, 0], 1, 0, 3, 0.25, 1), ([1, -2], 1, 1, 0, 0.5, 0),
    ([-1, -2, -3, 1], 1, 3, 0, 0.75, -1), ([1, 2], 2, 0, 0, 1.0, 1),
])
def test_direction_denominator_includes_selected_zero_coefficients(values, positive, negative, zero, consistency, sign):
    assert direction_evidence(values) == {
        "positive": positive, "negative": negative, "zero": zero,
        "evidence_count": positive + negative, "consistency": consistency, "majority_sign": sign,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_direction_is_not_silently_missing(value):
    with pytest.raises(ValueError, match="有限"):
        direction_evidence([value])


def test_absolute_correlation_uses_components_not_cliques():
    generator = np.random.RandomState(7)
    center, noise = generator.normal(size=(2, 2000))
    features = pd.DataFrame({"first": center + 0.35 * noise, "middle": center,
                             "last": center - 0.35 * noise, "opposite": -center,
                             "constant": 1.0, "missing": np.nan})
    correlations = features.corr(method="spearman")
    assert correlations.loc["first", "middle"] > 0.9
    assert correlations.loc["middle", "last"] > 0.9
    assert correlations.loc["first", "last"] < 0.9
    assert correlation_groups(features, 0.9) == [
        {"group": "G001", "members": ["first", "middle", "last", "opposite"]},
        {"group": "G002", "members": ["constant"]},
        {"group": "G003", "members": ["missing"]},
    ]


def _record(split_no, order, coefficients):
    ranks = {feature: {**dict.fromkeys(METHOD_RANKS, float(rank)),
                       "平均排名": float(rank), "综合排名": float(rank)}
             for rank, feature in enumerate(order, 1)}
    return {"split": split_no, "selected_features": order[:2], "ranks": ranks,
            "nonconstant_features": order, "coefficients": coefficients}


def _controlled_summary():
    cfg = load_config(ROOT / "config.yaml")
    cfg["stability"]["n_resamples"] = 4
    cfg["stability"]["analysis"]["convergence_step"] = 2
    cfg["feature_selection"]["n_features_to_select"] = 2
    names = ["F001", "F002", "F003", "F004"]
    records = [
        _record(0, ["F002", "F001", "F003", "F004"], {"F002": 1.0, "F001": 1.0}),
        _record(1, ["F002", "F001", "F003", "F004"], {"F002": 2.0, "F001": -1.0}),
        _record(2, ["F002", "F003", "F001", "F004"], {"F002": 1.0, "F003": 0.0}),
        _record(3, ["F002", "F003", "F001", "F004"], {"F002": 2.0, "F003": 0.0}),
    ]
    groups = [{"group": "G001", "members": ["F001", "F003"]},
              {"group": "G002", "members": ["F002", "F004"]}]
    summary = summarize_stability(records, names, ["F001", "F004"], groups,
                                  resolve_stability(cfg), resolve_stability_analysis(cfg))
    return summary, records


def test_candidate_tiers_outside_reference_and_switching_groups():
    summary, _records = _controlled_summary()
    assert summary["counts_by_tier"] == {"高稳定候选": 1, "探索性候选": 1, "当前证据不足": 2}
    assert summary["outside_reference_high_frequency"] == ["F002"]
    candidates = {row["feature"]: row for row in summary["candidates"]}
    assert candidates["F001"]["direction"]["consistency"] == 0.5
    assert candidates["F003"]["selection_frequency"] == 0.5
    assert candidates["F003"]["direction"]["consistency"] is None
    assert candidates["F004"]["selection_frequency"] == 0.0
    assert summary["correlation_groups"][0]["selection_frequency"] == 1.0
    for entry in summary["frequency_threshold_sensitivity"].values():
        assert entry["high_stability_features"] == ["F002"]
        assert entry["outside_reference_high_frequency"] == ["F002"]
    checkpoints = summary["convergence"]["checkpoints"]
    assert [row["n_resamples"] for row in checkpoints] == [2, 4]
    assert checkpoints[-1]["frequency_threshold_sensitivity"] == summary["frequency_threshold_sensitivity"]
    assert checkpoints[-1]["max_frequency_change"] == 0.5
    assert summary["convergence"]["best_n_resamples"] is None


def test_group_frequency_does_not_double_count_members():
    summary, records = _controlled_summary()
    cfg = load_config(ROOT / "config.yaml")
    cfg["stability"]["n_resamples"] = 4
    cfg["feature_selection"]["n_features_to_select"] = 2
    names = [row["feature"] for row in summary["candidates"]]
    groups = [{"group": "G001", "members": names}]
    rebuilt = summarize_stability(records, names, ["F001"], groups,
                                  resolve_stability(cfg), resolve_stability_analysis(cfg))
    assert rebuilt["correlation_groups"][0]["selection_frequency"] == 1.0
    assert rebuilt["correlation_groups"][0]["mean_selected_members"] == 2.0


def test_pairwise_statistics_have_explicit_degenerate_and_overlap_semantics():
    first = _record(0, ["F001", "F002", "F003"], {"F001": 1, "F002": 1})
    second = _record(1, ["F003", "F002", "F001"], {"F003": 1, "F002": 1})
    stats = pairwise_stability([first, second], ["F001", "F002"])
    assert stats["spearman"]["median"] == -1.0
    assert stats["top_k_jaccard"]["median"] == 1 / 3
    assert stats["reference_overlap"][0]["top_k_jaccard"] == 1.0
    second["nonconstant_features"] = ["F002"]
    stats = pairwise_stability([first, second], ["F001", "F002"])
    assert stats["spearman"]["median"] is None
    assert stats["spearman"]["missing"] == 1 and stats["spearman"]["n"] == 0


def test_completed_artifacts_validate_and_preserve_original_protocol(fitted):
    cfg, features, labels, reference_train, reference_test, out_dir, result = fitted
    assert validate_stability_artifacts(cfg, features, labels, reference_train, reference_test, out_dir) == result
    assert result["protocol"]["same_population_as_metric_resampling"] is False
    assert result["protocol"]["reference_test_used"] is False
    assert len(result["summary"]["candidates"]) == len(features.columns)
    assert len(result["summary"]["pairwise"]["pairs"]) == 6
    plan = json.loads((out_dir / STABILITY_PLAN_FILE).read_text(encoding="utf-8"))
    assert plan["analysis"]["direction_denominator"] == "selected_splits_including_zero"
    assert plan["analysis"]["rank_correlation_scope"] == "common_nonconstant_features"
    assert "当前证据不足" in (out_dir / CANDIDATES_TEXT_FILE).read_text(encoding="utf-8")
    for path in out_dir.iterdir():
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_real_complete_analysis_ignores_reference_test_features_and_labels(fitted, tmp_path):
    cfg, features, labels, reference_train, reference_test, _out_dir, original = fitted
    changed_features, changed_labels = features.copy(), labels.copy()
    changed_features.loc[reference_test, :] = 1e12
    changed_labels.loc[reference_test] = 1 - changed_labels.loc[reference_test]
    metric_path = tmp_path / "resample_metrics.json"
    metric_path.write_bytes(b'{"existing_metric_population": true}\n')
    before = metric_path.read_bytes()
    changed = run_stability_analysis(cfg, changed_features, changed_labels, reference_train, reference_test, tmp_path)
    assert changed["summary"] == original["summary"]
    assert changed["reference_features"] == original["reference_features"]
    assert changed["fingerprint"] == original["fingerprint"]
    assert changed["sampling_protocol"] == original["sampling_protocol"]
    assert metric_path.read_bytes() == before


def test_complete_split_evidence_ignores_both_holdouts(fitted):
    cfg, features, labels, reference_train, reference_test, _out_dir, result = fitted
    original = result["records"][0]
    changed_features, changed_labels = features.copy(), labels.copy()
    excluded = reference_test.append(pd.Index(original["holdout_indices"]))
    changed_features.loc[excluded, :] = 1e12
    changed_labels.loc[excluded] = 1 - changed_labels.loc[excluded]
    changed = fit_stability_split(changed_features, changed_labels, pd.Index(original["train_indices"]),
                                  reference_train, reference_test, cfg, 0)
    for key in ("selected_features", "ranks", "coefficients", "nonconstant_features"):
        assert changed[key] == original[key]


def test_fixed_seed_serial_parallel_evidence_and_candidates_are_exact(fitted, tmp_path):
    cfg, features, labels, reference_train, reference_test, out_dir, original = fitted
    destination = tmp_path / "parallel"
    shutil.copytree(out_dir, destination)
    before_protocol = (destination / "stability_protocol.json").read_bytes()
    before_candidates = (destination / CANDIDATES_FILE).read_bytes()
    parallel_cfg = copy.deepcopy(cfg)
    parallel_cfg["stability"].update(n_jobs=2, enabled=True)
    repeated = run_stability_analysis(parallel_cfg, features, labels, reference_train, reference_test, destination)
    assert repeated["summary"] == original["summary"]
    for first, second in zip(original["records"], repeated["records"]):
        for key in ("selected_features", "ranks", "coefficients", "nonconstant_features"):
            assert first[key] == second[key]
    assert (destination / "stability_protocol.json").read_bytes() == before_protocol
    assert (destination / CANDIDATES_FILE).read_bytes() == before_candidates


@pytest.mark.parametrize("fault", [
    "summary", "direction", "ranks", "group", "missing_split", "test_row", "protocol", "source",
    "candidates_json", "candidates_text", "plan", "status", "runtime",
])
def test_artifact_fault_injection_is_rejected(fitted, tmp_path, fault):
    cfg, features, labels, reference_train, reference_test, out_dir, _result = fitted
    destination = tmp_path / fault
    shutil.copytree(out_dir, destination)
    path = destination / STABILITY_DETAIL_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fault == "summary":
        payload["summary"]["candidates"][0]["selection_frequency"] = 0.123
    elif fault == "direction":
        feature = payload["records"][0]["selected_features"][0]
        del payload["records"][0]["coefficients"][feature]
    elif fault == "ranks":
        feature = payload["records"][0]["selected_features"][0]
        payload["records"][0]["ranks"][feature]["平均排名"] += 1
    elif fault == "group":
        payload["summary"]["correlation_groups"][0]["members"] = ["unknown"]
    elif fault == "missing_split":
        payload["records"].pop()
    elif fault == "test_row":
        payload["records"][0]["train_indices"][0] = int(reference_test[0])
    elif fault == "protocol":
        payload["protocol"]["same_population_as_metric_resampling"] = True
    elif fault == "source":
        payload["fingerprint"]["source_sha256"]["stability.py"] = "stale"
    elif fault == "candidates_json":
        (destination / CANDIDATES_FILE).write_text("{}", encoding="utf-8")
    elif fault == "candidates_text":
        (destination / CANDIDATES_TEXT_FILE).write_text("错误清单", encoding="utf-8")
    elif fault == "plan":
        (destination / STABILITY_PLAN_FILE).write_text("{}", encoding="utf-8")
    elif fault == "status":
        payload["status"] = "running"
    elif fault == "runtime":
        payload["execution"]["elapsed_seconds"] = -1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_stability_artifacts(cfg, features, labels, reference_train, reference_test, destination)


def test_training_changes_invalidate_artifacts_but_test_changes_do_not(fitted):
    cfg, features, labels, reference_train, reference_test, out_dir, original = fitted
    changed = features.copy()
    changed.loc[reference_test, :] = -1e12
    assert validate_stability_artifacts(cfg, changed, labels, reference_train, reference_test, out_dir) == original
    changed.loc[reference_train[0], "F001"] += 0.5
    with pytest.raises(ValueError, match="陈旧"):
        validate_stability_artifacts(cfg, changed, labels, reference_train, reference_test, out_dir)
    with pytest.raises(ValueError, match="reference 特征清单"):
        validate_stability_artifacts(cfg, features, labels, reference_train, reference_test, out_dir, ["F006"])


def test_analysis_criteria_cannot_change_after_plan_is_frozen(fitted, tmp_path):
    cfg, features, labels, reference_train, reference_test, out_dir, _result = fitted
    destination = tmp_path / "frozen"
    shutil.copytree(out_dir, destination)
    changed = copy.deepcopy(cfg)
    changed["stability"]["analysis"]["direction_alpha"] = 2.0
    with pytest.raises(FileExistsError, match="不能静默覆盖判据"):
        run_stability_analysis(changed, features, labels, reference_train, reference_test, destination)


def test_training_label_signal_changes_real_direction_evidence(fitted):
    cfg, features, labels, reference_train, reference_test, _out_dir, result = fitted
    original = result["records"][0]
    train_indices = pd.Index(original["train_indices"])
    changed_labels = labels.copy()
    changed_labels.loc[train_indices] = 1 - changed_labels.loc[train_indices]
    changed = fit_stability_split(features, changed_labels, train_indices,
                                  reference_train, reference_test, cfg, 0)
    assert "F001" in original["selected_features"] and "F001" in changed["selected_features"]
    assert np.sign(original["coefficients"]["F001"]) == -np.sign(changed["coefficients"]["F001"])


def test_mutation_during_batch_cannot_produce_success_artifact(fitted, tmp_path, monkeypatch):
    cfg, features, labels, reference_train, reference_test, _out_dir, result = fitted
    fingerprints = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr("src.stability_analysis._fingerprint", lambda *args: next(fingerprints))
    monkeypatch.setattr("src.stability_analysis.fit_stability_split",
                        lambda *args: result["records"][args[-1]])
    with pytest.raises(RuntimeError, match="跑批期间"):
        run_stability_analysis(cfg, features, labels, reference_train, reference_test,
                               tmp_path, result["reference_features"])
    assert json.loads((tmp_path / STABILITY_DETAIL_FILE).read_text(encoding="utf-8"))["status"] == "running"
    assert not (tmp_path / CANDIDATES_FILE).exists()


@pytest.mark.parametrize("enabled,quick", [(True, False), (False, False), (True, True)])
def test_pipeline_dispatch_preserves_raw_column_universe(fitted, tmp_path, monkeypatch, enabled, quick):
    cfg, features, labels, reference_train, reference_test, _out_dir, _result = fitted
    cfg = copy.deepcopy(cfg)
    cfg["stability"]["enabled"] = enabled
    cfg["validation"]["temporal_holdout"]["enabled"] = False
    cfg["output"]["results_dir"] = str(tmp_path)
    cfg["run"]["quick_sample_size"] = len(features) + 1
    calls = []
    timestamps = pd.Series(pd.date_range("2008-01-01", periods=len(labels)), index=labels.index)
    monkeypatch.setattr("src.pipeline.load_secom", lambda *args: (features, labels, timestamps))
    reference = {
        "metrics": {}, "fitted": {"岭分类器": None}, "features": ["F001"],
        "selection": {"reference_model": "岭分类器", "criteria_agree": True}, "folds": 2,
        "X_train": features.loc[reference_train, ["F001"]],
        "X_test": features.loc[reference_test, ["F001"]],
        "y_train": labels.loc[reference_train], "y_test": labels.loc[reference_test],
    }
    monkeypatch.setattr("src.pipeline._fit_and_evaluate_split", lambda *args: reference)
    monkeypatch.setattr("src.pipeline._run_explain_stage", lambda *args: None)
    monkeypatch.setattr("src.pipeline.run_imbalance_comparison", lambda *args: None)
    monkeypatch.setattr("src.pipeline.run_ablation", lambda *args: None)
    monkeypatch.setattr("src.pipeline.compare_with_baseline", lambda *args: (True, []))

    def capture(config, raw_features, raw_labels, train_indices, test_indices, output_dir, reference_features):
        pd.testing.assert_frame_equal(raw_features, features)
        assert "F007" in raw_features.columns
        assert train_indices.equals(reference_train) and test_indices.equals(reference_test)
        assert reference_features == ["F001"]
        calls.append(True)
        return {"status": "ok"}

    monkeypatch.setattr("src.pipeline.run_stability_analysis", capture)
    result = run_pipeline(cfg, quick=quick)
    assert len(calls) == int(enabled and not quick)
    assert result["stability"] == ({"status": "ok"} if calls else None)


def test_execution_flags_do_not_rewrite_frozen_sampling_criteria(fitted, tmp_path):
    cfg, _features, labels, reference_train, reference_test, _out_dir, _result = fitted
    path = write_stability_protocol(labels, reference_train, reference_test, cfg, tmp_path)
    original = path.read_bytes()
    changed = copy.deepcopy(cfg)
    changed["stability"].update(enabled=True, n_jobs=4)
    assert write_stability_protocol(labels, reference_train, reference_test, changed, tmp_path) == path
    assert path.read_bytes() == original

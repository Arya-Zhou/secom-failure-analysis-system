"""稳定性开跑前的协议、判据和训练侧隔离测试；仅使用小型合成数据拟合。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.stability import (
    STABILITY_PROTOCOL_FILE, assert_training_only, build_stability_protocol,
    classify_candidate, resolve_stability, select_stability_features,
    selection_frequencies, stability_splits, write_stability_protocol,
)
from src.validation import artifact_split, resample_splits


@pytest.fixture
def config():
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def sample():
    generator = np.random.RandomState(42)
    labels = pd.Series(np.tile([0, 0, 0, 0, 1], 40), index=np.arange(1000, 1200))
    features = pd.DataFrame(generator.normal(size=(len(labels), 6)), index=labels.index,
                            columns=[f"F{number:03d}" for number in range(1, 7)])
    features["F001"] = labels * 8 + generator.normal(scale=0.01, size=len(labels))
    features["F007"] = np.nan
    reference_train, reference_test = artifact_split(labels, 0.2, 42)
    return features, labels, reference_train, reference_test


def test_committed_protocol_and_criteria_are_explicit(config):
    spec = resolve_stability(config)
    assert spec["enabled"] is False
    assert spec["input_scope"] == "reference_train_only"
    assert (spec["n_resamples"], spec["test_size"], spec["seed"], spec["n_jobs"]) == (20, 0.2, 42, 4)
    assert spec["frequency_thresholds"] == [0.6, 0.7, 0.8]
    assert spec["primary_frequency_threshold"] == 0.7
    assert spec["correlation_group_threshold"] == 0.9
    assert spec["correlation_statistic"] == "absolute_spearman"
    assert spec["correlation_grouping"] == "connected_components"
    assert spec["classification"] == {
        "exploratory_min_frequency": 0.0,
        "high_min_direction_consistency": 0.8,
        "missing_direction": "insufficient_evidence",
    }


@pytest.mark.parametrize("key,value", [
    ("enabled", "false"), ("input_scope", "all_samples"),
    ("n_resamples", 1), ("n_resamples", 20.5), ("n_resamples", True),
    ("n_jobs", 0), ("seed", -1), ("seed", 2 ** 32),
    ("test_size", 0), ("test_size", 1), ("test_size", float("nan")),
    ("frequency_thresholds", []), ("frequency_thresholds", [0.8, 0.6]),
    ("frequency_thresholds", [0.7, 0.7]), ("frequency_thresholds", [True]),
    ("primary_frequency_threshold", 0.65), ("correlation_group_threshold", 0),
    ("correlation_group_threshold", float("inf")),
    ("correlation_statistic", "pearson"), ("correlation_grouping", "unknown"),
    ("classification", None),
])
def test_invalid_protocol_is_rejected(config, key, value):
    config["stability"][key] = value
    with pytest.raises(ValueError, match="stability"):
        resolve_stability(config)


@pytest.mark.parametrize("key", [
    "enabled", "input_scope", "n_resamples", "test_size", "seed", "n_jobs",
    "frequency_thresholds", "primary_frequency_threshold", "correlation_group_threshold",
    "correlation_statistic", "correlation_grouping", "classification",
])
def test_missing_protocol_field_is_rejected(config, key):
    del config["stability"][key]
    with pytest.raises(ValueError, match="stability"):
        resolve_stability(config)


@pytest.mark.parametrize("key,value", [
    ("exploratory_min_frequency", 0.6),
    ("high_min_direction_consistency", float("nan")),
    ("missing_direction", "high_stability"),
])
def test_invalid_classification_rules_are_rejected(config, key, value):
    config["stability"]["classification"][key] = value
    with pytest.raises(ValueError):
        resolve_stability(config)


@pytest.mark.parametrize("shortcut", ["quick", "override", "methods"])
def test_shortcuts_cannot_replace_four_method_selection(config, shortcut):
    if shortcut == "quick":
        config["run"]["quick"] = True
    elif shortcut == "override":
        config["feature_selection"]["override_features_path"] = "saved_features.txt"
    else:
        config["feature_selection"]["methods"].remove("rfe")
    with pytest.raises(ValueError, match="stability"):
        resolve_stability(config)


@pytest.mark.parametrize("key,value", [("methods", [None]), ("n_features_to_select", 0),
                                       ("n_features_to_select", True)])
def test_invalid_selection_configuration_is_rejected(config, key, value):
    config["feature_selection"][key] = value
    with pytest.raises(ValueError, match="stability"):
        resolve_stability(config)


def test_inner_population_is_not_metric_resampling(config, sample):
    _features, labels, reference_train, reference_test = sample
    actual = stability_splits(labels, reference_train, reference_test, config)
    assert len(actual) == 20
    expected = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    for (train_indices, holdout_indices), (train_positions, holdout_positions) in zip(
            actual, expected.split(np.zeros(len(reference_train)), labels.loc[reference_train])):
        assert train_indices.equals(reference_train.take(train_positions))
        assert holdout_indices.equals(reference_train.take(holdout_positions))
        assert train_indices.intersection(reference_test).empty
        assert holdout_indices.intersection(reference_test).empty
        assert len(train_indices) == 128 and len(holdout_indices) == 32
    metric_splits = resample_splits(labels, 0.2, 42, 20)
    assert any(len(train_indices.intersection(reference_test)) for train_indices, _holdout in metric_splits)
    assert all(len(train_indices) == 160 for train_indices, _holdout in metric_splits)
    original = build_stability_protocol(labels, reference_train, reference_test, config)
    config["validation"]["resample"]["n_splits"] = 7
    assert build_stability_protocol(labels, reference_train, reference_test, config) == original


def test_frozen_outer_indices_ignore_test_labels(config, sample):
    _features, labels, reference_train, reference_test = sample
    original = build_stability_protocol(labels, reference_train, reference_test, config)
    perturbed = labels.astype(float)
    perturbed.loc[reference_test] = np.nan
    assert build_stability_protocol(perturbed, reference_train, reference_test, config) == original


def test_full_data_split_fault_is_rejected(config, sample, monkeypatch):
    _features, labels, reference_train, reference_test = sample
    metric_splits = resample_splits(labels, 0.2, 42, 20)
    monkeypatch.setattr("src.stability.resample_splits", lambda *args: metric_splits)
    with pytest.raises(ValueError, match="reference 测试侧"):
        stability_splits(labels, reference_train, reference_test, config)


def test_missing_split_fault_is_rejected(config, sample, monkeypatch):
    _features, labels, reference_train, reference_test = sample
    incomplete = resample_splits(labels.loc[reference_train], 0.2, 42, 19)
    monkeypatch.setattr("src.stability.resample_splits", lambda *args: incomplete)
    with pytest.raises(ValueError, match="划分数量"):
        stability_splits(labels, reference_train, reference_test, config)


@pytest.mark.parametrize("fault", ["test_row", "unknown_row", "duplicate_row", "empty"])
def test_bad_training_indices_are_rejected_before_fitting(config, sample, monkeypatch, fault):
    features, labels, reference_train, reference_test = sample
    train_indices = reference_train[:20]
    if fault == "test_row":
        train_indices = train_indices.append(reference_test[:1])
    elif fault == "unknown_row":
        train_indices = train_indices.append(pd.Index([-1]))
    elif fault == "duplicate_row":
        train_indices = train_indices.append(train_indices[:1])
    else:
        train_indices = train_indices[:0]

    def forbidden_fit(*args):
        pytest.fail("非法行索引到达了拟合入口")

    monkeypatch.setattr("src.stability.build_preprocess_pipeline", forbidden_fit)
    with pytest.raises(ValueError):
        select_stability_features(features, labels, train_indices, reference_train, reference_test, config)


def test_reference_partition_must_be_complete_and_disjoint(config, sample):
    _features, labels, reference_train, reference_test = sample
    with pytest.raises(ValueError, match="完整覆盖"):
        stability_splits(labels, reference_train[:-1], reference_test, config)
    with pytest.raises(ValueError, match="重叠"):
        stability_splits(labels, reference_train, reference_test.append(reference_train[:1]), config)


def test_real_selector_frequencies_ignore_both_holdout_sides(config, sample):
    features, labels, reference_train, reference_test = sample
    config["stability"]["n_resamples"] = 2
    config["feature_selection"]["n_features_to_select"] = 1
    original_selections, perturbed_selections = [], []
    for train_indices, holdout_indices in stability_splits(labels, reference_train, reference_test, config):
        selected, ranks = select_stability_features(
            features, labels, train_indices, reference_train, reference_test, config)
        changed_features, changed_labels = features.copy(), labels.copy()
        excluded = reference_test.append(holdout_indices)
        changed_features.loc[excluded, :] = 1e12
        changed_labels.loc[excluded] = 1 - changed_labels.loc[excluded]
        changed_selected, changed_ranks = select_stability_features(
            changed_features, changed_labels, train_indices, reference_train, reference_test, config)
        assert selected == changed_selected == ["F001"]
        pd.testing.assert_frame_equal(ranks, changed_ranks, check_exact=True)
        assert "F007" not in ranks.index
        assert {"F检验排名", "互信息排名", "RFE排名", "随机森林排名"}.issubset(ranks.columns)
        original_selections.append(selected)
        perturbed_selections.append(changed_selected)
    assert selection_frequencies(original_selections, features.columns, 2) == selection_frequencies(
        perturbed_selections, features.columns, 2)
    changed_features = features.copy()
    changed_features.loc[reference_train, "F002"] = features.loc[reference_train, "F001"]
    changed_features.loc[reference_train, "F001"] = features.loc[reference_train, "F003"]
    changed_selected, _ranks = select_stability_features(
        changed_features, labels, train_indices, reference_train, reference_test, config)
    assert changed_selected == ["F002"]


def test_frequency_includes_unselected_and_non_reference_features():
    assert selection_frequencies([["F001", "F003"], ["F003"]], ["F001", "F002", "F003"], 2) == {
        "F001": 0.5, "F002": 0.0, "F003": 1.0,
    }


@pytest.mark.parametrize("selections", [[], [["F001"]], [["F001"], []],
                                         [["F001", "F001"], ["F001"]], [["F002"], ["F001"]]])
def test_incomplete_or_invalid_frequency_records_are_rejected(selections):
    with pytest.raises(ValueError):
        selection_frequencies(selections, ["F001"], 2)


@pytest.mark.parametrize("frequency,direction,expected", [
    (0.0, 1.0, "当前证据不足"), (1.0, None, "当前证据不足"),
    (0.05, 1.0, "探索性候选"), (0.65, 1.0, "探索性候选"),
    (0.7, 0.799, "探索性候选"), (0.7, 0.8, "高稳定候选"),
    (1.0, 1.0, "高稳定候选"),
])
def test_candidate_tier_boundaries(config, frequency, direction, expected):
    assert classify_candidate(frequency, direction, resolve_stability(config)) == expected


def test_all_sensitivity_tiers_use_frozen_thresholds(config):
    spec = resolve_stability(config)
    assert [classify_candidate(0.7, 0.8, spec, threshold)
            for threshold in spec["frequency_thresholds"]] == ["高稳定候选", "高稳定候选", "探索性候选"]
    with pytest.raises(ValueError, match="预先定义"):
        classify_candidate(0.7, 0.8, spec, 0.65)


def test_classification_rules_are_mutually_exclusive_and_complete(config):
    spec = resolve_stability(config)
    for count in range(21):
        frequency = count / 20
        for direction in (None, 0.0, 0.5, 0.79, 0.8, 1.0):
            for threshold in spec["frequency_thresholds"]:
                insufficient = count == 0 or direction is None
                high = not insufficient and frequency >= threshold and direction >= 0.8
                exploratory = not insufficient and not high
                assert sum((insufficient, high, exploratory)) == 1
                expected = "当前证据不足" if insufficient else "高稳定候选" if high else "探索性候选"
                assert classify_candidate(frequency, direction, spec, threshold) == expected


@pytest.mark.parametrize("frequency,direction", [(float("nan"), 1.0), (1.0, float("nan")),
                                                 (-0.1, 0.8), (1.1, 0.8), (0.7, 1.1)])
def test_invalid_candidate_evidence_does_not_become_a_tier(config, frequency, direction):
    with pytest.raises(ValueError):
        classify_candidate(frequency, direction, resolve_stability(config))


def test_protocol_is_persisted_separately_without_training(config, sample, tmp_path, monkeypatch):
    _features, labels, reference_train, reference_test = sample
    metric_path = tmp_path / "resample_metrics.json"
    metric_bytes = b'{"status":"existing_metric_population"}\n'
    metric_path.write_bytes(metric_bytes)

    def forbidden_selection(*args, **kwargs):
        pytest.fail("记录协议时不能启动特征选择")

    monkeypatch.setattr("src.stability.select_features", forbidden_selection)
    path = write_stability_protocol(labels, reference_train, reference_test, config, tmp_path)
    assert path.name == STABILITY_PROTOCOL_FILE
    assert metric_path.read_bytes() == metric_bytes
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "protocol_only" and payload["analysis_executed"] is False
    assert payload["protocol"]["n_population"] == len(reference_train)
    assert payload["protocol"]["reference_test_used"] is False
    assert payload["protocol"]["same_population_as_metric_resampling"] is False
    assert "metrics_by_split" not in payload and "parameter_candidates" not in payload
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    before = path.read_bytes()
    assert write_stability_protocol(labels, reference_train, reference_test, config, tmp_path) == path
    assert path.read_bytes() == before
    changed_config = copy.deepcopy(config)
    changed_config["stability"]["primary_frequency_threshold"] = 0.8
    with pytest.raises(FileExistsError, match="不能静默覆盖判据"):
        write_stability_protocol(labels, reference_train, reference_test, changed_config, tmp_path)
    assert path.read_bytes() == before and metric_path.read_bytes() == metric_bytes

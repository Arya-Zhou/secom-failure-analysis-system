"""模型比较的训练侧边界、角色隔离、配对判据与交付证据回归。"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation import score_samples
from src.imbalance import (
    apply_operating_point, load_operating_point, save_fitted_operating_point,
    verify_operating_point_evaluation,
)
from src.model_comparison import (
    COMPARISON_METRICS, DELIVERY_CANDIDATES, FEATURE_CONTROL, MODEL_NAMES,
    REFERENCE_BASELINES, build_comparison_estimator, fit_comparison_split,
    metrics_from_predictions, model_recipe, resolve_model_comparison, summarize_model_comparison,
)
from src.model_comparison_analysis import (
    COMPARISON_DETAIL_FILE, COMPARISON_FILE, COMPARISON_MARKDOWN, COMPARISON_MODEL_FILE,
    COMPARISON_PLAN_FILE, COMPARISON_SELECTION_FILE, render_model_comparison,
    run_model_comparison, validate_model_comparison_artifacts,
)
from src.modeling import get_models, make_model, registry_names
from src.stability_analysis import run_stability_analysis
from src.validation import artifact_split, paired_diff_interval


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    cfg = load_config(ROOT / "config.yaml")
    cfg["stability"].update(n_resamples=4, n_jobs=1)
    cfg["model_comparison"]["n_jobs"] = 1
    cfg["feature_selection"]["n_features_to_select"] = 3
    cfg["explain"]["enabled"] = False
    cfg["model"]["parameters"]["balanced_random_forest"]["n_estimators"] = 12
    cfg["model"]["parameters"]["hist_gradient_boosting"]["max_iter"] = 15
    generator = np.random.RandomState(42)
    labels = pd.Series(np.tile([0, 0, 0, 0, 1], 40), index=np.arange(2000, 2200))
    features = pd.DataFrame(generator.normal(size=(200, 6)), index=labels.index,
                            columns=[f"F{number:03d}" for number in range(1, 7)])
    features["F001"] += 3 * labels
    features["F002"] -= labels
    features.loc[features.index[::9], "F003"] = np.nan
    features["F007"], features["F008"] = np.nan, 1.0
    reference_train, reference_test = artifact_split(labels, 0.2, 42)
    output = tmp_path_factory.mktemp("model_comparison")
    stability = run_stability_analysis(cfg, features, labels, reference_train, reference_test, output)
    result = run_model_comparison(cfg, features, labels, reference_train, reference_test, output)
    return cfg, features, labels, reference_train, reference_test, output, stability, result


def _copy_output(fitted, tmp_path):
    destination = tmp_path / "artifacts"
    shutil.copytree(fitted[5], destination)
    return destination


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_new_models_registered_without_expanding_default_experiments():
    cfg = load_config(ROOT / "config.yaml")
    assert set(registry_names()) == set(DELIVERY_CANDIDATES + REFERENCE_BASELINES)
    assert isinstance(make_model("hist_gradient_boosting", 42, "balanced"), HistGradientBoostingClassifier)
    assert isinstance(make_model("elasticnet", 42, "balanced"), LogisticRegression)
    assert list(get_models(cfg, 42)) == ["logistic", "random_forest", "ridge"]
    assert cfg["imbalance"]["comparison"]["models"] == cfg["model"]["active"]
    assert cfg["ablation"]["models"] == cfg["model"]["active"]
    cfg["model"]["active"] = "all"
    assert set(get_models(cfg, 42)) == set(registry_names())


@pytest.mark.parametrize("key,value", [
    ("enabled", "false"), ("input_scope", "all_rows"), ("reference_model", "random_forest"),
    ("interval", "bootstrap"), ("primary_metric", "ber"), ("replacement_rule", "min_test_cost"),
    ("delivery_candidates", ["logistic", "balanced_random_forest"]),
    ("reference_baselines", ["random_forest"]), ("n_jobs", 0), ("n_jobs", True), ("n_jobs", 1.5),
])
def test_ambiguous_recipe_is_rejected(key, value):
    cfg = load_config(ROOT / "config.yaml")
    cfg["model_comparison"][key] = value
    with pytest.raises(ValueError, match="model_comparison"):
        resolve_model_comparison(cfg)


@pytest.mark.parametrize("key", ["random_state", "class_weight"])
def test_model_parameters_cannot_override_seed_or_weight(key):
    with pytest.raises(ValueError, match="不得覆盖"):
        make_model("elasticnet", 42, "balanced", {key: None})


def test_recipes_have_fixed_roles_and_no_strategy_search():
    cfg = load_config(ROOT / "config.yaml")
    for name in MODEL_NAMES:
        recipe = model_recipe(cfg, name)
        assert recipe["role"] == ("delivery_candidate" if name in DELIVERY_CANDIDATES
                                  else "reference_baseline" if name in REFERENCE_BASELINES
                                  else "feature_selection_control")
        assert not recipe["threshold_search"] and not recipe["calibration_search"]
    assert model_recipe(cfg, "balanced_random_forest")["parameters"]["class_weight"] is None
    assert model_recipe(cfg, "hist_gradient_boosting")["parameters"]["early_stopping"] is False
    assert model_recipe(cfg, "elasticnet")["parameters"] == model_recipe(cfg, FEATURE_CONTROL)["parameters"]


def test_actual_artifacts_recompute_and_only_selected_candidate_is_delivered(fitted):
    cfg, features, labels, train, test, output, _stability, result = fitted
    assert validate_model_comparison_artifacts(cfg, features, labels, train, test, output) == result
    report = json.loads((output / COMPARISON_FILE).read_text(encoding="utf-8"))
    selected = result["summary"]["selection"]["selected_model"]
    delivered = [name for name, row in report["models"].items() if row["artifacts"]]
    assert delivered == [selected]
    assert selected in DELIVERY_CANDIDATES
    assert all(report["models"][name]["artifacts"] == {} for name in REFERENCE_BASELINES)
    assert result["execution"]["feature_selector_refits"] == 0
    assert result["execution"]["reused_selection_records"] == 4
    assert result["analysis_plan"]["protocol"]["outer_test_used_for_selection"] is False
    with threadpool_limits(limits=1):
        loaded = load_operating_point(output / COMPARISON_MODEL_FILE,
                                      manifest=output / COMPARISON_FILE, require_manifest=True)
        predictions = apply_operating_point(loaded, features.loc[test])
    assert predictions.tolist() == result["outer_holdout"]["models"][selected]["predictions"]


def test_paired_intervals_are_recomputed_from_same_split_differences(fitted):
    cfg, _features, _labels, _train, _test, _output, _stability, result = fitted
    for pair, metrics in result["summary"]["paired_diff"].items():
        left, right = pair.split(" - ")
        for metric in COMPARISON_METRICS:
            expected = paired_diff_interval(
                [row["models"][left]["metrics"][metric] for row in result["records"]],
                [row["models"][right]["metrics"][metric] for row in result["records"]],
                "percentile", cfg["random_state"])
            assert metrics[metric] == expected
    assert len(result["summary"]["paired_diff"]) == 21
    assert result["analysis_plan"]["protocol"]["interval_percentiles"] == [5.0, 95.0]


@pytest.mark.parametrize("difference,replaced", [(-0.2, True), (0.0, False), (0.2, False)])
def test_replacement_requires_strictly_negative_paired_upper_bound(fitted, difference, replaced):
    cfg, *_unused, result = fitted
    records = copy.deepcopy(result["records"])
    for row in records:
        for model in row["models"].values():
            model["metrics"]["cost_per_wafer"] = 1.0
        row["models"]["hist_gradient_boosting"]["metrics"]["cost_per_wafer"] += difference
        for name in (*REFERENCE_BASELINES, FEATURE_CONTROL):
            row["models"][name]["metrics"]["cost_per_wafer"] = 0.0
    selection = summarize_model_comparison(records, cfg)["selection"]
    assert selection["selected_model"] == ("hist_gradient_boosting" if replaced else "ridge")
    assert not set(selection["eligible_replacements"]).intersection((*REFERENCE_BASELINES, FEATURE_CONTROL))


def test_cross_zero_pair_does_not_force_a_ranking(fitted):
    cfg, *_unused, result = fitted
    records = copy.deepcopy(result["records"])
    for split, row in enumerate(records):
        for model in row["models"].values():
            model["metrics"]["cost_per_wafer"] = 1.0
        row["models"]["hist_gradient_boosting"]["metrics"]["cost_per_wafer"] += [-0.2, 0.1, -0.1, 0.05][split]
    summary = summarize_model_comparison(records, cfg)
    assert summary["vs_reference"]["hist_gradient_boosting"]["cost_per_wafer"]["excludes_zero"] is False
    assert summary["selection"]["selected_model"] == "ridge"


def test_outer_test_perturbation_does_not_change_training_selection(fitted, tmp_path):
    cfg, features, labels, train, test, _output, _stability, result = fitted
    changed_features, changed_labels = features.copy(), labels.copy()
    changed_features.loc[test, "F001"] = -100.0
    changed_labels.loc[test] = 1 - changed_labels.loc[test]
    updated = run_model_comparison(cfg, changed_features, changed_labels, train, test, _copy_output(fitted, tmp_path))
    assert updated["records"] == result["records"]
    assert updated["summary"] == result["summary"]
    assert updated["selection_lock"] == result["selection_lock"]
    assert updated["outer_holdout"] != result["outer_holdout"]


def test_inner_holdout_does_not_enter_fitted_preprocessing_or_coefficients(fitted, monkeypatch):
    cfg, features, labels, _train, _test, _output, stability, result = fitted
    sampling = stability["records"][0]
    train_indices, held_indices = sampling["train_indices"], sampling["holdout_indices"]
    observed = []
    original_fit = SimpleImputer.fit

    def traced_fit(self, values, target=None):
        observed.append(values.index.tolist())
        return original_fit(self, values, target)

    monkeypatch.setattr(SimpleImputer, "fit", traced_fit)
    held = features.loc[held_indices].copy()
    held["F001"] = -100.0
    models, _fitted = fit_comparison_split(
        cfg, features.loc[train_indices], labels.loc[train_indices], held, 1 - labels.loc[held_indices],
        sampling["selected_features"])
    assert observed == [train_indices] * len(MODEL_NAMES)
    for name in MODEL_NAMES:
        baseline = result["records"][0]["models"][name]
        assert models[name]["coefficients"] == baseline["coefficients"]
        assert models[name]["selector_features"] == baseline["selector_features"]
    assert models["logistic"]["metrics"] != result["records"][0]["models"]["logistic"]["metrics"]


def test_serial_and_parallel_have_identical_evidence(fitted, tmp_path):
    cfg, features, labels, train, test, _output, _stability, result = fitted
    parallel_cfg = copy.deepcopy(cfg)
    parallel_cfg["model_comparison"]["n_jobs"] = 2
    repeated = run_model_comparison(parallel_cfg, features, labels, train, test, _copy_output(fitted, tmp_path))
    assert repeated["records"] == result["records"]
    assert repeated["summary"] == result["summary"]
    assert repeated["outer_holdout"] == result["outer_holdout"]


@pytest.mark.parametrize("case,expected", [
    ("missing_split", "划分记录不完整"), ("indices", "冻结划分"), ("missing_model", "角色或候选"),
    ("features", "特征与该轮"), ("coefficients", "嵌入式特征或系数"), ("metrics", "行级预测重算"),
    ("paired", "配对区间或选择"), ("selection", "配对区间或选择"), ("lock", "选择锁"),
    ("fingerprint", "来源已陈旧"), ("role", "角色、指标或交付清单"),
    ("external", "角色、指标或交付清单"), ("markdown", "报告文本"), ("model", "SHA-256"),
])
def test_inconsistent_artifacts_are_rejected(fitted, tmp_path, case, expected):
    cfg, features, labels, train, test, _output, _stability, _result = fitted
    output = _copy_output(fitted, tmp_path)
    detail_path, report_path = output / COMPARISON_DETAIL_FILE, output / COMPARISON_FILE
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if case == "missing_split":
        detail["records"].pop()
    elif case == "indices":
        detail["records"][0]["validation_indices"].reverse()
    elif case == "missing_model":
        del detail["records"][0]["models"]["balanced_random_forest"]
    elif case == "features":
        detail["records"][0]["models"]["logistic"]["selector_features"].reverse()
    elif case == "coefficients":
        detail["records"][0]["models"]["elasticnet"]["active_features"] = ["unknown"]
    elif case == "metrics":
        detail["records"][0]["models"]["logistic"]["metrics"]["ber"] += 0.1
    elif case == "paired":
        detail["summary"]["vs_reference"]["hist_gradient_boosting"]["ber"]["low"] += 0.1
    elif case == "selection":
        detail["summary"]["selection"]["selected_model"] = "balanced_random_forest"
    elif case == "lock":
        _write(output / COMPARISON_SELECTION_FILE, {"selection": "wrong"})
    elif case == "fingerprint":
        detail["fingerprint"]["training_data_sha256"] = "wrong"
    elif case == "role":
        report["models"]["random_forest"]["role"] = "delivery_candidate"
    elif case == "external":
        report["external_benchmark"]["external"]["derived_ber"] = 0.01
    elif case == "markdown":
        (output / COMPARISON_MARKDOWN).write_text("rank by test BER", encoding="utf-8")
    elif case == "model":
        with (output / COMPARISON_MODEL_FILE).open("ab") as stream:
            stream.write(b"changed")
    _write(detail_path, detail)
    _write(report_path, report)
    with pytest.raises(ValueError, match=expected):
        validate_model_comparison_artifacts(cfg, features, labels, train, test, output)


def test_changed_frozen_hyperparameters_are_rejected_before_refitting(fitted, tmp_path, monkeypatch):
    cfg, features, labels, train, test, _output, _stability, _result = fitted
    changed = copy.deepcopy(cfg)
    changed["model"]["parameters"]["elasticnet"]["C"] = 0.2

    def forbidden(*args, **kwargs):
        pytest.fail("陈旧协议不应开始训练")

    monkeypatch.setattr("src.model_comparison_analysis.fit_comparison_split", forbidden)
    with pytest.raises(FileExistsError, match="冻结计划已不同"):
        run_model_comparison(changed, features, labels, train, test, _copy_output(fitted, tmp_path))


def test_selection_is_locked_before_outer_test_is_evaluated(fitted, tmp_path, monkeypatch):
    cfg, features, labels, train, test, _output, _stability, result = fitted
    output = _copy_output(fitted, tmp_path)
    (output / COMPARISON_SELECTION_FILE).rename(output / "previous_selection.json")
    original = fit_comparison_split
    calls = []

    def observe(config, train_features, train_labels, held_features, held_labels, vote_features, **kwargs):
        if kwargs.get("keep_fitted"):
            lock = json.loads((output / COMPARISON_SELECTION_FILE).read_text(encoding="utf-8"))
            assert lock["selection"] == result["summary"]["selection"]
            assert held_features.index.tolist() == test.tolist()
            calls.append("outer_after_lock")
        else:
            assert not set(held_features.index).intersection(test)
            assert not set(train_features.index).intersection(test)
        return original(config, train_features, train_labels, held_features, held_labels, vote_features, **kwargs)

    monkeypatch.setattr("src.model_comparison_analysis.fit_comparison_split", observe)
    run_model_comparison(cfg, features, labels, train, test, output)
    assert calls == ["outer_after_lock"]


def test_uncached_selection_is_not_silently_replaced(fitted, tmp_path, monkeypatch):
    cfg, features, labels, train, test, _output, _stability, _result = fitted

    def forbidden(*args, **kwargs):
        pytest.fail("无来源记录时不得开始模型比较")

    monkeypatch.setattr("src.model_comparison_analysis.fit_comparison_split", forbidden)
    with pytest.raises(FileNotFoundError):
        run_model_comparison(cfg, features, labels, train, test, tmp_path)


@pytest.mark.parametrize("filename", [COMPARISON_DETAIL_FILE, COMPARISON_FILE, COMPARISON_PLAN_FILE,
                                     COMPARISON_SELECTION_FILE, COMPARISON_MODEL_FILE])
def test_incomplete_artifacts_fail(fitted, tmp_path, filename):
    cfg, features, labels, train, test, _output, _stability, _result = fitted
    output = _copy_output(fitted, tmp_path)
    (output / filename).rename(output / (filename + ".missing"))
    with pytest.raises(FileNotFoundError):
        validate_model_comparison_artifacts(cfg, features, labels, train, test, output)


def test_default_decision_boundary_is_part_of_prediction_evidence():
    with pytest.raises(ValueError, match="默认决策边界"):
        metrics_from_predictions([0, 1], [1, 0], [0.1, 0.9], "probability", {"fn": 10, "fp": 1}, 0)


def test_evaluation_checker_is_not_an_unguarded_inference_entry(fitted):
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        verify_operating_point_evaluation({}, fitted[1].loc[fitted[4]], [], [], "probability")


def test_evaluation_checker_rejects_changed_scores(fitted):
    cfg, features, _labels, _train, test, output, _stability, result = fitted
    selected = result["summary"]["selection"]["selected_model"]
    row = result["outer_holdout"]["models"][selected]
    with threadpool_limits(limits=1):
        loaded = load_operating_point(output / COMPARISON_MODEL_FILE,
                                      manifest=output / COMPARISON_FILE, require_manifest=True)
        scores = list(row["scores"])
        scores[0] += 0.001
        with pytest.raises(ValueError, match="未逐位复现"):
            verify_operating_point_evaluation(loaded, features.loc[test], row["predictions"], scores, row["score_space"])


def test_external_benchmark_has_actual_numbers_and_all_three_limits(fitted):
    report = json.loads((fitted[5] / COMPARISON_FILE).read_text(encoding="utf-8"))
    benchmark = report["external_benchmark"]
    selected = report["summary"]["selection"]["selected_model"]
    assert benchmark["external"]["derived_ber"] == pytest.approx((2 - 0.5806 - 0.8318) / 2)
    assert benchmark["local"]["ber"] == report["models"][selected]["outer_holdout"]["ber"]
    assert benchmark["paired_comparison_available"] is False
    assert len(benchmark["limitations"]) == 3
    text = render_model_comparison(report)
    assert "参考基线（不参与选择、不产模型交付物）" in text
    assert "不是独立样本均值置信区间" in text
    assert "5%–95%" in text
    assert "不能配对、不能据此排名" in text


@pytest.mark.parametrize("case", ["short", "nonfinite", "invalid_probability"])
def test_invalid_row_evidence_is_rejected(case):
    scores = [0.1, 0.9]
    if case == "short":
        scores.pop()
    elif case == "nonfinite":
        scores[0] = float("nan")
    else:
        scores[0] = 1.5
    with pytest.raises(ValueError):
        metrics_from_predictions([0, 1], [0, 1], scores, "probability", {"fn": 10, "fp": 1}, 0)


def test_embedded_input_differs_from_vote_but_hyperparameters_match(fitted):
    cfg, features, _labels, train, _test, _output, stability, _result = fitted
    embedded, all_features = build_comparison_estimator(cfg, "elasticnet", features.loc[train], stability["reference_features"])
    voted, vote_features = build_comparison_estimator(cfg, FEATURE_CONTROL, features.loc[train], stability["reference_features"])
    assert len(all_features) > len(vote_features)
    assert "F007" not in all_features
    assert embedded.named_steps["model"].get_params() == voted.named_steps["model"].get_params()


def test_reference_is_the_existing_train_selected_linear_model():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg["model_comparison"]["reference_model"] == "ridge"


def test_fitted_model_preserves_the_selector_column_names(fitted):
    cfg, features, labels, train, _test, _output, stability, _result = fitted
    estimator, selected = build_comparison_estimator(cfg, "logistic", features.loc[train], stability["reference_features"])
    with threadpool_limits(limits=1):
        estimator.fit(features.loc[train], labels.loc[train])
    assert estimator.named_steps["model"].feature_names_in_.tolist() == selected


@pytest.mark.parametrize("name", DELIVERY_CANDIDATES)
def test_every_delivery_candidate_roundtrips_the_full_named_pipeline(fitted, tmp_path, name):
    cfg, features, labels, train, test, _output, stability, _result = fitted
    estimator, selected = build_comparison_estimator(cfg, name, features.loc[train], stability["reference_features"])
    costs = cfg["costs"]
    with threadpool_limits(limits=1):
        estimator.fit(features.loc[train], labels.loc[train])
        predictions = estimator.predict(features.loc[test])
        scores, space = score_samples(estimator, features.loc[test])
        metrics = metrics_from_predictions(labels.loc[test], predictions, scores, space, costs, cfg["evaluation"]["fp_budget"])
        filename, digest = save_fitted_operating_point(
            estimator, name, selected, metrics["confusion"], costs, tmp_path, COMPARISON_MODEL_FILE)
        manifest = tmp_path / "manifest.json"
        _write(manifest, {"models": {name: {"artifacts": {
            "operating_point": filename, "operating_point_sha256": digest,
        }}}})
        loaded = load_operating_point(tmp_path / filename, manifest=manifest, require_manifest=True)
        verify_operating_point_evaluation(loaded, features.loc[test], predictions, scores, space)
        assert np.array_equal(apply_operating_point(loaded, features.loc[test]), predictions)

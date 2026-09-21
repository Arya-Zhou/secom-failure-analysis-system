"""合成数据校准测试：分别验证装配正确性与落盘后篡改检测，不依赖真实数据文件。"""
from __future__ import annotations

import ast
import inspect
import json
import pickle
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src import calibration as cal  # noqa: E402
from src import imbalance as imb  # noqa: E402
from src.evaluation import (  # noqa: E402
    SCORE_SPACE_CALIBRATED, SCORE_SPACE_PROBABILITY, score_samples,
)

SEED = 42
N_FEATURES = 6
CV_FOLDS = 3


# ---------- 公共合成数据与配置 ----------

@pytest.fixture(scope="module")
def synth_data():
    """raw 形态的不平衡二分类合成数据：未标准化、注入缺失值、含一列常量。"""
    X, y = make_classification(
        n_samples=260, n_features=8, n_informative=5, weights=[0.85, 0.15],
        flip_y=0.02, random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    X = X * rng.uniform(0.5, 30.0, size=X.shape[1]) + rng.uniform(-5.0, 5.0, size=X.shape[1])
    X[rng.rand(*X.shape) < 0.08] = np.nan
    X = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1])])
    X["C0"] = 1.0
    y = pd.Series(y, name="label")
    n_test = 60
    return (X.iloc[:-n_test], y.iloc[:-n_test], X.iloc[-n_test:], y.iloc[-n_test:])


def _cfg(strategies=None, models=("logistic",), **cal_over):
    calibration = {"method": "sigmoid", "contrast_method": "isotonic",
                   "cv_folds": CV_FOLDS}
    calibration.update(cal_over)
    if strategies is None:
        strategies = ["baseline", imb.CALIBRATED_PRIMARY, imb.CALIBRATED_CONTRAST]
    return {
        "preprocessing": {"impute_strategy": "median", "scale": True},
        "feature_selection": {"n_features_to_select": N_FEATURES},
        "costs": {"fn": 10.0, "fp": 1.0},
        "imbalance": {
            "strategy": "class_weight",
            "calibration": calibration,
            "cost_sensitivity": [5, 10, 20],
            "comparison": {"enabled": True, "strategies": list(strategies),
                           "models": list(models), "cv_folds": CV_FOLDS,
                           "smote_k_neighbors": 3},
        },
    }


@pytest.fixture(scope="module")
def fitted(synth_data):
    """已拟合的主口径校准 estimator + 训练侧完全折外的校准概率。"""
    X_train, y_train, _, _ = synth_data
    est = imb.build_strategy_estimator(
        imb.CALIBRATED_PRIMARY, "logistic", SEED, _cfg(), N_FEATURES)
    oof = imb._oof_scores(est, X_train, y_train, CV_FOLDS, SEED)
    est.fit(X_train, y_train)
    return est, oof


@pytest.fixture(scope="module")
def comparison(synth_data, tmp_path_factory):
    """跑一次含校准两行的完整对比，供产物级断言复用（module 级，只跑一次）。"""
    X_train, y_train, X_test, y_test = synth_data
    out = tmp_path_factory.mktemp("cal_cmp")
    res = imb.run_imbalance_comparison(
        _cfg(), X_train, y_train, X_test, y_test, out, SEED)
    return res, out


# ---------- 1. 折外性质：阈值必须选在完全折外的校准概率上 ----------

def test_oof_calibrated_probabilities_are_not_in_sample(synth_data, fitted):
    """CalibratedClassifierCV 全训练集拟合后对同一训练集 predict_proba **不是**折外。"""
    X_train, y_train, _, _ = synth_data
    est, oof = fitted
    in_sample = est.predict_proba(X_train)[:, 1]
    assert oof.shape == in_sample.shape == (len(y_train),)
    assert not np.allclose(oof, in_sample), (
        "折外概率与同集内概率完全一致：说明取的其实是同集内分数，"
        "'用了 CV 校准器就自然折外'这个假设正是本断言要否掉的")


def test_deployment_threshold_is_reproducible_from_oof_scores(synth_data, comparison):
    """落盘阈值必须能由「训练侧折外分数 + canonical 成本」逐位重算出来。"""
    X_train, y_train, _, _ = synth_data
    res, _ = comparison
    r = res["models"]["逻辑回归"]["strategies"][imb.CALIBRATED_PRIMARY]
    est = imb.build_strategy_estimator(
        imb.CALIBRATED_PRIMARY, "logistic", SEED, _cfg(), N_FEATURES)
    oof = imb._oof_scores(est, X_train, y_train, CV_FOLDS, SEED)
    thr, info = imb.select_threshold_by_cost(y_train, oof, 10.0, 1.0)
    assert r["threshold_selected_on"] == "train_oof"
    assert r["threshold"] == pytest.approx(thr, abs=0.0)
    assert r["threshold_selection"]["train_fn"] == info["train_fn"]
    assert r["threshold_selection"]["train_fp"] == info["train_fp"]


def test_calibrator_wraps_whole_pipeline_not_only_model(synth_data):
    """校准器包的是整条链路：折内重新拟合填充/标准化/特征选择，不是只包最后那一步。"""
    est = imb.build_strategy_estimator(
        imb.CALIBRATED_PRIMARY, "logistic", SEED, _cfg(), N_FEATURES)
    assert isinstance(est, CalibratedClassifierCV)
    inner = cal.base_pipeline(est)
    assert isinstance(inner, ImbPipeline)
    assert [n for n, _ in inner.steps] == ["impute", "scale", "select", "model"]
    # 基础 estimator 取不加权裸链路（口径②）：加权会先把分数分布整体推走
    assert inner.named_steps["model"].class_weight is None


# ---------- 2. 正确性：校准后概率本身对不对 ----------

def test_calibrated_probability_correlates_positively_with_label(synth_data, fitted):
    """折外校准概率与真实标签的相关方向必须为正。"""
    _, y_train, _, _ = synth_data
    _, oof = fitted
    corr = float(np.corrcoef(oof, np.asarray(y_train, dtype=float))[0, 1])
    assert corr > 0, f"折外校准概率与标签负相关（{corr:+.4f}）：正类索引很可能反了"


def test_calibrated_brier_beats_constant_prior(synth_data, fitted):
    """折外 Brier 必须优于「常数预测先验」这个平凡基线。"""
    _, y_train, _, _ = synth_data
    _, oof = fitted
    y = np.asarray(y_train, dtype=float)
    prior = float(y.mean())
    assert float(np.mean((oof - y) ** 2)) < float(np.mean((prior - y) ** 2))


def test_calibration_is_monotone_and_preserves_ranking(synth_data, fitted):
    """校准是单调非递减映射；未并出新并列时 AUC 必须逐位不变。"""
    X_train, y_train, _, _ = synth_data
    est, oof = fitted
    diag = cal.calibration_diagnostics(est, X_train, y_train, oof)
    assert diag["monotonic_non_decreasing"], (
        f"最大下降 {diag['max_decrease']:.3e}：单调性被破坏说明正类索引反了")
    if not diag["ties_introduced"]:
        assert abs(diag["auc_delta"]) <= cal.AUC_TOLERANCE
    # sigmoid 是严格单调，本数据下不该并出新并列
    assert diag["n_distinct_calibrated"] == diag["n_distinct_raw"]


def test_isotonic_ties_are_recorded_not_hidden(synth_data):
    """isotonic 的常数平台可能并出新并列，须记录本组数据的并列与 AUC 变化。"""
    X_train, y_train, _, _ = synth_data
    est = imb.build_strategy_estimator(
        imb.CALIBRATED_CONTRAST, "logistic", SEED, _cfg(), N_FEATURES)
    oof = imb._oof_scores(est, X_train, y_train, CV_FOLDS, SEED)
    est.fit(X_train, y_train)
    diag = cal.calibration_diagnostics(est, X_train, y_train, oof)
    assert diag["monotonic_non_decreasing"]
    assert diag["ties_introduced"], "本组合成数据应在 isotonic 的常数平台上并出新并列"
    assert diag["n_distinct_calibrated"] < diag["n_distinct_raw"]
    assert not diag["structural_problems"], (
        "并列引起的 AUC 变化不是缺陷：AUC 给并列记 0.5 分，"
        "把一对本来判反的样本并成并列反而涨分")


def test_calibrated_probability_range_shape_and_nan(synth_data, fitted):
    """校准后概率落在 [0,1]、无 NaN、形状与样本数一致。"""
    X_train, y_train, _, X_test = synth_data[0], synth_data[1], synth_data[2], synth_data[2]
    est, oof = fitted
    diag = cal.calibration_diagnostics(est, X_train, y_train, oof)
    assert diag["in_unit_interval"] and diag["finite"] and diag["shape_ok"]
    p = est.predict_proba(synth_data[2])[:, 1]
    assert p.shape == (len(synth_data[3]),) and np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0


class _InvertedCalibrator:
    """把校准器的输出取反：p → 1-p。故障注入用，模拟正类索引接反。"""

    def __init__(self, inner):
        self.inner = inner

    def predict(self, T):
        return 1.0 - np.asarray(self.inner.predict(T), dtype=float)


def test_monotonicity_check_catches_flipped_positive_index(synth_data, fitted):
    """故障注入：把校准后概率整体取反（等价于正类索引反了），单调性断言必须转红。"""
    X_train, y_train, _, _ = synth_data
    est, oof = fitted
    good = cal.calibration_diagnostics(est, X_train, y_train, oof)
    assert not good["structural_problems"], "前提：未注入故障时该组断言全过"

    # 在校准器这一环注入反向：基础链路分数一字不动，只有校准后概率反了 ——
    # 正是"折内与最终两步的正类索引约定不一致"的症状，也是行为指纹抓不到的那一类
    flipped = deepcopy(est)
    inner_cal = flipped.calibrated_classifiers_[0].calibrators[0]
    flipped.calibrated_classifiers_[0].calibrators[0] = _InvertedCalibrator(inner_cal)

    bad = cal.calibration_diagnostics(flipped, X_train, y_train, 1.0 - oof)
    assert not bad["monotonic_non_decreasing"], "注入反向映射后单调性断言仍为真：该断言是空转的"
    assert any("正类索引" in p for p in bad["structural_problems"])


def test_structural_problem_aborts_the_whole_comparison(synth_data, tmp_path, monkeypatch):
    """装配错误必须让整条对比流程失败，而不是记一笔继续跑完。"""
    X_train, y_train, X_test, y_test = synth_data
    real = cal.calibration_diagnostics

    def broken(est, X, y, oof):
        d = real(est, X, y, oof)
        d["monotonic_non_decreasing"] = False
        d["max_decrease"] = 0.5
        d["structural_problems"] = cal.structural_problems(d)
        d["passed"] = False
        return d

    monkeypatch.setattr(imb, "calibration_diagnostics", broken)
    with pytest.raises(ValueError, match="校准层装配错误"):
        imb.run_imbalance_comparison(
            _cfg(), X_train, y_train, X_test, y_test, tmp_path, SEED)


def test_quality_failure_keeps_artifact_out_of_selection_pool():
    """质量项不过关的校准行不得进选择池——"全过才可标为可部署"是过滤器不是承诺。"""
    ok, reason = imb._selection_eligibility(
        imb.CALIBRATED_PRIMARY,
        {"passed": False, "structural_problems": [],
         "quality_problems": ["折外 Brier 不优于常数预测先验"]})
    assert ok is False and "Brier" in reason
    ok2, reason2 = imb._selection_eligibility(
        imb.CALIBRATED_PRIMARY,
        {"passed": True, "structural_problems": [], "quality_problems": []})
    assert ok2 is True and reason2 is None
    # 对照行永远不进池，与它自己过不过无关
    ok3, reason3 = imb._selection_eligibility(
        imb.CALIBRATED_CONTRAST,
        {"passed": True, "structural_problems": [], "quality_problems": []})
    assert ok3 is False and "对照" in reason3


# ---------- 3. 结构：校准器在唯一决策函数内部 ----------

def test_score_space_is_calibrated_and_survives_clone(synth_data):
    """校准 estimator 的分数空间单列一个名字，且 clone 之后仍然如此（不靠实例属性标记）。"""
    X_train, y_train, X_test, _ = synth_data
    est = imb.build_strategy_estimator(
        imb.CALIBRATED_PRIMARY, "logistic", SEED, _cfg(), N_FEATURES)
    plain = imb.build_strategy_estimator("baseline", "logistic", SEED, _cfg(), N_FEATURES)
    est.fit(X_train, y_train)
    plain.fit(X_train, y_train)
    assert score_samples(est, X_test)[1] == SCORE_SPACE_CALIBRATED
    assert score_samples(plain, X_test)[1] == SCORE_SPACE_PROBABILITY
    cloned = clone(est).fit(X_train, y_train)
    assert score_samples(cloned, X_test)[1] == SCORE_SPACE_CALIBRATED, (
        "clone 后分数空间退回未校准语义：该判定挂在实例属性上了，"
        "sklearn clone 只复制 get_params，属性会被丢掉")


def test_calibration_is_not_an_identity_map(synth_data, fitted):
    """非空转：校准后概率与校准前分数不是同一串数，否则"校准器在路径里"无从谈起。"""
    _, _, X_test, _ = synth_data
    est, _ = fitted
    calibrated, _ = score_samples(est, X_test)
    raw, raw_space = cal.raw_scores(est, X_test)
    assert raw_space == SCORE_SPACE_PROBABILITY
    assert not np.allclose(calibrated, raw)


def test_decide_rejects_bypassing_the_calibrator(synth_data, fitted):
    """把内层裸链路塞回决策函数（绕开校准器）必须 fail-loud，不得静默按原阈值出预测。"""
    _, _, X_test, _ = synth_data
    est, _ = fitted
    inner = cal.base_pipeline(est)
    with pytest.raises(ValueError, match="分数空间"):
        imb._decide(inner, X_test, 0.07, SCORE_SPACE_CALIBRATED)


def test_decision_graph_is_derived_from_the_object(synth_data, fitted):
    """推理图按已拟合对象重算，含校准器那一环；重采样步骤不进图（predict 时被跳过）。"""
    est, _ = fitted
    assert cal.decision_graph(est, 0.07) == [
        "impute", "scale", "select", "model", "calibrator", "threshold"]
    plain = imb.build_strategy_estimator("smote", "logistic", SEED, _cfg(), N_FEATURES)
    assert cal.decision_graph(plain, None) == [
        "impute", "scale", "select", "model", "argmax"]


def test_decide_has_no_calibration_branch(synth_data, fitted):
    """_decide 不得按校准类型分叉，保证自检与上线使用同一条决策路径。"""
    src = inspect.getsource(imb._decide)
    tree = ast.parse(textwrap.dedent(src))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"is_calibrated", "calibration_block", "base_pipeline",
                 "raw_scores", "calibrated_classifiers_", "calibrators"}
    assert not (names & forbidden), (
        f"_decide 里出现了校准相关分支 {names & forbidden}：校准必须对决策函数透明")
    # 且自检与上线确实走同一个函数：两者对同一输入给出同一串预测
    est, _ = fitted
    _, _, X_test, _ = synth_data
    probe_pred = imb._decide(est, X_test, 0.07, SCORE_SPACE_CALIBRATED)
    assert (probe_pred == (est.predict_proba(X_test)[:, 1] >= 0.07).astype(int)).all()


# ---------- 4. 探针：必须覆盖阈值两侧 ----------

def test_probe_coverage_recorded_and_straddles_threshold(comparison):
    """每个含阈值的产物都须记录探针对阈值两侧的覆盖，且两侧都非空。"""
    res, _ = comparison
    checked = 0
    for m in res["models"].values():
        for strat, r in m["strategies"].items():
            cov = r["probe_coverage"]
            assert cov["n_rows"] == imb.PROBE_ROWS
            if r["threshold"] is None:
                assert cov["straddles_threshold"] is None
                continue
            checked += 1
            assert cov["below_threshold"] >= 1 and cov["at_or_above_threshold"] >= 1
            assert cov["straddles_threshold"] is True
    assert checked >= 2, "本用例需要至少两个含阈值的工作点才谈得上覆盖"


def test_straddle_assertion_is_not_vacuous():
    """故障注入：全落一侧的覆盖必须抛错——否则阈值篡改检测是哑的。"""
    one_sided = {"n_rows": 100, "score_min": 0.1, "score_max": 0.2,
                 "distinct_scores": 100, "below_threshold": 100,
                 "at_or_above_threshold": 0, "straddles_threshold": False}
    with pytest.raises(ValueError, match="全部落在阈值同一侧"):
        cal.assert_probe_straddles_threshold(one_sided, "测试用例")
    ok = dict(one_sided, below_threshold=40, at_or_above_threshold=60,
              straddles_threshold=True)
    cal.assert_probe_straddles_threshold(ok, "测试用例")   # 不应抛


def test_isotonic_probe_output_count_is_recorded(comparison):
    """不同输出概率数须落账，但不将它解释为探针覆盖的区间数。"""
    res, _ = comparison
    r = res["models"]["逻辑回归"]["strategies"][imb.CALIBRATED_CONTRAST]
    cov = r["probe_coverage"]
    assert 1 <= cov["distinct_scores"] <= imb.PROBE_ROWS


def test_isotonic_distinct_outputs_do_not_count_covered_intervals():
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        [0, 1, 2, 3], [0, 0, 1, 1])
    inputs = np.array([1.25, 1.5, 1.75])
    probabilities = calibrator.predict(inputs)
    np.testing.assert_array_equal(probabilities, [0.25, 0.5, 0.75])
    intervals = np.searchsorted(calibrator.X_thresholds_, inputs)
    assert np.unique(intervals).size == 1
    coverage = cal.probe_coverage(probabilities, 0.5)
    assert coverage["distinct_scores"] == 3
    assert coverage["below_threshold"] == 1
    assert coverage["at_or_above_threshold"] == 2


@pytest.mark.parametrize("method", [None, "sigmoid", "isotonic"])
def test_report_describes_isotonic_interpolation_without_segment_counts(
        comparison, tmp_path, method):
    result = deepcopy(comparison[0])
    deployment = result["models"]["逻辑回归"]["deployment"]
    deployment["calibration"] = {"method": method} if method else None
    deployment["probe_coverage"]["distinct_scores"] = 3
    report = tmp_path / "comparison.md"
    imb._write_markdown(result, report)
    text = report.read_text(encoding="utf-8")
    assert "节点间线性插值" in text
    assert "常数平台" in text
    assert "未被探针覆盖" in text
    assert "不能保证检出" in text
    assert "分段常数" not in text
    if method == "isotonic":
        assert "3 个不同输出概率" in text
        assert "不是已覆盖区间数" in text


# ---------- 5. 篡改检测：七类负例，逐条核对报错原因 ----------

def _artifact(comparison, model="逻辑回归"):
    res, out = comparison
    return out / res["models"][model]["artifacts"]["operating_point"]


def _tampered_copy(path, tmp_path, mutate, recompute_digest=False):
    """在副本上施加改动，返回新路径。"""
    op = pickle.loads(Path(path).read_bytes())
    mutate(op)
    if recompute_digest:
        op["payload_digest"] = imb.payload_digest(op)
    dst = tmp_path / Path(path).name
    dst.write_bytes(pickle.dumps(op))
    return dst


def _must_reject(path, tmp_path, mutate, reason, recompute_digest=False):
    dst = _tampered_copy(path, tmp_path, mutate, recompute_digest)
    with pytest.raises(ValueError, match=reason):
        imb.load_operating_point(dst, manifest=None)


def test_reject_replaced_calibrator(comparison, synth_data, tmp_path):
    """替换校准器但保留基础分类器：分数变、指纹拦下；校准方法字段还是对的，故只有指纹能抓。"""
    X_train, y_train, _, _ = synth_data
    other = imb.build_strategy_estimator(
        imb.CALIBRATED_CONTRAST, "logistic", SEED, _cfg(), N_FEATURES)
    other.fit(X_train, y_train)
    donor = other.calibrated_classifiers_[0].calibrators[0]

    def swap(op):
        assert cal.is_calibrated(op["pipeline"]), "本用例需要校准类工作点"
        op["pipeline"].calibrated_classifiers_[0].calibrators[0] = donor

    _must_reject(_artifact(comparison), tmp_path, swap, "行为指纹不符")


def test_reject_changed_calibration_method_field(comparison, tmp_path):
    """改产物记录的 calibration.method：按对象重算即对不上，且报错点名该字段。"""
    def bend(op):
        op["calibration"] = dict(op["calibration"], method="isotonic")

    _must_reject(_artifact(comparison), tmp_path, bend, "calibration",
                 recompute_digest=True)


def test_reject_flipped_positive_class_index(comparison, tmp_path):
    """调换类别顺序：正类索引由对象重算得出，直接点名拒绝。"""
    def flip(op):
        est = op["pipeline"]
        est.classes_ = np.asarray(est.classes_)[::-1].copy()
        assert cal.positive_class(est) == 0, "前提：篡改确实改掉了正类索引"

    _must_reject(_artifact(comparison), tmp_path, flip, "positive_class",
                 recompute_digest=True)


def test_reject_changed_score_space(comparison, tmp_path):
    """把 score_space 改成未校准语义：决策函数与实际链路对不上，fail-loud。"""
    def bend(op):
        op["score_space"] = SCORE_SPACE_PROBABILITY

    _must_reject(_artifact(comparison), tmp_path, bend, "推理字段摘要不符")
    # 就算把摘要一并重算，决策函数仍会因分数空间与链路实际不符而拒绝
    sub = tmp_path / "b"
    sub.mkdir()
    dst = _tampered_copy(_artifact(comparison), sub, bend, recompute_digest=True)
    with pytest.raises(ValueError, match="分数空间"):
        imb.load_operating_point(dst, manifest=None)


def test_reject_threshold_change_even_with_recomputed_digest(comparison, tmp_path):
    """改阈值并重算局部摘要：摘要层被绕过，行为指纹的最终预测摘要仍会拦下。"""
    def bend(op):
        assert op["threshold"] is not None
        op["threshold"] = float(op["threshold"]) + 0.3

    _must_reject(_artifact(comparison), tmp_path, bend, "行为指纹不符",
                 recompute_digest=True)


def test_reject_cost_change_via_manifest(comparison, tmp_path):
    """改成本档位并重算摘要：成本不影响预测，两个内部层都无感，只有外部清单能识别。"""
    res, out = comparison
    src = _artifact(comparison)

    def bend(op):
        op["costs"] = {"fn": 20.0, "fp": 1.0}

    # 内部两层：成本不进决策路径，摘要重算后确实放行 —— 如实验证这一点
    inner = _tampered_copy(src, tmp_path, bend, recompute_digest=True)
    imb.load_operating_point(inner, manifest=None)
    # 外部清单层：文件字节变了，哈希对不上
    (tmp_path / imb.COMPARISON_JSON).write_bytes(
        (out / imb.COMPARISON_JSON).read_bytes())
    with pytest.raises(ValueError, match="SHA-256 与清单不符"):
        imb.load_operating_point(inner, manifest=tmp_path / imb.COMPARISON_JSON)


def test_reject_decide_bypassing_calibrator_in_artifact(comparison, tmp_path):
    """把产物里的链路换成内层裸链路（绕开校准器）：推理图重算即少一环。"""
    def bypass(op):
        op["pipeline"] = cal.base_pipeline(op["pipeline"])

    dst = _tampered_copy(_artifact(comparison), tmp_path, bypass,
                         recompute_digest=True)
    with pytest.raises(ValueError, match="推理语义不符") as exc:
        imb.load_operating_point(dst, manifest=None)
    msg = str(exc.value)
    assert "'enabled': True" in msg and "'enabled': False" in msg, (
        f"报错未点出校准层被摘掉这件事：{msg}")


def test_reject_missing_schema_version(comparison, tmp_path):
    """结构版本必填、无默认值：缺失即 fail-loud，不许兜底成"按新语义读旧字节"。"""
    def drop(op):
        del op["artifact_schema_version"]

    dst = _tampered_copy(_artifact(comparison), tmp_path, drop)
    with pytest.raises(ValueError, match="artifact_schema_version"):
        imb.load_operating_point(dst, manifest=None)


def test_reject_wrong_schema_version(comparison, tmp_path):
    def bend(op):
        op["artifact_schema_version"] = "2026.08-uncalibrated"

    _must_reject(_artifact(comparison), tmp_path, bend, "结构版本",
                 recompute_digest=True)


def test_digest_covers_every_new_semantic_field(comparison):
    """新增语义字段逐个进摘要：改任一个，payload_digest 必须变。"""
    op = pickle.loads(_artifact(comparison).read_bytes())
    base = imb.payload_digest(op)
    mutations = {
        "artifact_schema_version": "x",
        "calibration": dict(op["calibration"], method="isotonic"),
        "decision_graph": op["decision_graph"][:-1],
        "costs": {"fn": 99.0, "fp": 1.0},
        "score_space": SCORE_SPACE_PROBABILITY,
        "probe_coverage": dict(op["probe_coverage"], below_threshold=0),
        "probe_diagnostics": dict(op["probe_diagnostics"], raw_scores_sha256="x"),
    }
    for key, value in mutations.items():
        tampered = dict(op)
        tampered[key] = value
        assert imb.payload_digest(tampered) != base, f"{key} 未进推理字段摘要"


def test_raw_score_hint_separates_base_chain_from_calibrator(comparison, synth_data,
                                                             tmp_path):
    """校准前分数只作分诊：换校准器时提示指向校准器，换基础链路时指向基础链路。"""
    X_train, y_train, _, _ = synth_data
    other = imb.build_strategy_estimator(
        imb.CALIBRATED_CONTRAST, "logistic", SEED, _cfg(), N_FEATURES)
    other.fit(X_train, y_train)

    op = pickle.loads(_artifact(comparison).read_bytes())
    op["pipeline"].calibrated_classifiers_[0].calibrators[0] = (
        other.calibrated_classifiers_[0].calibrators[0])
    assert "改动在校准器" in imb._calibration_layer_hint(op)

    op2 = pickle.loads(_artifact(comparison).read_bytes())
    inner = cal.base_pipeline(op2["pipeline"])
    inner.named_steps["model"].coef_ = inner.named_steps["model"].coef_ * 2.0
    assert "改动在基础链路" in imb._calibration_layer_hint(op2)


# ---------- 6. 成本敏感性 ----------

def test_cost_grid_requires_canonical_ratio():
    """canonical 成本比必须出现在敏感性表里，否则主表与本表对不上。"""
    grid = imb.cost_grid([5, 10, 20], fp_cost=1.0, canonical_fn=10.0)
    assert [g["ratio"] for g in grid] == [5.0, 10.0, 20.0]
    assert [g["is_canonical"] for g in grid] == [False, True, False]
    with pytest.raises(ValueError, match="未包含 canonical 成本比"):
        imb.cost_grid([5, 20], fp_cost=1.0, canonical_fn=10.0)


def test_cost_sensitivity_products_and_conclusion(comparison):
    """三档共用同一组折外分数只重选阈值；产物落盘并给出"会不会换工作点"的结论。"""
    res, out = comparison
    assert (out / imb.COST_SENSITIVITY_JSON).exists()
    assert (out / imb.COST_SENSITIVITY_MD).exists()
    cs = json.loads((out / imb.COST_SENSITIVITY_JSON).read_text(encoding="utf-8"))
    assert cs["canonical"] == {"fn": 10.0, "fp": 1.0}
    m = cs["models"]["逻辑回归"]
    assert m["strategy"] == imb.CALIBRATED_PRIMARY
    assert m["score_space"] == SCORE_SPACE_CALIBRATED
    assert [row["ratio"] for row in m["rows"]] == [5.0, 10.0, 20.0]
    assert sum(row["is_canonical"] for row in m["rows"]) == 1
    assert isinstance(m["changes_operating_point"], bool)
    assert m["distinct_thresholds"] == len({round(r["threshold"], 12)
                                            for r in m["rows"]})
    # 主表绑的那一档必须与 canonical 行逐位相同：两张表不许各说各话
    canonical = next(r for r in m["rows"] if r["is_canonical"])
    main = res["models"]["逻辑回归"]["strategies"][imb.CALIBRATED_PRIMARY]
    assert canonical["threshold"] == pytest.approx(main["threshold"], abs=0.0)


def test_cost_sensitivity_thresholds_move_with_cost_ratio(comparison):
    """成本比越高（漏检越贵），选出的阈值不应更高——方向反了说明代价函数用错了。"""
    _, out = comparison
    cs = json.loads((out / imb.COST_SENSITIVITY_JSON).read_text(encoding="utf-8"))
    rows = cs["models"]["逻辑回归"]["rows"]
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True), (
        f"阈值随 FN:FP 上升而上升（{thresholds}）：漏检更贵时应当更早报警")


@pytest.fixture
def cost_artifacts(comparison):
    result, output = comparison
    result = deepcopy(result)
    sensitivity = json.loads((output / imb.COST_SENSITIVITY_JSON).read_text(encoding="utf-8"))
    result["models"]["岭分类器"] = deepcopy(result["models"]["逻辑回归"])
    sensitivity["models"]["岭分类器"] = deepcopy(sensitivity["models"]["逻辑回归"])
    result["cost_sensitivity"]["changes_operating_point"]["岭分类器"] = (
        sensitivity["models"]["岭分类器"]["changes_operating_point"])
    return result, sensitivity


def _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity):
    output = tmp_path / "outputs"
    output.mkdir()
    (tmp_path / "config.yaml").write_text(json.dumps(_cfg()), encoding="utf-8")
    (output / imb.COMPARISON_JSON).write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (output / imb.COST_SENSITIVITY_JSON).write_text(
        json.dumps(sensitivity, ensure_ascii=False), encoding="utf-8")
    (output / imb.COST_SENSITIVITY_MD).write_text("# 成本敏感性\n", encoding="utf-8")
    source = (ROOT / "verify.sh").read_text(encoding="utf-8")
    stage = source.split('echo "===== 阶段 5.11/6:', 1)[1]
    script = stage.split("$PY - <<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(script, str(ROOT / "verify.sh") + ":5.11", "exec"), {})


def test_cost_verification_accepts_complete_artifacts(cost_artifacts, tmp_path, monkeypatch):
    _verify_calibration_artifacts(tmp_path, monkeypatch, *cost_artifacts)


def test_cost_verification_compares_primary_row_not_deployed_strategy(
        cost_artifacts, tmp_path, monkeypatch):
    result, sensitivity = cost_artifacts
    for model in result["models"].values():
        model["selection_pool"] = ["baseline"]
        model["best_by_cv_cost"] = "baseline"
        model["calibration_contrast"] = imb._calibration_contrast(
            model["strategies"], "baseline", ["baseline"])
        assert model["strategies"]["baseline"]["threshold"] is None
    _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


@pytest.mark.parametrize("mutation", ["empty", "missing", "extra"])
def test_cost_verification_rejects_incomplete_model_sets(
        cost_artifacts, tmp_path, monkeypatch, mutation):
    result, sensitivity = cost_artifacts
    if mutation == "empty":
        sensitivity["models"] = {}
    elif mutation == "missing":
        sensitivity["models"].pop("岭分类器")
    else:
        sensitivity["models"]["额外模型"] = deepcopy(sensitivity["models"]["逻辑回归"])
    with pytest.raises(SystemExit, match="模型集合"):
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


@pytest.mark.parametrize("field", ["fn", "fp", "tn", "tp", "cost", "threshold"])
def test_cost_verification_rejects_wrong_canonical_numbers(
        cost_artifacts, tmp_path, monkeypatch, field):
    result, sensitivity = cost_artifacts
    canonical = next(row for row in sensitivity["models"]["逻辑回归"]["rows"]
                     if row["is_canonical"])
    if field in ("fn", "fp", "tn", "tp"):
        canonical["test"][field] += 1
    elif field == "cost":
        canonical["test_cost"] += 1
    else:
        canonical["threshold"] = float(np.nextafter(canonical["threshold"], np.inf))
    with pytest.raises(SystemExit, match="canonical.*与主表不符"):
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "reordered"])
def test_cost_verification_rejects_incomplete_or_duplicate_rows(
        cost_artifacts, tmp_path, monkeypatch, mutation):
    result, sensitivity = cost_artifacts
    rows = sensitivity["models"]["逻辑回归"]["rows"]
    if mutation == "empty":
        rows.clear()
    elif mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows[0] = deepcopy(rows[1])
    else:
        rows.reverse()
    with pytest.raises(SystemExit, match="成本档位必须完整、升序且唯一"):
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


@pytest.mark.parametrize(("path", "value", "reason"), [
    (("status",), "failed", "成本敏感性 status"),
    (("grid", 0, "fn"), 15, "成本敏感性 grid"),
    (("canonical", "fn"), 20, "成本敏感性 canonical"),
    (("models", "逻辑回归", "strategy"), "baseline", "strategy 与主表不符"),
    (("models", "逻辑回归", "calibration_method"), "isotonic", "calibration_method 与主表不符"),
    (("models", "逻辑回归", "score_space"), "probability", "score_space 与主表不符"),
    (("models", "逻辑回归", "n_test"), 999, "n_test 与主表不符"),
    (("models", "逻辑回归", "n_test_positive"), 999, "n_test_positive 与主表不符"),
    (("models", "逻辑回归", "rows", 0, "fn_cost"), 15, "fn_cost 与config不符"),
    (("models", "逻辑回归", "rows", 0, "fp_cost"), 2, "fp_cost 与config不符"),
    (("models", "逻辑回归", "rows", 0, "is_canonical"), True, "canonical 标记"),
    (("models", "逻辑回归", "rows", 0, "threshold"), float("nan"), "阈值必须为有限数值"),
    (("models", "逻辑回归", "rows", 0, "test", "fn"), -1, "test 计数必须为非负整数"),
    (("models", "逻辑回归", "rows", 0, "train_oof", "fn"), 0.5,
     "train_oof 计数必须为非负整数"),
    (("models", "逻辑回归", "rows", 0, "train_oof", "cost"), -1,
     "train_oof cost 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "test_cost"), -1, "test_cost 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "test", "召回率"), -1, "召回率 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "test", "BER"), -1, "BER 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "cost_per_wafer"), -1,
     "cost_per_wafer 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "trivial_all_pass_cost"), -1,
     "trivial_all_pass_cost 与重算结果不符"),
    (("models", "逻辑回归", "rows", 0, "improvement_vs_all_pass"), 2,
     "improvement_vs_all_pass 与重算结果不符"),
    (("models", "逻辑回归", "distinct_thresholds"), -1,
     "distinct_thresholds 与重算结果不符"),
    (("models", "逻辑回归", "changes_operating_point"), None,
     "changes_operating_point 与重算结果不符"),
    (("models", "逻辑回归", "changes_test_confusion"), None,
     "changes_test_confusion 与重算结果不符"),
])
def test_cost_verification_rejects_inconsistent_metadata_and_derived_values(
        cost_artifacts, tmp_path, monkeypatch, path, value, reason):
    result, sensitivity = cost_artifacts
    target = sensitivity
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SystemExit, match=reason):
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


@pytest.mark.parametrize("section", ["train_oof", "test"])
def test_cost_verification_rejects_self_consistent_but_mismatched_canonical(
        cost_artifacts, tmp_path, monkeypatch, section):
    result, sensitivity = cost_artifacts
    canonical = next(row for row in sensitivity["models"]["逻辑回归"]["rows"]
                     if row["is_canonical"])
    if section == "train_oof":
        canonical["train_oof"]["fn"] += 1
        canonical["train_oof"]["cost"] += canonical["fn_cost"]
    else:
        counts = canonical["test"]
        delta = 1 if counts["tp"] > 0 else -1
        counts["fn"] += delta
        counts["tp"] -= delta
        n_positive = counts["fn"] + counts["tp"]
        n_negative = counts["fp"] + counts["tn"]
        counts["召回率"] = counts["tp"] / n_positive
        counts["BER"] = (counts["fn"] / n_positive + counts["fp"] / n_negative) / 2
        canonical["test_cost"] += delta * canonical["fn_cost"]
        canonical["cost_per_wafer"] = canonical["test_cost"] / (n_positive + n_negative)
        trivial_cost = canonical["trivial_all_pass_cost"]
        canonical["improvement_vs_all_pass"] = (trivial_cost - canonical["test_cost"]) / trivial_cost
    with pytest.raises(SystemExit, match="canonical.*与主表不符") as error:
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)
    assert "与重算结果不符" not in str(error.value)


def test_cost_verification_rejects_mismatched_main_summary(
        cost_artifacts, tmp_path, monkeypatch):
    result, sensitivity = cost_artifacts
    summary = result["cost_sensitivity"]["changes_operating_point"]
    summary["逻辑回归"] = not summary["逻辑回归"]
    with pytest.raises(SystemExit, match="主表成本敏感性摘要 changes_operating_point"):
        _verify_calibration_artifacts(tmp_path, monkeypatch, result, sensitivity)


# ---------- 7. 配置口径 ----------

def test_resolve_calibration_validates_methods():
    assert cal.resolve_calibration(_cfg()) == {
        "method": "sigmoid", "contrast_method": "isotonic", "cv_folds": CV_FOLDS}
    with pytest.raises(ValueError, match="method 仅支持"):
        cal.resolve_calibration(_cfg(method="platt"))
    with pytest.raises(ValueError, match="contrast_method 与 method 相同"):
        cal.resolve_calibration(_cfg(contrast_method="sigmoid"))
    with pytest.raises(ValueError, match="cv_folds"):
        cal.resolve_calibration(_cfg(cv_folds=1))


def test_contrast_row_is_reported_but_never_deployed(comparison):
    """对照行不部署；另记录保留主口径、额外加入对照是否改变选择。"""
    res, _ = comparison
    m = res["models"]["逻辑回归"]
    assert imb.CALIBRATED_CONTRAST in m["strategies"]
    assert imb.CALIBRATED_CONTRAST not in m["selection_pool"]
    assert m["best_by_cv_cost"] != imb.CALIBRATED_CONTRAST
    c = m["calibration_contrast"]
    assert c["primary_method"] == "sigmoid" and c["contrast_method"] == "isotonic"
    assert isinstance(c["would_change_selection"], bool)
    assert c["selected_with_contrast_in_pool"] in imb.STRATEGIES


def test_contrast_adds_candidate_without_replacing_primary():
    strategies = {
        strategy: {"cv": {"CV期望代价": cost}, "expected_cost": cost,
                   "calibration": {"method": method},
                   "calibration_diagnostics": {"passed": True}}
        for strategy, cost, method in (
            (imb.CALIBRATED_PRIMARY, 669, "sigmoid"),
            (imb.CALIBRATED_CONTRAST, 722, "isotonic"),
            ("threshold_moving", 695, None),
        )
    }
    contrast = imb._calibration_contrast(
        strategies, imb.CALIBRATED_PRIMARY, [imb.CALIBRATED_PRIMARY, "threshold_moving"])
    assert contrast["selected_with_contrast_in_pool"] == imb.CALIBRATED_PRIMARY
    assert contrast["would_change_selection"] is False
    replacement_pool = [imb.CALIBRATED_CONTRAST, "threshold_moving"]
    assert min(replacement_pool, key=lambda strategy: strategies[strategy]["cv"]["CV期望代价"]) == (
        "threshold_moving")

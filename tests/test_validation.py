"""评估协议层专项：划分等价性、迁移守护、区间与配对差值、时间序 splitter 的显式失败。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src.config import load_config  # noqa: E402
from src.evaluation import extended_test_metrics, recall_at_fp_budget  # noqa: E402
from src.pipeline import (  # noqa: E402
    INTERVAL_METRICS, RESAMPLE_FILE, TEMPORAL_FILE, run_pipeline,
    temporal_escalation, _load_random_intervals,
)
from src.validation import (  # noqa: E402
    ARTIFACT_SPLIT_INDEX, artifact_split, bootstrap_interval, interval,
    paired_diff_interval, percentile_interval, point_vs_interval,
    prior_drift_table, reference_percentile, resample_splits,
    resolve_artifact_split, resolve_resample, resolve_temporal_holdout,
    temporal_holdout_split,
)
from test_selection import (  # noqa: E402
    SEED, TEST_SIZE, _cfg, _write_secom_files, synth_raw,  # noqa: F401
)


def _labels(n: int = 1567, pos_rate: float = 0.0664) -> pd.Series:
    """与 SECOM 同规模同失败率的标签；划分器行为只跟标签与种子有关，与特征取值无关。"""
    n_pos = int(round(n * pos_rate))
    y = np.zeros(n, dtype=int)
    y[np.random.RandomState(0).choice(n, n_pos, replace=False)] = 1
    return pd.Series(y)


# ---------- 1. reference 划分 := 重采样总体的第 0 个划分（附录 A-3 的可执行版） ----------

def test_artifact_split_equals_train_test_split():
    """这条等价性是"区间与正式产物同出一次运行"的全部依据，必须是断言而不是备注。"""
    y = _labels()
    tr_ref, te_ref = train_test_split(
        y.index, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    tr, te = artifact_split(y, TEST_SIZE, SEED)
    assert list(tr) == list(tr_ref)
    assert list(te) == list(te_ref)


@pytest.mark.parametrize("n_splits", [1, 2, 20])
def test_split_zero_is_independent_of_population_size(n_splits):
    """开不开重采样、R 取多少，第 0 个划分都必须是同一个——否则产物随 R 漂移。"""
    y = _labels()
    tr, te = resample_splits(y, TEST_SIZE, SEED, n_splits)[ARTIFACT_SPLIT_INDEX]
    tr_ref, te_ref = artifact_split(y, TEST_SIZE, SEED)
    assert list(tr) == list(tr_ref) and list(te) == list(te_ref)


def test_resample_splits_are_distinct_and_stratified():
    y = _labels()
    splits = resample_splits(y, TEST_SIZE, SEED, 20)
    assert len({tuple(te) for _, te in splits}) == 20, "重采样划分出现重复，区间会被低估"
    for _, te in splits:
        rate = float(y.loc[te].mean())
        assert 0.03 < rate < 0.11, f"分层失效：测试段失败率 {rate:.3f}"


# ---------- 2. 配置层：迁移守护与取值校验 ----------

def test_legacy_split_section_is_rejected():
    """旧 split.stratify 与新 validation 共存时显式报错，不静默择一。"""
    cfg = {"split": {"test_size": 0.2, "stratify": True},
           "validation": {"temporal_holdout": {"enabled": True}},
           "artifact_split": {"protocol": "stratified_shuffle",
                              "test_size": 0.2, "seed": 42}}
    with pytest.raises(ValueError, match="共存"):
        resolve_artifact_split(cfg)


@pytest.mark.parametrize("spec, match", [
    (None, "缺少 artifact_split"),
    ({"protocol": "random", "test_size": 0.2, "seed": 42}, "仅支持"),
    ({"protocol": "stratified_shuffle", "seed": 42}, "缺少 test_size"),
])
def test_artifact_split_config_validation(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_artifact_split({"artifact_split": spec} if spec else {})


def test_resample_defaults_to_disabled():
    """默认关是有意的（R=20 是小时级开销）；默认值漂成 true 会让每次跑 main 都变慢一个量级。"""
    assert resolve_resample({})["enabled"] is False
    assert resolve_resample({})["n_splits"] == 20


def test_resample_rejects_unknown_interval():
    with pytest.raises(ValueError, match="仅支持"):
        resolve_resample({"validation": {"resample": {"interval": "gaussian"}}})


def test_temporal_holdout_rejects_stratify_knob():
    """时间序协议不接受 stratify：给了这个旋钮就等于允许把时间顺序打乱。"""
    with pytest.raises(ValueError, match="不接受 stratify"):
        resolve_temporal_holdout({"validation": {"temporal_holdout": {"stratify": True}}})


def test_repo_config_matches_new_namespace():
    cfg = load_config(ROOT / "config.yaml")
    assert "split" not in cfg, "旧 split 段仍在 config.yaml 里"
    spec = resolve_artifact_split(cfg)
    assert spec["seed"] == int(cfg["random_state"]), (
        "artifact_split.seed 与 random_state 不同：重采样总体与全局种子会分叉")
    rs = resolve_resample(cfg)
    assert rs["enabled"] is False and rs["n_splits"] == 20
    th = resolve_temporal_holdout(cfg)
    assert th["test_fraction"] == 0.2
    assert th["enabled"] is True, (
        "时间序对照默认关掉了：它只多跑一条划分，关掉会让 README 引用的时间序数字"
        "出自某次手工开启的历史运行")


# ---------- 3. 时间序 splitter：按时间切，类别不足显式失败 ----------

def _ts(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2008-07-19", periods=n, freq="h"))


def test_temporal_split_is_ordered_and_disjoint():
    n = 100
    ts = _ts(n).sample(frac=1.0, random_state=1)      # 打乱行序，划分须靠 timestamp 而非行号
    y = pd.Series(np.where(np.arange(n) % 7 == 0, 1, 0), index=ts.index)
    tr, te = temporal_holdout_split(ts, y, 0.2)
    assert len(te) == 20 and len(tr) == 80
    assert not set(tr) & set(te)
    assert ts.loc[tr].max() < ts.loc[te].min(), "训练段有样本晚于测试段：时间顺序没生效"


def test_temporal_split_gap_removes_tail_of_train():
    n, gap = 100, 10
    ts = _ts(n)
    y = pd.Series(np.where(np.arange(n) % 7 == 0, 1, 0), index=ts.index)
    tr, te = temporal_holdout_split(ts, y, 0.2, gap=gap)
    assert len(tr) == 70 and len(te) == 20


def test_temporal_split_fails_when_a_side_lacks_a_class():
    """正负样本不足时必须炸，而不是悄悄给出一个只有单类的"结果"。"""
    n = 100
    ts = _ts(n)
    y = pd.Series(np.zeros(n, dtype=int), index=ts.index)
    y.iloc[:5] = 1                                    # 正样本全在前段，测试段无失效
    with pytest.raises(ValueError, match="缺少另一类"):
        temporal_holdout_split(ts, y, 0.2)


def test_temporal_split_requires_complete_timestamps():
    n = 50
    ts = _ts(n)
    ts.iloc[3] = pd.NaT
    y = pd.Series(np.where(np.arange(n) % 5 == 0, 1, 0), index=ts.index)
    with pytest.raises(ValueError, match="完整 timestamp"):
        temporal_holdout_split(ts, y, 0.2)


# ---------- 4. 区间、配对差值、百分位 ----------

def test_percentile_interval_brackets_the_point():
    vals = [0.20, 0.24, 0.25, 0.26, 0.30]
    iv = percentile_interval(vals)
    assert iv["low"] <= iv["point"] <= iv["high"]
    assert iv["n"] == 5 and iv["method"] == "percentile"


def test_bootstrap_interval_is_narrower_than_percentile():
    """两者估计的不是同一个量：bootstrap 是均值的不确定度，必然窄于分布本身。"""
    rng = np.random.RandomState(0)
    vals = list(rng.normal(0.25, 0.05, size=50))
    p, b = percentile_interval(vals), bootstrap_interval(vals, seed=SEED)
    assert (b["high"] - b["low"]) < (p["high"] - p["low"])


def test_intervals_ignore_none_values():
    """岭分类器没有 Brier：整列 None 时给 None，部分 None 时只用算得出来的那部分。"""
    assert percentile_interval([None, None]) is None
    assert percentile_interval([0.1, None, 0.3])["n"] == 2


def test_paired_diff_detects_a_constant_difference():
    a = [0.30, 0.32, 0.28, 0.31]
    b = [x - 0.05 for x in a]
    d = paired_diff_interval(a, b, "percentile", SEED)
    assert d["point"] == pytest.approx(0.05)
    assert d["excludes_zero"] is True
    same = paired_diff_interval(a, a, "percentile", SEED)
    assert same["low"] == same["high"] == 0.0 and same["excludes_zero"] is False


def test_paired_diff_requires_same_population():
    with pytest.raises(ValueError, match="同一批划分"):
        paired_diff_interval([0.1, 0.2], [0.1], "percentile", SEED)


def test_reference_percentile_positions_the_reference():
    vals = [0.1, 0.2, 0.3, 0.4]
    assert reference_percentile(0.4, vals) == 100.0
    assert reference_percentile(0.25, vals) == 50.0
    assert reference_percentile(None, vals) is None


def test_interval_rejects_unknown_method():
    with pytest.raises(ValueError, match="未知的区间方法"):
        interval([0.1], "gaussian", SEED)


# ---------- 5. 新增指标字段 ----------

def test_recall_at_fp_budget_counts_only_within_budget():
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])   # 排序: 正 负 正 负 负 正
    assert recall_at_fp_budget(y, score, 0) == pytest.approx(1 / 3)
    assert recall_at_fp_budget(y, score, 1) == pytest.approx(2 / 3)
    assert recall_at_fp_budget(y, score, 3) == pytest.approx(1.0)
    assert recall_at_fp_budget(np.zeros(4), score[:4], 2) == 0.0


def test_extended_metrics_report_brier_only_for_probabilities():
    """岭分类器只有 decision_function：Brier 记 None，不拿决策分数冒充概率（校准是 B3 的事）。"""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    X = pd.DataFrame(np.random.RandomState(0).normal(size=(60, 4)))
    y = pd.Series((X[0] + X[1] > 0).astype(int))
    logit = LogisticRegression().fit(X, y)
    ridge = RidgeClassifier().fit(X, y)
    m_logit = extended_test_metrics(logit, X, y, 10.0, 1.0, 5)
    m_ridge = extended_test_metrics(ridge, X, y, 10.0, 1.0, 5)
    assert m_logit["分数空间"] == "probability" and 0.0 <= m_logit["Brier分数"] <= 1.0
    assert m_ridge["分数空间"] == "decision_margin" and m_ridge["Brier分数"] is None
    cm = m_logit["测试集混淆"]
    assert m_logit["期望代价"] == pytest.approx(cm["fn"] * 10.0 + cm["fp"] * 1.0)


# ---------- 6. 同源断言：划分 #0 的折级指标 ≡ reference 产物指标 ----------

@pytest.fixture(scope="module")
def resample_run(synth_raw, tmp_path_factory) -> dict:
    """开启重采样跑一次合成数据全流程（R=3，秒级）。"""
    X, y = synth_raw
    tmp = tmp_path_factory.mktemp("resample")
    feat, lab = _write_secom_files(X, y, tmp / "d")
    cfg = _cfg(feat, lab, tmp / "d" / "out")
    cfg["validation"] = {"resample": {
        "enabled": True, "n_splits": 3, "interval": "percentile",
        "paired_diff": True, "n_jobs": 1}}
    return run_pipeline(cfg, quick=True)


def test_same_source_assertion_holds(resample_run):
    """划分 #0 在重采样循环里被重算一遍，必须与 reference 产物逐位相同（容差 0）。"""
    ss = resample_run["resample"]["same_source_assertion"]
    assert ss["ok"] is True and ss["max_abs_diff"] == 0.0, ss["mismatches"]


def test_resample_artifact_is_complete(resample_run):
    res = resample_run["resample"]
    out = Path(resample_run["output_dir"]) / RESAMPLE_FILE
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["protocol"]["n_splits"] == 3
    assert [p["split"] for p in on_disk["metrics_by_split"]] == [0, 1, 2]
    for model, iv in res["intervals"].items():
        assert set(iv) == set(INTERVAL_METRICS), model
        for metric, got in iv.items():
            if got is not None:
                assert got["low"] <= got["point"] <= got["high"], (model, metric)
    assert res["paired_diff"], "未产出配对差值区间"
    assert sum(res["selection_frequency"].values()) == 3


def test_reference_point_matches_the_pipeline_metrics(resample_run):
    """产物里记的 reference 点估计就是主流程 metrics 本身，不是另算的一份。"""
    for model, point in resample_run["resample"]["reference_point"].items():
        for metric, val in point.items():
            assert val == resample_run["metrics"][model][metric]


def test_reference_percentile_is_recorded(resample_run):
    pct = resample_run["resample"]["reference_percentile"]
    assert set(pct) == set(resample_run["metrics"])
    ref = resample_run["reference_model"]
    assert pct[ref]["测试集BER"] is not None


# ---------- 7. 时间序协议接入主流程（阶段 B2）：先验漂移、点对区间、升级规则 ----------

def test_prior_drift_table_matches_known_distribution():
    """按已知构造核对：等样本数分箱、失败率、最大/最小倍数、首箱占全部失效的比例。"""
    n = 100
    ts = _ts(n).sample(frac=1.0, random_state=3)   # 打乱行序：分箱须按 timestamp
    y = pd.Series(np.zeros(n, dtype=int), index=ts.index)
    order = ts.sort_values().index
    y.loc[order[:10]] = 1        # 首箱 10/20 = 50%
    y.loc[order[20:25]] = 1      # 次箱 5/20 = 25%
    y.loc[order[80:81]] = 1      # 末箱 1/20 = 5%
    dr = prior_drift_table(ts, y, n_bins=5)
    assert [b["n"] for b in dr["bins"]] == [20] * 5
    assert [b["failures"] for b in dr["bins"]] == [10, 5, 0, 0, 1]
    assert dr["bins"][0]["failure_rate"] == pytest.approx(0.5)
    assert dr["max_over_min_ratio"] == pytest.approx(0.5 / 0.05)
    assert dr["first_bin_share_of_failures"] == pytest.approx(10 / 16)
    assert dr["failures"] == 16 and dr["n"] == n


def test_prior_drift_table_requires_complete_timestamps():
    ts = _ts(30)
    ts.iloc[2] = pd.NaT
    y = pd.Series(np.where(np.arange(30) % 5 == 0, 1, 0), index=ts.index)
    with pytest.raises(ValueError, match="完整 timestamp"):
        prior_drift_table(ts, y)


def test_point_vs_interval_marks_inside_and_outside():
    """时间序只有一条划分、没有自己的区间，所以这里读的是"点落在哪"，两个方向都断。"""
    iv = percentile_interval([0.20, 0.24, 0.25, 0.26, 0.30])
    inside = point_vs_interval(0.25, iv)
    outside = point_vs_interval(0.90, iv)
    assert inside["within_interval"] is True
    assert inside["delta_vs_interval_point"] == pytest.approx(0.0)
    assert outside["within_interval"] is False and outside["delta_vs_interval_point"] > 0
    assert point_vs_interval(0.25, None) is None
    assert point_vs_interval(None, iv) is None


def _cell(value: float, low: float, high: float) -> dict:
    return point_vs_interval(value, {
        "point": (low + high) / 2, "low": low, "high": high,
        "low_pct": 5.0, "high_pct": 95.0})


def test_temporal_escalation_only_fires_on_the_worse_side():
    """双向断言：更差一侧之外必须升级，更好一侧之外必须不升级。
    只断前半句的话，一个"落在区间外就升级"的实现同样能全绿，而它会把好消息报成风险。"""
    ref = "岭分类器"
    worse = {ref: {"测试集BER": _cell(0.60, 0.24, 0.43)}}
    better = {ref: {"测试集BER": _cell(0.10, 0.24, 0.43)}}
    inside = {ref: {"测试集BER": _cell(0.30, 0.24, 0.43)}}
    assert temporal_escalation(worse, ref)["required"] is True
    assert temporal_escalation(better, ref)["required"] is False
    assert temporal_escalation(inside, ref)["required"] is False
    # 召回的方向相反：低于下界才是恶化
    assert temporal_escalation({ref: {"召回率": _cell(0.10, 0.38, 0.77)}}, ref)["required"]
    assert not temporal_escalation({ref: {"召回率": _cell(0.95, 0.38, 0.77)}}, ref)["required"]
    # 非 reference 模型不触发升级：README 的头条数字只有 reference 那一行
    assert temporal_escalation(worse, "逻辑回归")["required"] is False


def test_temporal_escalation_records_which_model_it_judged():
    """记录实际判定的主流程 reference，避免独立重算误用时间序 reference。"""
    ref = "岭分类器"
    worse = {ref: {"测试集BER": _cell(0.60, 0.24, 0.43)}}
    assert temporal_escalation(worse, ref)["evaluated_on"] == ref
    other = temporal_escalation(worse, "逻辑回归")
    assert other["evaluated_on"] == "逻辑回归" and other["required"] is False
    assert temporal_escalation({}, ref)["evaluated_on"] == ref


@pytest.fixture(scope="module")
def temporal_run(synth_raw, tmp_path_factory) -> dict:
    """同一次运行里同时开重采样（R=3）与时间序：对照的两侧因此必然同源同代码。"""
    X, y = synth_raw
    tmp = tmp_path_factory.mktemp("temporal")
    feat, lab = _write_secom_files(X, y, tmp / "d")
    cfg = _cfg(feat, lab, tmp / "d" / "out")
    cfg["validation"] = {
        "resample": {"enabled": True, "n_splits": 3, "interval": "percentile",
                     "paired_diff": True, "n_jobs": 1},
        "temporal_holdout": {"enabled": True, "test_fraction": 0.2, "gap": 0,
                             "prior_drift_bins": 5},
    }
    return run_pipeline(cfg, quick=True)


def test_temporal_artifact_is_complete(temporal_run):
    tp = temporal_run["temporal"]
    on_disk = json.loads(
        (Path(temporal_run["output_dir"]) / TEMPORAL_FILE).read_text(encoding="utf-8"))
    assert on_disk["status"] == "ok"
    assert on_disk["protocol"]["name"] == "temporal_holdout"
    assert on_disk["protocol"]["n_test_positive"] >= 1
    assert on_disk["protocol"]["train_span"][1] <= on_disk["protocol"]["test_span"][0]
    assert len(on_disk["prior_drift"]["bins"]) == 5
    assert set(on_disk["models"]) == set(temporal_run["metrics"])
    assert on_disk["selection"]["selection_basis"] == tp["selection"]["selection_basis"]
    # 升级规则判的必须是**主流程 reference**，不是时间序协议自己选出的那个
    assert on_disk["escalation"]["evaluated_on"] == temporal_run["reference_model"]
    # 两协议并列落盘、互不覆盖
    assert (Path(temporal_run["output_dir"]) / RESAMPLE_FILE).exists()


def test_temporal_protocol_is_a_different_split_than_the_random_one(temporal_run):
    """时间序不是随机协议换个种子：它的测试集必须是时间上的末段，与划分 #0 不同。"""
    p = temporal_run["temporal"]["protocol"]
    assert p["n_test"] == temporal_run["split"]["n_test"]
    assert p["n_test_positive"] != temporal_run["temporal"]["prior_drift"]["failures"]
    assert p["test_span"][0] > p["train_span"][0]


def test_temporal_comparison_comes_from_this_run(temporal_run):
    cmp_ = temporal_run["temporal"]["random_protocol_comparison"]
    assert cmp_ is not None and cmp_["interval_source"] == "in_run"
    for model, cells in cmp_["models"].items():
        assert set(cells) == set(INTERVAL_METRICS), model
        cell = cells["测试集BER"]
        assert cell["value"] == temporal_run["temporal"]["models"][model]["测试集BER"]
        assert cell["percentile_in_random"] is not None
    assert "escalation" in cmp_ and isinstance(cmp_["escalation"]["reasons"], list)


@pytest.mark.parametrize("mutate, expect", [
    (lambda r: r.update(status="failed"), "status"),
    (lambda r: r["protocol"].update(n_splits=999), "n_splits"),
    (lambda r: r["protocol"].update(seed=7), "seed"),
    (lambda r: r["protocol"].update(quick=False), "quick"),
])
def test_mismatched_resample_artifact_is_not_used_as_the_comparison(
        temporal_run, tmp_path, mutate, expect):
    """故障注入：协议不符或陈旧的区间产物一律降级为"无对照"，不能拿去和本轮时间序并排。"""
    src = json.loads(
        (Path(temporal_run["output_dir"]) / RESAMPLE_FILE).read_text(encoding="utf-8"))
    mutate(src)
    (tmp_path / RESAMPLE_FILE).write_text(json.dumps(src), encoding="utf-8")
    res, reason = _load_random_intervals(
        None, tmp_path,
        {"test_size": TEST_SIZE, "seed": SEED}, {"n_splits": 3},
        temporal_run["metrics"], True)
    assert res is None and expect in reason


def test_stale_resample_artifact_is_not_used_as_the_comparison(temporal_run, tmp_path):
    """陈旧性：产物记的 reference 点估计与本轮主流程不符 → 那是旧代码留下的区间。"""
    src = json.loads(
        (Path(temporal_run["output_dir"]) / RESAMPLE_FILE).read_text(encoding="utf-8"))
    model = next(iter(src["reference_point"]))
    src["reference_point"][model]["测试集BER"] += 0.05
    (tmp_path / RESAMPLE_FILE).write_text(json.dumps(src), encoding="utf-8")
    res, reason = _load_random_intervals(
        None, tmp_path,
        {"test_size": TEST_SIZE, "seed": SEED}, {"n_splits": 3},
        temporal_run["metrics"], True)
    assert res is None and "须重跑" in reason
    # 未改动的同一份产物必须被接受，否则上面那条红灯只是因为它永远拒绝
    (tmp_path / RESAMPLE_FILE).write_text(
        (Path(temporal_run["output_dir"]) / RESAMPLE_FILE).read_text(encoding="utf-8"),
        encoding="utf-8")
    ok, source = _load_random_intervals(
        None, tmp_path,
        {"test_size": TEST_SIZE, "seed": SEED}, {"n_splits": 3},
        temporal_run["metrics"], True)
    assert ok is not None and source == "artifact"


def test_pipeline_fails_loud_when_temporal_tail_has_no_failures(synth_raw, tmp_path):
    """故障注入（集成级）：把失效全搬到时间前段，时间序协议必须让整个流程失败，
    而不是给出一个测试段只有单类的"指标"。"""
    X, y = synth_raw
    y_sorted = np.concatenate([np.sort(y)[::-1]])   # 正样本全排在最前 = 时间最早
    feat, lab = _write_secom_files(X, y_sorted, tmp_path / "d")
    cfg = _cfg(feat, lab, tmp_path / "d" / "out")
    cfg["validation"] = {
        "resample": {"enabled": False},
        "temporal_holdout": {"enabled": True, "test_fraction": 0.2, "gap": 0},
    }
    with pytest.raises(ValueError, match="缺少另一类"):
        run_pipeline(cfg, quick=True)

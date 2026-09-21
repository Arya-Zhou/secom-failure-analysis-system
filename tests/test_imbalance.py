"""不平衡策略对比专项测试：合成数据，秒级完成，不依赖真实数据文件。"""
from __future__ import annotations

import ast
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src import calibration as cal  # noqa: E402
from src import imbalance as imb  # noqa: E402
from src.evaluation import (  # noqa: E402
    classification_metrics, evaluate_train_test, predict_with_threshold,
    score_samples,
)
from src.modeling import get_models, make_model  # noqa: E402

SEED = 42
N_FEATURES = 6  # 测试配置的折内特征选择维数（小于总列数，保证选择器真的在选）


# ---------- 公共合成数据 ----------

@pytest.fixture(scope="module")
def synth_data():
    """raw 形态的不平衡二分类合成数据：未标准化、注入缺失值、含一列常量。"""
    X, y = make_classification(
        n_samples=240, n_features=8, n_informative=5, weights=[0.85, 0.15],
        flip_y=0.02, random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    X = X * rng.uniform(0.5, 30.0, size=X.shape[1]) + rng.uniform(-5.0, 5.0, size=X.shape[1])
    X[rng.rand(*X.shape) < 0.08] = np.nan
    X = pd.DataFrame(X, columns=[f"F{i}" for i in range(X.shape[1])])
    X["C0"] = 1.0  # 常量列：折内 F 检验分数为 NaN，SelectKBest 不应选它
    y = pd.Series(y, name="label")
    n_test = 60
    return (X.iloc[:-n_test], y.iloc[:-n_test], X.iloc[-n_test:], y.iloc[-n_test:])


def _cfg(**overrides):
    """对比实验用完整配置（含 preprocessing / feature_selection 子节）。"""
    cmp = {
        "enabled": True,
        "strategies": list(imb.STRATEGIES),
        "models": ["logistic"],
        "cv_folds": 3,
        "smote_k_neighbors": 3,
    }
    cmp.update(overrides)
    return {
        "preprocessing": {"impute_strategy": "median", "scale": True},
        "feature_selection": {"n_features_to_select": N_FEATURES},
        "costs": {"fn": 10.0, "fp": 1.0},
        "imbalance": {
            "strategy": "class_weight",
            "calibration": {"method": "sigmoid", "contrast_method": "isotonic",
                            "cv_folds": 3},
            "cost_sensitivity": [5, 10, 20],
            "comparison": cmp,
        },
    }


def _filled(X_train, X_test):
    """evaluation 函数直测用：训练集中位数填充。"""
    med = X_train.median()
    return X_train.fillna(med), X_test.fillna(med)


# ---------- 1. 策略构建（全链路 Pipeline） ----------

def test_build_estimator_full_chain_each_strategy():
    """每种策略都是 填充→标准化→特征选择→[重采样]→模型 的单一 Pipeline；校准类再包一层校准器。"""
    for strat in imb.STRATEGIES:
        est = imb.build_strategy_estimator(strat, "logistic", SEED, _cfg(), N_FEATURES)
        calibrated = strat in imb.CALIBRATED_STRATEGIES
        assert cal.is_calibrated(est) is calibrated, f"{strat} 的校准层归属不符"
        inner = cal.base_pipeline(est)
        assert isinstance(inner, ImbPipeline), f"{strat} 内层应为全链路 imblearn Pipeline"
        names = [n for n, _ in inner.steps]
        expected = ["impute", "scale", "select"]
        if strat in ("smote", "undersample"):
            expected.append("sampler")
        expected.append("model")
        assert names == expected, f"{strat} 步骤应为 {expected}，实际 {names}"
        if strat in ("smote", "undersample"):
            assert hasattr(inner.named_steps["sampler"], "fit_resample")
        assert inner.named_steps["select"].k == N_FEATURES


def test_build_estimator_unknown_strategy_raises():
    with pytest.raises(ValueError, match="未知不平衡策略"):
        imb.build_strategy_estimator("oversample_magic", "logistic", SEED, _cfg(), N_FEATURES)


def test_class_weight_only_for_class_weight_strategy():
    est_cw = imb.build_strategy_estimator("class_weight", "logistic", SEED, _cfg(), N_FEATURES)
    est_base = imb.build_strategy_estimator("baseline", "logistic", SEED, _cfg(), N_FEATURES)
    assert est_cw.named_steps["model"].class_weight == "balanced"
    assert est_base.named_steps["model"].class_weight is None


# ---------- 2. 防泄漏：全链路只在训练侧 fit ----------

class _SpySMOTE(SMOTE):
    """记录每次 fit_resample 收到的样本量，用于验证重采样从未见过验证折/测试集。"""

    seen_sizes: list[int] = []  # 类属性：跨 clone 共享（sklearn clone 会重建实例）

    def fit_resample(self, X, y):
        _SpySMOTE.seen_sizes.append(len(X))
        return super().fit_resample(X, y)


def test_resampling_confined_to_train_folds(synth_data, monkeypatch):
    """行为级防泄漏验证：CV 期间采样器只见过训练折，绝不见全量/验证折。"""
    X_train, y_train, X_test, y_test = synth_data
    _SpySMOTE.seen_sizes = []
    monkeypatch.setattr(
        imb, "make_sampler",
        lambda strategy, seed, cfg: _SpySMOTE(random_state=seed, k_neighbors=3)
        if strategy == "smote" else None,
    )
    folds = 3
    r = imb._evaluate_one_strategy(
        "smote", "logistic", X_train, y_train, X_test, y_test,
        SEED, folds, 10.0, 1.0, _cfg(), N_FEATURES,
    )
    sizes = _SpySMOTE.seen_sizes
    n_train = len(y_train)
    # CV 每折只喂训练折（约 2/3），最终拟合喂完整训练集；
    assert len(sizes) == folds + 1, f"应为 {folds} 次 CV + 1 次最终拟合，实际 {sizes}"
    fold_sizes, final_size = sizes[:folds], sizes[folds]
    assert final_size == n_train, "最终拟合应使用完整训练集"
    for s in fold_sizes:
        assert s < n_train, f"CV 折内样本量 {s} 不应达到训练集全量 {n_train}"
    assert sum(fold_sizes) == (folds - 1) * n_train, "各折训练折样本量之和应等于 (折数-1)×训练集"
    # 测试集从未参与重采样：任何一次调用的样本量都不含测试集
    assert all(s <= n_train for s in sizes)
    assert r["test"]["BER"] >= 0.0  # 流程完整走通


class _SpySelect(SelectKBest):
    """记录每次 fit 收到的样本量：验证上游特征选择只在训练侧拟合。"""

    seen_sizes: list[int] = []

    def fit(self, X, y=None):
        _SpySelect.seen_sizes.append(len(X))
        return super().fit(X, y)


def test_upstream_transforms_fit_within_folds(synth_data, monkeypatch):
    """
    行为级防泄漏验证（上游表示层）：特征选择只在 CV 训练折与最终训练集上 fit，
    验证折与测试集从未参与。
    """
    X_train, y_train, X_test, y_test = synth_data
    _SpySelect.seen_sizes = []
    monkeypatch.setattr(imb, "SelectKBest", _SpySelect)
    folds = 3
    r = imb._evaluate_one_strategy(
        "baseline", "logistic", X_train, y_train, X_test, y_test,
        SEED, folds, 10.0, 1.0, _cfg(), N_FEATURES,
    )
    sizes = _SpySelect.seen_sizes
    n_train, n_total = len(y_train), len(y_train) + len(y_test)
    assert len(sizes) == folds + 1, f"应为 {folds} 次 CV + 1 次最终拟合，实际 {sizes}"
    assert sizes[-1] == n_train, "最终拟合应使用完整训练集"
    for s in sizes[:folds]:
        assert s < n_train, f"CV 期间特征选择只能见训练折，实际见 {s}/{n_train}"
    assert all(s < n_total for s in sizes), "特征选择从未见过含测试集的数据"
    assert r["test"]["BER"] >= 0.0


# ---------- 3. 阈值选择 ----------

def test_select_threshold_minimizes_cost():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # 候选代价（fn=10, fp=1）：t=0.1→2；t=0.35→1；t=0.4→11；t=0.8→10；哨兵→20
    t, info = imb.select_threshold_by_cost(y, scores, 10.0, 1.0)
    assert t == pytest.approx(0.35)
    assert info["train_fn"] == 0 and info["train_fp"] == 1
    assert info["train_cost"] == pytest.approx(1.0)


def test_select_threshold_deterministic():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    scores = rng.rand(200)
    r1 = imb.select_threshold_by_cost(y, scores, 10.0, 1.0)
    r2 = imb.select_threshold_by_cost(y, scores, 10.0, 1.0)
    assert r1 == r2


def test_select_threshold_tie_prefers_recall():
    """同代价平局时取最低阈值（召回优先）。"""
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    # fn=fp=1 时 t=0.05、t=0.8、哨兵 三者代价均为 2 → 取最低阈值 0.05
    t, info = imb.select_threshold_by_cost(y, scores, 1.0, 1.0)
    assert t == pytest.approx(0.05)
    assert info["train_fn"] == 0


def test_select_threshold_higher_fn_cost_lowers_threshold():
    rng = np.random.RandomState(1)
    n = 300
    y = (rng.rand(n) < 0.15).astype(int)
    scores = np.clip(y * 0.3 + rng.rand(n) * 0.7, 0, 1)  # 有区分度但有重叠
    t_cheap, _ = imb.select_threshold_by_cost(y, scores, 2.0, 1.0)
    t_costly, _ = imb.select_threshold_by_cost(y, scores, 50.0, 1.0)
    assert t_costly <= t_cheap, "漏检成本越高，阈值应越低（即更保守，多报少漏）"


def test_select_threshold_single_class_raises():
    with pytest.raises(ValueError, match="两类样本"):
        imb.select_threshold_by_cost(np.zeros(10), np.linspace(0, 1, 10), 10.0, 1.0)


def test_nested_threshold_cv_deterministic_and_complete(synth_data):
    """嵌套 CV：同参两次结果一致；OOF 混淆计数覆盖全部训练样本。"""
    X_train, y_train, _, _ = synth_data
    est = imb.build_strategy_estimator(
        "threshold_moving", "logistic", SEED, _cfg(), N_FEATURES)
    cv1 = imb._nested_threshold_cv(est, X_train, y_train, 3, SEED, 10.0, 1.0)
    cv2 = imb._nested_threshold_cv(est, X_train, y_train, 3, SEED, 10.0, 1.0)
    assert cv1 == cv2
    assert {"CV_BER均值", "CV_BER标准差", "CV召回率均值", "CV混淆", "CV期望代价"} == set(cv1)
    cm = cv1["CV混淆"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_train)


# ---------- 4. 成本与指标 ----------

def test_expected_cost_known_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0, 0, 0])  # FN=2, FP=1
    cost, cm = imb.expected_cost(y_true, y_pred, 10.0, 1.0)
    assert cost == pytest.approx(21.0)
    assert cm == {"tn": 4, "fp": 1, "fn": 2, "tp": 1}


def test_classification_metrics_keys_and_auc_none():
    y = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 1, 1])
    m = classification_metrics(y, pred)
    assert set(m) == {"准确率", "精确率", "召回率", "F1分数", "BER", "AUC"}
    assert m["AUC"] is None, "无分数时 AUC 应为 None"
    m2 = classification_metrics(y, pred, y_score=np.array([0.1, 0.9, 0.4, 0.8]))
    assert m2["AUC"] == pytest.approx(1.0)


def test_evaluate_train_test_threshold_none_unchanged(synth_data):
    """threshold=None 时行为与历史逐位一致（守住基线比对）。"""
    X_train_raw, y_train, X_test_raw, y_test = synth_data
    X_train, X_test = _filled(X_train_raw, X_test_raw)
    model = make_model("logistic", SEED, "balanced").fit(X_train, y_train)
    m_default = evaluate_train_test(model, X_train, y_train, X_test, y_test)
    m_none = evaluate_train_test(model, X_train, y_train, X_test, y_test, threshold=None)
    assert m_default == m_none


def test_evaluate_train_test_with_extreme_threshold(synth_data):
    """极低阈值 → 全部判失败 → 召回 1、精确率≈正类占比。"""
    X_train_raw, y_train, X_test_raw, y_test = synth_data
    X_train, X_test = _filled(X_train_raw, X_test_raw)
    model = make_model("logistic", SEED, None).fit(X_train, y_train)
    m = evaluate_train_test(model, X_train, y_train, X_test, y_test, threshold=-1.0)
    assert m["召回率"] == pytest.approx(1.0)
    pred = predict_with_threshold(model, X_test, -1.0)
    assert pred.sum() == len(y_test)


# ---------- 5. 上线接口：已拟合链路行为指纹与工作点部署 ----------

def _fit_chain(X, y, cfg=None, model="logistic", n_features=N_FEATURES):
    """构建并拟合一条对比实验链路，供指纹对比用。"""
    est = imb.build_strategy_estimator(
        "baseline", model, SEED, cfg or _cfg(), n_features)
    return est.fit(X, y)


def _probe(est):
    """按链路自身已学统计量构造探针输入。"""
    return imb._build_probe_input(est, imb.PROBE_SEED, imb.PROBE_ROWS)


def _fp(est, probe_input=None, threshold=None):
    """计算行为指纹；probe_input 显式传入时两条链路共用同一份探针输入。"""
    Xp = _probe(est) if probe_input is None else probe_input
    _, space = score_samples(est, Xp)
    return imb.behavior_fingerprint(
        est, Xp, threshold=threshold, score_space=space, seed=imb.PROBE_SEED)


def _tamper_class_mapping(pipe) -> bool:
    """反转末端分类器的类别映射。"""
    model = pipe.steps[-1][1]
    lb = getattr(model, "_label_binarizer", None)
    if lb is not None:
        lb.classes_ = np.asarray(lb.classes_)[::-1].copy()
        return True
    if hasattr(model, "classes_"):
        model.classes_ = np.asarray(model.classes_)[::-1].copy()
        return True
    return False


def _tamper_final_estimator(pipe):
    """对末端 estimator 的已学状态施加一处改动来改变预测，与具体模型无关。"""
    model = pipe.steps[-1][1]
    if hasattr(model, "estimators_"):
        for tree in model.estimators_:
            v = tree.tree_.value
            v[:, :, :] = v[:, :, ::-1]
    elif hasattr(model, "coef_"):
        model.coef_ = -np.asarray(model.coef_)
    elif isinstance(model, HistGradientBoostingClassifier):
        model._baseline_prediction = -model._baseline_prediction
        for iteration in model._predictors:
            for predictor in iteration:
                predictor.nodes["value"] = -predictor.nodes["value"]
    else:
        raise AssertionError(
            f"未覆盖的模型类型: {type(model).__name__}，请补齐篡改手法")


def test_behavior_fingerprint_stable_for_same_fitted_chain(synth_data):
    """同结构 + 同参数 + 同训练数据 → 行为指纹相同（可复现，不含随机噪声）。"""
    X_train, y_train, _, _ = synth_data
    f1 = _fp(_fit_chain(X_train, y_train))
    f2 = _fp(_fit_chain(X_train, y_train))
    assert f1 == f2 and len(f1["scores_sha256"]) == 16
    assert f1["n_rows"] == imb.PROBE_ROWS


def test_behavior_fingerprint_stable_across_processes(synth_data, tmp_path):
    """行为指纹必须跨进程稳定。"""
    import subprocess
    import textwrap

    X_train, y_train, _, _ = synth_data
    est = _fit_chain(X_train, y_train)
    probe_input = _probe(est)
    _, space = score_samples(est, probe_input)
    expected = imb.behavior_fingerprint(
        est, probe_input, threshold=None, score_space=space, seed=imb.PROBE_SEED)
    pkl = tmp_path / "chain.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"pipeline": est, "probe_input": probe_input, "space": space}, f)

    code = textwrap.dedent(f"""
        import json, pickle, sys
        sys.path.insert(0, {str(ROOT)!r})
        from src.imbalance import behavior_fingerprint, PROBE_SEED
        with open({str(pkl)!r}, "rb") as f:
            d = pickle.load(f)
        print(json.dumps(behavior_fingerprint(
            d["pipeline"], d["probe_input"], threshold=None,
            score_space=d["space"], seed=PROBE_SEED), sort_keys=True), end="")
    """)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == expected, (
        f"跨进程行为指纹不一致：本进程 {expected}，子进程 {out.stdout}")


def test_behavior_probe_covers_impute_statistics(synth_data):
    """探针输入必须含缺失值。"""
    X_train, y_train, _, _ = synth_data
    est = _fit_chain(X_train, y_train)
    probe_input = _probe(est)
    assert probe_input.isna().any(axis=0).all(), "每一列都应至少有一个缺失值"

    base = _fp(est, probe_input)
    imputer = est.named_steps["impute"]
    imputer.statistics_ = imputer.statistics_ + 1.0
    # 沿用同一份探针输入：否则探针会随已学统计量一起变，测不到"分数变了"
    after = _fp(est, probe_input)
    assert after["input_sha256"] == base["input_sha256"], "同一份探针输入"
    assert after["scores_sha256"] != base["scores_sha256"], \
        "改掉填充统计量必须改变分数摘要"


@pytest.mark.parametrize("model_name", list(imb.registry_names()))
def test_behavior_fingerprint_detects_tampered_model_state(synth_data, model_name):
    """任意注册模型：末端 estimator 的已学状态被改后，行为指纹必须改变。"""
    X_train, y_train, X_test, _ = synth_data
    est = _fit_chain(X_train, y_train, model=model_name)
    probe_input = _probe(est)
    base = _fp(est, probe_input)
    base_pred = est.predict(X_test)

    _tamper_final_estimator(est)

    assert (est.predict(X_test) != base_pred).any(), "前提检查：篡改确实改变了预测"
    after = _fp(est, probe_input)
    assert after["input_sha256"] == base["input_sha256"], "同一份探针输入"
    assert after["scores_sha256"] != base["scores_sha256"], \
        "模型内部状态改变必须改变分数摘要"


def test_behavior_fingerprint_detects_different_fit_range(synth_data):
    """特征集合相同但拟合数据不同时，行为指纹必须不同。"""
    X_train, y_train, X_test, y_test = synth_data
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    est_train = _fit_chain(X_train, y_train)
    est_all = _fit_chain(X_all, y_all)
    # 先确认两条链路选到的特征集合确实一致，否则测的就不是"同特征"这一点
    feats_train = imb._selected_features(est_train, X_train.columns)
    feats_all = imb._selected_features(est_all, X_all.columns)
    if feats_train == feats_all:
        probe_input = _probe(est_train)      # 固定同一份探针输入作为受控变量
        assert (_fp(est_train, probe_input)["scores_sha256"]
                != _fp(est_all, probe_input)["scores_sha256"]), \
            "同特征集合、不同拟合数据必须产生不同分数摘要"


def test_behavior_fingerprint_detects_preprocessing_and_hyperparam_changes(synth_data):
    """预处理参数、随机种子、模型超参任一不同 → 行为指纹不同。"""
    X_train, y_train, _, _ = synth_data
    ref = _fit_chain(X_train, y_train)
    probe_input = _probe(ref)                # 全部对比固定同一份探针输入
    base = _fp(ref, probe_input)["scores_sha256"]

    def scores_of(est):
        return _fp(est, probe_input)["scores_sha256"]

    cfg_mean = _cfg()
    cfg_mean["preprocessing"]["impute_strategy"] = "mean"
    assert scores_of(_fit_chain(X_train, y_train, cfg_mean)) != base, "填充策略不同必须换指纹"

    cfg_noscale = _cfg()
    cfg_noscale["preprocessing"]["scale"] = False
    assert scores_of(_fit_chain(X_train, y_train, cfg_noscale)) != base, "是否标准化必须换指纹"

    assert scores_of(_fit_chain(X_train, y_train, n_features=N_FEATURES - 2)) != base

    # 随机种子：随机森林对种子敏感，拟合结果不同即指纹不同
    rf_a = _fit_chain(X_train, y_train, model="random_forest")
    rf_b = imb.build_strategy_estimator(
        "baseline", "random_forest", SEED + 1, _cfg(), N_FEATURES).fit(X_train, y_train)
    rf_probe = _probe(rf_a)
    assert (_fp(rf_a, rf_probe)["scores_sha256"]
            != _fp(rf_b, rf_probe)["scores_sha256"]), "随机种子不同必须换指纹"


def _run_and_get_artifact(tmp_path, synth_data, **kw):
    """跑一次对比实验，返回 (配置里第一个模型的产物路径, 结果 dict)。"""
    X_train, y_train, X_test, y_test = synth_data
    cfg = _cfg(**kw)
    res = imb.run_imbalance_comparison(
        cfg, X_train, y_train, X_test, y_test, tmp_path, SEED)
    model = cfg["imbalance"]["comparison"]["models"][0]
    return tmp_path / f"imbalance_operating_point_{model}.pkl", res


def test_load_rejects_tampered_payload_digest(synth_data, tmp_path):
    """记录的推理字段摘要与实际不符 → 加载即 fail-loud。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    op = imb.load_operating_point(path)
    op["payload_digest"] = "deadbeefdeadbeef"
    with open(path, "wb") as f:
        pickle.dump(op, f)
    with pytest.raises(ValueError, match="推理字段摘要不符"):
        imb.load_operating_point(path, manifest=None)


def test_load_rejects_tampered_threshold(synth_data, tmp_path):
    """阈值被改写而链路一字未动 → 必须拒绝。"""
    path, _ = _run_and_get_artifact(
        tmp_path, synth_data, models=["logistic"], strategies=["threshold_moving"])
    op = imb.load_operating_point(path)
    assert op["threshold"] is not None
    op["threshold"] = float(op["threshold"]) + 0.3
    with open(path, "wb") as f:
        pickle.dump(op, f)
    with pytest.raises(ValueError, match="推理字段摘要不符"):
        imb.load_operating_point(path, manifest=None)


def test_load_rejects_tampered_model_state(synth_data, tmp_path):
    """模型内部状态被改而两个摘要字段原样保留 → 必须由行为指纹拒绝。"""
    # 只跑一种策略：本测试验的是树内部状态改变能否被指纹抓到，与策略无关；
    # 随机森林 × 校准类的嵌套 CV 在此纯属陪跑（实测多花约 3 分钟）
    path, _ = _run_and_get_artifact(
        tmp_path, synth_data, models=["random_forest"], strategies=["baseline"])
    op = imb.load_operating_point(path)
    for tree in cal.base_pipeline(op["pipeline"]).named_steps["model"].estimators_:
        v = tree.tree_.value
        v[:, :, :] = v[:, :, ::-1]
    with open(path, "wb") as f:
        pickle.dump(op, f)      # payload_digest 与 probe 记录都原样保留
    with pytest.raises(ValueError, match="行为指纹不符"):
        imb.load_operating_point(path, manifest=None)


@pytest.mark.parametrize("model_name", list(imb.registry_names()))
def test_load_rejects_tampered_class_mapping(synth_data, tmp_path, model_name):
    """类别映射被反转后分数一字未变、最终预测全部翻转，加载必须拒绝。"""
    _, _, X_test, _ = synth_data
    path, _ = _run_and_get_artifact(
        tmp_path, synth_data, models=[model_name], strategies=["class_weight"])
    op = imb.load_operating_point(path)
    assert op["threshold"] is None, "本测试针对无阈值（走 predict）的工作点"
    base_pred = imb.apply_operating_point(op, X_test)
    base_probe = op["probe"]

    assert _tamper_class_mapping(op["pipeline"]), (
        f"未覆盖的模型类型: {type(op['pipeline'].steps[-1][1]).__name__}，"
        "请补齐类别映射篡改手法")

    after = imb.behavior_fingerprint(
        op["pipeline"], op["probe_input"], threshold=op["threshold"],
        score_space=op["score_space"], seed=base_probe["seed"])
    assert after["scores_sha256"] == base_probe["scores_sha256"], \
        "前提：该篡改不改变任何分数（只看分数的自检会整批放行）"
    assert after["predictions_sha256"] != base_probe["predictions_sha256"], \
        "最终预测摘要必须改变，正是分数层漏掉的那一层"
    assert (imb.apply_operating_point(op, X_test) != base_pred).all(), \
        "前提：真实输入上的判定全部翻转"

    with open(path, "wb") as f:
        pickle.dump(op, f)      # payload_digest 与 probe 记录都原样保留
    # 正类索引可由已拟合对象重算，故加载时按名点出该字段；行为指纹是同一轮的兜底，
    # 上面几行已单独验过它确实抓得到（分数摘要不变、最终预测摘要变）
    with pytest.raises(ValueError, match="positive_class"):
        imb.load_operating_point(path, manifest=None)


@pytest.mark.parametrize("strategy", ["class_weight", "threshold_moving"])
def test_fingerprint_predictions_come_from_deployment_path(
    synth_data, tmp_path, strategy,
):
    """产物记录的预测摘要必须等于 apply_operating_point 在同一输入上的输出摘要。"""
    path, _ = _run_and_get_artifact(
        tmp_path / strategy, synth_data, strategies=[strategy])
    op = imb.load_operating_point(path)
    assert (op["threshold"] is None) == (strategy == "class_weight")
    y_probe = imb.apply_operating_point(op, op["probe_input"])
    assert op["probe"]["predictions_sha256"] == imb._sha16(y_probe)


def test_load_rejects_swapped_probe_input(synth_data, tmp_path):
    """探针输入被换成另一块数据时必须拒绝。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    op = imb.load_operating_point(path)
    rng = np.random.RandomState(0)
    op["probe_input"] = op["probe_input"] * 0 + rng.standard_normal(
        op["probe_input"].shape)
    with open(path, "wb") as f:
        pickle.dump(op, f)
    with pytest.raises(ValueError, match="行为指纹不符"):
        imb.load_operating_point(path, manifest=None)


def test_load_rejects_replaced_artifact_via_manifest(synth_data, tmp_path):
    """整份产物被替换成另一份内部自洽的产物时，只有外部清单能识别。"""
    _, res = _run_and_get_artifact(
        tmp_path, synth_data, models=["logistic", "random_forest"],
        strategies=["baseline"])   # 本测试只需要两份互不相同的产物
    a = tmp_path / res["models"]["逻辑回归"]["artifacts"]["operating_point"]
    b = tmp_path / res["models"]["随机森林"]["artifacts"]["operating_point"]
    a.write_bytes(b.read_bytes())        # 用 A 的文件名装 B 的内容

    imb.load_operating_point(a, manifest=None)      # 自检层：内部自洽，放行
    with pytest.raises(ValueError, match="SHA-256 与清单不符"):
        imb.load_operating_point(a)                 # 清单层：文件哈希不符，拒绝


def test_load_without_manifest_warns(synth_data, tmp_path, caplog):
    """清单缺失时降级为只做自检，但必须留痕，不允许静默降级。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    (tmp_path / imb.COMPARISON_JSON).unlink()
    with caplog.at_level("WARNING"):
        imb.load_operating_point(path)
    assert "未经外部清单核对" in caplog.text, "清单缺失必须告警"


def test_load_with_explicit_missing_manifest_fails_loud(synth_data, tmp_path):
    """显式指定的清单不存在 → 报错，拒绝当成"没有清单"而悄悄降级。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    with pytest.raises(FileNotFoundError):
        imb.load_operating_point(path, manifest=tmp_path / "not_here.json")


def test_load_rejects_renamed_probe_columns(synth_data, tmp_path):
    """只把链路与探针的列名一致改掉（值、列序、统计量均未动）时必须拒绝。"""
    X_train, y_train, X_test, _ = synth_data
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    op = imb.load_operating_point(path)

    head = cal.base_pipeline(op["pipeline"]).steps[0][1]
    orig = list(head.feature_names_in_)
    renamed = [f"renamed_{i}" for i in range(len(orig))]
    head.feature_names_in_ = np.asarray(renamed, dtype=object)
    op["probe_input"] = pd.DataFrame(op["probe_input"].values, columns=renamed)
    op["payload_digest"] = imb.payload_digest(op)   # 攻击者可重算字段摘要

    # 前提：改名之后，拿原始列名的真实数据已经不能用了
    with pytest.raises(ValueError, match="feature names"):
        imb.apply_operating_point(op, X_test[orig])

    with open(path, "wb") as f:
        pickle.dump(op, f)
    with pytest.raises(ValueError, match="行为指纹不符"):
        imb.load_operating_point(path, manifest=None)


def test_probe_digest_column_boundaries_unambiguous():
    """不同的列名分组必须得到不同的摘要，与列名中含什么字符无关。"""
    values = np.arange(4, dtype=np.float64).reshape(2, 2)
    ambiguous_pairs = [
        (["a\x00b", "c"], ["a", "b\x00c"]),
        (["ab", "c"], ["a", "bc"]),
        (["f1", "f2"], ["f2", "f1"]),
    ]
    for left, right in ambiguous_pairs:
        a = imb._sha16_frame(pd.DataFrame(values, columns=left))
        b = imb._sha16_frame(pd.DataFrame(values, columns=right))
        assert a != b, f"列名 {left} 与 {right} 摘要相同：列名边界有歧义"


def test_explicit_manifest_without_entry_fails_loud(synth_data, tmp_path):
    """显式指定的清单存在、却没登记这份产物时必须报错，不得降级为告警。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    empty = tmp_path / "empty_manifest.json"
    empty.write_text(json.dumps({"models": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="未登记产物"):
        imb.load_operating_point(path, manifest=empty)


def test_require_manifest_rejects_degraded_paths(synth_data, tmp_path):
    """require_manifest=True：任何取不到清单登记的情况都必须报错。"""
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    imb.load_operating_point(path, require_manifest=True)   # 清单齐备：放行

    with pytest.raises(ValueError, match="require_manifest"):
        imb.load_operating_point(path, manifest=None, require_manifest=True)

    (tmp_path / imb.COMPARISON_JSON).unlink()
    with pytest.raises(ValueError, match="未找到外部清单"):
        imb.load_operating_point(path, require_manifest=True)


def test_artifact_records_runtime_env_for_diagnosis(synth_data, tmp_path):
    """产物记录生成环境，且该记录只用于分诊，环境不同不影响放行。"""
    path, res = _run_and_get_artifact(tmp_path, synth_data)
    op = imb.load_operating_point(path)
    for key in ("python", "numpy", "sklearn", "imblearn", "platform",
                "libc", "threadpools"):
        assert op["runtime_env"].get(key), f"产物未记录运行环境项: {key}"
    assert res["runtime_env"] == op["runtime_env"], "对比结果 json 应同样留痕"

    hint = imb._env_diff_hint(op)   # 当前环境即产物生成环境，无差异
    assert "可排除" not in hint, f"无差异提示不得超出快照能证明的范围: {hint}"

    # 环境记录被改写不改变行为 → 只能由推理字段摘要拦下（而非被当成行为不符）
    op["runtime_env"] = dict(op["runtime_env"], sklearn="0.0.0-fake")
    with open(path, "wb") as f:
        pickle.dump(op, f)
    with pytest.raises(ValueError, match="推理字段摘要不符"):
        imb.load_operating_point(path, manifest=None)


# 工作点产物内部对象：直接取出它们就等于绕开受控入口自己出预测
_OP_INTERNALS = ("pipeline", "probe_input")
_DESERIALIZERS = {("pickle", "load"), ("pickle", "loads"), ("joblib", "load")}


def _const_str(node):
    """取常量字符串节点的值，非常量字符串返回 None。"""
    return node.value if isinstance(node, ast.Constant) and isinstance(
        node.value, str) else None


def _deployment_path_offenders(source: str) -> list[str]:
    """AST 扫描源码，返回绕过受控部署入口的写法（best-effort 属于静态提示，不是保证面）。"""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript):
            key = _const_str(node.slice)
            if key in _OP_INTERNALS:
                offenders.append(f'直接取用工作点内部对象 ["{key}"]')
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if (isinstance(func.value, ast.Name)
                        and (func.value.id, func.attr) in _DESERIALIZERS):
                    offenders.append(
                        f"自行反序列化产物 {func.value.id}.{func.attr}()（绕过加载校验）")
                if func.attr in ("get", "pop", "setdefault") and node.args:
                    key = _const_str(node.args[0])
                    if key in _OP_INTERNALS:
                        offenders.append(f'直接取用工作点内部对象 .{func.attr}("{key}")')
        if (isinstance(node, ast.Name) and node.id == "_decide") or (
                isinstance(node, ast.Attribute) and node.attr == "_decide") or (
                isinstance(node, ast.FunctionDef) and node.name == "_decide"):
            offenders.append("引用或重定义了私有决策函数 _decide")
    return sorted(set(offenders))


def test_deployment_scanner_catches_known_bypasses():
    """先证明扫描器抓得住已知的绕过写法，再拿它去扫仓库：零命中不等于没问题。"""
    bypasses = {
        "加载后直接 predict": (
            'op = imb.load_operating_point(p)\n'
            'y = op["pipeline"].predict(X)\n'),
        "自行反序列化后直接 predict": (
            'import pickle\n'
            'op = pickle.load(open(p, "rb"))\n'
            'y = op["pipeline"].predict(X)\n'),
        "经 .get 取链路": (
            'op = imb.load_operating_point(p)\n'
            'y = op.get("pipeline").predict(X)\n'),
        "经 .setdefault 取链路": (
            'op = imb.load_operating_point(p)\n'
            'y = op.setdefault("pipeline", None).predict(X)\n'),
        "joblib 反序列化": 'import joblib\nop = joblib.load(p)\n',
        "复用私有决策函数": (
            'from src.imbalance import _decide\n'
            'y = _decide(pl, X, t, s)\n'),
        "复制一份决策函数": 'def _decide(pipeline, X, threshold, space):\n    return 0\n',
        # 以下三种把自行反序列化写成了别名形态，第 2 类判定认不出；但下一步都写成
        # op["pipeline"] 直接下标，落在第 1 类命中集内，与用什么名字导入无关
        "from import 别名 + 取裸链路": (
            'from pickle import load\n'
            'op = load(open(p, "rb"))\n'
            'y = op["pipeline"].predict(X)\n'),
        "模块别名 + 取裸链路": (
            'import pickle as serializer\n'
            'op = serializer.load(open(p, "rb"))\n'
            'y = op["pipeline"].predict(X)\n'),
        "产物当参数传入后取裸链路": (
            'def serve(op, X):\n'
            '    return op["pipeline"].predict(X)\n'),
    }
    for name, src in bypasses.items():
        assert _deployment_path_offenders(src), f"扫描器漏掉了绕过写法: {name}"

    sanctioned = (
        '"""工作点存在 outputs/imbalance_operating_point_<模型>.pkl。"""\n'
        'op = imb.load_operating_point(path, require_manifest=True)\n'
        'y = imb.apply_operating_point(op, X)\n'
        'cfg["pipeline_steps"] = 3\n'
    )
    assert not _deployment_path_offenders(sanctioned), (
        "受控入口的正确用法被误判，扫描器会红到失去信息量")

    # 无关字典同名键会命中，是有意选的方向而非缺陷：消除该误报要改成只认加载函数赋值来的
    # 名字，那会把"产物当参数传入"从命中变漏检。钉成断言，防后续顺手优化掉取舍而不知代价。
    assert _deployment_path_offenders('steps = cfg["pipeline"]\n'), (
        "键名判定被收窄了：请先确认没有把参数传递形态一并漏掉")


def test_scanner_boundary_matches_documented_ast_rules():
    """把 _deployment_path_offenders 的边界钉成断言：命中集与补集逐例实测必须相符。"""
    in_rule = {
        "规则1 常量字符串下标": 'y = op["pipeline"].predict(X)',
        "规则1 被取值对象不限": 'y = op.copy()["pipeline"].predict(X)',
        "规则1 另一个受保护键": 'X = op["probe_input"]',
        "规则2 .get 首参为常量键": 'y = op.get("pipeline").predict(X)',
        "规则2 .pop 首参为常量键": 'y = op.pop("pipeline").predict(X)',
        "规则2 .setdefault 首参为常量键": (
            'y = op.setdefault("pipeline", None).predict(X)'),
    }
    out_of_rule = {
        "__getitem__ 取常量键": 'y = op.__getitem__("pipeline").predict(X)',
        "非绑定 dict.__getitem__": (
            'y = dict.__getitem__(op, "pipeline").predict(X)'),
        ".get 但首参是对象不是键": 'y = dict.get(op, "pipeline").predict(X)',
        "itemgetter 取常量键": (
            'import operator\n'
            'y = operator.itemgetter("pipeline")(op).predict(X)'),
        "映射模式匹配": (
            'match op:\n'
            '    case {"pipeline": pipeline}:\n'
            '        y = pipeline.predict(X)'),
        "getattr 动态取方法": 'y = getattr(op, "get")("pipeline").predict(X)',
        "推导式按常量键筛 items()": (
            'y = [v for k, v in op.items() if k == "pipeline"][0].predict(X)'),
        "解包成关键字参数": (
            'def serve(pipeline, **kw):\n'
            '    return pipeline.predict(X)\n'
            'y = serve(**op)'),
        "变量当键": 'k = "pipeline"\ny = op[k].predict(X)',
        "绕开键名取值": 'y = list(op.values())[0].predict(X)',
        "遍历取值": 'for k, v in op.items():\n    y = v.predict(X)',
    }
    for name, src in in_rule.items():
        assert _deployment_path_offenders(src), (
            f"边界说明称命中、实测却漏: {name}，实现被收窄，"
            "请同步 _deployment_path_offenders 的 docstring")
    for name, src in out_of_rule.items():
        assert not _deployment_path_offenders(src), (
            f"边界说明称不命中、实测却命中: {name}，实现被扩宽，"
            "请同步 _deployment_path_offenders 的 docstring")


def _files_to_scan():
    """递归遍历仓库内所有会被部署路径复用的源码文件。"""
    skip_dirs = {"tests", "vvsecom", "outputs", "__pycache__", ".git"}
    entry = ROOT / "src" / "imbalance.py"
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if fn.endswith(".py") and p != entry:
                yield p


def test_no_second_deployment_entry_point():
    """仓库内不得出现第二条工作点部署入口（静态提示层）。"""
    offenders = []
    for p in _files_to_scan():
        offenders += [f"{p.relative_to(ROOT)}: {hit}"
                      for hit in _deployment_path_offenders(
                          p.read_text(encoding="utf-8"))]
    assert not offenders, (
        "发现受控入口之外的部署路径: " + "；".join(offenders)
        + "。工作点只能经 load_operating_point / apply_operating_point 使用")


def test_deployment_scanner_scope_covers_subpackages(tmp_path, monkeypatch):
    """扫描范围本身也要故障注入：往子包与仓库根各放一份绕过写法，必须都被抓住。"""
    (tmp_path / "src" / "service").mkdir(parents=True)
    (tmp_path / "src" / "imbalance.py").write_text(
        'def _decide(pipeline, X, threshold, space):\n    return 0\n',
        encoding="utf-8")                       # 受控入口自身：必须被豁免
    (tmp_path / "src" / "service" / "inference.py").write_text(
        'op = load(open(p, "rb"))\ny = op["pipeline"].predict(X)\n',
        encoding="utf-8")
    (tmp_path / "serve.py").write_text(
        'import joblib\nop = joblib.load(p)\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        'op = pickle.load(f)\n', encoding="utf-8")   # 负例注入场：必须被豁免

    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    scanned = {p.relative_to(tmp_path).as_posix() for p in _files_to_scan()}
    assert scanned == {"serve.py", "src/service/inference.py"}, \
        f"扫描范围不符：实际 {sorted(scanned)}"


def test_self_deserialized_artifact_cannot_be_applied(synth_data, tmp_path):
    """自行反序列化的产物不得出预测，该判据与代码怎么写、放在哪个目录无关。"""
    _, _, X_test, _ = synth_data
    path, _ = _run_and_get_artifact(tmp_path, synth_data)

    raw = pickle.loads(path.read_bytes())    # 跳过受控入口：三层校验一层没跑
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        imb.apply_operating_point(raw, X_test)

    good = imb.load_operating_point(path)
    expected = imb.apply_operating_point(good, X_test)
    assert (imb.apply_operating_point(imb._mark_loaded(raw), X_test)
            == expected).all(), "补上来源标记后行为应完全一致：该校验只判来源，不改决策路径"

    resaved = tmp_path / "resaved.pkl"
    with open(resaved, "wb") as f:
        pickle.dump(good, f)                 # 落盘的是**带标记**的那份
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        imb.apply_operating_point(pickle.loads(resaved.read_bytes()), X_test)


def test_self_deserialized_artifact_cannot_use_score_or_input_name_helpers(
        synth_data, tmp_path):
    """列名与分数同样只能经受控入口读取，不能靠自行反序列化绕过。"""
    _, _, X_test, _ = synth_data
    path, _ = _run_and_get_artifact(tmp_path, synth_data)
    raw = pickle.loads(path.read_bytes())
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        imb.score_operating_point(raw, X_test)
    with pytest.raises(ValueError, match="未经 load_operating_point"):
        imb.operating_point_input_names(raw)

    good = imb.load_operating_point(path)
    scores, space = imb.score_operating_point(good, X_test)
    expected_scores, expected_space = score_samples(good["pipeline"], X_test)
    assert space == expected_space
    assert np.array_equal(scores, expected_scores)
    assert imb.operating_point_input_names(good) == list(good["pipeline"].feature_names_in_)


def test_apply_operating_point_score_space_guard(synth_data, tmp_path):
    """产物记录的分数空间与链路实际不符 → 拒绝出预测。"""
    X_train, y_train, X_test, y_test = synth_data
    path, _ = _run_and_get_artifact(
        tmp_path, synth_data, models=["logistic"], strategies=["threshold_moving"])
    op = imb.load_operating_point(path)
    assert op["threshold"] is not None
    op["score_space"] = "decision_margin"  # 与逻辑回归实际的 probability 不符
    with pytest.raises(ValueError, match="分数空间"):
        imb.apply_operating_point(op, X_test)


def test_main_flow_has_no_threshold_config_path():
    """阈值配置接口已整体移除，不存在被误填的入口。"""
    for gone in ("resolve_decision_threshold", "assert_threshold_matches_chain",
                 "operating_point_signature", "fitted_pipeline_fingerprint",
                 "_FITTED_ATTRS"):
        assert not hasattr(imb, gone), f"{gone} 应已随旧部署路径一并移除"
    cfg_text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "decision_threshold:" not in cfg_text, "config 不应再有阈值配置键"


def test_main_strategy_sampling_fails_loud():
    """主流程配置 smote/undersample 必须显式报错，不得静默退化为裸模型。"""
    for bad in ("smote", "undersample"):
        cfg = {"imbalance": {"strategy": bad}, "model": {"active": "all"}}
        with pytest.raises(ValueError, match="仅支持"):
            get_models(cfg, SEED)
    cfg_ok = {"imbalance": {"strategy": "none"}, "model": {"active": "logistic"}}
    models = get_models(cfg_ok, SEED)
    assert list(models) == ["logistic"]
    assert models["logistic"].class_weight is None


def test_main_pipeline_fits_each_model_once(monkeypatch, tmp_path):
    """主流程 quick 模式下没有任何 estimator 实例被重复 fit。"""
    from collections import Counter

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    from src.config import load_config
    from src import pipeline as pl

    cfg = load_config(str(ROOT / "config.yaml"))
    for key in ("features_path", "labels_path"):
        if not (ROOT / cfg["data"][key]).resolve().exists():
            pytest.skip(f"缺少数据文件 {cfg['data'][key]}，跳过主流程集成测试")
    cfg["run"]["quick"] = True
    cfg["explain"]["enabled"] = False       # 本测试只关心训练次数
    # 产物写临时目录：本测试只数 fit 次数，写进真实 outputs/ 会用 quick 产物
    # 覆盖掉 verify 阶段 5 刚跑出来的 full 产物（temporal_metrics.json 首当其冲）
    cfg["output"]["results_dir"] = str(tmp_path / "out")

    fit_log, refs = [], []                  # refs 保引用，防 id 被回收复用

    def make_counting(cls, orig):
        def counting_fit(self, X, y=None, **kw):
            refs.append(self)
            fit_log.append((cls.__name__, id(self)))
            return orig(self, X, y, **kw)
        return counting_fit

    for cls in (LogisticRegression, RandomForestClassifier, RidgeClassifier):
        monkeypatch.setattr(cls, "fit", make_counting(cls, cls.fit))

    pl.run_pipeline(cfg, quick=True)

    counts = Counter(fit_log)
    assert counts, "未捕获到任何模型 fit 调用"
    repeated = {k: v for k, v in counts.items() if v > 1}
    assert not repeated, f"存在被重复 fit 的 estimator 实例: {repeated}"


def _dep_stub(**over):
    """部署块入参的最小骨架：只带 _deployment_block 真正读的字段。"""
    r = {"score_space": "probability", "threshold": None,
         "main_equivalent_strategy": None,
         "calibration": {"enabled": False, "method": None,
                         "cv_protocol": None, "positive_class": 1},
         "decision_graph": ["impute", "scale", "select", "model", "argmax"],
         "probe_coverage": {"n_rows": 4, "distinct_scores": 4,
                            "straddles_threshold": None},
         "behavior_probe": {"seed": 1, "n_rows": 4, "score_space": "probability",
                            "scores_sha256": "abc", "predictions_sha256": "def"}}
    r.update(over)
    return r


def test_deployment_block_variants():
    """部署块四分支：校准类/含阈值/采样类均不可经配置复现，纯策略类可沿用训练策略。"""
    r_thr = _dep_stub(threshold=0.123, main_equivalent_strategy="none")
    dep = imb._deployment_block("threshold_moving", "logistic", r_thr, "op.pkl")
    assert dep["supported_in_main"] is False and dep["main_config"] is None
    assert dep["artifact"] == "op.pkl" and dep["note"]
    assert dep["reproduces_experiment_metrics_in_main"] is False
    assert dep["threshold_via_config"] is False

    r_cw = _dep_stub(main_equivalent_strategy="class_weight")
    dep_cw = imb._deployment_block("class_weight", "ridge", r_cw, "op.pkl")
    assert dep_cw["supported_in_main"] is True
    assert dep_cw["main_config"] == {"strategy": "class_weight"}
    assert dep_cw["reproduces_experiment_metrics_in_main"] is False, "特征链路不同，不承诺指标一致"

    r_sm = _dep_stub()
    dep_sm = imb._deployment_block("smote", "ridge", r_sm, "op.pkl")
    assert dep_sm["supported_in_main"] is False and dep_sm["main_config"] is None

    # 校准类单独一支：主流程没有校准层，说明里必须点出分数空间与成本假设的绑定
    r_cal = _dep_stub(
        threshold=0.07, score_space="calibrated_probability",
        calibration={"enabled": True, "method": "sigmoid",
                     "cv_protocol": "StratifiedKFold(n_splits=5, shuffle=True, "
                                    "random_state=42)", "positive_class": 1},
        decision_graph=["impute", "scale", "select", "model", "calibrator", "threshold"])
    dep_cal = imb._deployment_block(imb.CALIBRATED_PRIMARY, "ridge", r_cal, "op.pkl")
    assert dep_cal["supported_in_main"] is False and dep_cal["main_config"] is None
    assert "校准器" in dep_cal["note"] and "cost_sensitivity" in dep_cal["note"]
    assert dep_cal["calibration"]["method"] == "sigmoid"
    assert dep_cal["decision_graph"][-2:] == ["calibrator", "threshold"]


# ---------- 6. 端到端产物 ----------

def test_run_comparison_outputs(synth_data, tmp_path):
    X_train, y_train, X_test, y_test = synth_data
    res = imb.run_imbalance_comparison(
        _cfg(), X_train, y_train, X_test, y_test, tmp_path, SEED)

    assert res["status"] == "ok"
    jp = tmp_path / imb.COMPARISON_JSON
    assert jp.exists() and (tmp_path / imb.COMPARISON_MD).exists()
    saved = json.loads(jp.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["selection_basis"] == "train_cv_expected_cost"
    assert saved["n_features"] == N_FEATURES

    m = saved["models"]["逻辑回归"]
    assert set(m["strategies"]) == set(imb.STRATEGIES), "每种策略结果齐全"
    for strat, r in m["strategies"].items():
        assert set(r["test"]) == {"准确率", "精确率", "召回率", "F1分数", "BER", "AUC"}
        assert {"CV_BER均值", "CV_BER标准差", "CV召回率均值",
                "CV混淆", "CV期望代价"} <= set(r["cv"])
        assert "scores" not in r and "y_pred" not in r, "绘图用数组不得落盘"
        assert r["expected_cost"] >= 0
    tm = m["strategies"]["threshold_moving"]
    assert tm["threshold"] is not None and tm["threshold_selected_on"] == "train_oof"
    assert tm["score_space"] == "probability"
    assert "delta_vs_baseline" in tm and "期望代价变化" in tm["delta_vs_baseline"]
    assert m["best_by_cv_cost"] in imb.STRATEGIES
    assert "deployment" in m and isinstance(m["deployment"]["supported_in_main"], bool)
    for key in ("pr_curve", "confusion", "operating_point"):
        assert (tmp_path / m["artifacts"][key]).exists(), f"产物缺失: {key}"
    # 产物文件哈希记在 json（信任锚必须在产物之外），且与磁盘上的文件一致
    sha = m["artifacts"]["operating_point_sha256"]
    assert sha == imb.artifact_sha256(tmp_path / m["artifacts"]["operating_point"])
    # 每个策略都记录了折内实际所选特征与整条已拟合链路的行为指纹
    for strat, r in m["strategies"].items():
        assert len(r["selected_features"]) == N_FEATURES
        assert "C0" not in r["selected_features"], "常量列不应被 F 检验选中"
        probe = r["behavior_probe"]
        assert len(probe["scores_sha256"]) == 16 and probe["n_rows"] == imb.PROBE_ROWS


def test_operating_point_artifact_reproduces_metrics(synth_data, tmp_path):
    """闭环证据：加载落盘的工作点重新预测，混淆矩阵逐位复现。"""
    X_train, y_train, X_test, y_test = synth_data
    res = imb.run_imbalance_comparison(
        _cfg(), X_train, y_train, X_test, y_test, tmp_path, SEED)
    m = res["models"]["逻辑回归"]
    best = m["best_by_cv_cost"]
    pkl = tmp_path / m["artifacts"]["operating_point"]
    assert pkl.exists()

    op = imb.load_operating_point(pkl)   # 含清单哈希核对 + 产物自检
    assert op["strategy"] == best and op["model"] == "logistic"

    y_pred = imb.apply_operating_point(op, X_test)
    _, cm = imb.expected_cost(y_test, y_pred, 10.0, 1.0)
    assert cm == m["strategies"][best]["confusion"], "加载产物后应逐位复现落盘混淆矩阵"
    assert cm == op["test_confusion"]

    assert op["selected_features"] == m["strategies"][best]["selected_features"]
    assert op["probe"] == m["strategies"][best]["behavior_probe"]
    assert op["probe"] == imb.behavior_fingerprint(
        op["pipeline"], op["probe_input"], threshold=op["threshold"],
        score_space=op["score_space"], seed=op["probe"]["seed"])
    # 探针输入随产物走，且其摘要（值 + 列名）进了推理字段摘要：替换探针输入或只改列名
    # 都会被发现
    assert op["probe"]["input_sha256"] == imb._sha16_frame(op["probe_input"])
    # 记录的最终预测摘要就是上线路径的输出摘要（自检与上线同源）
    assert op["probe"]["predictions_sha256"] == imb._sha16(
        imb.apply_operating_point(op, op["probe_input"]))


def test_run_comparison_best_selected_on_train_cv(synth_data, tmp_path):
    """最优工作点必须等于训练侧 CV 期望代价的 argmin，测试集不参与选择。"""
    X_train, y_train, X_test, y_test = synth_data
    res = imb.run_imbalance_comparison(
        _cfg(), X_train, y_train, X_test, y_test, tmp_path, SEED)
    m = res["models"]["逻辑回归"]
    # 选择池 = 可进交付的那些策略：同表对照与正确性未过关的校准行被排除在外
    pool = m["selection_pool"]
    assert set(pool) <= set(m["strategies"]) and pool
    assert imb.CALIBRATED_CONTRAST not in pool
    cv_costs = {s: m["strategies"][s]["cv"]["CV期望代价"] for s in pool}
    assert m["best_by_cv_cost"] == min(cv_costs, key=lambda k: (cv_costs[k], k))


def test_run_comparison_ridge_decision_margin(synth_data, tmp_path):
    """无 predict_proba 的模型走决策间隔空间，阈值移动仍可用。"""
    X_train, y_train, X_test, y_test = synth_data
    cfg = _cfg(models=["ridge"], strategies=["baseline", "threshold_moving"])
    res = imb.run_imbalance_comparison(
        cfg, X_train, y_train, X_test, y_test, tmp_path, SEED)
    tm = res["models"]["岭分类器"]["strategies"]["threshold_moving"]
    assert tm["score_space"] == "decision_margin"
    assert tm["threshold"] is not None


def test_run_comparison_disabled_leaves_record(synth_data, tmp_path):
    X_train, y_train, X_test, y_test = synth_data
    cfg = _cfg(enabled=False)
    res = imb.run_imbalance_comparison(
        cfg, X_train, y_train, X_test, y_test, tmp_path, SEED)
    assert res["status"] == "disabled"
    saved = json.loads((tmp_path / imb.COMPARISON_JSON).read_text(encoding="utf-8"))
    assert saved["status"] == "disabled" and saved.get("reason")


def test_run_comparison_unknown_strategy_fails_loud(synth_data, tmp_path):
    X_train, y_train, X_test, y_test = synth_data
    cfg = _cfg(strategies=["baseline", "magic"])
    with pytest.raises(ValueError, match="未知不平衡策略"):
        imb.run_imbalance_comparison(
            cfg, X_train, y_train, X_test, y_test, tmp_path, SEED)

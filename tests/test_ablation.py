"""消融实验专项测试：合成数据，秒级完成，不依赖真实数据文件。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # 使 `src` 可导入，无需安装为包

from src import ablation as ab  # noqa: E402
from src.explain import shap_feature_ranking  # noqa: E402
from src.modeling import make_model  # noqa: E402

SEED = 42
N_FEATURES = 5  # 消融每格保留的特征数，小于总列数才谈得上"在选"


@pytest.fixture(scope="module")
def synth_data():
    """raw 形态的不平衡二分类合成数据：未标准化、注入缺失值。"""
    X, y = make_classification(
        n_samples=240, n_features=10, n_informative=6, weights=[0.85, 0.15],
        flip_y=0.02, random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    X = X * rng.uniform(0.5, 30.0, size=X.shape[1]) + rng.uniform(-5.0, 5.0, size=X.shape[1])
    X[rng.rand(*X.shape) < 0.05] = np.nan
    X = pd.DataFrame(X, columns=[f"F{i:03d}" for i in range(X.shape[1])])
    y = pd.Series(y, name="label")
    n_test = 60
    return (X.iloc[:-n_test], y.iloc[:-n_test], X.iloc[-n_test:], y.iloc[-n_test:])


def _cfg(**ablation_overrides):
    """消融实验用完整配置（含 preprocessing / feature_selection / explain 子节）。"""
    ab_cfg = {"enabled": True, "models": ["logistic"]}
    ab_cfg.update(ablation_overrides)
    return {
        "preprocessing": {"impute_strategy": "median", "scale": True},
        # RFE 在合成数据上也不慢，但四方法都跑一遍会拖慢整个测试模块，故只用两种
        "feature_selection": {
            "n_features_to_select": N_FEATURES,
            "methods": ["f_test", "random_forest"],
        },
        "imbalance": {"strategy": "class_weight"},
        "costs": {"fn": 10.0, "fp": 1.0},
        "explain": {"background_size": 40},
        "ablation": ab_cfg,
    }


def _filled_scaled(X_train, X_test):
    """SHAP 排名直测用：训练集中位数填充 + 训练集统计量标准化。"""
    med = X_train.median()
    tr, te = X_train.fillna(med), X_test.fillna(med)
    mu, sd = tr.mean(), tr.std().replace(0.0, 1.0)
    return (tr - mu) / sd, (te - mu) / sd


# ---------- 1. SHAP 特征排名 ----------

def test_shap_ranking_returns_all_features_in_order(synth_data):
    """不给 top_k 时返回全部特征，且按平均 |SHAP| 降序（重排后仍是同一集合）。"""
    X_train, y_train, X_test, _ = synth_data
    tr, _te = _filled_scaled(X_train, X_test)
    model = make_model("logistic", SEED, "balanced").fit(tr, y_train)
    ranked, method = shap_feature_ranking(
        model, tr, list(tr.columns), tr.head(40), seed=SEED)
    assert set(ranked) == set(tr.columns)
    assert len(ranked) == tr.shape[1]
    assert method == "linear"  # 逻辑回归应走解析解，不该退化到 kernel


def test_shap_ranking_respects_top_k(synth_data):
    """top_k 截断后必须是完整排名的前缀，不是随便取的子集。"""
    X_train, y_train, X_test, _ = synth_data
    tr, _te = _filled_scaled(X_train, X_test)
    model = make_model("logistic", SEED, "balanced").fit(tr, y_train)
    full, _ = shap_feature_ranking(model, tr, list(tr.columns), tr.head(40), seed=SEED)
    top, _ = shap_feature_ranking(
        model, tr, list(tr.columns), tr.head(40), seed=SEED, top_k=N_FEATURES)
    assert top == full[:N_FEATURES]


def test_shap_ranking_rejects_column_mismatch(synth_data):
    """待解释数据与背景列序不一致必须显式失败，否则会把 SHAP 值配到错的特征上。"""
    X_train, y_train, X_test, _ = synth_data
    tr, _te = _filled_scaled(X_train, X_test)
    model = make_model("logistic", SEED, "balanced").fit(tr, y_train)
    shuffled = tr.head(40)[list(reversed(tr.columns))]
    with pytest.raises(ValueError, match="特征列不一致"):
        shap_feature_ranking(model, tr, list(tr.columns), shuffled, seed=SEED)


def test_shap_ranking_rejects_empty_background(synth_data):
    """背景为空时 SHAP 基线无意义，必须报错而非算出一份看似正常的排名。"""
    X_train, y_train, X_test, _ = synth_data
    tr, _te = _filled_scaled(X_train, X_test)
    model = make_model("logistic", SEED, "balanced").fit(tr, y_train)
    with pytest.raises(ValueError, match="背景数据为空"):
        shap_feature_ranking(model, tr, list(tr.columns), tr.head(0), seed=SEED)


# ---------- 2. 2x2 网格的完整性 ----------

def test_grid_has_all_four_cells(synth_data, tmp_path):
    """四个格全部产出，且每格记录的特征数等于配置值。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    assert res["status"] == "ok"
    cells = res["models"]["logistic"]["cells"]
    assert set(cells) == {ab.cell_key(s, w) for s in ab.SELECTORS for w in ab.WEIGHTINGS}
    for cell in cells.values():
        assert cell["n_features"] == N_FEATURES
        assert set(cell["confusion"]) == {"tn", "fp", "fn", "tp"}
        assert cell["expected_cost"] == pytest.approx(
            cell["confusion"]["fn"] * 10.0 + cell["confusion"]["fp"] * 1.0)


def test_vote_cells_share_one_feature_set(synth_data, tmp_path):
    """投票选特征与训练策略无关，两个 vote 格必须用同一份特征集。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    feats = res["models"]["logistic"]["features"]
    assert feats[ab.cell_key("vote", "none")] == feats[ab.cell_key("vote", "class_weight")]
    assert feats[ab.cell_key("vote", "none")] == res["features_vote"]


def test_effects_match_cell_differences(synth_data, tmp_path):
    """effects 里的三个差值必须由对应两格的指标算出，不能是另行统计的数字。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    m = res["models"]["logistic"]
    cells, eff = m["cells"], m["effects"]
    pairs = {
        "weighting_on_vote": (("vote", "none"), ("vote", "class_weight")),
        "selector_on_none": (("vote", "none"), ("shap", "none")),
        "selector_on_weighted": (("vote", "class_weight"), ("shap", "class_weight")),
    }
    for key, (a, b) in pairs.items():
        ca, cb = cells[ab.cell_key(*a)], cells[ab.cell_key(*b)]
        assert eff[key]["BER变化"] == pytest.approx(cb["测试集BER"] - ca["测试集BER"])
        assert eff[key]["召回变化"] == pytest.approx(cb["召回率"] - ca["召回率"])
        assert eff[key]["漏检变化"] == cb["confusion"]["fn"] - ca["confusion"]["fn"]


def test_artifacts_written(synth_data, tmp_path):
    """json / md / png 三份产物齐备，json 可解析且与返回值一致。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    for fname in ("ablation_comparison.json", "ablation_comparison.md",
                  res["artifacts"]["grid_png"]):
        assert (tmp_path / fname).exists(), f"缺产物 {fname}"
    on_disk = json.loads((tmp_path / "ablation_comparison.json").read_text(encoding="utf-8"))
    assert on_disk["models"]["logistic"]["cells"] == res["models"]["logistic"]["cells"]


def test_disabled_records_status(synth_data, tmp_path):
    """开关关闭时落盘 disabled 状态而非什么都不写，验收脚本才能区分"关了"与"跑挂了"。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(enabled=False), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    assert res["status"] == "disabled"
    on_disk = json.loads((tmp_path / "ablation_comparison.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "disabled"


# ---------- 3. 防泄漏：测试集不得影响任何选择 ----------

def test_test_set_does_not_affect_feature_selection(synth_data, tmp_path):
    """替换测试集特征值后，两种特征集与探针信息必须逐位不变（下方反向断言防空转）。"""
    X_train, y_train, X_test, y_test = synth_data
    rng = np.random.RandomState(0)
    X_test_perturbed = X_test + rng.normal(0, 50.0, size=X_test.shape)

    base = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                           tmp_path / "a", SEED, quick=True)
    pert = ab.run_ablation(_cfg(), X_train, y_train, X_test_perturbed, y_test,
                           tmp_path / "b", SEED, quick=True)

    assert base["features_vote"] == pert["features_vote"]
    assert base["models"]["logistic"]["features"] == pert["models"]["logistic"]["features"]
    assert base["models"]["logistic"]["shap_probe"] == pert["models"]["logistic"]["shap_probe"]
    # 反向确认扰动确实进到了链路里：测试集指标应当被它改变，否则上面的相等是空转
    assert (base["models"]["logistic"]["cells"][ab.cell_key("vote", "class_weight")]
            != pert["models"]["logistic"]["cells"][ab.cell_key("vote", "class_weight")])


def test_shap_probe_records_class_weight_per_cell(synth_data, tmp_path):
    """探针的 class_weight 必须跟随本格设置，否则"选特征"与"训练"两个口径不一致。"""
    X_train, y_train, X_test, y_test = synth_data
    res = ab.run_ablation(_cfg(), X_train, y_train, X_test, y_test,
                          tmp_path, SEED, quick=True)
    probes = res["models"]["logistic"]["shap_probe"]
    assert probes["none"]["probe_class_weight"] is None
    assert probes["class_weight"]["probe_class_weight"] == "balanced"
    assert probes["none"]["probe_n_features"] == X_train.shape[1]

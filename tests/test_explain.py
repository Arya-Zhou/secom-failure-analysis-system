"""SHAP 可解释性模块专项测试（合成数据，秒级，不依赖真实数据文件）。

测试原则：断言失败时输出实际值 vs 期望值，不以放宽容差或 skip 制造绿灯；
kernel 兜底链路显式覆盖（KNN 路径）。

结构：
  - 参数化核心测试：explainer 创建、SHAP 维度、非全零、解释空间内自洽断言
  - 定向产物测试：ridge（无 predict_proba）单晶圆报告；rf（tree）全局 json+png
  - 显式报错：错误列序、维度不匹配、空背景
  - TP/FN 极值选样逻辑
  - pipeline 集成：required 失败语义与 manifest 审计
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.explain import (
    _explained_output,
    _make_explainer,
    _model_score,
    _positive_class_values,
    explain_global,
    explain_wafer,
    pick_case_positions,
)
from src import pipeline as pl

SEED = 42
N_FEAT = 8
FEATURES = [f"F{i:03d}" for i in range(N_FEAT)]


def _data():
    X_arr, y_arr = make_classification(
        n_samples=150, n_features=N_FEAT, n_informative=5,
        weights=[0.85, 0.15], random_state=SEED,
    )
    return pd.DataFrame(X_arr, columns=FEATURES), pd.Series(y_arr)


X_ALL, Y_ALL = _data()
BG = X_ALL.sample(40, random_state=SEED)
POS = int(np.flatnonzero(Y_ALL.to_numpy() == 1)[0])  # 一个正类（失效）样本


@pytest.fixture(scope="module")
def fitted():
    models = {
        "ridge": RidgeClassifier(class_weight="balanced"),
        "logistic": LogisticRegression(class_weight="balanced", max_iter=2000),
        "rf": RandomForestClassifier(
            n_estimators=30, class_weight="balanced", random_state=SEED),
    }
    return {k: m.fit(X_ALL, Y_ALL) for k, m in models.items()}


# ---------- 参数化核心测试：三模型 × (创建/维度/非零/自洽) ----------

EXPECTED_METHOD = {"ridge": "linear", "logistic": "linear", "rf": "tree"}


@pytest.mark.parametrize("name", ["ridge", "logistic", "rf"])
def test_core_shap_properties(name, fitted):
    model = fitted[name]
    explainer, method, space = _make_explainer(model, BG, seed=SEED)
    assert method == EXPECTED_METHOD[name]
    assert space  # 解释空间必须有标注

    sv = _positive_class_values(explainer.shap_values(X_ALL.iloc[:20]))
    assert sv.shape == (20, N_FEAT)
    assert not np.allclose(sv, 0), f"{name}: SHAP 全为 0（背景数据错误的典型症状）"

    # 自洽断言：解释空间内 baseline + sum(SHAP) ≈ 模型输出（解析解应精确）
    x_2d = X_ALL.iloc[[POS]]
    sv1 = _positive_class_values(explainer.shap_values(x_2d))[0]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if isinstance(base, (list, np.ndarray)) else float(base)
    recon = base + float(sv1.sum())
    expected = _explained_output(model, x_2d, method)
    assert recon == pytest.approx(expected, abs=1e-6, rel=1e-4), (
        f"{name}/{method} 自洽失败: baseline+sum={recon:.6f} "
        f"解释空间输出={expected:.6f} 偏差={abs(recon - expected):.2e}"
    )


def test_kernel_fallback_path():
    """kernel 兜底链路（无 coef_ / estimators_ 的模型），采样近似容差放宽但仍须自洽。"""
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_ALL, Y_ALL)
    explainer, method, space = _make_explainer(knn, BG, seed=SEED)
    assert method == "kernel"
    assert space == "positive-class probability"

    x_2d = X_ALL.iloc[[POS]]
    sv1 = _positive_class_values(explainer.shap_values(x_2d))[0]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if isinstance(base, (list, np.ndarray)) else float(base)
    recon = base + float(sv1.sum())
    expected = _explained_output(knn, x_2d, method)
    assert recon == pytest.approx(expected, abs=1e-2), (
        f"kernel 自洽失败: baseline+sum={recon:.6f} 期望={expected:.6f}"
    )


def test_model_score_semantics(fitted):
    """展示分数语义：概率就是概率、margin 就是决策分数，不冒充。"""
    x_2d = X_ALL.iloc[[POS]]
    score, label = _model_score(fitted["logistic"], x_2d)
    assert "概率" in label and 0.0 <= score <= 1.0
    score_r, label_r = _model_score(fitted["ridge"], x_2d)
    assert "决策分数" in label_r  # ridge 无 predict_proba，不得显示为概率


# ---------- 定向产物测试 ----------

def test_ridge_wafer_artifacts(fitted, tmp_path):
    """ridge（无 predict_proba）也能产出完整单晶圆报告 + 贡献图。"""
    res = explain_wafer(
        fitted["ridge"], X_ALL.iloc[POS], int(Y_ALL.iloc[POS]), FEATURES, "ridge",
        tmp_path, background=BG, wafer_id=POS, case="FN", seed=SEED,
    )
    txt = tmp_path / f"shap_explanation_wafer_{POS}_FN.txt"
    png = tmp_path / f"shap_contribution_wafer_{POS}_FN.png"
    assert txt.exists() and txt.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    body = txt.read_text(encoding="utf-8")
    for kw in ("案例类型: FN", "SHAP 解释空间", "自洽校验", "拉动失效风险", "抑制失效风险"):
        assert kw in body, f"报告缺少关键内容: {kw}"
    assert res["consistency_ok"], (
        f"ridge 报告自洽超容差: recon={res['reconstruction']:.6f} "
        f"expected={res['explained_output']:.6f}"
    )
    assert res["output_space"] == "decision margin"


def test_rf_global_artifacts(fitted, tmp_path):
    """rf（tree）全局产物：json 特征数与元数据齐全，png 非空。"""
    res = explain_global(
        fitted["rf"], X_ALL.iloc[:30], FEATURES, "rf", tmp_path,
        background=BG, seed=SEED,
    )
    js_path = tmp_path / "shap_values_rf.json"
    png_path = tmp_path / "shap_summary_bar_rf.png"
    assert png_path.exists() and png_path.stat().st_size > 0
    data = json.loads(js_path.read_text(encoding="utf-8"))
    assert len(data["features"]) == N_FEAT, (
        f"JSON 特征数 {len(data['features'])} != 输入特征数 {N_FEAT}")
    for key in ("shap_method", "output_space", "random_seed",
                "background_size", "n_samples_explained"):
        assert key in data, f"JSON 缺少元数据字段: {key}"
    assert data["shap_method"] == "tree"
    assert res["shap_method"] == "tree" and res["output_space"]
    assert any(v > 0 for v in res["importance"].values()), "全局 SHAP 全为 0"


# ---------- 显式报错 ----------

def test_input_validation_raises(fitted, tmp_path):
    # 维度不匹配
    with pytest.raises(ValueError):
        explain_global(fitted["ridge"], X_ALL, FEATURES[:-1], "r", tmp_path,
                       background=BG, seed=SEED)
    # 列序错乱（错列序会导致错误归因，必须显式失败）
    with pytest.raises(ValueError):
        explain_wafer(fitted["ridge"], X_ALL.iloc[POS], 1, FEATURES, "r", tmp_path,
                      background=BG[list(reversed(FEATURES))], wafer_id=0, seed=SEED)
    # 空背景（对应 explain.background_size <= 0）
    with pytest.raises(ValueError):
        explain_global(fitted["ridge"], X_ALL, FEATURES, "r", tmp_path,
                       background=BG.iloc[0:0], seed=SEED)


# ---------- TP/FN 极值选样逻辑 ----------

def test_pick_case_positions():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.2, 0.1, 0.8])
    picks = pick_case_positions(y_true, y_pred, scores)
    assert picks["TP"] == 0          # 唯一 TP
    assert picks["FN"] == 1          # 两个 FN 中分数最高者（最接近阈值）

    # 无 FN 时返回 None（调用方记录 warning，跳过不算失败）
    picks2 = pick_case_positions(np.array([1, 0]), np.array([1, 0]), np.array([0.7, 0.1]))
    assert picks2["TP"] == 0
    assert picks2["FN"] is None


# ---------- SHAP 新旧输出格式转换 ----------

def test_positive_class_values_formats():
    two_d = np.ones((3, 4))
    assert _positive_class_values(two_d).shape == (3, 4)

    as_list = [np.zeros((3, 4)), np.ones((3, 4))]        # 老接口：[负类, 正类]
    out = _positive_class_values(as_list)
    assert out.shape == (3, 4) and np.all(out == 1)

    three_d = np.stack([np.zeros((3, 4)), np.ones((3, 4))], axis=-1)  # 新接口 (n,f,c)
    out3 = _positive_class_values(three_d)
    assert out3.shape == (3, 4) and np.all(out3 == 1)


# ---------- pipeline 集成层：required 失败语义与 manifest 审计 ----------

def _stage_cfg(required=True, enabled=True, bg=20):
    return {"explain": {"enabled": enabled, "required": required, "background_size": bg}}


def _boom(*args, **kwargs):
    raise RuntimeError("注入的 SHAP 失败")


def test_required_true_propagates_injected_failure(fitted, tmp_path, monkeypatch):
    """针对曾实际发生的"异常被吞、验证假绿"问题的回归测试：required=true 时失败必须上抛。"""
    monkeypatch.setattr(pl, "explain_global", _boom)
    with pytest.raises(RuntimeError):
        pl._run_explain_stage(_stage_cfg(required=True), fitted["ridge"], "ridge",
                              X_ALL, X_ALL, Y_ALL, FEATURES, tmp_path, SEED, "t1")
    # finally 落盘：失败状态与顶层失败原因均须可审计
    mf = json.loads((tmp_path / "shap_manifest.json").read_text(encoding="utf-8"))
    assert mf["status"] == "failed"
    assert mf.get("error"), "顶层失败原因应写入 manifest"


def test_required_false_degrades_but_marks_failed(fitted, tmp_path, monkeypatch):
    """required=false 允许主流程继续，但 manifest 必须保留 failed 供验证层拦截。"""
    monkeypatch.setattr(pl, "explain_global", _boom)
    mf = pl._run_explain_stage(_stage_cfg(required=False), fitted["ridge"], "ridge",
                               X_ALL, X_ALL, Y_ALL, FEATURES, tmp_path, SEED, "t2")
    assert mf["status"] == "failed"
    assert mf.get("error"), "降级路径同样应记录顶层失败原因"
    on_disk = json.loads((tmp_path / "shap_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "failed"


def test_explain_stage_manifest_ok(fitted, tmp_path):
    """正常路径：manifest 状态 ok，generated 案例文件齐全且自洽，skipped 必有原因。"""
    mf = pl._run_explain_stage(_stage_cfg(), fitted["ridge"], "ridge",
                               X_ALL, X_ALL, Y_ALL, FEATURES, tmp_path, SEED, "t3")
    assert mf["status"] == "ok"
    # manifest 顶层 schema：含解释元数据（shap_method / output_space）
    for key in ("enabled", "status", "run_tag", "model", "random_seed",
                "background_size", "shap_method", "output_space", "global", "cases"):
        assert key in mf, f"manifest 缺少顶层字段: {key}"
    assert mf["shap_method"] == "linear"
    assert mf["output_space"] == "decision margin"
    for key in ("png", "json"):
        assert (tmp_path / mf["global"][key]).exists()
    assert mf["cases"], "至少应有 TP/FN 状态记录"
    for case, info in mf["cases"].items():
        if info["status"] == "generated":
            assert (tmp_path / info["report"]).exists()
            assert (tmp_path / info["plot"]).exists()
            assert info["consistency_ok"], f"{case} 自洽未通过: {info}"
        else:
            assert info["status"] == "skipped" and info["reason"]


def test_explain_stage_disabled_writes_minimal_manifest(fitted, tmp_path):
    """enabled=false 也要落盘最小 manifest，验证层据此输出"跳过"而非报错。"""
    mf = pl._run_explain_stage({"explain": {"enabled": False}}, fitted["ridge"], "ridge",
                               X_ALL, X_ALL, Y_ALL, FEATURES, tmp_path, SEED, "t4")
    assert mf["enabled"] is False and mf["reason"]
    on_disk = json.loads((tmp_path / "shap_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["enabled"] is False

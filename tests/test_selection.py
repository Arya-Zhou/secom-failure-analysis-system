"""选择口径专项：reference 模型只能由训练侧信息选出。合成数据写成 SECOM 格式临时文件，秒级。"""
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

from src.config import canonical_costs, load_config  # noqa: E402
from src.evaluation import resolve_cv_folds  # noqa: E402
from src.pipeline import SELECTION_BASIS, SELECTION_KEY, run_pipeline  # noqa: E402
from src.validation import artifact_split  # noqa: E402

SEED = 42
N_COLS = 591     # load_secom 按 SECOM 原始维数命名列，合成数据须对齐
N_ROWS = 300
TEST_SIZE = 0.2


def _write_secom_files(X: np.ndarray, y: np.ndarray, out: Path) -> tuple[Path, Path]:
    """把数组写成 SECOM 原始格式：空格分隔的特征文件 + "标签 时间戳" 的标签文件。"""
    out.mkdir(parents=True, exist_ok=True)
    feat = out / "secom.data"
    lab = out / "secom_labels.data"
    pd.DataFrame(X).to_csv(feat, sep=" ", header=False, index=False, na_rep="NaN")
    stamps = pd.date_range("2008-07-19 11:55:00", periods=len(y), freq="h")
    lab.write_text(
        "".join(f'{1 if v == 1 else -1} "{t.strftime("%d/%m/%Y %H:%M:%S")}"\n'
                for v, t in zip(y, stamps)),
        encoding="utf-8",
    )
    return feat, lab


@pytest.fixture(scope="module")
def synth_raw() -> tuple[np.ndarray, np.ndarray]:
    """不平衡二分类合成数据，591 列（信息列在前，其余为噪声），含缺失值。"""
    X_inf, y = make_classification(
        n_samples=N_ROWS, n_features=20, n_informative=8, weights=[0.88, 0.12],
        flip_y=0.02, random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    X = np.hstack([X_inf, rng.normal(size=(N_ROWS, N_COLS - X_inf.shape[1]))])
    X = X * rng.uniform(0.5, 30.0, size=X.shape[1]) + rng.uniform(-5.0, 5.0, size=X.shape[1])
    X[rng.rand(*X.shape) < 0.03] = np.nan
    X[:, -1] = np.nan  # 全空列：让 drop_all_nan_columns 这一步也走到
    return X, y


def _cfg(feat: Path, lab: Path, out_dir: Path) -> dict:
    """跑通主流程的最小配置。quick=True 只为跳过 RFE 与下游实验，样本数不缩减。"""
    return {
        "random_state": SEED,
        "data": {"features_path": str(feat), "labels_path": str(lab),
                 "timestamp_format": "%d/%m/%Y %H:%M:%S"},
        "preprocessing": {"impute_strategy": "median", "scale": True},
        "feature_selection": {"n_features_to_select": 15,
                              "methods": ["f_test", "random_forest"],
                              "override_features_path": None},
        "costs": {"fn": 10.0, "fp": 1.0},
        "artifact_split": {"protocol": "stratified_shuffle",
                           "test_size": TEST_SIZE, "seed": SEED},
        "validation": {"resample": {"enabled": False}},
        "imbalance": {"strategy": "class_weight", "comparison": {"enabled": False}},
        "model": {"active": ["logistic", "random_forest", "ridge"], "cv_folds": 5},
        "evaluation": {"primary_metric": "ber", "fp_budget": 5},
        "explain": {"enabled": False},
        "ablation": {"enabled": False},
        "output": {"results_dir": str(out_dir), "log_level": "INFO"},
        "run": {"quick": True, "quick_sample_size": 10 ** 6},
    }


def _run(X: np.ndarray, y: np.ndarray, tmp: Path, name: str) -> dict:
    feat, lab = _write_secom_files(X, y, tmp / name)
    return run_pipeline(_cfg(feat, lab, tmp / name / "out"), quick=True)


@pytest.fixture(scope="module")
def baseline_run(synth_raw, tmp_path_factory) -> dict:
    X, y = synth_raw
    return _run(X, y, tmp_path_factory.mktemp("sel_base"), "base")


# ---------- 1. 选择依据可审计 ----------

def test_selection_block_records_basis_and_ranking(baseline_run):
    sel = baseline_run["selection"]
    assert sel["selection_basis"] == SELECTION_BASIS == "train_cv_expected_cost"
    assert sel["reference_model"] == baseline_run["reference_model"]
    assert sel["reference_model"] == sel["ranking_by_cv_cost"][0]
    assert set(sel["ranking_by_cv_cost"]) == set(baseline_run["metrics"])
    assert set(sel["ranking_by_cv_ber"]) == set(baseline_run["metrics"])
    assert sel["criteria_agree"] == (
        sel["ranking_by_cv_cost"] == sel["ranking_by_cv_ber"])
    assert sel["costs"] == {"fn": 10.0, "fp": 1.0}


def test_reference_model_is_argmin_of_train_cv_cost(baseline_run):
    """选出的必须就是训练侧 CV 代价最小的那个——把口径重算一遍，不信任落盘结论。"""
    metrics = baseline_run["metrics"]
    want = min(sorted(metrics), key=lambda k: metrics[k]["CV折外期望代价"])
    assert baseline_run["reference_model"] == want


def test_cv_cost_matches_its_own_confusion(baseline_run):
    """CV 折外期望代价必须与同一份折外混淆自洽，否则两个数字里至少一个是别处来的。"""
    fn_c, fp_c = 10.0, 1.0
    for name, m in baseline_run["metrics"].items():
        cm = m["CV折外混淆"]
        assert m["CV折外期望代价"] == pytest.approx(cm["fn"] * fn_c + cm["fp"] * fp_c), name


def test_selection_and_metrics_land_in_artifact(baseline_run):
    out = Path(baseline_run["output_dir"])
    mfile = next(out.glob("metrics_quick_*.json"))
    on_disk = json.loads(mfile.read_text(encoding="utf-8"))
    assert on_disk[SELECTION_KEY] == baseline_run["selection"]
    ref = baseline_run["reference_model"]
    assert [p.name for p in out.glob(f"reference_model_{ref}_quick_*.pkl")], (
        "reference 模型产物未按 reference_model_<模型> 命名落盘")
    assert not list(out.glob("best_model_*")), "best_model_* 是已废弃的命名"


# ---------- 2. 防泄漏：测试集不得影响选择（双向断言） ----------

def test_test_set_does_not_affect_model_selection(synth_raw, tmp_path):
    """扰动测试集后选择与训练侧指标逐位不变，而测试集指标必须变（反向断言防空转）。"""
    X, y = synth_raw
    # 与 pipeline.py 同一个划分口径（重采样总体第 0 个）；若两者失配，扰动会落到训练侧，
    # 反向断言当场转红
    y_ser = pd.Series(y)
    _, test_idx = artifact_split(y_ser, TEST_SIZE, SEED)

    X_pert = X.copy()
    rng = np.random.RandomState(0)
    X_pert[np.asarray(test_idx)] += rng.normal(0, 50.0, size=(len(test_idx), X.shape[1]))

    base = _run(X, y, tmp_path, "base")
    pert = _run(X_pert, y, tmp_path, "pert")

    assert base["reference_model"] == pert["reference_model"]
    assert base["selection"] == pert["selection"]
    assert base["selected_features"] == pert["selected_features"]
    for name in base["metrics"]:
        for key in ("CV_BER均值", "CV_BER标准差", "CV折外期望代价", "CV折外混淆", "训练集BER"):
            assert base["metrics"][name][key] == pert["metrics"][name][key], (
                f"{name}.{key} 被测试集改变了")

    changed = [n for n in base["metrics"]
               if base["metrics"][n]["测试集BER"] != pert["metrics"][n]["测试集BER"]]
    assert changed, "扰动没有改变任何测试集指标，说明它压根没进到链路里（上面的相等是空转）"


# ---------- 3. 成本假设与折数：配置层的守护 ----------

def test_canonical_costs_rejects_legacy_location():
    """旧位置残留即报错：两处选择各拿一份成本会悄悄漂成两把尺子。"""
    with pytest.raises(ValueError, match="已上移为顶层 costs"):
        canonical_costs({"costs": {"fn": 10.0, "fp": 1.0},
                         "imbalance": {"comparison": {"costs": {"fn": 5.0, "fp": 1.0}}}})


@pytest.mark.parametrize("cfg", [{}, {"costs": {}}, {"costs": {"fn": 10.0}}])
def test_canonical_costs_requires_explicit_values(cfg):
    with pytest.raises(ValueError, match="必填、无默认值"):
        canonical_costs(cfg)


def test_resolve_cv_folds_clamps_to_minority_class():
    y_big = pd.Series([0] * 100 + [1] * 30)
    assert resolve_cv_folds(10, y_big) == 10
    y_small = pd.Series([0] * 100 + [1] * 4)
    assert resolve_cv_folds(10, y_small) == 4
    with pytest.raises(ValueError, match="无法做分层交叉验证"):
        resolve_cv_folds(10, pd.Series([0] * 100 + [1]))


def test_config_has_no_switch_to_disable_train_side_cv():
    """守住"选择依赖它就不给开关"这条：run_cv 一旦被加回来，选择就能静默失去训练侧依据。"""
    cfg = load_config(ROOT / "config.yaml")
    assert "run_cv" not in cfg["evaluation"]
    assert canonical_costs(cfg) == (10.0, 1.0)

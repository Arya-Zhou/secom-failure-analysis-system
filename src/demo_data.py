"""演示数据读取：报告模式读已跟踪快照，预测模式再准备工作点输入。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .data_io import load_secom
from .preprocessing import drop_all_nan_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "docs" / "demo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RIDGE_OP_NAME = "imbalance_operating_point_ridge.pkl"
COMPARISON_JSON = "imbalance_comparison.json"

REQUIRED_REPORT_FILES = (
    "resample_metrics.json",
    "temporal_metrics.json",
    "parameter_candidates.txt",
    "drift_report.md",
    "imbalance_comparison.md",
    "shap_fn_wafer.png",
    "shap_tp_wafer.png",
    "shap_global_ridge.png",
    "shap_fn_report.txt",
    "drift_overview.png",
)

# 策略工作点测试侧四象限冻结样本；分数来自当前 Ridge class_weight 产物。
FROZEN_SAMPLES = (
    {"case": "FN", "wafer_id": 518, "y_true": 1, "y_pred": 0, "score": -0.19777629614980508},
    {"case": "TP", "wafer_id": 96, "y_true": 1, "y_pred": 1, "score": 0.19072472564783993},
    {"case": "FP", "wafer_id": 689, "y_true": 0, "y_pred": 1, "score": 1.3868596723438131},
    {"case": "TN", "wafer_id": 1433, "y_true": 0, "y_pred": 0, "score": -1.735855856689775},
)

BANNER = "离线失效分析演示，成本 10:1 是假设"
CHAIN_NOTE = "本页样本来自独立成本策略链：折内 F 检验 + class_weight 岭分类器。"
SHAP_NOTE = "局部 SHAP 来自主分析 reference 链，展示该样本的特征贡献。"
GLOBAL_SHAP_NOTE = "全局 SHAP 来自主分析 reference 链，按平均绝对贡献排序。"
SAMPLE_NOTE = "样本来自已完成评估的策略链测试侧。"
GLOBAL_CHOICE = "全局特征贡献"
QUADRANT_GLOSSARY = """**判定记号**
- **T**（True）：预测与真实一致
- **F**（False）：预测与真实不一致
- **P**（Positive）：模型判定为失效
- **N**（Negative）：模型判定为正常
- **TP**：真实失效，模型命中
- **FN**：真实失效，模型漏检
- **FP**：真实正常，模型误报
- **TN**：真实正常，模型放行"""


def resolve_path(path_str: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """相对路径相对仓库根目录解析。"""
    path = Path(path_str)
    return path if path.is_absolute() else (root / path).resolve()


def missing_report_files(demo_dir: Path = DEMO_DIR) -> list[str]:
    """报告模式缺少的已跟踪快照文件名。"""
    return [name for name in REQUIRED_REPORT_FILES if not (demo_dir / name).exists()]


def demo_file(name: str, demo_dir: Path = DEMO_DIR) -> Path:
    """读取报告快照；缺文件时给出中文路径。"""
    path = demo_dir / name
    if not path.exists():
        raise FileNotFoundError(f"缺少演示快照 {name}，期望路径：{path}")
    return path


def read_text(name: str, demo_dir: Path = DEMO_DIR) -> str:
    """按 UTF-8 读取演示文本快照。"""
    return demo_file(name, demo_dir).read_text(encoding="utf-8")


def read_json(name: str, demo_dir: Path = DEMO_DIR):
    """读取演示 JSON 快照。"""
    import json
    return json.loads(read_text(name, demo_dir))


def data_paths(config_path: Path = CONFIG_PATH) -> tuple[Path, Path, str]:
    """返回特征、标签路径与时间戳格式。"""
    cfg = load_config(config_path)
    features = resolve_path(cfg["data"]["features_path"])
    labels = resolve_path(cfg["data"]["labels_path"])
    return features, labels, cfg["data"]["timestamp_format"]


def operating_point_paths(root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    """Ridge 策略工作点产物与同目录清单。"""
    out = root / "outputs"
    return out / RIDGE_OP_NAME, out / COMPARISON_JSON


def predict_inputs_ready(root: Path = PROJECT_ROOT, config_path: Path = CONFIG_PATH) -> dict:
    """本机预测模式所需的数据、产物与清单是否存在。"""
    features, labels, _ = data_paths(config_path)
    pkl, manifest = operating_point_paths(root)
    return {
        "features": features.exists(),
        "labels": labels.exists(),
        "operating_point": pkl.exists(),
        "manifest": manifest.exists(),
        "features_path": str(features),
        "labels_path": str(labels),
        "operating_point_path": str(pkl),
        "manifest_path": str(manifest),
    }


def predict_mode_enabled(status: dict | None = None) -> bool:
    """数据、pkl、清单齐备才启用本机预测。"""
    status = predict_inputs_ready() if status is None else status
    return all(status[key] for key in ("features", "labels", "operating_point", "manifest"))


def load_raw_features(config_path: Path = CONFIG_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """按主流程口径加载原始 591 列特征与标签。"""
    features, labels, fmt = data_paths(config_path)
    X, y, _timestamps = load_secom(str(features), str(labels), fmt)
    return X, y


def prepare_operating_point_features(
    X: pd.DataFrame,
    expected_names: list[str] | None = None,
) -> pd.DataFrame:
    """删除全空列后断言列名、数量和顺序与工作点一致。"""
    prepared, _dropped = drop_all_nan_columns(X)
    if expected_names is not None:
        expected = list(expected_names)
        actual = list(prepared.columns)
        if actual != expected:
            raise ValueError(
                f"特征列与工作点不一致：得到 {len(actual)} 列，工作点需要 {len(expected)} 列。"
                "请先删除全空列并保持 F001 起的列序，不要把原始 591 列直接送入预测。"
            )
    return prepared


def frozen_sample(wafer_id: int) -> dict:
    """按行号取冻结样本；未登记则失败。"""
    for row in FROZEN_SAMPLES:
        if row["wafer_id"] == wafer_id:
            return dict(row)
    allowed = ", ".join(str(row["wafer_id"]) for row in FROZEN_SAMPLES)
    raise ValueError(f"晶圆 {wafer_id} 不在冻结演示清单中，可选行号：{allowed}")


def frozen_row(X: pd.DataFrame, wafer_id: int) -> pd.DataFrame:
    """从已加载表中取冻结样本的一行。"""
    frozen_sample(wafer_id)
    if wafer_id not in X.index:
        raise KeyError(f"数据集中找不到晶圆行号 {wafer_id}")
    return X.loc[[wafer_id]]

"""不平衡策略对比实验：按训练侧 CV 期望代价选最优工作点，输入原始特征划分，输出对比结果与可部署工作点。"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下生成图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .config import canonical_costs
from .calibration import (
    assert_probe_straddles_threshold, base_pipeline, build_calibrated_estimator,
    calibration_block, calibration_diagnostics, cost_grid, decision_graph,
    is_calibrated, probe_coverage, raw_scores, resolve_calibration,
)
from .evaluation import (
    SCORE_SPACES, balanced_error_rate, classification_metrics, expected_cost,
    score_samples,
)
from .modeling import MODEL_DISPLAY_NAMES, make_model, registry_names

logger = logging.getLogger(__name__)

# 校准策略：主口径进工作点选择池，对照只报数字。两者的校准方法都由 config 给，名字不写死方法。
CALIBRATED_PRIMARY = "calibrated_threshold"
CALIBRATED_CONTRAST = "calibrated_threshold_contrast"
CALIBRATED_STRATEGIES = (CALIBRATED_PRIMARY, CALIBRATED_CONTRAST)

# 策略登记表：display 用于日志/中文产物；en 用于图内标签
STRATEGIES = ("baseline", "class_weight", "smote", "undersample", "threshold_moving",
              CALIBRATED_PRIMARY, CALIBRATED_CONTRAST)
STRATEGY_DISPLAY = {
    "baseline": "原始基线",
    "class_weight": "class_weight",
    "smote": "SMOTE",
    "undersample": "欠采样",
    "threshold_moving": "阈值移动",
    CALIBRATED_PRIMARY: "折外校准+阈值",
    CALIBRATED_CONTRAST: "折外校准+阈值·对照",
}
STRATEGY_EN = {
    "baseline": "Baseline",
    "class_weight": "class_weight",
    "smote": "SMOTE",
    "undersample": "Undersample",
    "threshold_moving": "Threshold moving",
    CALIBRATED_PRIMARY: "Calibrated + threshold",
    CALIBRATED_CONTRAST: "Calibrated (contrast)",
}
# 分类色固定分配（经 CVD 安全校验的色板，顺序不随策略数变化）
STRATEGY_COLORS = {
    "baseline": "#2a78d6",
    "class_weight": "#eb6834",
    "smote": "#1baf7a",
    "undersample": "#eda100",
    "threshold_moving": "#e87ba4",
    CALIBRATED_PRIMARY: "#7b52c9",
    CALIBRATED_CONTRAST: "#00968f",
}
# 阈值类策略：部署阈值取训练侧折外分数，CV 评估走嵌套 CV（外层每折的阈值由该折训练部分选出）
THRESHOLD_STRATEGIES = ("threshold_moving",) + CALIBRATED_STRATEGIES

COMPARISON_JSON = "imbalance_comparison.json"
COMPARISON_MD = "imbalance_comparison.md"
COST_SENSITIVITY_JSON = "cost_sensitivity.json"
COST_SENSITIVITY_MD = "cost_sensitivity.md"

SELECTION_BASIS = "train_cv_expected_cost"

# 产物结构版本：加载时必填、无默认值、缺失即报错。加校准层后 payload 多了 calibration /
# decision_graph 等语义字段，给缺失项兜默认值等于用新语义读旧字节，而内容比对那几层全会放行。
ARTIFACT_SCHEMA_VERSION = "2026.09-calibrated"

# 各策略在主流程 imbalance.strategy 下的等价训练策略；
# None = 主流程训练路径不支持该策略（采样类，见 modeling.MAIN_IMBALANCE_STRATEGIES）
MAIN_EQUIVALENT_STRATEGY = {
    "baseline": "none",
    "threshold_moving": "none",  # 阈值移动基于裸基线模型，训练侧不加权
    "class_weight": "class_weight",
    "smote": None,
    "undersample": None,
    CALIBRATED_PRIMARY: None,    # 主流程训练路径没有校准层，只能靠加载产物部署
    CALIBRATED_CONTRAST: None,
}

# 行为探针参数：判定两条已拟合链路是否同一条用行为等价（非属性清单，因 estimator 已学状态
# 枚举不完，如随机森林的树内容）。探针输入随产物落盘，改这两个常量只影响新产物；行数 100 取实测检出率/开销折中。
PROBE_SEED = 20260731
PROBE_ROWS = 100


def _build_probe_input(pipeline, seed: int, n_rows: int):
    """按链路自身已学统计量合成探针输入，只在产物落盘时调用一次。"""
    head = base_pipeline(pipeline).steps[0][1]
    n_features = int(head.n_features_in_)
    names = getattr(head, "feature_names_in_", None)
    columns = ([str(c) for c in names] if names is not None
               else [f"f{i}" for i in range(n_features)])

    center = np.nan_to_num(
        np.asarray(getattr(head, "statistics_", np.zeros(n_features)), dtype=np.float64))
    scaler = base_pipeline(pipeline).named_steps.get("scale")
    scale = (np.asarray(scaler.scale_, dtype=np.float64)
             if scaler is not None else np.ones(n_features))
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    rng = np.random.RandomState(seed)
    X = center + rng.standard_normal((n_rows, n_features)) * scale
    if n_rows >= 2:
        X[0] = center - 4.0 * scale
        X[1] = center + 4.0 * scale
    cols = np.arange(n_features)
    X[cols % n_rows, cols] = np.nan
    return pd.DataFrame(X, columns=columns)


def _sha16(arr) -> str:
    """float64 连续字节的 sha256 前 16 位（数组内容的确定性摘要）。"""
    buf = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(buf.tobytes()).hexdigest()[:16]


def _sha16_frame(df) -> str:
    """探针输入的摘要：值与列名一起进摘要。"""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(df, dtype=np.float64)).tobytes())
    columns = getattr(df, "columns", None)
    h.update(b"\xff")  # 值与列名之间的分隔符，避免两段字节拼接产生歧义
    if columns is not None:
        for col in columns:
            raw = str(col).encode("utf-8")
            h.update(len(raw).to_bytes(8, "big"))
            h.update(raw)
    return h.hexdigest()[:16]


def _threadpool_snapshot():
    """BLAS / OpenMP 线程池的实现、版本与线程数快照，取不到即如实标注。"""
    try:
        import threadpoolctl
    except ImportError:
        return "threadpoolctl 未安装（本项未记录）"
    return sorted(
        f"{e.get('user_api')}/{e.get('internal_api')} "
        f"{e.get('version')} x{e.get('num_threads')}"
        for e in threadpoolctl.threadpool_info())


def _runtime_env() -> dict:
    """生成环境快照，只作诊断，不作校验闸门。"""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "imblearn": imblearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": " ".join(x for x in platform.libc_ver() if x) or "未知",
        "threadpools": _threadpool_snapshot(),
    }


def _env_diff_hint(op: dict) -> str:
    """指纹不符时的分诊提示：列出产物记录环境与当前环境的差异项。"""
    recorded = op.get("runtime_env") or {}
    if not recorded:
        return ""
    current = _runtime_env()
    diff = [f"{k} {recorded.get(k)!r}→{current.get(k)!r}"
            for k in recorded if recorded.get(k) != current.get(k)]
    if not diff:
        return "（已记录的运行环境项与产物一致；未记录的环境因素不在此结论内）"
    return ("（运行环境差异：" + "；".join(diff)
            + " —— 请先排查这些，再怀疑文件被改动）")


def _decide(pipeline, X, threshold, score_space) -> np.ndarray:
    """工作点的唯一决策函数：无阈值走 predict，有阈值在其分数空间上切。"""
    if threshold is None:
        return np.asarray(pipeline.predict(X))
    scores, space = score_samples(pipeline, X)
    if space != score_space:
        raise ValueError(
            f"产物记录的分数空间 {score_space!r} 与链路实际 {space!r} 不符")
    return (scores >= threshold).astype(int)


def behavior_fingerprint(
    pipeline, probe_input, *, threshold, score_space, seed=None,
) -> dict:
    """已拟合工作点在给定探针输入上的行为指纹：探针输入、分数、最终预测三项摘要。"""
    scores, space = score_samples(pipeline, probe_input)
    preds = _decide(pipeline, probe_input, threshold, score_space)
    arr = np.asarray(probe_input, dtype=np.float64)
    return {
        "seed": seed,
        "n_rows": int(arr.shape[0]),
        "n_features": int(arr.shape[1]),
        "score_space": space,
        "input_sha256": _sha16_frame(probe_input),
        "scores_sha256": _sha16(scores),
        "predictions_sha256": _sha16(preds),
    }


# 摘要覆盖推理字段；runtime_env、probe_diagnostics 仅防静默改写，不作环境匹配或校准正确性判据。
_DIGEST_FIELDS = (
    "artifact_schema_version", "model", "strategy", "main_equivalent_strategy",
    "threshold", "score_space", "calibration", "decision_graph", "costs",
    "selected_features", "test_confusion", "expected_cost",
    "probe", "probe_coverage", "probe_diagnostics", "runtime_env",
)

# 加载时从已拟合对象重算校准语义与推理图；它们与行为指纹互补，每次加载均校验。
# 探针输入和阈值分别由输入指纹、推理字段摘要校验，避免覆盖检查掩盖更明确的错误。
_RECOMPUTED_FIELDS = ("calibration", "decision_graph")


def payload_digest(payload: dict) -> str:
    """产物推理字段的规范化摘要（sha256 前 16 位）。"""
    missing = [k for k in _DIGEST_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"工作点产物缺少必需字段: {missing}")
    body = {k: payload[k] for k in _DIGEST_FIELDS}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def artifact_sha256(path) -> str:
    """产物文件的字节 SHA-256（读磁盘字节，不重新序列化）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_schema_version(op: dict) -> None:
    """结构版本必填、无默认值：给缺失项兜默认值等于用新语义读旧字节，内容比对层全会放行。"""
    version = op.get("artifact_schema_version")
    if version is None:
        raise ValueError(
            "工作点产物缺少 artifact_schema_version：本版加载器按含校准层的推理语义"
            "解释 score_space 与 threshold，无从确认旧产物用的是同一套语义，"
            "拒绝使用（请用当前代码重新生成产物）")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"工作点产物结构版本 {version!r} 与本版加载器 {ARTIFACT_SCHEMA_VERSION!r} 不符："
            "字段语义可能已变，拒绝使用（请用当前代码重新生成产物）")


def decision_semantics(pipeline, threshold) -> dict:
    """按已拟合对象重算「决定分数语义与最终预测」的那两个字段（不需要打分，故加载时零开销）。"""
    return {
        "calibration": calibration_block(pipeline),
        "decision_graph": decision_graph(pipeline, threshold),
    }


def _calibration_layer_hint(op: dict) -> str:
    """指纹不符时分辨改动落在基础链路还是校准器。只作分诊，不参与放行。"""
    recorded = (op.get("probe_diagnostics") or {}).get("raw_scores_sha256")
    if not recorded or not is_calibrated(op.get("pipeline")):
        return ""   # 没有校准层就没有"两环之分"，给提示反而是误导
    fresh, _ = raw_scores(op["pipeline"], op["probe_input"])
    if _sha16(fresh) == recorded:
        return "（校准前原始分数未变、校准后输出变了：改动在校准器或其之后的决策层）"
    return "（校准前原始分数也变了：改动在基础链路——预处理/标准化/特征选择/分类器）"


def verify_operating_point(op: dict) -> None:
    """产物自检：结构版本 → 按对象重算推理语义 → 推理字段摘要 → 行为指纹，任一不符即抛错。"""
    _check_schema_version(op)
    fresh_sem = decision_semantics(op["pipeline"], op["threshold"])
    for key in _RECOMPUTED_FIELDS:
        if op.get(key) != fresh_sem[key]:
            raise ValueError(
                f"工作点产物推理语义不符：{key} 记录 {op.get(key)!r}，按产物重算 "
                f"{fresh_sem[key]!r} —— 该字段直接决定分数语义与最终预测，拒绝使用")
    actual = payload_digest(op)
    if actual != op.get("payload_digest"):
        raise ValueError(
            f"工作点产物推理字段摘要不符：记录 {op.get('payload_digest')!r}，"
            f"实际 {actual!r} —— 阈值/分数空间/校准语义/所选特征/探针记录等字段已被改动，拒绝使用")
    recorded = op["probe"]
    fresh = behavior_fingerprint(
        op["pipeline"], op["probe_input"], threshold=op["threshold"],
        score_space=op["score_space"], seed=recorded.get("seed"))
    if fresh != recorded:
        raise ValueError(
            f"工作点产物行为指纹不符：记录 {recorded}，实际 {fresh} —— 该工作点"
            "对同一探针输入给出了不同的分数或最终预测（或探针输入本身被换过），"
            f"不是记录的那条已拟合链路，拒绝使用"
            f"{_calibration_layer_hint(op)}{_env_diff_hint(op)}")


def _manifest_sha256(path: Path, manifest, require: bool) -> str | None:
    """从外部清单取该产物记录的文件 SHA-256，取不到时按严格度报错或返回 None。"""
    if manifest is None:
        if require:
            raise ValueError(
                f"require_manifest=True 与 manifest=None 冲突：调用方既要求强制"
                f"清单核对、又显式跳过了它（{path.name}）")
        return None

    explicit = manifest != "auto"
    mpath = Path(manifest) if explicit else (path.parent / COMPARISON_JSON)
    if not mpath.exists():
        if explicit:
            raise FileNotFoundError(f"指定的外部清单不存在: {mpath}")
        if require:
            raise ValueError(
                f"require_manifest=True 但同目录未找到外部清单 {mpath}"
                f"（{path.name}）：无法核对整份产物是否被替换，拒绝加载")
        return None

    data = json.loads(mpath.read_text(encoding="utf-8"))
    for m in (data.get("models") or {}).values():
        art = m.get("artifacts") or {}
        if art.get("operating_point") == path.name:
            sha = art.get("operating_point_sha256")
            if sha:
                return sha
            break
    if explicit or require:
        raise ValueError(
            f"外部清单 {mpath} 未登记产物 {path.name} 的文件 SHA-256 —— 清单与"
            "产物错配（显式指定清单或 require_manifest=True 时不允许降级），拒绝使用")
    return None


# 加载来源标记：只有经 load_operating_point() 校验通过的产物才带它。用每进程 import 时新建
# 的哨兵对象（非布尔/字符串，后者能被写进 pkl 一起交付、反序列化照样等值）；哨兵经 pickle 往返后不是同一对象，`is` 一定为假。
_LOADER_MARK = "_verified_by_loader"
_LOADER_TOKEN = object()


def _mark_loaded(op: dict) -> dict:
    """标记该产物已由本模块加载并校验，返回同一个对象。"""
    op[_LOADER_MARK] = _LOADER_TOKEN
    return op


def load_operating_point(path, manifest="auto", require_manifest=False) -> dict:
    """加载工作点产物并校验，不符即报错；清单未签名，挡的是损坏、错配、拿错文件，非篡改。"""
    path = Path(path)
    recorded = _manifest_sha256(path, manifest, require_manifest)
    if recorded is None:
        logger.warning(
            "工作点产物 %s 未经外部清单核对（清单缺失或未登记该产物）：仍会做"
            "产物自检，但无法据此判断整份文件是否被替换", path.name)
    else:
        actual = artifact_sha256(path)
        if actual != recorded:
            raise ValueError(
                f"工作点产物文件 SHA-256 与清单不符：清单 {recorded!r}，实际 "
                f"{actual!r}（{path}）—— 整份产物已被替换或改动，拒绝使用")
    with open(path, "rb") as f:
        op = pickle.load(f)
    verify_operating_point(op)
    return _mark_loaded(op)


def _ensure_loaded(op: dict, refuse: str) -> None:
    """受控入口共用的来源闸：未经过 load_operating_point 的产物一律拒绝。"""
    if op.get(_LOADER_MARK) is not _LOADER_TOKEN:
        raise ValueError(
            "该工作点产物未经 load_operating_point() 加载：自行反序列化跳过了"
            "外部清单核对与产物自检（行为指纹 + 推理字段摘要），无法确认拿到的"
            "是记录的那条已拟合链路，" + refuse + "。请改用 load_operating_point("
            "path) 加载后再调用本函数")


def apply_operating_point(op: dict, X) -> np.ndarray:
    """按工作点产物出预测，与自检共用同一个决策函数 _decide，调用方不重建任何预处理。"""
    _ensure_loaded(op, "拒绝出预测")
    return _decide(op["pipeline"], X, op["threshold"], op["score_space"])


def operating_point_input_names(op: dict) -> list[str]:
    """工作点拟合时的输入列名，不向调用方暴露内部 pipeline 对象。"""
    _ensure_loaded(op, "拒绝读取输入列名")
    names = getattr(op["pipeline"], "feature_names_in_", None)
    if names is None:
        raise ValueError("工作点缺少 feature_names_in_，无法核对输入列")
    return list(names)


def score_operating_point(op: dict, X) -> tuple[np.ndarray, str]:
    """按工作点产物出分数，与预测共用已加载校验，调用方不接触内部 pipeline。"""
    _ensure_loaded(op, "拒绝出分数")
    return score_samples(op["pipeline"], X)


def verify_operating_point_evaluation(op, features, predictions, scores, score_space) -> None:
    """在受控入口内部核对评估证据，不向调用方暴露工作点内部模型或分数。"""
    actual = apply_operating_point(op, features)
    actual_scores, actual_space = score_operating_point(op, features)
    if (not np.array_equal(actual, predictions) or not np.array_equal(actual_scores, scores)
            or actual_space != score_space):
        raise ValueError("工作点未逐位复现评估预测或分数")


def make_sampler(strategy: str, seed: int, cmp_cfg: dict):
    """重采样器工厂：仅 smote / undersample 返回采样器，其余返回 None。"""
    if strategy == "smote":
        k = int(cmp_cfg.get("smote_k_neighbors", 5))
        return SMOTE(random_state=seed, k_neighbors=k)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    return None


def build_strategy_estimator(
    strategy: str, model_name: str, seed: int, cfg: dict, n_features: int,
):
    """按策略构建全链路未训练 estimator：填充 → 标准化 → 特征选择 → [重采样] → 模型 → [校准器]。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知不平衡策略: {strategy}，可选: {list(STRATEGIES)}")
    pre_cfg = cfg.get("preprocessing") or {}
    cmp_cfg = (cfg.get("imbalance") or {}).get("comparison") or {}

    # keep_empty_features：折内可能出现"该折训练部分全空"的列，保留为 0 常量
    # 而非丢列，保证折间列数稳定；常量列的 F 分数为 NaN，SelectKBest 不会选它
    steps: list = [("impute", SimpleImputer(
        strategy=pre_cfg.get("impute_strategy", "median"),
        keep_empty_features=True))]
    if pre_cfg.get("scale", True):
        steps.append(("scale", StandardScaler()))
    steps.append(("select", SelectKBest(f_classif, k=int(n_features))))

    class_weight = "balanced" if strategy == "class_weight" else None
    sampler = make_sampler(strategy, seed, cmp_cfg)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", make_model(model_name, seed, class_weight)))
    pipeline = ImbPipeline(steps)
    if strategy not in CALIBRATED_STRATEGIES:
        return pipeline

    # 校准的基础 estimator 取不加权裸链路：校准要学的是分数到概率的映射，
    # 而 class_weight 会先把分数分布整体推走，两个手段叠在一起就分不出各自的贡献。
    cal_cfg = resolve_calibration(cfg)
    method = (cal_cfg["method"] if strategy == CALIBRATED_PRIMARY
              else cal_cfg["contrast_method"])
    if method is None:
        raise ValueError(
            f"策略 {strategy} 需要 imbalance.calibration.contrast_method，但它是 null："
            "对照已关闭时不应把该策略列进 imbalance.comparison.strategies")
    return build_calibrated_estimator(pipeline, method, cal_cfg["cv_folds"], seed)


def select_threshold_by_cost(
    y_true, scores, fn_cost: float, fp_cost: float,
) -> tuple[float, dict]:
    """在给定分数上选期望代价最小的决策阈值，预测正类的条件为 score >= 阈值。"""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("选阈值需要正负两类样本")

    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos

    # 阈值取 s_sorted[i] 时：FN = 分数严格小于它的正类数，FP = 分数 >= 它的负类数
    pos_cum = np.concatenate([[0], np.cumsum(y_sorted == 1)])  # 前 i 个里的正类数
    neg_cum = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    # 每个唯一分数只在其首次出现位置作为候选（重复分数的 FN/FP 相同）
    first_idx = np.flatnonzero(np.concatenate([[True], s_sorted[1:] != s_sorted[:-1]]))
    fn_counts = pos_cum[first_idx]
    fp_counts = n_neg - neg_cum[first_idx]
    thresholds = s_sorted[first_idx]
    # 哨兵：阈值高于最大分数 → 全部预测为通过（FN = 全部正类）
    thresholds = np.append(thresholds, s_sorted[-1] + 1.0)
    fn_counts = np.append(fn_counts, n_pos)
    fp_counts = np.append(fp_counts, 0)

    costs = fn_counts * fn_cost + fp_counts * fp_cost
    best = int(np.argmin(costs))  # argmin 取首个最小值 = 最低阈值 = 召回优先
    info = {
        "train_fn": int(fn_counts[best]),
        "train_fp": int(fp_counts[best]),
        "train_cost": float(costs[best]),
        "n_candidates": int(len(thresholds)),
    }
    return float(thresholds[best]), info


def _fold_ber_recall(y_va, y_pred_va) -> tuple[float, float]:
    """单折 BER 与失败类召回。"""
    ber = balanced_error_rate(y_va, y_pred_va)
    pos = int((np.asarray(y_va) == 1).sum())
    tp = int(((np.asarray(y_va) == 1) & (np.asarray(y_pred_va) == 1)).sum())
    return ber, (tp / pos if pos else 0.0)


def _summarize_cv(y, oof_pred, per_fold, fn_cost, fp_cost) -> dict:
    """CV 汇总：逐折 BER 与召回均值，加折外(OOF)混淆计数与期望代价。"""
    bers, recalls = zip(*per_fold)
    cost, cm = expected_cost(y, oof_pred, fn_cost, fp_cost)
    return {
        "CV_BER均值": float(np.mean(bers)),
        "CV_BER标准差": float(np.std(bers)),
        "CV召回率均值": float(np.mean(recalls)),
        "CV混淆": cm,
        "CV期望代价": cost,
    }


def _cv_oof_metrics(estimator, X, y, folds, seed, fn_cost, fp_cost) -> dict:
    """分层 CV 折外指标：逐折 clone、在训练折 fit、对验证折 predict。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    per_fold = []
    for tr_idx, va_idx in skf.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.asarray(est.predict(X.iloc[va_idx]))
        oof_pred[va_idx] = pred
        per_fold.append(_fold_ber_recall(y.iloc[va_idx], pred))
    return _summarize_cv(y, oof_pred, per_fold, fn_cost, fp_cost)


def _oof_scores(estimator, X, y, folds, seed) -> np.ndarray:
    """训练数据的折外分数（概率或决策间隔，由 estimator 能力决定）。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    method = "predict_proba" if hasattr(estimator, "predict_proba") else "decision_function"
    oof = cross_val_predict(clone(estimator), X, y, cv=skf, method=method)
    return oof[:, 1] if oof.ndim == 2 else np.ravel(oof)


def _nested_threshold_cv(estimator, X, y, folds, seed, fn_cost, fp_cost) -> dict:
    """threshold_moving 的嵌套 CV：外层每折的阈值由该折训练部分的内层 OOF 选出。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    per_fold = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        inner_scores = _oof_scores(estimator, X_tr, y_tr, folds, seed)
        thr, _ = select_threshold_by_cost(y_tr, inner_scores, fn_cost, fp_cost)
        est = clone(estimator)
        est.fit(X_tr, y_tr)
        s_va, _ = score_samples(est, X.iloc[va_idx])
        pred = (s_va >= thr).astype(int)
        oof_pred[va_idx] = pred
        per_fold.append(_fold_ber_recall(y.iloc[va_idx], pred))
    return _summarize_cv(y, oof_pred, per_fold, fn_cost, fp_cost)


def _selected_features(est, columns) -> list[str]:
    """从已拟合的链路取出折内特征选择实际保留的列名。"""
    mask = base_pipeline(est).named_steps["select"].get_support()
    return [str(c) for c, keep in zip(columns, mask) if keep]


def _selection_eligibility(strategy: str, diagnostics: dict | None) -> tuple[bool, str | None]:
    """能否进工作点选择池。「正确性全过才可标为可部署」写成过滤器而不是文档里的一句话。"""
    if strategy == CALIBRATED_CONTRAST:
        return False, "同表对照，只报数字、不产交付物"
    if diagnostics is not None and not diagnostics["passed"]:
        return False, ("校准正确性检查未通过：" + "；".join(
            diagnostics["structural_problems"] + diagnostics["quality_problems"]))
    return True, None


def _evaluate_one_strategy(
    strategy, model_name, X_train, y_train, X_test, y_test,
    seed, folds, fn_cost, fp_cost, cfg, n_features,
) -> dict:
    """单个 (模型, 策略) 组合：训练侧 CV 指标、训练集拟合、测试集一次评估。"""
    est = build_strategy_estimator(strategy, model_name, seed, cfg, n_features)
    threshold, sel_info, oof = None, None, None

    if strategy in THRESHOLD_STRATEGIES:
        cv = _nested_threshold_cv(est, X_train, y_train, folds, seed, fn_cost, fp_cost)
        # 部署阈值仅用训练侧 OOF 分数选择，CV 代价由上面的嵌套 CV 评估。
        # 校准器全训练集拟合后的同集 predict_proba 不是 OOF，校准策略也必须重做折外预测。
        oof = _oof_scores(est, X_train, y_train, folds, seed)
        threshold, sel_info = select_threshold_by_cost(y_train, oof, fn_cost, fp_cost)
    else:
        cv = _cv_oof_metrics(est, X_train, y_train, folds, seed, fn_cost, fp_cost)

    est.fit(X_train, y_train)
    scores, space = score_samples(est, X_test)
    if threshold is not None:
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = np.asarray(est.predict(X_test))
    metrics = classification_metrics(y_test, y_pred, scores)
    cost, cm = expected_cost(y_test, y_pred, fn_cost, fp_cost)

    # 正确性诊断与篡改检测是两类防线：指纹只证明产物没被改，证明不了它当初装配对了
    diagnostics = None
    if strategy in CALIBRATED_STRATEGIES:
        diagnostics = calibration_diagnostics(est, X_train, y_train, oof)
        if diagnostics["structural_problems"]:
            raise ValueError(
                f"{model_name}/{strategy} 校准层装配错误："
                + "；".join(diagnostics["structural_problems"]))
    eligible, ineligible_reason = _selection_eligibility(strategy, diagnostics)

    # 工作点标识：上线路径对固定探针的行为指纹，以及主流程等价训练策略
    # （采样类与校准类在主流程无等价路径，取 None）
    features = _selected_features(est, X_train.columns)
    main_strategy = MAIN_EQUIVALENT_STRATEGY.get(strategy)
    probe_input = _build_probe_input(est, PROBE_SEED, PROBE_ROWS)
    probe = behavior_fingerprint(
        est, probe_input, threshold=threshold, score_space=space, seed=PROBE_SEED)
    semantics = decision_semantics(est, threshold)
    probe_scores, _ = score_samples(est, probe_input)
    coverage = probe_coverage(probe_scores, threshold)
    assert_probe_straddles_threshold(coverage, f"{model_name}/{strategy}")
    probe_raw, raw_space = raw_scores(est, probe_input)

    return {
        "test": metrics,
        "cv": cv,
        "confusion": cm,
        "expected_cost": cost,
        "threshold": threshold,
        "threshold_selected_on": "train_oof" if threshold is not None else None,
        "threshold_selection": sel_info,
        "score_space": space,
        "calibration": semantics["calibration"],
        "decision_graph": semantics["decision_graph"],
        "calibration_diagnostics": diagnostics,
        "selection_eligible": eligible,
        "selection_ineligible_reason": ineligible_reason,
        "selected_features": features,
        "behavior_probe": probe,
        "probe_coverage": coverage,
        # 校准前分数只作故障分诊（分辨改动在基础链路还是校准器），不参与放行判断
        "probe_diagnostics": {"raw_score_space": raw_space,
                              "raw_scores_sha256": _sha16(probe_raw)},
        "main_equivalent_strategy": main_strategy,
        "scores": scores,      # 仅供绘图，落盘前剔除
        "y_pred": y_pred,      # 同上
        "estimator": est,          # 仅供工作点产物落盘，不进 json
        "probe_input": probe_input,  # 同上（随产物走，不靠加载时重算）
        "oof_scores": oof,     # 仅供成本敏感性复用同一组训练侧折外分数，不进 json
    }


def _save_operating_point(model_name, best, r, out_dir, costs, filename=None) -> tuple[str, str]:
    """把最优工作点的已拟合链路整体落盘，返回 (文件名, 文件 SHA-256)。"""
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": model_name,
        "strategy": best,
        "main_equivalent_strategy": r["main_equivalent_strategy"],
        "pipeline": r["estimator"],
        "threshold": r["threshold"],
        "score_space": r["score_space"],
        # 校准语义与推理图：加载时按已拟合对象重算一遍再比对，不采信这两行记录值
        "calibration": r["calibration"],
        "decision_graph": r["decision_graph"],
        # 阈值是这一组成本假设下的最优解，成本换档阈值就换；两者必须一起交付
        "costs": dict(costs),
        "selected_features": r["selected_features"],
        "test_confusion": r["confusion"],
        "expected_cost": r["expected_cost"],
        "probe": r["behavior_probe"],
        "probe_coverage": r["probe_coverage"],
        "probe_diagnostics": r["probe_diagnostics"],
        # 探针输入随产物走，不靠加载时按种子重算：重算会让自检能力依赖生成探针的代码
        # 从未变过，一旦改了构造方式，既有产物会以"行为指纹不符"误报
        "probe_input": r["probe_input"],
        # 生成环境只在指纹不符时用于分诊（是换了环境还是文件被改），不参与放行判断
        "runtime_env": _runtime_env(),
    }
    payload["payload_digest"] = payload_digest(payload)
    fname = filename if filename is not None else f"imbalance_operating_point_{model_name}.pkl"
    fpath = Path(out_dir) / fname
    with open(fpath, "wb") as f:
        pickle.dump(payload, f)
    return fname, artifact_sha256(fpath)


def save_fitted_operating_point(estimator, model_name, selected_features, confusion,
                                costs, out_dir, filename) -> tuple[str, str]:
    """保存已锁定、无额外阈值的完整链路，复用工作点的清单与行为指纹契约。"""
    probe_input = _build_probe_input(estimator, PROBE_SEED, PROBE_ROWS)
    probe_scores, space = score_samples(estimator, probe_input)
    semantics = decision_semantics(estimator, None)
    record = {
        "estimator": estimator, "main_equivalent_strategy": None, "threshold": None,
        "score_space": space, **semantics, "selected_features": list(selected_features),
        "confusion": dict(confusion),
        "expected_cost": float(confusion["fn"] * costs["fn"] + confusion["fp"] * costs["fp"]),
        "behavior_probe": behavior_fingerprint(
            estimator, probe_input, threshold=None, score_space=space, seed=PROBE_SEED),
        "probe_coverage": probe_coverage(probe_scores, None), "probe_input": probe_input,
        "probe_diagnostics": {"raw_score_space": space, "raw_scores_sha256": _sha16(probe_scores)},
    }
    return _save_operating_point(model_name, "class_weight", record, out_dir, costs, filename)


def _deployment_block(best: str, model_name: str, r: dict, artifact: str) -> dict:
    """生成最优工作点的部署说明，只声明实际成立的事。"""
    common = {
        "artifact": artifact,
        "artifact_usage": (
            "load_operating_point(路径) 加载并校验（外部清单文件哈希 + 产物"
            "自检）→ apply_operating_point(op, X) 出预测；产物自带填充/标准化"
            "/特征选择/模型/校准器的已拟合状态与阈值"),
        "behavior_probe_sha256": r["behavior_probe"]["scores_sha256"],
        "behavior_probe_predictions_sha256": r["behavior_probe"]["predictions_sha256"],
        "threshold": r["threshold"],
        "score_space": r["score_space"],
        "decision_graph": r["decision_graph"],
        "calibration": r["calibration"],
        "probe_coverage": r["probe_coverage"],
        "reproduces_experiment_metrics_in_main": False,
        "threshold_via_config": False,  # 阈值不经配置传递，见模块 docstring
    }
    main_strategy = r["main_equivalent_strategy"]
    if r["calibration"]["enabled"]:
        return {
            **common,
            "supported_in_main": False,
            "main_config": None,
            "note": (
                "该工作点在推理图里含校准器（分数空间 calibrated_probability），"
                "主流程训练路径没有这一层，配置里也不存在能把它打开的开关；"
                "部署只能加载 artifact。阈值是校准后概率上的切点，换成本假设即换阈值，"
                "成本档位随产物落盘（costs 字段），敏感性见 cost_sensitivity 产物"),
        }
    if r["threshold"] is not None:
        return {
            **common,
            "supported_in_main": False,
            "main_config": None,
            "note": (
                "该工作点含阈值，而阈值是这条已拟合链路的属性（分数分布随 fit "
                "范围/预处理参数/超参/种子而变），无法通过配置承载；部署请加载 "
                "artifact。若要在主流程链路上用阈值移动，须在该链路自身的训练侧"
                "折外分数上重新选阈值，不能沿用本数值"),
        }
    if main_strategy is None:
        return {
            **common,
            "supported_in_main": False,
            "main_config": None,
            "note": (
                f"主流程训练路径不支持 {best}（仅本对比实验实现折内重采样）；"
                "部署请加载 artifact，或先在主流程实现采样管线"),
        }
    return {
        **common,
        "supported_in_main": True,
        "main_config": {"strategy": main_strategy},
        "note": (
            "该工作点不含阈值，训练策略可在主流程直接沿用；但主流程特征链路"
            "与实验不同（四方法投票 vs 折内 F 检验），指标不会与本表逐位相同，"
            "须在主流程重新评估后再对外引用"),
    }


def _labeled(strategy: str, r: dict, mapping: dict) -> str:
    """策略名后缀上实际用的校准方法：主口径与对照都叫"折外校准"，不写方法就分不出哪行是哪个。"""
    method = ((r or {}).get("calibration") or {}).get("method")
    return f"{mapping[strategy]} [{method}]" if method else mapping[strategy]


def _plot_pr_curves(model_name, strat_results, y_test, out_dir) -> str:
    """各策略 PR 曲线与实际工作点。"""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for strat, r in strat_results.items():
        color = STRATEGY_COLORS[strat]
        prec, rec, _ = precision_recall_curve(y_test, r["scores"])
        thresholded = strat in THRESHOLD_STRATEGIES
        marker = "D" if thresholded else "o"
        if not thresholded:
            ax.plot(rec, prec, color=color, lw=2, label=_labeled(strat, r, STRATEGY_EN))
        # 工作点 = 该策略实际预测的 (recall, precision)
        cm = r["confusion"]
        op_rec = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
        op_prec = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
        label = f"{_labeled(strat, r, STRATEGY_EN)} (op. point)" if thresholded else None
        ax.plot([op_rec], [op_prec], marker=marker, ms=9, color=color,
                markeredgecolor="white", markeredgewidth=1.5, linestyle="none",
                label=label, zorder=5)
    # 无技能基准线：正类占比（随机分类器的精确率）
    pos_rate = float(np.mean(np.asarray(y_test) == 1))
    ax.axhline(pos_rate, color="#999999", lw=1, linestyle="--",
               label=f"No-skill ({pos_rate:.3f})")
    ax.set_xlabel("Recall (fail class)")
    ax.set_ylabel("Precision (fail class)")
    ax.set_title(f"PR Curves & Operating Points — {model_name}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fname = f"imbalance_pr_curve_{model_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_confusion_grid(model_name, strat_results, out_dir) -> str:
    """各策略混淆矩阵并排对比（单色渐变表数量，文字标注计数）。"""
    n = len(strat_results)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (strat, r) in zip(axes, strat_results.items()):
        cm = r["confusion"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() or 1)
        for i in range(2):
            for j in range(2):
                v = int(mat[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                        color="white" if v > mat.max() * 0.6 else "#1a1a19")
        ax.set_xticks([0, 1], labels=["Pred pass", "Pred fail"], fontsize=8)
        ax.set_yticks([0, 1], labels=["True pass", "True fail"], fontsize=8)
        ax.set_title(_labeled(strat, r, STRATEGY_EN), fontsize=9)
    fig.suptitle(f"Confusion Matrices by Strategy — {model_name}", fontsize=12)
    fig.tight_layout()
    fname = f"imbalance_confusion_{model_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _delta_vs_baseline(strat_results: dict) -> None:
    """就地补充各策略相对 baseline 的量化对比（召回提升/误报增加/代价变化）。"""
    base = strat_results.get("baseline")
    if base is None:
        return
    for strat, r in strat_results.items():
        if strat == "baseline":
            continue
        r["delta_vs_baseline"] = {
            "召回率提升": r["test"]["召回率"] - base["test"]["召回率"],
            "误报增加": r["confusion"]["fp"] - base["confusion"]["fp"],
            "漏检减少": base["confusion"]["fn"] - r["confusion"]["fn"],
            "期望代价变化": r["expected_cost"] - base["expected_cost"],
        }


def _calibration_contrast(strat_results: dict, best: str, pool: list) -> dict | None:
    """保留主口径、额外加入对照是否改变选择；不模拟移除主口径后替换校准方法。"""
    primary = strat_results.get(CALIBRATED_PRIMARY)
    contrast = strat_results.get(CALIBRATED_CONTRAST)
    if primary is None or contrast is None:
        return None
    hypothetical = min(list(pool) + [CALIBRATED_CONTRAST],
                       key=lambda s: (strat_results[s]["cv"]["CV期望代价"], s))
    return {
        "primary_method": primary["calibration"]["method"],
        "contrast_method": contrast["calibration"]["method"],
        "cv_cost_primary": primary["cv"]["CV期望代价"],
        "cv_cost_contrast": contrast["cv"]["CV期望代价"],
        "cv_cost_delta": contrast["cv"]["CV期望代价"] - primary["cv"]["CV期望代价"],
        "test_cost_primary": primary["expected_cost"],
        "test_cost_contrast": contrast["expected_cost"],
        "primary_diagnostics_passed": bool(primary["calibration_diagnostics"]["passed"]),
        "contrast_diagnostics_passed": bool(contrast["calibration_diagnostics"]["passed"]),
        "selected_with_contrast_in_pool": hypothetical,
        "would_change_selection": hypothetical != best,
        "note": (
            "对照行只报数字、不产交付物：主口径由 imbalance.calibration.method 定死，"
            "不让两个校准方法按同一批 CV 代价互相竞争"),
    }


def _cost_sensitivity_for_model(r: dict, strategy: str, y_train, y_test,
                                grid: list) -> dict:
    """复用同一组训练侧折外分数，只重选阈值：不重训、不重跑 CV、不重选策略。"""
    oof, scores = r["oof_scores"], np.asarray(r["scores"], dtype=float)
    y_arr = np.asarray(y_test).astype(int)
    n_test, n_pos = int(y_arr.size), int((y_arr == 1).sum())
    rows = []
    for g in grid:
        thr, sel = select_threshold_by_cost(y_train, oof, g["fn"], g["fp"])
        pred = (scores >= thr).astype(int)
        cost, cm = expected_cost(y_test, pred, g["fn"], g["fp"])
        # 平凡策略「全部判通过」：一片不复检，代价全部来自漏检。它随成本比同比例缩放，
        # 故"相对它改善了多少"可以跨档读（计数类的 FN/FP 同理，带单位的代价则不行）
        trivial = float(n_pos * g["fn"])
        rows.append({
            "ratio": g["ratio"], "fn_cost": g["fn"], "fp_cost": g["fp"],
            "is_canonical": g["is_canonical"], "threshold": thr,
            "train_oof": {"fn": sel["train_fn"], "fp": sel["train_fp"],
                          "cost": sel["train_cost"]},
            "test": {**cm,
                     "召回率": float(cm["tp"] / n_pos) if n_pos else 0.0,
                     "BER": float(balanced_error_rate(y_test, pred))},
            "test_cost": cost,
            "cost_per_wafer": cost / n_test if n_test else None,
            "trivial_all_pass_cost": trivial,
            "improvement_vs_all_pass": ((trivial - cost) / trivial) if trivial else None,
        })
    keys = {(round(row["threshold"], 12)) for row in rows}
    confusions = {(row["test"]["fn"], row["test"]["fp"]) for row in rows}
    return {
        "strategy": strategy,
        "calibration_method": (r.get("calibration") or {}).get("method"),
        "score_space": r["score_space"],
        "n_test": n_test, "n_test_positive": n_pos,
        "rows": rows,
        "distinct_thresholds": len(keys),
        "changes_operating_point": len(keys) > 1,
        "changes_test_confusion": len(confusions) > 1,
    }


def _write_cost_sensitivity_md(result: dict, path: Path) -> None:
    """成本敏感性表落盘为 markdown。"""
    lines = [
        "# 成本敏感性：换成本比会不会换工作点",
        "",
        f"- 口径：复用**同一组训练侧折外分数**（{result['basis']}），只重选阈值 —— "
        "不重训、不重跑 CV、不重选策略",
        f"- canonical 成本比 FN:FP = {result['canonical']['fn']:g}:"
        f"{result['canonical']['fp']:g}；**主对比表与正式 pkl 只绑这一档**，"
        "其余档位只在本表出现",
        "- **不同档位的期望代价与每片平均代价不可横比**：单位随成本比而变。"
        "跨档只读 FN / FP / 相对改善",
        "- 相对改善的基准是平凡策略「全部判通过」（一片不复检，代价全来自漏检），"
        "它随成本比同比例缩放，故这一列可以跨档读",
        "",
    ]
    for display, m in result["models"].items():
        lines.append(f"## {display}")
        lines.append("")
        method = m.get("calibration_method")
        lines.append(f"策略：{STRATEGY_DISPLAY.get(m['strategy'], m['strategy'])}"
                     + (f" [{method}]" if method else "")
                     + f"（分数空间 {m['score_space']}），测试集 {m['n_test']} 片"
                     f"（失败 {m['n_test_positive']} 片）")
        lines.append("")
        lines.append("| FN:FP | 阈值 | 训练侧折外 FN/FP | 测试 FN | 测试 FP | 召回 | BER "
                     "| 每片平均代价 | 相对「全部判通过」改善 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in m["rows"]:
            mark = " ★" if row["is_canonical"] else ""
            imp = row["improvement_vs_all_pass"]
            imp_txt = f"{imp * 100:+.1f}%" if imp is not None else "n/a"
            per_wafer = row["cost_per_wafer"]
            per_txt = f"{per_wafer:.3f}" if per_wafer is not None else "n/a"
            lines.append(
                f"| {row['ratio']:g}:1{mark} | {row['threshold']:.4f} "
                f"| {row['train_oof']['fn']}/{row['train_oof']['fp']} "
                f"| {row['test']['fn']} | {row['test']['fp']} "
                f"| {row['test']['召回率']:.3f} | {row['test']['BER']:.3f} "
                f"| {per_txt} | {imp_txt} |")
        lines.append("")
        verdict = ("**会**" if m["changes_operating_point"] else "**不会**")
        lines.append(
            f"换成本比{verdict}选出不同工作点："
            f"三档共 {m['distinct_thresholds']} 个不同阈值，"
            f"测试集混淆{'有' if m['changes_test_confusion'] else '无'}差异。"
            + ("故「最优工作点」一词必须带成本假设限定——它是成本比的函数，不是链路的固有属性。"
               if m["changes_operating_point"] else
               "本数据下三档给出同一个切点，但这是实测结果不是性质，换数据须重测。"))
        lines.append("")
    lines.append("## 边界")
    lines.append("")
    lines.append(
        "- 本表只重选**阈值**，不重排**策略**：阈值类策略的 CV 折外混淆本身就是成本比的"
        "函数（换成本比会在每个折内选出不同阈值），拿落盘的那一份去换算得到的不是"
        "那个成本比下的 CV 代价，故策略层面的重排序不在本表内")
    lines.append(
        "- 阈值取自训练侧折外分数，测试集只做一次评估；三档共用同一批折外分数，"
        "故三行之间的差异只来自成本比，不含重训噪声")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cost_sensitivity(cfg: dict, per_model: dict, y_train, y_test, out_dir: Path,
                         fn_cost: float, fp_cost: float) -> dict:
    """成本敏感性主入口：三档成本比在同一组折外分数上各选一次阈值，落独立产物。"""
    ratios = ((cfg.get("imbalance") or {}).get("cost_sensitivity") or [])
    if not ratios:
        raise ValueError(
            "config 缺少 imbalance.cost_sensitivity（成本档位列表）："
            "阈值是成本假设的函数，只给一档就无从知道换个假设会不会换工作点")
    grid = cost_grid(ratios, fp_cost, fn_cost)
    result = {
        "status": "ok",
        "basis": "折外校准概率：主口径校准工作点在训练侧的完全折外分数",
        "canonical": {"fn": fn_cost, "fp": fp_cost},
        "grid": grid,
        "models": {},
    }
    for display, r in per_model.items():
        result["models"][display] = _cost_sensitivity_for_model(
            r["result"], r["strategy"], y_train, y_test, grid)
    (out_dir / COST_SENSITIVITY_JSON).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_cost_sensitivity_md(result, out_dir / COST_SENSITIVITY_MD)
    logger.info("[成本敏感性] 结果已保存: %s / %s",
                out_dir / COST_SENSITIVITY_JSON, out_dir / COST_SENSITIVITY_MD)
    return result


def _write_markdown(result: dict, path: Path) -> None:
    """对比表落盘为 markdown"""
    fn_cost = result["costs"]["fn"]
    fp_cost = result["costs"]["fp"]
    lines = [
        "# 不平衡策略对比（漏检风险控制）",
        "",
        f"- 统一口径：与主流程相同的 80/20 分层划分索引（seed={result['random_seed']}，"
        f"同一批测试晶圆），测试集 {result['test_set']['n']} 片"
        f"（失败 {result['test_set']['n_fail']} 片）",
        # 「防泄漏」单独立着就是全称断言：显式限定它覆盖拟合范围，
        # 时间维度的 look-ahead bias 由时间序对照协议度量（temporal_metrics.json）
        "- 防泄漏（覆盖拟合范围；时间维度见时间序对照协议）：填充/标准化/特征选择（F 检验选 "
        f"{result['n_features']} 维）/重采样全部封装进同一 Pipeline，只在 "
        f"{result['cv_folds']} 折分层 CV 的训练折与最终训练集上拟合；"
        "校准类策略的基础分类器与校准器同样只在折内拟合，其部署阈值取自**完全折外**的"
        "校准概率（外层每折在自己的训练部分重新拟合分类器与校准器）；"
        "测试集不参与任何拟合与选择",
        "- 工作点选择：只用训练侧信息 —— **选择池内**各策略按 CV 折外期望代价取最小"
        "（阈值类策略的 CV 代价来自嵌套 CV，无选择偏差）；池外的行照常报数字但不产交付物，"
        "每行为何在池外逐条写在下面；测试集对每个策略只做一次最终评估，"
        "作为所选工作点的独立验证",
        f"- 成本假设（相对值，可在 config 调整）：漏检 FN={fn_cost:g}，误报 FP={fp_cost:g}"
        " —— 失效晶圆流出的代价远高于良品复检",
        "",
    ]
    for display, m in result["models"].items():
        lines.append(f"## {display}")
        lines.append("")
        lines.append(
            "| 策略 | CV期望代价(训练侧) | Accuracy | Recall | Precision | F1 "
            "| AUC | BER | FN | FP | 测试期望代价 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for strat, r in m["strategies"].items():
            t, cm = r["test"], r["confusion"]
            auc = f"{t['AUC']:.3f}" if t["AUC"] is not None else "N/A"
            note = "" if r.get("selection_eligible", True) else "（对照，不进选择）"
            lines.append(
                f"| {_labeled(strat, r, STRATEGY_DISPLAY)}{note} "
                f"| {r['cv']['CV期望代价']:g} "
                f"| {t['准确率']:.3f} | {t['召回率']:.3f} "
                f"| {t['精确率']:.3f} | {t['F1分数']:.3f} | {auc} | {t['BER']:.3f} "
                f"| {cm['fn']} | {cm['fp']} | {r['expected_cost']:g} |")
        best = m["best_by_cv_cost"]
        br = m["strategies"][best]
        lines.append("")
        conclusion = (
            f"最优工作点（按训练侧 CV 期望代价，在可选池 "
            f"{[_labeled(s, m['strategies'][s], STRATEGY_DISPLAY) for s in m['selection_pool']]} "
            f"内选出）：**{_labeled(best, br, STRATEGY_DISPLAY)}**"
            f"（CV 代价 {br['cv']['CV期望代价']:g}；"
            f"测试集独立验证：代价 {br['expected_cost']:g}）")
        d = br.get("delta_vs_baseline")
        if d:
            conclusion += (
                f" —— 测试集上相比原始基线：召回 {d['召回率提升']:+.3f}，"
                f"漏检 {-d['漏检减少']:+d} 片，误报 {d['误报增加']:+d} 片，"
                f"期望代价 {d['期望代价变化']:+g}")
        lines.append(conclusion)
        if br.get("threshold") is not None:
            lines.append(
                f"（部署阈值 {br['threshold']:.4f}，分数空间 {br['score_space']}，"
                "由训练集折外分数按最小期望代价选出）")
        for strat, r in m["strategies"].items():
            if r.get("selection_ineligible_reason"):
                lines.append(
                    f"- {_labeled(strat, r, STRATEGY_DISPLAY)} 未进选择池："
                    f"{r['selection_ineligible_reason']}")
        contrast = m.get("calibration_contrast")
        if contrast:
            lines.append("")
            lines.append("校准方法对照（主口径 vs 同表对照）：")
            lines.append(
                f"- CV 折外期望代价：{contrast['primary_method']} "
                f"{contrast['cv_cost_primary']:g} vs {contrast['contrast_method']} "
                f"{contrast['cv_cost_contrast']:g}（差 {contrast['cv_cost_delta']:+g}）")
            lines.append(
                f"- 若保留主口径、额外把对照放进选择池，选出的会是 "
                f"**{STRATEGY_DISPLAY[contrast['selected_with_contrast_in_pool']]}"
                f"**，与当前选择"
                f"{'不同' if contrast['would_change_selection'] else '相同'}；"
                "这不代表移除主口径后替换校准方法的结果")
            lines.append(f"- 说明：{contrast['note']}")
        diag_rows = [(s, r) for s, r in m["strategies"].items()
                     if r.get("calibration_diagnostics")]
        if diag_rows:
            lines.append("")
            lines.append("校准正确性检查（与篡改检测是两类防线，互相顶替不了）：")
            for strat, r in diag_rows:
                d = r["calibration_diagnostics"]
                verdict = "通过" if d["passed"] else "未通过"
                auc = ("N/A" if d["auc_raw"] is None or d["auc_calibrated"] is None
                       else f"{d['auc_raw']:.4f} → {d['auc_calibrated']:.4f}")
                corr = ("N/A" if d["oof_corr_with_label"] is None
                        else f"{d['oof_corr_with_label']:+.4f}")
                lines.append(
                    f"- {_labeled(strat, r, STRATEGY_DISPLAY)}：{verdict}"
                    f"｜单调非递减 {d['monotonic_non_decreasing']}"
                    f"｜AUC 校准前后 {auc}"
                    f"（不同取值 {d['n_distinct_raw']} → {d['n_distinct_calibrated']}，"
                    f"{'并出了新并列' if d['ties_introduced'] else '未并出新并列'}）"
                    f"｜折外 Brier {d['oof_brier']:.6f} vs 常数先验 "
                    f"{d['brier_constant_prior']:.6f}"
                    f"（改善 {d['oof_brier_improvement']:+.6f}）"
                    f"｜折外概率与标签相关 {corr}")
                for p in d["structural_problems"] + d["quality_problems"]:
                    lines.append(f"  - 未过：{p}")
        dep = m.get("deployment") or {}
        if dep:
            lines.append("")
            lines.append("部署：")
            lines.append(
                f"- 工作点产物（整条已拟合链路 + 阈值；行为指纹 分数 "
                f"`{dep.get('behavior_probe_sha256')}` / 最终预测 "
                f"`{dep.get('behavior_probe_predictions_sha256')}`）："
                f"`{dep.get('artifact')}`")
            lines.append(f"- 用法：{dep.get('artifact_usage', '')}")
            if dep.get("supported_in_main") and dep.get("main_config"):
                lines.append(
                    f"- 主流程可沿用该训练策略："
                    f"`imbalance.strategy: \"{dep['main_config']['strategy']}\"`")
            else:
                lines.append("- 该工作点**不可**通过改主流程配置复现")
            if dep.get("note"):
                lines.append(f"- 说明：{dep['note']}")
            lines.append(
                "- 特征链路差异：实验用折内 F 检验选特征，主流程用四方法投票，"
                "两者特征集合不同，主流程指标须重新评估，不等同本表数字")
            lines.append(
                "- 产物校验的保证范围（如实声明）：加载时先按本目录 "
                "`imbalance_comparison.json` 记录的文件 SHA-256 核对整份产物"
                "（该步排在反序列化之前；清单缺失则跳过并告警），再做产物自检 —— "
                "结构版本必填、无默认值、缺失即拒绝；校准语义与推理图按已拟合对象"
                "**重算**后比对（不采信记录值）；行为指纹比对上线路径对内置探针输入"
                "给出的**校准后分数与最终预测**（自检与上线调用的是同一个决策函数，"
                "不存在两条路径分歧的缝；校准器在该函数内部，不是它外面的后处理）；"
                "推理字段摘要覆盖阈值、分数空间、校准方法与正类索引、推理图、所选特征、"
                "成本假设、探针记录与其阈值两侧覆盖、校准前分数摘要与运行环境。")
            lines.append(
                "- **必定检出**：重训、换训练数据范围、超参、随机种子、预处理"
                "统计量、特征选择结果、模型系数或树内容、类别映射、阈值、分数"
                "空间、校准器**整体**替换、校准方法改写、正类索引调换、绕过校准器"
                "直接读基础分数、探针列名、整份产物替换、文件损坏 —— 这些正是事故"
                "真实发生的形态。")
            cov = dep.get("probe_coverage") or {}
            method = (dep.get("calibration") or {}).get("method")
            if method == "isotonic":
                piecewise = (
                    "本产物的校准器是 isotonic（节点间线性插值，可含常数平台），"
                    f"{cov.get('n_rows', PROBE_ROWS)} 行探针产生 "
                    f"{cov.get('distinct_scores', '若干')} 个不同输出概率，"
                    "这个数量不是已覆盖区间数；未被探针覆盖区间内的映射改动不能保证检出。")
            elif method:
                piecewise = (
                    f"本产物的校准器是 {method}；若改用 isotonic（节点间线性插值，"
                    "可含常数平台），未被探针覆盖区间内的映射改动不能保证检出。")
            else:
                piecewise = (
                    "若使用 isotonic（节点间线性插值，可含常数平台），"
                    "未被探针覆盖区间内的映射改动不能保证检出。")
            lines.append(
                "- **明确不覆盖**：按探针反向构造的单点改动（例如只改探针未"
                f"路由到的单个树叶）。{piecewise}行为指纹是**证据不是数学证明**，"
                "保证的是在这批探针输入上逐位一致。这是有限抽样的固有极限，不是实现"
                "缺陷 —— 覆盖它需要签名信任锚与写权限管控，而非更多探针行数。"
                "**本产物清单未签名**，产物与清单被一并改动仍会通过；该机制"
                "针对产物损坏、版本错配、拿错文件，不针对定向攻击。")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_imbalance_comparison(
    cfg: dict, X_train, y_train, X_test, y_test, out_dir, seed: int,
) -> dict:
    """不平衡策略对比主入口（full 模式流程阶段，由 pipeline 调用）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / COMPARISON_JSON

    cmp_cfg = (cfg.get("imbalance") or {}).get("comparison") or {}
    if not cmp_cfg.get("enabled", True):
        result = {"status": "disabled", "reason": "config imbalance.comparison.enabled=false"}
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("imbalance.comparison.enabled=false：跳过策略对比（已记录）")
        return result

    logger.info("===== 不平衡策略对比（漏检风险控制）=====")
    strategies = list(cmp_cfg.get("strategies", STRATEGIES))
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"未知不平衡策略: {unknown}，可选: {list(STRATEGIES)}")
    models_cfg = cmp_cfg.get("models", "all")
    model_names = registry_names() if models_cfg == "all" else list(models_cfg)
    folds = int(cmp_cfg.get("cv_folds", 5))
    fn_cost, fp_cost = canonical_costs(cfg)
    fs_cfg = cfg.get("feature_selection") or {}
    n_features = min(int(fs_cfg.get("n_features_to_select", 40)), X_train.shape[1])
    logger.info(
        "严格链路：raw 特征 %d 维 → 折内 F 检验选 %d 维；策略选择依据=训练侧 CV 期望代价",
        X_train.shape[1], n_features)

    result: dict = {
        "status": "failed",  # 成功结束时改写为 ok
        "random_seed": int(seed),
        "cv_folds": folds,
        "costs": {"fn": fn_cost, "fp": fp_cost},
        "strategies": strategies,
        "selection_basis": SELECTION_BASIS,
        "n_features": int(n_features),
        # 审计与排障用：行为指纹不符时据此分辨"换了环境"与"文件被改"
        "runtime_env": _runtime_env(),
        "leakage_control": (
            "填充/标准化/特征选择/重采样封装进同一 Pipeline，只在训练折/训练集"
            "拟合；校准器同样折内拟合；阈值只用训练侧 OOF 分数（校准类为完全折外的"
            "校准概率）；策略选择只用训练侧 CV 期望代价"),
        "test_set": {"n": int(len(y_test)), "n_fail": int((y_test == 1).sum())},
        "models": {},
    }
    cost_sensitivity_inputs: dict = {}
    try:
        for model_name in model_names:
            display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            strat_results: dict = {}
            for strat in strategies:
                r = _evaluate_one_strategy(
                    strat, model_name, X_train, y_train, X_test, y_test,
                    seed, folds, fn_cost, fp_cost, cfg, n_features,
                )
                strat_results[strat] = r
                logger.info(
                    "%s / %s: CV代价=%g | 测试: BER=%.3f 召回=%.3f FN=%d FP=%d 代价=%g",
                    display, STRATEGY_DISPLAY[strat], r["cv"]["CV期望代价"],
                    r["test"]["BER"], r["test"]["召回率"], r["confusion"]["fn"],
                    r["confusion"]["fp"], r["expected_cost"],
                )
            _delta_vs_baseline(strat_results)
            pr_png = _plot_pr_curves(model_name, strat_results, y_test, out_dir)
            cm_png = _plot_confusion_grid(model_name, strat_results, out_dir)
            # 策略选择只允许用训练侧信息：按 CV 折外期望代价取最小。选择池排除同表对照
            # 与正确性未过关的校准行——"全过才可标为可部署"落成过滤器，而不是文档里的承诺
            pool = [s for s, r in strat_results.items() if r["selection_eligible"]]
            if not pool:
                raise ValueError(
                    f"{display}: 没有任何策略可进工作点选择池 —— "
                    + "；".join(f"{s}: {r['selection_ineligible_reason']}"
                               for s, r in strat_results.items()))
            best = min(pool, key=lambda s: (strat_results[s]["cv"]["CV期望代价"], s))
            contrast = _calibration_contrast(strat_results, best, pool)
            # 成本敏感性复用主口径校准工作点的训练侧折外分数（存在时），落独立产物
            cs_source = strat_results.get(CALIBRATED_PRIMARY)
            if cs_source is not None:
                cost_sensitivity_inputs[display] = {
                    "result": cs_source, "strategy": CALIBRATED_PRIMARY}
            # 工作点整链路落盘后再剔除不可序列化字段
            op_pkl, op_sha = _save_operating_point(
                model_name, best, strat_results[best], out_dir, result["costs"])
            deployment = _deployment_block(
                best, model_name, strat_results[best], op_pkl)
            result["models"][display] = {
                "strategies": strat_results,
                "selection_pool": pool,
                "best_by_cv_cost": best,
                "calibration_contrast": contrast,
                "deployment": deployment,
                "artifacts": {
                    "pr_curve": pr_png, "confusion": cm_png,
                    "operating_point": op_pkl,
                    # 产物文件哈希的信任锚只能放在产物之外，故记在本 json
                    "operating_point_sha256": op_sha,
                },
            }
            logger.info(
                "%s 最优工作点（按训练侧 CV 期望代价，池 %s）: %s | 测试集独立验证代价=%g",
                display, pool, STRATEGY_DISPLAY[best],
                strat_results[best]["expected_cost"])
        if cost_sensitivity_inputs:
            cs = run_cost_sensitivity(
                cfg, cost_sensitivity_inputs, y_train, y_test, out_dir,
                fn_cost, fp_cost)
            # 本 json 只留引用与结论位，明细在独立产物里，两份不各存一套数字
            result["cost_sensitivity"] = {
                "artifact": COST_SENSITIVITY_JSON,
                "markdown": COST_SENSITIVITY_MD,
                "ratios": [g["ratio"] for g in cs["grid"]],
                "changes_operating_point": {
                    d: m["changes_operating_point"] for d, m in cs["models"].items()},
            }
        else:
            logger.info("未启用 %s 策略：跳过成本敏感性（无可复用的折外校准分数）",
                        CALIBRATED_PRIMARY)
        result["status"] = "ok"
        _write_markdown(result, out_dir / COMPARISON_MD)
        logger.info("[不平衡对比] 结果已保存: %s / %s", json_path, out_dir / COMPARISON_MD)
    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("不平衡策略对比失败")
        raise
    finally:
        # 不可序列化 / 只供中间步骤的字段统一在此剔除，成功与失败路径共用一处
        for m in result.get("models", {}).values():
            for r in (m.get("strategies") or {}).values():
                for key in ("scores", "y_pred", "estimator", "probe_input", "oof_scores"):
                    r.pop(key, None)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    return result

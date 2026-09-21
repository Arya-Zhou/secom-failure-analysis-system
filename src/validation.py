"""评估协议层：一个重采样总体、两个角色——第 0 个划分产正式产物，全体划分产区间。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# reference（正式产物）划分在重采样总体中的序号。产物与区间同源的全部依据就是这个 0。
ARTIFACT_SPLIT_INDEX = 0

# 区间下/上分位。固定为模块常量而非配置项：它是报告口径，换一次全部历史数字失去可比性。
INTERVAL_LOW_PCT = 5.0
INTERVAL_HIGH_PCT = 95.0

# bootstrap 重抽次数（对 R 个折级值的均值做区间）。
BOOTSTRAP_ROUNDS = 2000

INTERVAL_METHODS = ("percentile", "bootstrap")


def resolve_artifact_split(cfg: dict) -> dict:
    """读 artifact_split 并做迁移守护：旧 split 段与新 validation 段共存即报错。"""
    legacy = cfg.get("split")
    if legacy is not None and cfg.get("validation") is not None:
        raise ValueError(
            "检测到旧的 split 配置段（含 split.stratify）与新的 validation 段共存："
            "划分口径有两个来源，无法判断以哪个为准。请删除 split 段，"
            "改用 artifact_split（reference 划分）+ validation.resample（重采样总体）")
    spec = cfg.get("artifact_split")
    if not isinstance(spec, dict):
        raise ValueError("config 缺少 artifact_split（reference 产物的划分口径）")
    protocol = spec.get("protocol")
    if protocol != "stratified_shuffle":
        raise ValueError(
            f"artifact_split.protocol 仅支持 'stratified_shuffle'，收到 {protocol!r}："
            "reference 划分必须是重采样总体的第 0 个划分，换协议即失去同源关系")
    if spec.get("test_size") is None or spec.get("seed") is None:
        raise ValueError("artifact_split 缺少 test_size 或 seed")
    return {
        "protocol": protocol,
        "test_size": float(spec["test_size"]),
        "seed": int(spec["seed"]),
    }


def resolve_resample(cfg: dict) -> dict:
    """读 validation.resample。默认关闭：R 次重采样是分钟到小时级开销，不进日常运行路径。"""
    node = ((cfg.get("validation") or {}).get("resample") or {})
    interval = node.get("interval", "percentile")
    if interval not in INTERVAL_METHODS:
        raise ValueError(
            f"validation.resample.interval 仅支持 {list(INTERVAL_METHODS)}，收到 {interval!r}")
    n_splits = int(node.get("n_splits", 20))
    if n_splits < 1:
        raise ValueError("validation.resample.n_splits 必须 >= 1")
    return {
        "enabled": bool(node.get("enabled", False)),
        "n_splits": n_splits,
        "interval": interval,
        "paired_diff": bool(node.get("paired_diff", True)),
        "n_jobs": int(node.get("n_jobs", 1)),
    }


def resolve_temporal_holdout(cfg: dict) -> dict:
    """读 validation.temporal_holdout。本协议不提供 stratify——打乱时间顺序就不是时间序了。"""
    node = ((cfg.get("validation") or {}).get("temporal_holdout") or {})
    if "stratify" in node:
        raise ValueError(
            "validation.temporal_holdout 不接受 stratify：分层会打乱时间顺序，"
            "使该协议退化为随机划分。正负样本不足时本协议显式失败，不靠分层兜底")
    test_fraction = float(node.get("test_fraction", 0.2))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("validation.temporal_holdout.test_fraction 必须落在 (0, 1)")
    gap = int(node.get("gap", 0))
    if gap < 0:
        raise ValueError("validation.temporal_holdout.gap 不能为负")
    bins = int(node.get("prior_drift_bins", 5))
    if bins < 2:
        raise ValueError("validation.temporal_holdout.prior_drift_bins 必须 >= 2")
    return {
        "enabled": bool(node.get("enabled", False)),
        "test_fraction": test_fraction,
        "gap": gap,
        "prior_drift_bins": bins,
    }


def make_resample_population(test_size: float, seed: int, n_splits: int):
    """重采样总体本身。reference 划分 := 该总体的第 0 个划分（附录 A-3 实测与
    train_test_split(random_state=seed, stratify=y) 逐位等价，且与 n_splits 取值无关）。"""
    return StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed)


def resample_splits(y, test_size: float, seed: int, n_splits: int) -> list[tuple]:
    """列出总体的全部划分，元素为 (训练集标签索引, 测试集标签索引)。只用标签：分层划分与特征取值无关。"""
    pop = make_resample_population(test_size, seed, n_splits)
    idx = pd.Index(y.index)
    return [(idx[tr], idx[te]) for tr, te in pop.split(np.zeros(len(y)), y)]


def artifact_split(y, test_size: float, seed: int, n_splits: int = 1) -> tuple:
    """取 reference 划分 = 总体第 0 个划分。n_splits 只影响总体规模，不影响第 0 个。"""
    return resample_splits(y, test_size, seed, max(1, n_splits))[ARTIFACT_SPLIT_INDEX]


def temporal_holdout_split(timestamps, y, test_fraction: float, gap: int = 0) -> tuple:
    """按 timestamp 排序取前段训练、后段测试；任一侧缺类别即显式失败（不为分层打乱时序）。"""
    ts = pd.Series(timestamps).loc[y.index]
    if ts.isna().any():
        raise ValueError(f"时间序划分需要完整 timestamp，有 {int(ts.isna().sum())} 条缺失")
    order = ts.sort_values(kind="stable").index
    n = len(order)
    n_test = int(round(n * test_fraction))
    if n_test < 1 or n_test >= n:
        raise ValueError(f"时间序划分: test_fraction={test_fraction} 在 n={n} 下切不出两段")
    train_idx = order[: n - n_test - gap]
    test_idx = order[n - n_test:]
    if len(train_idx) < 1:
        raise ValueError(f"时间序划分: gap={gap} 把训练段吃空了（n={n}, n_test={n_test}）")
    for name, part in (("训练段", train_idx), ("测试段", test_idx)):
        classes = set(pd.Series(y).loc[part].unique())
        if classes != {0, 1}:
            raise ValueError(
                f"时间序划分的{name}只含类别 {sorted(classes)}，缺少另一类："
                "本协议不提供 stratify，正负样本不足即失败")
    return train_idx, test_idx


def prior_drift_table(timestamps, y, n_bins: int = 5) -> dict:
    """按 timestamp 等样本数分箱的失败率表：时间序协议为何与随机协议不可互换，先看这张表。"""
    ts = pd.Series(timestamps).loc[y.index]
    if ts.isna().any():
        raise ValueError(f"先验漂移刻画需要完整 timestamp，有 {int(ts.isna().sum())} 条缺失")
    if n_bins < 2:
        raise ValueError("prior_drift_table 的 n_bins 必须 >= 2")
    order = ts.sort_values(kind="stable").index
    labels = pd.Series(y).loc[order].to_numpy().astype(int)
    stamps = ts.loc[order]
    bins = []
    for k, (idx_part, lab_part) in enumerate(
            zip(np.array_split(np.asarray(order), n_bins),
                np.array_split(labels, n_bins)), start=1):
        n, fails = int(lab_part.size), int((lab_part == 1).sum())
        bins.append({
            "bin": k, "n": n, "failures": fails,
            "failure_rate": float(fails / n) if n else None,
            "start": str(stamps.loc[idx_part[0]]), "end": str(stamps.loc[idx_part[-1]]),
        })
    rates = [b["failure_rate"] for b in bins if b["failure_rate"]]
    total_fail = int((labels == 1).sum())
    return {
        "n_bins": int(n_bins), "n": int(labels.size), "failures": total_fail,
        "overall_failure_rate": float(total_fail / labels.size) if labels.size else None,
        "bins": bins,
        "max_over_min_ratio": float(max(rates) / min(rates)) if rates else None,
        "first_bin_share_of_failures": (
            float(bins[0]["failures"] / total_fail) if total_fail else None),
        "span": [str(stamps.iloc[0]), str(stamps.iloc[-1])],
    }


def point_vs_interval(value, iv: dict | None) -> dict | None:
    """把一个单点放进另一协议的区间里读。时间序只有一条划分、没有自己的区间，
    所以这里刻意不叫"区间重叠"——两侧不对称，说成重叠就是把单点当成了区间。"""
    if iv is None or value is None or np.isnan(float(value)):
        return None
    v = float(value)
    return {
        "value": v, "interval_point": iv["point"],
        "interval": [iv["low"], iv["high"]],
        "low_pct": iv["low_pct"], "high_pct": iv["high_pct"],
        "within_interval": bool(iv["low"] <= v <= iv["high"]),
        "delta_vs_interval_point": v - iv["point"],
    }


def _clean(values) -> np.ndarray:
    """丢掉 None / NaN（如岭分类器无 Brier）；区间只对算得出来的那部分成立。"""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[~np.isnan(arr)]


def percentile_interval(values) -> dict | None:
    """分位区间：刻画指标在重采样下的分布本身（不是均值的不确定度）。"""
    arr = _clean(values)
    if arr.size == 0:
        return None
    return {
        "method": "percentile", "n": int(arr.size),
        "point": float(np.median(arr)), "mean": float(np.mean(arr)),
        "low": float(np.percentile(arr, INTERVAL_LOW_PCT)),
        "high": float(np.percentile(arr, INTERVAL_HIGH_PCT)),
        "low_pct": INTERVAL_LOW_PCT, "high_pct": INTERVAL_HIGH_PCT,
    }


def bootstrap_interval(values, seed: int, rounds: int = BOOTSTRAP_ROUNDS) -> dict | None:
    """bootstrap 区间：刻画均值的不确定度，比分位区间窄，两者不可混用。"""
    arr = _clean(values)
    if arr.size == 0:
        return None
    rng = np.random.RandomState(seed)
    means = arr[rng.randint(0, arr.size, size=(rounds, arr.size))].mean(axis=1)
    return {
        "method": "bootstrap", "n": int(arr.size), "rounds": int(rounds),
        "point": float(np.mean(arr)), "mean": float(np.mean(arr)),
        "low": float(np.percentile(means, INTERVAL_LOW_PCT)),
        "high": float(np.percentile(means, INTERVAL_HIGH_PCT)),
        "low_pct": INTERVAL_LOW_PCT, "high_pct": INTERVAL_HIGH_PCT,
    }


def interval(values, method: str, seed: int) -> dict | None:
    if method == "percentile":
        return percentile_interval(values)
    if method == "bootstrap":
        return bootstrap_interval(values, seed)
    raise ValueError(f"未知的区间方法: {method!r}")


def paired_diff_interval(values_a, values_b, method: str, seed: int) -> dict | None:
    """配对差值区间：同一批划分上逐个相减再求区间。两组独立区间重叠不等于无差异。"""
    a, b = np.asarray(values_a, dtype=object), np.asarray(values_b, dtype=object)
    if a.shape != b.shape:
        raise ValueError("配对差值要求两组来自同一批划分，长度必须相同")
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not np.isnan(float(x)) and not np.isnan(float(y))]
    if not pairs:
        return None
    diffs = [float(x) - float(y) for x, y in pairs]
    res = interval(diffs, method, seed)
    if res is not None:
        # 区间不含 0 才谈得上"有差异"；这一位是配对差值唯一要读的结论
        res["excludes_zero"] = bool(res["low"] > 0 or res["high"] < 0)
    return res


def reference_percentile(ref_value, values) -> float | None:
    """reference 指标在 R 个折级值中的百分位。落在 5-95 之外要查因（见 §4.4.3）。"""
    arr = _clean(values)
    if ref_value is None or arr.size == 0 or np.isnan(float(ref_value)):
        return None
    return float((arr <= float(ref_value)).mean() * 100.0)

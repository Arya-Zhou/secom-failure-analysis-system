"""从逐轮训练证据计算候选分档、相关组、排名一致性与描述性收敛趋势。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .stability import classify_candidate, selection_frequencies

TIERS = ("高稳定候选", "探索性候选", "当前证据不足")
METHOD_RANKS = ("F检验排名", "互信息排名", "RFE排名", "随机森林排名")


def correlation_groups(train_features: pd.DataFrame, threshold: float) -> list[dict]:
    """按绝对 Spearman 阈值连边取连通分量；常量或全空列保留为独立组。"""
    names = list(train_features.columns)
    if not names or len(names) != len(set(names)) or not 0 < threshold <= 1:
        raise ValueError("相关组需要唯一特征列及 (0, 1] 阈值")
    correlations = train_features.corr(method="spearman").to_numpy()
    adjacent = np.isfinite(correlations) & (np.abs(correlations) >= threshold)
    remaining = set(range(len(names)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        members = set()
        while pending:
            current = pending.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            members.add(current)
            pending.extend(set(np.flatnonzero(adjacent[current])).intersection(remaining))
        groups.append({"group": f"G{len(groups) + 1:03d}",
                       "members": [names[position] for position in sorted(members)]})
    return groups


def direction_evidence(coefficients: list[float]) -> dict:
    """一致率以入选轮次为分母；零系数不提供方向且不提高一致率。"""
    values = np.asarray(coefficients, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("方向系数必须为有限数值")
    positive, negative = int((values > 0).sum()), int((values < 0).sum())
    evidence_count = positive + negative
    return {
        "positive": positive, "negative": negative, "zero": int((values == 0).sum()),
        "evidence_count": evidence_count,
        "consistency": max(positive, negative) / len(values) if evidence_count else None,
        "majority_sign": (int(np.sign(positive - negative)) if evidence_count else None),
    }


def _validate_records(records: list[dict], feature_names: list[str], spec: dict,
                      analysis: dict) -> None:
    if [record["split"] for record in records] != list(range(spec["n_resamples"])):
        raise ValueError("逐轮证据必须按顺序包含全部预定划分")
    selection_frequencies([record["selected_features"] for record in records],
                          feature_names, spec["n_resamples"])
    for record in records:
        ranks, selected = record["ranks"], record["selected_features"]
        if not ranks or set(ranks).difference(feature_names):
            raise ValueError("排名包含未知特征或缺少可用特征")
        if set(record["coefficients"]) != set(selected):
            raise ValueError("方向系数必须与本轮入选特征逐项对应")
        informative = record["nonconstant_features"]
        if len(informative) != len(set(informative)) or set(informative).difference(ranks):
            raise ValueError("非恒定特征必须属于本轮排名全集")
        scores = {}
        for feature in feature_names:
            if feature not in ranks:
                continue
            row = ranks[feature]
            if set(row) != {*METHOD_RANKS, "平均排名", "综合排名"}:
                raise ValueError("每轮必须保留四方法投票的完整排名")
            values = np.asarray(list(row.values()), dtype=float)
            if not np.isfinite(values).all() or (values < 1).any() or (values > len(ranks)).any():
                raise ValueError("投票排名必须为有效有限排名")
            average = sum(row[method] for method in METHOD_RANKS) / len(METHOD_RANKS)
            if row["平均排名"] != average:
                raise ValueError("平均排名与四方法投票不一致")
            scores[feature] = average
        combined = pd.Series(scores).rank()
        if any(ranks[feature]["综合排名"] != combined[feature] for feature in scores):
            raise ValueError("综合排名与平均排名不一致")
        expected = sorted(scores, key=scores.get)[:analysis["top_k"]]
        if selected != expected:
            raise ValueError("入选记录与四方法投票前 k 项不一致")
        direction_evidence(list(record["coefficients"].values()))


def _candidate_rows(records, feature_names, reference_features, groups, spec) -> list[dict]:
    frequencies = selection_frequencies([record["selected_features"] for record in records],
                                        feature_names, len(records))
    memberships = {feature: group["group"] for group in groups for feature in group["members"]}
    if set(memberships) != set(feature_names) or sum(len(group["members"]) for group in groups) != len(feature_names):
        raise ValueError("相关组必须无重叠地覆盖特征全集")
    candidates = []
    for feature in feature_names:
        coefficients = [record["coefficients"][feature] for record in records
                        if feature in record["coefficients"]]
        ranks = [record["ranks"][feature]["综合排名"] for record in records
                 if feature in record["ranks"]]
        direction = direction_evidence(coefficients)
        frequency = frequencies[feature]
        candidates.append({
            "feature": feature, "selected_count": len(coefficients),
            "selection_frequency": frequency, "in_reference": feature in reference_features,
            "available_splits": len(ranks), "median_rank": float(np.median(ranks)) if ranks else None,
            "direction": direction, "correlation_group": memberships[feature],
            "tier": classify_candidate(frequency, direction["consistency"], spec),
            "tiers_by_threshold": {
                str(threshold): classify_candidate(frequency, direction["consistency"], spec, threshold)
                for threshold in spec["frequency_thresholds"]},
        })
    return sorted(candidates, key=lambda row: (-row["selection_frequency"], row["feature"]))


def _sensitivity(candidates, spec) -> dict:
    return {
        str(threshold): {
            "counts_by_tier": {tier: sum(row["tiers_by_threshold"][str(threshold)] == tier
                                         for row in candidates) for tier in TIERS},
            "high_stability_features": [row["feature"] for row in candidates
                                        if row["tiers_by_threshold"][str(threshold)] == TIERS[0]],
            "outside_reference_high_frequency": [row["feature"] for row in candidates
                                                  if not row["in_reference"]
                                                  and row["selection_frequency"] >= threshold],
        } for threshold in spec["frequency_thresholds"]
    }


def distribution(values: list) -> dict:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("分布中不允许非有限值")
    result = {"n": len(finite), "missing": len(values) - len(finite)}
    result.update(dict.fromkeys(("min", "p05", "median", "p95", "max"), None))
    if len(finite):
        result.update(zip(("min", "p05", "median", "p95", "max"),
                          np.quantile(finite, [0, 0.05, 0.5, 0.95, 1]).tolist()))
    return result


def jaccard(first, second) -> float | None:
    first, second = set(first), set(second)
    return len(first.intersection(second)) / len(first.union(second)) if first or second else None


def pairwise_stability(records: list[dict], reference_features: list[str]) -> dict:
    """Spearman 只比较双方非恒定特征；Top-k 重合度不补入缺失特征。"""
    pairs = []
    for first, second in combinations(records, 2):
        common = sorted(set(first["nonconstant_features"]).intersection(second["nonconstant_features"]))
        first_ranks = pd.Series([first["ranks"][feature]["平均排名"] for feature in common], dtype=float)
        second_ranks = pd.Series([second["ranks"][feature]["平均排名"] for feature in common], dtype=float)
        spearman = None
        if len(common) >= 2 and first_ranks.nunique() > 1 and second_ranks.nunique() > 1:
            spearman = float(first_ranks.corr(second_ranks, method="spearman"))
        pairs.append({"first_split": first["split"], "second_split": second["split"],
                      "n_common_nonconstant": len(common), "spearman": spearman,
                      "top_k_jaccard": jaccard(first["selected_features"], second["selected_features"])})
    reference_overlap = [{"split": record["split"],
                          "top_k_jaccard": jaccard(record["selected_features"], reference_features)}
                         for record in records]
    return {
        "pairs": pairs,
        "spearman": distribution([pair["spearman"] for pair in pairs]),
        "top_k_jaccard": distribution([pair["top_k_jaccard"] for pair in pairs]),
        "reference_overlap": reference_overlap,
        "reference_top_k_jaccard": distribution([row["top_k_jaccard"] for row in reference_overlap]),
        "interpretation": "descriptive_pairwise_distribution_not_independent_confidence_intervals",
    }


def summarize_stability(records, feature_names, reference_features, groups, spec, analysis) -> dict:
    """以逐轮证据为唯一来源生成三档清单及三档敏感性，不按结果更换阈值。"""
    if not reference_features or len(reference_features) != len(set(reference_features)):
        raise ValueError("reference 特征清单必须非空且唯一")
    if set(reference_features).difference(feature_names):
        raise ValueError("reference 特征清单超出特征全集")
    _validate_records(records, feature_names, spec, analysis)
    candidates = _candidate_rows(records, feature_names, reference_features, groups, spec)
    sensitivity = _sensitivity(candidates, spec)
    primary = sensitivity[str(spec["primary_frequency_threshold"])]
    group_summary = []
    for group in groups:
        counts = [len(set(group["members"]).intersection(record["selected_features"])) for record in records]
        group_summary.append({**group, "selected_count": sum(count > 0 for count in counts),
                              "selection_frequency": sum(count > 0 for count in counts) / len(records),
                              "mean_selected_members": float(np.mean(counts))})
    checkpoints = list(range(analysis["convergence_step"], len(records), analysis["convergence_step"]))
    checkpoints.append(len(records))
    trend, previous_frequencies, previous_high = [], None, None
    for count in checkpoints:
        rows = _candidate_rows(records[:count], feature_names, reference_features, groups, spec)
        current = _sensitivity(rows, spec)
        primary_checkpoint = current[str(spec["primary_frequency_threshold"])]
        frequencies = {row["feature"]: row["selection_frequency"] for row in rows}
        changes = ([abs(frequencies[feature] - previous_frequencies[feature]) for feature in feature_names]
                   if previous_frequencies is not None else None)
        high = primary_checkpoint["high_stability_features"]
        trend.append({
            "n_resamples": count, "frequency_threshold_sensitivity": current,
            "max_frequency_change": max(changes) if changes is not None else None,
            "mean_frequency_change": float(np.mean(changes)) if changes is not None else None,
            "high_stability_jaccard_to_previous": jaccard(high, previous_high) if previous_high is not None else None,
            "high_stability_added": sorted(set(high).difference(previous_high or [])),
            "high_stability_removed": sorted(set(previous_high or []).difference(high)),
        })
        previous_frequencies, previous_high = frequencies, high
    return {
        "candidates": candidates, "counts_by_tier": primary["counts_by_tier"],
        "frequency_threshold_sensitivity": sensitivity,
        "outside_reference_high_frequency": primary["outside_reference_high_frequency"],
        "correlation_groups": group_summary,
        "pairwise": pairwise_stability(records, reference_features),
        "convergence": {"status": "descriptive_only", "checkpoints": trend,
                        "best_n_resamples": None},
    }

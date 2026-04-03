from collections.abc import Sequence
from typing import cast

from ..schema.drugrec_task import DrugRecCase, GNNMetrics, GNNRecResult


def _get_ranked_drugids(
    result: GNNRecResult,
    top_k: int | None = None,
) -> list[str]:
    """提取排序结果中的药物 ID 列表。"""
    ranked_drugs = result["ranked_drugs"]
    if top_k is None:
        return [ranked_drug["drugid"] for ranked_drug in ranked_drugs]
    return [ranked_drug["drugid"] for ranked_drug in ranked_drugs[:top_k]]


def _get_hit(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算是否至少命中一个金标准药物。"""
    return 1.0 if gold_drugids.intersection(ranked_drugids) else 0.0


def _get_mrr(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算首个命中药物的倒数排名。"""
    for index, drugid in enumerate(ranked_drugids, start=1):
        if drugid in gold_drugids:
            return 1.0 / index
    return 0.0


def _get_precision_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算 top-k 药物集合的准确率。"""
    if not ranked_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(ranked_drugids)


def _get_recall_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算 top-k 药物集合的召回率。"""
    if not gold_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(gold_drugids)


def _get_f1(precision: float, recall: float) -> float:
    """根据准确率和召回率计算 F1。"""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_jaccard_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算 top-k 药物集合与金标准集合的 Jaccard。"""
    ranked_drugid_set = set(ranked_drugids)
    union = gold_drugids.union(ranked_drugid_set)
    if not union:
        return 0.0
    intersection = gold_drugids.intersection(ranked_drugid_set)
    return len(intersection) / len(union)


def _get_evidence_mrr(result: GNNRecResult) -> float:
    """计算证据排序结果的首个命中倒数排名。"""
    for ranked_evidence in result["ranked_evidences"]:
        if ranked_evidence["label"] == 1:
            return 1.0 / ranked_evidence["rank"]
    return 0.0


def _get_evidence_hit_at_5(result: GNNRecResult) -> float:
    """计算前五条证据是否命中正例。"""
    return 1.0 if any(
        ranked_evidence["label"] == 1
        for ranked_evidence in result["ranked_evidences"][:5]
    ) else 0.0


def get_gnn_metrics(
    case: DrugRecCase,
    result: GNNRecResult,
) -> GNNMetrics:
    """根据主模型结果计算药物与证据指标。"""
    gold_drugids = set(case["gold_drugids"])
    ranked_drugids = _get_ranked_drugids(result)
    top_5_drugids = _get_ranked_drugids(result, top_k=5)
    precision_at_5 = _get_precision_at_k(gold_drugids, top_5_drugids)
    recall_at_5 = _get_recall_at_k(gold_drugids, top_5_drugids)
    return {
        "hit": _get_hit(gold_drugids, ranked_drugids),
        "mrr": _get_mrr(gold_drugids, ranked_drugids),
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "f1_at_5": _get_f1(precision_at_5, recall_at_5),
        "jaccard_at_5": _get_jaccard_at_k(gold_drugids, top_5_drugids),
        "evidence_mrr": _get_evidence_mrr(result),
        "evidence_hit_at_5": _get_evidence_hit_at_5(result),
    }


def aggregate_gnn_metrics(
    metrics_list: Sequence[GNNMetrics],
) -> GNNMetrics:
    """对一组病例的全部 GNN 指标取平均。"""
    if not metrics_list:
        return {}
    metric_keys = sorted(
        {
            metric_key
            for metrics in metrics_list
            for metric_key in metrics
        }
    )
    return cast(
        GNNMetrics,
        {
            metric_key: sum(
                metrics[metric_key]
                for metrics in metrics_list
                if metric_key in metrics
            ) / len(metrics_list)
            for metric_key in metric_keys
        },
    )


__all__ = [
    "aggregate_gnn_metrics",
    "get_gnn_metrics",
]

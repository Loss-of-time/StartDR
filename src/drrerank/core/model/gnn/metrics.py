from collections.abc import Sequence
from dataclasses import fields

from ...schema import DrugRecCase
from .schema import GNNMetrics, GNNRecResult


def get_ranked_drugids(
    result: GNNRecResult,
    top_k: int | None = None,
) -> list[str]:
    ranked_drugs = result.ranked_drugs
    if top_k is None:
        return [ranked_drug.drugid for ranked_drug in ranked_drugs]
    return [ranked_drug.drugid for ranked_drug in ranked_drugs[:top_k]]


def get_hit(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    return 1.0 if gold_drugids.intersection(ranked_drugids) else 0.0


def get_mrr(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    for index, drugid in enumerate(ranked_drugids, start=1):
        if drugid in gold_drugids:
            return 1.0 / index
    return 0.0


def get_precision_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    if not ranked_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(ranked_drugids)


def get_recall_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    if not gold_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(gold_drugids)


def get_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_jaccard_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    ranked_drugid_set = set(ranked_drugids)
    union = gold_drugids.union(ranked_drugid_set)
    if not union:
        return 0.0
    intersection = gold_drugids.intersection(ranked_drugid_set)
    return len(intersection) / len(union)


def get_evidence_mrr(result: GNNRecResult) -> float:
    for ranked_evidence in result.ranked_evidences:
        if ranked_evidence.label == 1:
            return 1.0 / ranked_evidence.rank
    return 0.0


def get_evidence_hit_at_5(result: GNNRecResult) -> float:
    return 1.0 if any(
        ranked_evidence.label == 1
        for ranked_evidence in result.ranked_evidences[:5]
    ) else 0.0


def get_gnn_metrics(
    case: DrugRecCase,
    result: GNNRecResult,
) -> GNNMetrics:
    gold_drugids = set(case.gold_drugids)
    ranked_drugids = get_ranked_drugids(result)
    top_5_drugids = get_ranked_drugids(result, top_k=5)
    precision_at_5 = get_precision_at_k(gold_drugids, top_5_drugids)
    recall_at_5 = get_recall_at_k(gold_drugids, top_5_drugids)
    return GNNMetrics(
        hit=get_hit(gold_drugids, ranked_drugids),
        mrr=get_mrr(gold_drugids, ranked_drugids),
        precision_at_5=precision_at_5,
        recall_at_5=recall_at_5,
        f1_at_5=get_f1(precision_at_5, recall_at_5),
        jaccard_at_5=get_jaccard_at_k(gold_drugids, top_5_drugids),
        evidence_mrr=get_evidence_mrr(result),
        evidence_hit_at_5=get_evidence_hit_at_5(result),
    )


def aggregate_gnn_metrics(
    metrics_list: Sequence[GNNMetrics],
) -> GNNMetrics:
    if not metrics_list:
        return GNNMetrics()
    aggregated: dict[str, float] = {}
    for metric_field in fields(GNNMetrics):
        values = [
            value
            for metrics in metrics_list
            if (value := getattr(metrics, metric_field.name)) is not None
        ]
        if values:
            aggregated[metric_field.name] = sum(values) / len(metrics_list)
    return GNNMetrics(**aggregated)

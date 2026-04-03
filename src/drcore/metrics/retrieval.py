from collections.abc import Sequence

from ..schema import (
    DrugRecMedicine,
    DrugRecRecord,
    MetricsResult,
    SummaryResult,
)


def get_gold_medicines(patient: DrugRecRecord) -> list[DrugRecMedicine]:
    return patient["medicine"]


def get_gold_ids(patient: DrugRecRecord) -> set[str]:
    return {medicine["drugid"] for medicine in get_gold_medicines(patient)}


def hit_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    return 1.0 if gold_ids.intersection(retrieved_ids) else 0.0


def recall_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    if not gold_ids:
        return 0.0
    return len(gold_ids.intersection(retrieved_ids)) / len(gold_ids)


def mrr_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    for index, drugid in enumerate(retrieved_ids, start=1):
        if drugid in gold_ids:
            return 1.0 / index
    return 0.0


def get_metrics_result(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> MetricsResult:
    return {
        "hit": hit_at_k(gold_ids, retrieved_ids),
        "recall": recall_at_k(gold_ids, retrieved_ids),
        "mrr": mrr_at_k(gold_ids, retrieved_ids),
    }


def aggregate_metrics(metrics_list: Sequence[MetricsResult]) -> MetricsResult:
    if not metrics_list:
        return {
            "hit": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
        }
    total = len(metrics_list)
    return {
        "hit": sum(metrics["hit"] for metrics in metrics_list) / total,
        "recall": sum(metrics["recall"] for metrics in metrics_list) / total,
        "mrr": sum(metrics["mrr"] for metrics in metrics_list) / total,
    }


def get_summary(metrics_list: Sequence[MetricsResult]) -> SummaryResult:
    return {
        "patient_count": len(metrics_list),
        "failure_count": sum(1 for metrics in metrics_list if metrics["hit"] == 0.0),
    }

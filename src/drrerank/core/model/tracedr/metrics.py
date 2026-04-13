from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Protocol


# Protocol 的意思是：不要求对象必须继承某个类，只要“长得像”这个协议
class RankedAnswerLike(Protocol):
    id: str


@dataclass(slots=True)
class TraceDRMetrics:
    loss: float = 0.0
    p_at_1: float = 0.0
    mrr: float = 0.0
    h_at_5: float = 0.0
    answer_presence: float = 0.0
    jaccard: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    f1_at_5: float = 0.0


def get_ranked_answer_ids(
    answers: Sequence[RankedAnswerLike],
    top_k: int | None = None,
) -> list[str]:
    if top_k is None:
        return [answer.id for answer in answers]
    return [answer.id for answer in answers[:top_k]]


def get_hit(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    return 1.0 if gold_answer_ids.intersection(ranked_answer_ids) else 0.0


def get_mrr(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    for index, answer_id in enumerate(ranked_answer_ids, start=1):
        if answer_id in gold_answer_ids:
            return 1.0 / index
    return 0.0


def get_precision_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    if not ranked_answer_ids:
        return 0.0
    hit_count = len(gold_answer_ids.intersection(ranked_answer_ids))
    return hit_count / len(ranked_answer_ids)


def get_recall_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    if not gold_answer_ids:
        return 0.0
    hit_count = len(gold_answer_ids.intersection(ranked_answer_ids))
    return hit_count / len(gold_answer_ids)


def get_f1(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_jaccard_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    ranked_answer_id_set = set(ranked_answer_ids)
    union = gold_answer_ids.union(ranked_answer_id_set)
    if not union:
        return 0.0
    intersection = gold_answer_ids.intersection(ranked_answer_id_set)
    return len(intersection) / len(union)


def calculate_metrics(
    question_id: str,
    answers: Sequence[RankedAnswerLike],
    gold_answers: Sequence[str],
    k: int = 5,
) -> TraceDRMetrics:
    del question_id

    gold_answer_ids = set(gold_answers)
    ranked_answer_ids = get_ranked_answer_ids(answers)
    top_1_answer_ids = get_ranked_answer_ids(answers, top_k=1)
    top_k_answer_ids = get_ranked_answer_ids(answers, top_k=k)

    precision_at_k = get_precision_at_k(gold_answer_ids, top_k_answer_ids)
    recall_at_k = get_recall_at_k(gold_answer_ids, top_k_answer_ids)

    return TraceDRMetrics(
        p_at_1=get_precision_at_k(gold_answer_ids, top_1_answer_ids),
        mrr=get_mrr(gold_answer_ids, ranked_answer_ids),
        h_at_5=get_hit(gold_answer_ids, top_k_answer_ids),
        answer_presence=get_hit(gold_answer_ids, ranked_answer_ids),
        jaccard=get_jaccard_at_k(gold_answer_ids, top_k_answer_ids),
        precision_at_5=precision_at_k,
        recall_at_5=recall_at_k,
        f1_at_5=get_f1(precision_at_k, recall_at_k),
    )


def aggregate_metrics(
    metrics_list: Sequence[TraceDRMetrics],
) -> TraceDRMetrics:
    if not metrics_list:
        raise ValueError("指标列表为空，无法聚合。")

    aggregated: dict[str, float] = {}
    for metric_field in fields(TraceDRMetrics):
        values = [getattr(metrics, metric_field.name) for metrics in metrics_list]
        aggregated[metric_field.name] = sum(values) / len(values)
    return TraceDRMetrics(**aggregated)

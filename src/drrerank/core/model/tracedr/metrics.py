"""TraceDR、GAT、KGD 共享指标计算。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Protocol

from scipy.sparse import csr_matrix

from ...schema import DrugRecMedicine


class RankedAnswerLike(Protocol):
    """最小排序结果协议。"""

    id: str


@dataclass(slots=True)
class TraceDRMetrics:
    """TraceDR 系列模型通用指标。"""

    loss: float = 0.0
    p_at_1: float = 0.0
    mrr: float = 0.0
    h_at_5: float = 0.0
    answer_presence: float = 0.0
    ddi_rate: float = 0.0
    jaccard_similarity: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    f1_at_5: float = 0.0


def _get_ranked_answer_ids(
    answers: Sequence[RankedAnswerLike],
    top_k: int | None = None,
) -> list[str]:
    if top_k is None:
        return [answer.id for answer in answers]
    return [answer.id for answer in answers[:top_k]]


def _get_hit(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    return 1.0 if gold_answer_ids.intersection(ranked_answer_ids) else 0.0


def _get_mrr(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    index: int
    answer_id: str
    for index, answer_id in enumerate(ranked_answer_ids, start=1):
        if answer_id in gold_answer_ids:
            return 1.0 / index
    return 0.0


def _get_precision_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    if not ranked_answer_ids:
        return 0.0
    hit_count: int = len(gold_answer_ids.intersection(ranked_answer_ids))
    return hit_count / len(ranked_answer_ids)


def _get_recall_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    if not gold_answer_ids:
        return 0.0
    hit_count: int = len(gold_answer_ids.intersection(ranked_answer_ids))
    return hit_count / len(gold_answer_ids)


def _get_f1(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_jaccard_similarity_at_k(
    gold_answer_ids: set[str],
    ranked_answer_ids: Sequence[str],
) -> float:
    ranked_answer_id_set: set[str] = set(ranked_answer_ids)
    union: set[str] = gold_answer_ids.union(ranked_answer_id_set)
    if not union:
        return 0.0
    intersection: set[str] = gold_answer_ids.intersection(ranked_answer_id_set)
    return len(intersection) / len(union)


def _build_interaction_name_set(
    on_medicines: Sequence[DrugRecMedicine],
) -> set[str]:
    interaction_names: set[str] = set()
    medicine: DrugRecMedicine
    for medicine in on_medicines:
        interaction_name: str
        for interaction_name in (interaction.name for interaction in medicine.interaction):
            if interaction_name != "":
                interaction_names.add(interaction_name)
    return interaction_names


def _get_interaction_based_ddi_rate(
    predicted_answer_ids: Sequence[str],
    candidate_drug_map: Mapping[str, DrugRecMedicine],
    on_medicines: Sequence[DrugRecMedicine],
    k: int,
) -> float:
    if k <= 0 or not on_medicines:
        return 0.0

    interaction_names: set[str] = _build_interaction_name_set(on_medicines)
    if not interaction_names:
        return 0.0

    interaction_count: int = 0
    answer_id: str
    for answer_id in predicted_answer_ids:
        medicine: DrugRecMedicine | None = candidate_drug_map.get(answer_id)
        if medicine is None:
            continue
        # 目的：沿用参考实现口径，只要推荐药任一成分命中当前在用药相互作用名单，就记为一次 DDI。
        if any(
            ingredient.ingredient in interaction_names
            for ingredient in medicine.ingredients
            if ingredient.ingredient is not None
        ):
            interaction_count += 1
    return interaction_count / k


def _get_pairwise_ddi_rate(
    predicted_answer_ids: Sequence[str],
    ddi_adj: csr_matrix,
    drugid_to_index: Mapping[str, int],
) -> float:
    unique_answer_ids: list[str] = list(dict.fromkeys(predicted_answer_ids))
    predicted_indices: list[int] = [
        drugid_to_index[answer_id]
        for answer_id in unique_answer_ids
        if answer_id in drugid_to_index
    ]
    if len(predicted_indices) < 2:
        return 0.0

    interaction_count: int = 0
    pair_count: int = 0
    left_index: int
    right_index: int
    left_drug_id: int
    right_drug_id: int
    for left_index, left_drug_id in enumerate(predicted_indices):
        for right_index in range(left_index + 1, len(predicted_indices)):
            right_drug_id = predicted_indices[right_index]
            pair_count += 1
            if ddi_adj[left_drug_id, right_drug_id] != 0:
                interaction_count += 1
    return interaction_count / pair_count if pair_count else 0.0


def _get_ddi_rate(
    predicted_answer_ids: Sequence[str],
    k: int,
    candidate_drug_map: Mapping[str, DrugRecMedicine] | None = None,
    on_medicines: Sequence[DrugRecMedicine] | None = None,
    ddi_adj: csr_matrix | None = None,
    drugid_to_index: Mapping[str, int] | None = None,
) -> float:
    if candidate_drug_map is not None and on_medicines is not None:
        return _get_interaction_based_ddi_rate(
            predicted_answer_ids=predicted_answer_ids,
            candidate_drug_map=candidate_drug_map,
            on_medicines=on_medicines,
            k=k,
        )
    if ddi_adj is not None and drugid_to_index is not None:
        return _get_pairwise_ddi_rate(
            predicted_answer_ids=predicted_answer_ids,
            ddi_adj=ddi_adj,
            drugid_to_index=drugid_to_index,
        )
    return 0.0


def calculate_metrics(
    question_id: str,
    answers: Sequence[RankedAnswerLike],
    gold_answers: Sequence[str],
    k: int = 5,
    candidate_drug_map: Mapping[str, DrugRecMedicine] | None = None,
    on_medicines: Sequence[DrugRecMedicine] | None = None,
    ddi_adj: csr_matrix | None = None,
    drugid_to_index: Mapping[str, int] | None = None,
) -> TraceDRMetrics:
    """计算单样本排序指标。

    Args:
        question_id: 当前问题 id，仅用于兼容现有调用签名。
        answers: 排序后的候选答案序列。
        gold_answers: 金标药物 id 列表。
        k: top-k 指标的截断深度。
        candidate_drug_map: TraceDR / GAT 用的候选药物明细。
        on_medicines: TraceDR / GAT 当前在用药列表。
        ddi_adj: KGD 用的 1-based DDI 邻接矩阵。
        drugid_to_index: KGD 用的药物编号到 1-based 索引映射。

    Returns:
        单样本指标结果。
    """

    del question_id

    gold_answer_ids: set[str] = set(gold_answers)
    ranked_answer_ids: list[str] = _get_ranked_answer_ids(answers)
    top_1_answer_ids: list[str] = _get_ranked_answer_ids(answers, top_k=1)
    top_k_answer_ids: list[str] = _get_ranked_answer_ids(answers, top_k=k)

    precision_at_k: float = _get_precision_at_k(gold_answer_ids, top_k_answer_ids)
    recall_at_k: float = _get_recall_at_k(gold_answer_ids, top_k_answer_ids)

    return TraceDRMetrics(
        p_at_1=_get_precision_at_k(gold_answer_ids, top_1_answer_ids),
        mrr=_get_mrr(gold_answer_ids, ranked_answer_ids),
        h_at_5=_get_hit(gold_answer_ids, top_k_answer_ids),
        answer_presence=_get_hit(gold_answer_ids, ranked_answer_ids),
        ddi_rate=_get_ddi_rate(
            predicted_answer_ids=top_k_answer_ids,
            k=k,
            candidate_drug_map=candidate_drug_map,
            on_medicines=on_medicines,
            ddi_adj=ddi_adj,
            drugid_to_index=drugid_to_index,
        ),
        jaccard_similarity=_get_jaccard_similarity_at_k(gold_answer_ids, top_k_answer_ids),
        precision_at_5=precision_at_k,
        recall_at_5=recall_at_k,
        f1_at_5=_get_f1(precision_at_k, recall_at_k),
    )


def aggregate_metrics(
    metrics_list: Sequence[TraceDRMetrics],
) -> TraceDRMetrics:
    """聚合多条 TraceDR 系列指标。

    Args:
        metrics_list: 单样本指标列表。

    Returns:
        字段均值后的聚合指标。
    """

    if not metrics_list:
        raise ValueError("指标列表为空，无法聚合。")

    aggregated: dict[str, float] = {}
    metric_field: object
    for metric_field in fields(TraceDRMetrics):
        field_name: str = metric_field.name
        values: list[float] = [getattr(metrics, field_name) for metrics in metrics_list]
        aggregated[field_name] = sum(values) / len(values)
    return TraceDRMetrics(**aggregated)

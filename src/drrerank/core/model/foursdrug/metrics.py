"""4SDrug 指标计算。"""

from dataclasses import fields

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from .schema import FourSDrugMetrics

type FourSDrugRankedIds = npt.NDArray[np.int64]
type FourSDrugProbabilityArray = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def _build_ranked_drug_ids(probabilities: FourSDrugProbabilityArray) -> FourSDrugRankedIds:
    """按分数降序构造 1-based 药物排序结果。"""

    return np.argsort(probabilities)[::-1].astype(np.int64) + 1


def _build_predicted_drug_ids(
    probabilities: FourSDrugProbabilityArray,
    threshold: float,
) -> list[int]:
    """按阈值构造预测药物列表。"""

    predicted_indices: FourSDrugRankedIds = np.flatnonzero(probabilities >= threshold).astype(
        np.int64,
    )
    return [int(index) + 1 for index in predicted_indices]


def _get_average_precision(
    gold_drug_ids: set[int],
    ranked_drug_ids: FourSDrugRankedIds,
) -> float:
    """计算单样本平均准确率。"""

    if not gold_drug_ids:
        return 0.0

    hit_count: int = 0
    precision_sum: float = 0.0
    rank: int
    drug_id: np.int64
    for rank, drug_id in enumerate(ranked_drug_ids, start=1):
        if int(drug_id) not in gold_drug_ids:
            continue
        hit_count += 1
        precision_sum += hit_count / rank
        if hit_count == len(gold_drug_ids):
            break
    return precision_sum / len(gold_drug_ids)


def _get_ddi_rate(
    predicted_drug_ids: list[int],
    ddi_adj: csr_matrix,
) -> float:
    """计算单样本预测结果的 DDI 率。"""

    if len(predicted_drug_ids) < 2:
        return 0.0

    interaction_count: int = 0
    pair_count: int = 0
    left_index: int
    right_index: int
    left_drug_id: int
    right_drug_id: int
    for left_index, left_drug_id in enumerate(predicted_drug_ids):
        for right_index in range(left_index + 1, len(predicted_drug_ids)):
            right_drug_id = predicted_drug_ids[right_index]
            pair_count += 1
            if ddi_adj[left_drug_id, right_drug_id] != 0:
                interaction_count += 1
    return interaction_count / pair_count if pair_count else 0.0


def calculate_metrics(
    probabilities: FourSDrugProbabilityArray,
    gold_drug_ids: list[int],
    ddi_adj: csr_matrix,
    threshold: float,
    loss: float = 0.0,
) -> FourSDrugMetrics:
    """计算单样本 4SDrug 指标。

    Args:
        probabilities: 模型输出概率，药物维为 0-based。
        gold_drug_ids: 金标药物 id，保持 1-based。
        ddi_adj: 1-based DDI 邻接矩阵。
        threshold: 多标签预测阈值。
        loss: 样本损失。

    Returns:
        单样本指标结果。
    """

    gold_set: set[int] = set(dict.fromkeys(gold_drug_ids))
    ranked_drug_ids: FourSDrugRankedIds = _build_ranked_drug_ids(probabilities)
    predicted_drug_ids: list[int] = _build_predicted_drug_ids(probabilities, threshold)
    predicted_set: set[int] = set(predicted_drug_ids)
    top_1_ids: set[int] = {int(ranked_drug_ids[0])} if ranked_drug_ids.size else set()
    top_5_ids: list[int] = [int(drug_id) for drug_id in ranked_drug_ids[:5]]
    top_5_set: set[int] = set(top_5_ids)

    intersection_count: int = len(gold_set.intersection(predicted_set))
    union_count: int = len(gold_set.union(predicted_set))
    precision: float = intersection_count / len(predicted_set) if predicted_set else 0.0
    recall: float = intersection_count / len(gold_set) if gold_set else 0.0
    f1: float = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0

    mrr: float = 0.0
    rank: int
    drug_id: np.int64
    for rank, drug_id in enumerate(ranked_drug_ids, start=1):
        if int(drug_id) in gold_set:
            mrr = 1.0 / rank
            break

    top_5_hit_count: int = len(gold_set.intersection(top_5_set))
    precision_at_5: float = top_5_hit_count / len(top_5_ids) if top_5_ids else 0.0
    recall_at_5: float = top_5_hit_count / len(gold_set) if gold_set else 0.0
    f1_at_5: float = (
        2.0 * precision_at_5 * recall_at_5 / (precision_at_5 + recall_at_5)
        if precision_at_5 + recall_at_5 > 0.0
        else 0.0
    )

    return FourSDrugMetrics(
        loss=loss,
        ja=intersection_count / union_count if union_count else 0.0,
        prauc=_get_average_precision(gold_set, ranked_drug_ids),
        precision=precision,
        recall=recall,
        f1=f1,
        avg_drugs=float(len(predicted_drug_ids)),
        ddi_rate=_get_ddi_rate(predicted_drug_ids, ddi_adj),
        p_at_1=1.0 if gold_set.intersection(top_1_ids) else 0.0,
        mrr=mrr,
        hit_at_5=1.0 if top_5_hit_count > 0 else 0.0,
        precision_at_5=precision_at_5,
        recall_at_5=recall_at_5,
        f1_at_5=f1_at_5,
    )


def aggregate_metrics(metrics_list: list[FourSDrugMetrics]) -> FourSDrugMetrics:
    """聚合多条 4SDrug 指标。"""

    if not metrics_list:
        raise ValueError("指标列表为空，无法聚合。")

    aggregated: dict[str, float] = {}
    metric_field: object
    for metric_field in fields(FourSDrugMetrics):
        field_name: str = metric_field.name
        field_values: list[float] = [getattr(metrics, field_name) for metrics in metrics_list]
        aggregated[field_name] = sum(field_values) / len(field_values)
    return FourSDrugMetrics(**aggregated)

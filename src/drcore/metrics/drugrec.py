from collections.abc import Sequence

from ..schema.drugrec_task import DrugRecCase, DrugRecMetrics, DrugRecResult


def get_ranked_drugids(
    result: DrugRecResult,
    top_k: int | None = None,
) -> list[str]:
    """提取排序结果中的药物 ID 列表。"""
    ranked_drugs = result["ranked_drugs"]
    if top_k is None:
        return [ranked_drug["drugid"] for ranked_drug in ranked_drugs]
    return [ranked_drug["drugid"] for ranked_drug in ranked_drugs[:top_k]]


def get_hit(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算是否至少命中一个金标准药物。"""
    return 1.0 if gold_drugids.intersection(ranked_drugids) else 0.0


def get_mrr(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算首个命中药物的倒数排名。"""
    for index, drugid in enumerate(ranked_drugids, start=1):
        if drugid in gold_drugids:
            return 1.0 / index
    return 0.0


def get_precision_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算 top-k 药物集合的准确率。"""
    if not ranked_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(ranked_drugids)


def get_recall_at_k(
    gold_drugids: set[str],
    ranked_drugids: Sequence[str],
) -> float:
    """计算 top-k 药物集合的召回率。"""
    if not gold_drugids:
        return 0.0
    hit_count = len(gold_drugids.intersection(ranked_drugids))
    return hit_count / len(gold_drugids)


def get_f1(precision: float, recall: float) -> float:
    """根据准确率和召回率计算 F1。"""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_jaccard_at_k(
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


def get_drugrec_metrics(
    case: DrugRecCase,
    result: DrugRecResult,
) -> DrugRecMetrics:
    """根据通用推荐结果计算正式离线指标。"""
    gold_drugids = set(case["gold_drugids"])
    ranked_drugids = get_ranked_drugids(result)
    top_5_drugids = get_ranked_drugids(result, top_k=5)
    precision_at_5 = get_precision_at_k(gold_drugids, top_5_drugids)
    recall_at_5 = get_recall_at_k(gold_drugids, top_5_drugids)
    return {
        "hit": get_hit(gold_drugids, ranked_drugids),
        "mrr": get_mrr(gold_drugids, ranked_drugids),
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5,
        "f1_at_5": get_f1(precision_at_5, recall_at_5),
        "jaccard_at_5": get_jaccard_at_k(gold_drugids, top_5_drugids),
    }


def aggregate_drugrec_metrics(
    metrics_list: Sequence[DrugRecMetrics],
) -> DrugRecMetrics:
    """对一组病例的通用推荐指标取平均。"""
    if not metrics_list:
        return {}
    metric_keys = sorted(
        {
            metric_key
            for metrics in metrics_list
            for metric_key in metrics
        }
    )
    return {
        metric_key: sum(
            metrics[metric_key]
            for metrics in metrics_list
            if metric_key in metrics
        ) / len(metrics_list)
        for metric_key in metric_keys
    }

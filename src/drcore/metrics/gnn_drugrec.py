from collections.abc import Sequence

from ..schema.drugrec_task import DrugRecCase, GNNMetrics, GNNRecResult


def get_evidence_mrr(result: GNNRecResult) -> float:
    """计算证据排序结果的首个命中倒数排名。"""
    ranked_evidences = result.get("ranked_evidences")
    if not ranked_evidences:
        return 0.0
    for ranked_evidence in ranked_evidences:
        if ranked_evidence["label"] == 1:
            return 1.0 / ranked_evidence["rank"]
    return 0.0


def get_evidence_hit_at_5(result: GNNRecResult) -> float:
    """计算前五条证据是否命中正例。"""
    ranked_evidences = result.get("ranked_evidences")
    if not ranked_evidences:
        return 0.0
    return 1.0 if any(
        ranked_evidence["label"] == 1
        for ranked_evidence in ranked_evidences[:5]
    ) else 0.0


def get_gnn_metrics(
    case: DrugRecCase,
    result: GNNRecResult,
) -> GNNMetrics:
    """根据 GNN 扩展结果计算图专属离线指标。"""
    _ = case
    if "ranked_evidences" not in result:
        return {}
    return {
        "evidence_mrr": get_evidence_mrr(result),
        "evidence_hit_at_5": get_evidence_hit_at_5(result),
    }


def aggregate_gnn_metrics(
    metrics_list: Sequence[GNNMetrics],
) -> GNNMetrics:
    """对一组病例的 GNN 扩展指标取平均。"""
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

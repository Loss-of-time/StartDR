"""离线评测模块。"""

from .gnn import aggregate_gnn_metrics, get_gnn_metrics

__all__ = [
    "aggregate_gnn_metrics",
    "get_gnn_metrics",
]

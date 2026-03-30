from typing import TypedDict


class MetricsResult(TypedDict):
    hit: float
    recall: float
    mrr: float


class SummaryResult(TypedDict):
    patient_count: int
    failure_count: int


class RetriverEvalConfig(TypedDict):
    retriver_name: str
    input_path: str
    top_k: int
    sample_count: int


class RetriverEvalReport(TypedDict):
    config: RetriverEvalConfig
    summary: SummaryResult
    metrics: MetricsResult


__all__ = [
    "MetricsResult",
    "RetriverEvalConfig",
    "RetriverEvalReport",
    "SummaryResult",
]

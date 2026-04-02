from typing import Literal, TypedDict

import torch
from jaxtyping import Float

from .drugrec import DatasetSplit, DrugRecMedicine
from .metrics import MetricsResult

RerankerName = Literal["gnn_reranker"]


class RankedDrug(TypedDict):
    drugid: str
    score: float
    rank: int
    drug: DrugRecMedicine
    retrieval_score: float | None # TODO 为什么有两个score和rank
    retrieval_rank: int
    label: int


class RerankResult(TypedDict):
    patient_id: str
    split: DatasetSplit
    reranker: RerankerName
    ranked_drugs: list[RankedDrug]


class TrainMetrics(TypedDict, total=False): # TODO total=False 是什么意思
    loss: float
    hit: float
    recall: float
    mrr: float


class TrainStepOutput(TypedDict):
    loss: Float[torch.Tensor, ""] # TODO 我 jaxtyping 为什么注释一个空
    loss_value: float
    metrics: TrainMetrics


class EvalStepOutput(TypedDict):
    metrics: MetricsResult
    results: list[RerankResult]


class RerankerCheckpoint(TypedDict):
    reranker: RerankerName
    model_state_dict: dict[str, torch.Tensor]
    init_kwargs: object


__all__ = [
    "EvalStepOutput",
    "RankedDrug",
    "RerankerCheckpoint",
    "RerankResult",
    "RerankerName",
    "TrainMetrics",
    "TrainStepOutput",
]

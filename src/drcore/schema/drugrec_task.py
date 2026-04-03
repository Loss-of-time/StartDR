from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal, TypedDict

import torch
from jaxtyping import Float

from .drugrec import DatasetSplit, DrugRecMedicine, DrugRecRecord
from .model.gnn_reranker import GNNNodeType, NumericFeatureStats
from .patient_candidate_set import CandidateDrug

type DrugRecModelName = Literal["gnn"]


class RankedDrug(TypedDict):
    drugid: str
    score: float
    rank: int
    drug: DrugRecMedicine
    label: int
    retrieval_score: float | None
    retrieval_rank: int


class DrugRecCase(TypedDict):
    patient_id: str
    split: DatasetSplit
    patient: DrugRecRecord
    gold_drugids: list[str]
    candidate_drugs: list[CandidateDrug]


class DrugRecResult(TypedDict):
    patient_id: str
    split: DatasetSplit
    model_name: DrugRecModelName
    ranked_drugs: list[RankedDrug]


class RankedEvidence(TypedDict):
    evidence_id: str
    score: float
    rank: int
    text: str
    label: int


class GNNNodeScore(TypedDict):
    node_id: str
    node_type: GNNNodeType
    score: float


class GNNRecResult(DrugRecResult, total=False):
    ranked_evidences: list[RankedEvidence]
    node_scores: list[GNNNodeScore]


class DrugRecMetrics(TypedDict, total=False):
    loss: float
    hit: float
    mrr: float
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float
    jaccard_at_5: float
    ddi_rate_at_5: float


class GNNMetrics(DrugRecMetrics, total=False):
    evidence_mrr: float
    evidence_hit_at_5: float


class TrainStepOutput(TypedDict):
    loss: Float[torch.Tensor, ""]
    loss_value: float
    metrics: DrugRecMetrics


class EvalStepOutput(TypedDict):
    results: Sequence[DrugRecResult]


class GNNModelInitKwargs(TypedDict):
    stats: NumericFeatureStats
    top_k: int
    hidden_size: int


type ModelStateDict = OrderedDict[str, torch.Tensor]


class DrugRecCheckpoint(TypedDict):
    model_name: DrugRecModelName
    model_state_dict: ModelStateDict
    init_kwargs: GNNModelInitKwargs


__all__ = [
    "DrugRecCase",
    "DrugRecCheckpoint",
    "DrugRecMetrics",
    "DrugRecModelName",
    "DrugRecResult",
    "EvalStepOutput",
    "GNNMetrics",
    "GNNModelInitKwargs",
    "GNNNodeScore",
    "GNNRecResult",
    "ModelStateDict",
    "RankedDrug",
    "RankedEvidence",
    "TrainStepOutput",
]

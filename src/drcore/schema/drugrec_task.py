from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from jaxtyping import Float

from .drugrec import DatasetSplit, DrugRecMedicine, DrugRecRecord
from .patient_candidate_set import CandidateDrug

type DrugRecModelName = Literal["gnn"]
type NodeId = str
type DrugId = str

GNNNodeType = Literal[
    "drug",
    "treat",
    "caution",
    "ingredient",
    "interaction",
]

GNNEdgeType = Literal[
    "drug_has_treat",
    "drug_has_caution",
    "drug_has_ingredient",
    "drug_has_interaction",
    "rev_drug_has_treat",
    "rev_drug_has_caution",
    "rev_drug_has_ingredient",
    "rev_drug_has_interaction",
]


class NumericFeatureStats(TypedDict):
    score_log_mean: float
    score_log_std: float


@dataclass(slots=True)
class GNNNode:
    node_id: NodeId
    node_type: GNNNodeType
    text: str


@dataclass(slots=True)
class DrugNodeNumericFeature:
    retrieval_score: float | None
    retrieval_rank: int | None
    is_candidate: int
    is_on_medicine: int


@dataclass(slots=True)
class GNNEdge:
    edge_type: GNNEdgeType
    src_node_id: NodeId
    dst_node_id: NodeId


@dataclass(slots=True)
class GNNCandidateTarget:
    drug_node_id: NodeId
    drugid: DrugId
    label: int


@dataclass(slots=True)
class GNNGraphSample:
    patient_id: str
    split: DatasetSplit
    patient_text: str
    gold_drugids: list[DrugId]
    nodes: list[GNNNode]
    edges: list[GNNEdge]
    drug_numeric_features: Mapping[NodeId, DrugNodeNumericFeature]
    candidate_targets: list[GNNCandidateTarget]


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


class DrugRecTrainSample(TypedDict):
    case: DrugRecCase


class GNNTrainSample(DrugRecTrainSample):
    graph_sample: GNNGraphSample


class GNNIntermediateMeta(TypedDict):
    split: DatasetSplit
    sample_count: int
    chunk_size: int
    source_path: str
    slot_names: list[str]


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
    "DrugNodeNumericFeature",
    "DrugRecCase",
    "DrugRecCheckpoint",
    "DrugRecMetrics",
    "DrugRecModelName",
    "DrugRecResult",
    "DrugRecTrainSample",
    "EvalStepOutput",
    "GNNMetrics",
    "GNNModelInitKwargs",
    "GNNCandidateTarget",
    "GNNEdge",
    "GNNEdgeType",
    "GNNGraphSample",
    "GNNIntermediateMeta",
    "GNNNode",
    "GNNNodeScore",
    "GNNNodeType",
    "GNNRecResult",
    "GNNTrainSample",
    "ModelStateDict",
    "NumericFeatureStats",
    "RankedDrug",
    "RankedEvidence",
    "TrainStepOutput",
]

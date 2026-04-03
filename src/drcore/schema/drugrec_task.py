from typing import Literal, TypedDict

import torch
from jaxtyping import Float

from .drugrec import DatasetSplit, DrugRecMedicine, DrugRecRecord
from .patient_candidate_set import CandidateDrug

type DrugId = str

GNNNodeType = Literal[
    "drug",
    "treat",
    "caution",
    "ingredient",
    "interaction",
]


class RankedDrug(TypedDict):
    drugid: str
    score: float
    rank: int
    drug: DrugRecMedicine
    label: int
    retrieval_score: float | None
    retrieval_rank: int


class RankedEvidence(TypedDict):
    evidence_id: str
    score: float
    rank: int
    text: str
    label: int


class DrugRecCase(TypedDict):
    patient_id: str
    split: DatasetSplit
    patient: DrugRecRecord
    gold_drugids: list[DrugId]
    candidate_drugs: list[CandidateDrug]


class GNNEntity(TypedDict):
    node_id: str
    node_type: GNNNodeType
    text: str
    label: int
    drugid: str | None


class GNNEvidence(TypedDict):
    evidence_id: str
    drugid: str
    text: str
    label: int


class GNNModelInput(TypedDict):
    patient_text: str
    entities: list[GNNEntity]
    evidences: list[GNNEvidence]
    ent_to_ev: Float[torch.Tensor, "entity evidence"]
    ev_to_ent: Float[torch.Tensor, "evidence entity"]
    entity_labels: Float[torch.Tensor, "entity"]
    evidence_labels: Float[torch.Tensor, "evidence"]
    candidate_entity_indices: list[int]


class GNNTrainSample(TypedDict):
    case: DrugRecCase
    model_input: GNNModelInput


class GNNIntermediateMeta(TypedDict):
    split: DatasetSplit
    sample_count: int
    source_path: str
    data_file: str


class GNNRecResult(TypedDict):
    patient_id: str
    split: DatasetSplit
    ranked_drugs: list[RankedDrug]
    ranked_evidences: list[RankedEvidence]


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


__all__ = [
    "DrugRecCase",
    "DrugRecMetrics",
    "GNNMetrics",
    "GNNIntermediateMeta",
    "GNNModelInput",
    "GNNEvidence",
    "GNNEntity",
    "GNNNodeType",
    "GNNRecResult",
    "GNNTrainSample",
    "RankedDrug",
    "RankedEvidence",
]

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float

from ...schema import DrugRecCase, RankedDrug, RankedEvidence

type EntityEvidenceMatrix = Float[torch.Tensor, "entity evidence"]
type EvidenceEntityMatrix = Float[torch.Tensor, "evidence entity"]
type EntityLabelTensor = Float[torch.Tensor, "entity"]
type EvidenceLabelTensor = Float[torch.Tensor, "evidence"]

GNNNodeType = Literal[
    "drug",
    "treat",
    "caution",
    "ingredient",
    "interaction",
]


@dataclass(slots=True)
class GNNEntity:
    node_id: str
    node_type: GNNNodeType
    text: str
    label: int
    drugid: str | None


@dataclass(slots=True)
class GNNEvidence:
    evidence_id: str
    drugid: str
    text: str
    label: int


@dataclass(slots=True)
class GNNModelInput:
    patient_text: str
    entities: list[GNNEntity]
    evidences: list[GNNEvidence]
    ent_to_ev: EntityEvidenceMatrix
    ev_to_ent: EvidenceEntityMatrix
    entity_labels: EntityLabelTensor
    evidence_labels: EvidenceLabelTensor
    candidate_entity_indices: list[int | None]


@dataclass(slots=True)
class GNNTrainSample:
    case: DrugRecCase
    model_input: GNNModelInput


@dataclass(slots=True)
class DrugRecMetrics:
    loss: float | None = None
    hit: float | None = None
    mrr: float | None = None
    precision_at_5: float | None = None
    recall_at_5: float | None = None
    f1_at_5: float | None = None
    jaccard_at_5: float | None = None
    ddi_rate_at_5: float | None = None


@dataclass(slots=True)
class GNNMetrics(DrugRecMetrics):
    evidence_mrr: float | None = None
    evidence_hit_at_5: float | None = None


@dataclass(slots=True)
class GNNRecResult:
    patient_id: str
    split: Literal["train", "dev", "test"]
    ranked_drugs: list[RankedDrug]
    ranked_evidences: list[RankedEvidence]

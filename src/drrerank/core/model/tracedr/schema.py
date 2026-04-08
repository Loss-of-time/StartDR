from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from torch import Tensor

from ...schema import DrugRecMedicine

type TraceDRQuestionId = int | str
type TraceDRNodeId = int | str
type TraceDREntityType = Literal["药品", "治疗", "禁用", "成分", "相互作用", ""]
type EntityEvidenceMatrix = Float[Tensor, "entity evidence"]
type EvidenceEntityMatrix = Float[Tensor, "evidence entity"]
type EntityMaskTensor = Float[Tensor, "entity"]
type EvidenceMaskTensor = Float[Tensor, "evidence"]
type EntityLabelTensor = Int[Tensor, "entity"]
type EvidenceLabelTensor = Int[Tensor, "evidence"]


@dataclass(slots=True)
class TraceDREntity:
    id: TraceDRNodeId | str
    label: str
    type: TraceDREntityType


@dataclass(slots=True)
class TraceDREvidence:
    evidence_text: str
    contain_entities: list[TraceDREntity]


@dataclass(slots=True)
class TraceDRModelSample:
    question_id: TraceDRQuestionId
    on_medicine: list[DrugRecMedicine]
    entities: list[TraceDREntity]
    entity_mask: EntityMaskTensor
    evidences: list[TraceDREvidence]
    evidence_mask: EvidenceMaskTensor
    ent_to_ev: EntityEvidenceMatrix
    ev_to_ent: EvidenceEntityMatrix
    entity_labels: EntityLabelTensor
    evidence_labels: EvidenceLabelTensor
    id_to_entity: np.ndarray
    id_to_evidence: np.ndarray
    tsf: str
    question: str
    gold_answers: list[str]

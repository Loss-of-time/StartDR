from dataclasses import dataclass, replace
from collections.abc import Sequence

from jaxtyping import Float, Int
from torch import Tensor

from ...schema import DrugRecMedicine, TraceDRSample

type EntityEvidenceMatrix = Float[Tensor, "entity evidence"]
type EvidenceEntityMatrix = Float[Tensor, "evidence entity"]
type EntityMaskTensor = Float[Tensor, "entity"]
type EvidenceMaskTensor = Float[Tensor, "evidence"]
type EntityLabelTensor = Int[Tensor, "entity"]
type EvidenceLabelTensor = Int[Tensor, "evidence"]


@dataclass(slots=True)
class GATEntity:
    id: int
    name: str
    instruction: DrugRecMedicine
    connect_property: list["GATEvidence"]
    is_answer: bool


@dataclass(slots=True)
class GATEvidence:
    id: int
    label: str


@dataclass(slots=True)
class GATModelSample:
    source_sample: TraceDRSample
    question_id: str
    tsf: str
    entities: list[GATEntity]
    entity_mask: EntityMaskTensor
    evidences: list[GATEvidence]
    evidence_mask: EvidenceMaskTensor
    ent_to_ev: EntityEvidenceMatrix
    ev_to_ent: EvidenceEntityMatrix
    entity_labels: EntityLabelTensor
    evidence_labels: EvidenceLabelTensor
    id_to_entity: Sequence[str | None]
    id_to_evidence: Sequence[str | None]
    gold_answers: list[str]

    def to_cuda(self) -> "GATModelSample":
        return replace(
            self,
            entity_mask=self.entity_mask.cuda(),
            evidence_mask=self.evidence_mask.cuda(),
            ent_to_ev=self.ent_to_ev.cuda(),
            ev_to_ent=self.ev_to_ent.cuda(),
            entity_labels=self.entity_labels.cuda(),
            evidence_labels=self.evidence_labels.cuda(),
        )

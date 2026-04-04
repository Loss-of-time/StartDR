from dataclasses import dataclass
from typing import Literal, cast

import torch
from cattrs import Converter
from jaxtyping import Float

type DatasetSplit = Literal["train", "dev", "test"]
type NullableString = str | None
type NullableInteger = int | None
type NullableCMAN = str | None
type DrugId = str
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
class DrugCaution:
    caution_level: NullableString
    caution_levelid: NullableInteger
    crowd: str
    crowd_id: int


@dataclass(slots=True)
class DrugIngredient:
    ingredient_id: NullableInteger
    ingredient: NullableString


@dataclass(slots=True)
class DrugInteraction:
    interaction_id: int
    name: str


@dataclass(slots=True)
class DrugTreat:
    treat: NullableString
    treat_id: NullableInteger


@dataclass(slots=True)
class DrugRecMedicine:
    CMAN: NullableCMAN
    caution: list[DrugCaution]
    drugid: str
    ingredients: list[DrugIngredient]
    interaction: list[DrugInteraction]
    name: str
    treat: list[DrugTreat]


@dataclass(slots=True)
class DrugRecRecord:
    age: int
    allergen: list[str]
    antecedents: list[str]
    diagnosis: list[str]
    gender: str
    group: list[str]
    id: str
    medicine: list[DrugRecMedicine]
    on_medicine: list[DrugRecMedicine]
    part: DatasetSplit
    symptom: list[str]


@dataclass(slots=True)
class CandidateDrug:
    drugid: str
    rank: int
    score: float | None
    drug: DrugRecMedicine
    is_gold: bool


@dataclass(slots=True)
class PatientCandidateSet:
    patient_id: str
    split: DatasetSplit
    retriever: str
    top_k: int
    retrieval_query: str
    patient: DrugRecRecord
    gold_drugids: list[str]
    candidate_drugs: list[CandidateDrug]


@dataclass(slots=True)
class RankedDrug:
    drugid: str
    score: float
    rank: int
    drug: DrugRecMedicine
    label: int
    retrieval_score: float | None
    retrieval_rank: int


@dataclass(slots=True)
class RankedEvidence:
    evidence_id: str
    score: float
    rank: int
    text: str
    label: int


@dataclass(slots=True)
class DrugRecCase:
    patient_id: str
    split: DatasetSplit
    patient: DrugRecRecord
    gold_drugids: list[DrugId]
    candidate_drugs: list[CandidateDrug]


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
    candidate_entity_indices: list[int]


@dataclass(slots=True)
class GNNTrainSample:
    case: DrugRecCase
    model_input: GNNModelInput


@dataclass(slots=True)
class GNNIntermediateMeta:
    split: DatasetSplit
    sample_count: int
    source_path: str
    data_file: str


@dataclass(slots=True)
class GNNRecResult:
    patient_id: str
    split: DatasetSplit
    ranked_drugs: list[RankedDrug]
    ranked_evidences: list[RankedEvidence]


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


_converter = Converter()
_converter.register_structure_hook(EntityEvidenceMatrix, lambda value, _: value)
_converter.register_structure_hook(EvidenceEntityMatrix, lambda value, _: value)
_converter.register_structure_hook(EntityLabelTensor, lambda value, _: value)
_converter.register_structure_hook(EvidenceLabelTensor, lambda value, _: value)


def structure(data: object, target: object) -> object:
    return _converter.structure(data, target)


def unstructure(data: object) -> object:
    return cast(object, _converter.unstructure(data))

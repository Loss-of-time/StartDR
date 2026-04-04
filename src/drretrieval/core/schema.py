from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from cattrs import Converter

type DatasetSplit = Literal["train", "dev", "test"]
type NullableString = str | None
type NullableInteger = int | None
type NullableCMAN = str | None
type PatientCandidateRetriever = Literal["bm25", "pyserini_bm25", "dense"]


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
    conflict: list[DrugRecMedicine] | None = None
    medicine_num: int | None = None


@dataclass(slots=True)
class RetrievedDrugCandidate:
    drugid: str
    score: float
    medicine: DrugRecMedicine | None = None
    metadata: dict[str, Any] | None = None


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        raise NotImplementedError

    def batch_retrieve(
        self,
        patients: Sequence[DrugRecRecord],
        top_k: int = 10,
    ) -> list[list[RetrievedDrugCandidate]]:
        retrieve = self.retrieve
        return [retrieve(patient, top_k=top_k) for patient in patients]


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
    retriever: PatientCandidateRetriever
    top_k: int
    retrieval_query: str
    patient: DrugRecRecord
    gold_drugids: list[str]
    candidate_drugs: list[CandidateDrug]


@dataclass(slots=True)
class MetricsResult:
    hit: float
    recall: float
    mrr: float


@dataclass(slots=True)
class SummaryResult:
    patient_count: int
    failure_count: int


@dataclass(slots=True)
class RetrieverEvalConfig:
    retriever_name: str
    input_path: str
    top_k: int
    sample_count: int


@dataclass(slots=True)
class RetrieverEvalReport:
    config: RetrieverEvalConfig
    summary: SummaryResult
    metrics: MetricsResult


_converter = Converter()


def structure(data: object, target: object) -> Any:
    return _converter.structure(data, target)


def unstructure(data: object) -> Any:
    return _converter.unstructure(data)

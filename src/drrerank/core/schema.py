from dataclasses import dataclass
from typing import Any, Literal, cast

from cattrs import Converter

type DatasetSplit = Literal["train", "dev", "test"]
type DrugId = str
type NullableString = str | None
type NullableInteger = int | None
type NullableCMAN = str | None


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
class TraceDRSample:
    people: DrugRecRecord
    top_k_drugs: dict[str, DrugRecMedicine]


@dataclass(slots=True)
class CandidateDrug:
    drugid: str
    rank: int
    score: float | None
    drug: DrugRecMedicine
    is_gold: bool


@dataclass(slots=True)
class RankedDrug:
    """精排后的候选药物记录。"""

    drugid: str
    score: float
    rank: int
    drug: DrugRecMedicine
    label: int
    retrieval_score: float | None
    retrieval_rank: int


@dataclass(slots=True)
class RankedEvidence:
    """精排后的证据记录。"""

    evidence_id: str
    score: float
    rank: int
    text: str
    label: int


@dataclass(slots=True)
class RankedCase:
    """单个病例的精排导出结果。"""

    patient_id: str
    split: DatasetSplit
    ranked_drugs: list[RankedDrug]
    ranked_evidences: list[RankedEvidence]


@dataclass(slots=True)
class DrugRecCase:
    patient_id: str
    split: DatasetSplit
    patient: DrugRecRecord
    gold_drugids: list[DrugId]
    candidate_drugs: list[CandidateDrug]


_converter = Converter()


def structure(data: object, target: object) -> Any:
    return _converter.structure(data, cast(Any, target))


def unstructure(data: object) -> object:
    return cast(object, _converter.unstructure(data))

from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from .io import load_jsonl, load_pickle
from .schema import (
    CandidateDrug,
    DatasetSplit,
    DrugCaution,
    DrugIngredient,
    DrugInteraction,
    DrugRecCase,
    DrugRecMedicine,
    DrugRecRecord,
    DrugTreat,
    TraceDRSample,
    structure,
)


class RawDrugCaution(TypedDict):
    caution_level: str | None
    caution_levelid: str | int | None
    crowd: str
    crowd_id: str | int


class RawDrugIngredient(TypedDict):
    ingredient: str | None
    ingredient_id: str | int | None


class RawDrugInteraction(TypedDict):
    interaction_id: str | int
    name: str


class RawDrugTreat(TypedDict):
    treat: str | None
    treat_id: str | int | None


class RawDrugRecMedicine(TypedDict):
    CMAN: str | None
    caution: list[RawDrugCaution]
    drugid: str | int
    ingredients: list[RawDrugIngredient]
    interaction: list[RawDrugInteraction]
    name: str
    treat: list[RawDrugTreat]


class RawDrugRecRecord(TypedDict):
    age: int | str
    allergen: list[str]
    antecedents: list[str]
    diagnosis: list[str]
    gender: str
    group: list[str]
    id: str | int
    medicine: list[RawDrugRecMedicine]
    on_medicine: list[RawDrugRecMedicine]
    symptom: list[str]
    conflict: NotRequired[list[RawDrugRecMedicine]]
    medicine_num: NotRequired[int | str]


class RawTraceDRRecord(TypedDict):
    people: RawDrugRecRecord
    top_k_drugs: Mapping[str, RawDrugRecMedicine]


type RawTraceDRDataset = Mapping[str, RawTraceDRRecord]


def to_optional_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "" or normalized.lower() == "none":
            return None
        return int(normalized)
    return int(value)


def build_medicine(raw_medicine: RawDrugRecMedicine) -> DrugRecMedicine:
    return DrugRecMedicine(
        CMAN=raw_medicine["CMAN"],
        caution=[
            DrugCaution(
                caution_level=raw_caution["caution_level"],
                caution_levelid=to_optional_int(raw_caution["caution_levelid"]),
                crowd=raw_caution["crowd"],
                crowd_id=int(raw_caution["crowd_id"]),
            )
            for raw_caution in raw_medicine["caution"]
        ],
        drugid=str(raw_medicine["drugid"]),
        ingredients=[
            DrugIngredient(
                ingredient=raw_ingredient["ingredient"],
                ingredient_id=to_optional_int(raw_ingredient["ingredient_id"]),
            )
            for raw_ingredient in raw_medicine["ingredients"]
        ],
        interaction=[
            DrugInteraction(
                interaction_id=int(raw_interaction["interaction_id"]),
                name=raw_interaction["name"],
            )
            for raw_interaction in raw_medicine["interaction"]
        ],
        name=raw_medicine["name"],
        treat=[
            DrugTreat(
                treat=raw_treat["treat"],
                treat_id=to_optional_int(raw_treat["treat_id"]),
            )
            for raw_treat in raw_medicine["treat"]
        ],
    )


def build_patient(
    raw_patient: RawDrugRecRecord,
    split: DatasetSplit,
) -> DrugRecRecord:
    return DrugRecRecord(
        age=int(raw_patient["age"]),
        allergen=raw_patient["allergen"],
        antecedents=raw_patient["antecedents"],
        diagnosis=raw_patient["diagnosis"],
        gender=raw_patient["gender"],
        group=raw_patient["group"],
        id=str(raw_patient["id"]),
        medicine=[build_medicine(raw_medicine) for raw_medicine in raw_patient["medicine"]],
        on_medicine=[build_medicine(raw_medicine) for raw_medicine in raw_patient["on_medicine"]],
        part=split,
        symptom=raw_patient["symptom"],
    )


def build_tracedr_sample_from_raw(
    raw_sample: RawTraceDRRecord,
    split: DatasetSplit,
) -> TraceDRSample:
    return TraceDRSample(
        people=build_patient(raw_sample["people"], split),
        top_k_drugs={
            str(drugid): build_medicine(raw_medicine)
            for drugid, raw_medicine in raw_sample["top_k_drugs"].items()
        },
    )


def build_candidate_drugs(
    top_k_drugs: Mapping[str, DrugRecMedicine],
    gold_drugid_set: set[str],
) -> list[CandidateDrug]:
    return [
        CandidateDrug(
            drugid=drugid,
            rank=rank,
            score=None,
            drug=drug,
            is_gold=drugid in gold_drugid_set,
        )
        for rank, (drugid, drug) in enumerate(top_k_drugs.items(), start=1)
    ]


def build_drugrec_case(sample: TraceDRSample) -> DrugRecCase:
    gold_drugids = list(dict.fromkeys(medicine.drugid for medicine in sample.people.medicine))
    gold_drugid_set = set(gold_drugids)
    return DrugRecCase(
        patient_id=sample.people.id,
        split=sample.people.part,
        patient=sample.people,
        gold_drugids=gold_drugids,
        candidate_drugs=build_candidate_drugs(sample.top_k_drugs, gold_drugid_set),
    )


def load_tracedr_samples(
    input_path: Path,
    limit: int | None = None,
) -> list[TraceDRSample]:
    return load_jsonl(
        path=input_path,
        parse_line=lambda row: structure(row, TraceDRSample),
        limit=limit,
    )


def load_tracedr_cases(
    input_path: Path,
    limit: int | None = None,
) -> list[DrugRecCase]:
    return [build_drugrec_case(sample) for sample in load_tracedr_samples(input_path, limit)]


def load_raw_tracedr_dataset(path: Path) -> RawTraceDRDataset:
    return cast(RawTraceDRDataset, load_pickle(path))

"""retrieval、rerank 与 RAG 之间的统一适配层。"""

from collections.abc import Sequence

from drrerank.core import schema as rerank_schema
from drretrieval.core import schema as retrieval_schema

from .schema import (
    DrugCaution,
    DrugIngredient,
    DrugInteraction,
    DrugRecMedicine,
    DrugRecRecord,
    DrugTreat,
    RagCandidate,
    RagCase,
    RagEvidence,
    TraceDRSample,
)

type ExternalDrugCaution = retrieval_schema.DrugCaution | rerank_schema.DrugCaution
type ExternalDrugIngredient = retrieval_schema.DrugIngredient | rerank_schema.DrugIngredient
type ExternalDrugInteraction = retrieval_schema.DrugInteraction | rerank_schema.DrugInteraction
type ExternalDrugTreat = retrieval_schema.DrugTreat | rerank_schema.DrugTreat
type ExternalDrugRecMedicine = retrieval_schema.DrugRecMedicine | rerank_schema.DrugRecMedicine
type ExternalDrugRecRecord = retrieval_schema.DrugRecRecord | rerank_schema.DrugRecRecord


def build_patient_query(patient: DrugRecRecord, delimiter: str = " || ") -> str:
    """构造与 TraceDR 一致的患者查询串。

    Args:
        patient: 患者记录。
        delimiter: 字段分隔符。

    Returns:
        供检索与 prompt 共用的查询文本。
    """

    on_medicine: str = ",".join(item.name for item in patient.on_medicine)
    group: str = ",".join(patient.group)
    symptom: str = ",".join(patient.symptom)
    diagnosis: str = ",".join(patient.diagnosis)
    antecedents: str = ",".join(patient.antecedents)
    allergen: str = ",".join(patient.allergen)
    return (
        f"{patient.age} {delimiter} {group} {delimiter} {patient.gender} {delimiter} "
        f"{diagnosis} {delimiter} {symptom} {delimiter} {antecedents} {delimiter} "
        f"{on_medicine} {delimiter} {allergen}"
    )


def build_medicine_evidence_text(medicine: DrugRecMedicine) -> str:
    """构造与 TraceDR 一致的药品证据文本。

    Args:
        medicine: 药品记录。

    Returns:
        单药证据文本。
    """

    treat_values: list[str] = [item.treat for item in medicine.treat if item.treat is not None]
    treatments_string: str = ", ".join(treat_values) if treat_values else "None"

    caution_values: list[str] = [
        f"{item.crowd}{item.caution_level}"
        for item in medicine.caution
        if item.caution_level is not None
    ]
    caution_string: str = ", ".join(caution_values) if caution_values else "None"

    interaction_values: list[str] = [item.name for item in medicine.interaction]
    interaction_string: str = ", ".join(interaction_values) if interaction_values else "None"

    ingredient_values: list[str] = [
        item.ingredient for item in medicine.ingredients if item.ingredient is not None
    ]
    ingredient_string: str = ", ".join(ingredient_values) if ingredient_values else "None"
    return (
        f"药名:{medicine.name} || 治疗:{treatments_string} || 禁用:{caution_string} || "
        f"成分:{ingredient_string} || 相互作用:{interaction_string}"
    )


def copy_caution(caution: ExternalDrugCaution) -> DrugCaution:
    """复制禁忌信息到 RAG 统一结构。"""

    return DrugCaution(
        caution_level=caution.caution_level,
        caution_levelid=caution.caution_levelid,
        crowd=caution.crowd,
        crowd_id=caution.crowd_id,
    )


def copy_ingredient(ingredient: ExternalDrugIngredient) -> DrugIngredient:
    """复制成分信息到 RAG 统一结构。"""

    return DrugIngredient(
        ingredient_id=ingredient.ingredient_id,
        ingredient=ingredient.ingredient,
    )


def copy_interaction(interaction: ExternalDrugInteraction) -> DrugInteraction:
    """复制相互作用信息到 RAG 统一结构。"""

    return DrugInteraction(
        interaction_id=interaction.interaction_id,
        name=interaction.name,
    )


def copy_treat(treat: ExternalDrugTreat) -> DrugTreat:
    """复制治疗信息到 RAG 统一结构。"""

    return DrugTreat(
        treat=treat.treat,
        treat_id=treat.treat_id,
    )


def copy_medicine(medicine: ExternalDrugRecMedicine) -> DrugRecMedicine:
    """复制药品信息到 RAG 统一结构。"""

    return DrugRecMedicine(
        CMAN=medicine.CMAN,
        caution=[copy_caution(item) for item in medicine.caution],
        drugid=medicine.drugid,
        ingredients=[copy_ingredient(item) for item in medicine.ingredients],
        interaction=[copy_interaction(item) for item in medicine.interaction],
        name=medicine.name,
        treat=[copy_treat(item) for item in medicine.treat],
    )


def copy_patient(patient: ExternalDrugRecRecord) -> DrugRecRecord:
    """复制患者信息到 RAG 统一结构。"""

    return DrugRecRecord(
        age=patient.age,
        allergen=list(patient.allergen),
        antecedents=list(patient.antecedents),
        diagnosis=list(patient.diagnosis),
        gender=patient.gender,
        group=list(patient.group),
        id=patient.id,
        medicine=[copy_medicine(item) for item in patient.medicine],
        on_medicine=[copy_medicine(item) for item in patient.on_medicine],
        part=patient.part,
        symptom=list(patient.symptom),
        conflict=list(getattr(patient, "conflict", [])) or None,
        medicine_num=getattr(patient, "medicine_num", None),
    )


def from_retrieval_sample(sample: retrieval_schema.TraceDRSample) -> TraceDRSample:
    """把 retrieval 样本转换为 RAG 统一候选集。"""

    return TraceDRSample(
        people=copy_patient(sample.people),
        top_k_drugs={drugid: copy_medicine(drug) for drugid, drug in sample.top_k_drugs.items()},
    )


def from_rerank_sample(sample: rerank_schema.TraceDRSample) -> TraceDRSample:
    """把 rerank 样本转换为 RAG 统一候选集。"""

    return TraceDRSample(
        people=copy_patient(sample.people),
        top_k_drugs={drugid: copy_medicine(drug) for drugid, drug in sample.top_k_drugs.items()},
    )


def build_rag_case(sample: TraceDRSample, candidate_limit: int | None = None) -> RagCase:
    """把 TraceDR 风格候选集转换成 RAG 统一输入。

    Args:
        sample: TraceDR 风格样本。
        candidate_limit: 保留的候选药物数量。

    Returns:
        可直接供 prompt 与费用估算共用的统一样本。
    """

    gold_drugids: list[str] = list(dict.fromkeys(item.drugid for item in sample.people.medicine))
    gold_drugid_set: set[str] = set(gold_drugids)
    candidates: list[RagCandidate] = []
    ranked_items: Sequence[tuple[str, DrugRecMedicine]] = tuple(sample.top_k_drugs.items())
    for rank, (drugid, drug) in enumerate(ranked_items, start=1):
        if candidate_limit is not None and rank > candidate_limit:
            break
        # 目的：先把 retrieval 结果规整为单药单证据，后续 rerank 只补排序字段即可。
        evidence: RagEvidence = RagEvidence(
            evidence_id=f"retrieval::{drugid}",
            drugid=drugid,
            text=build_medicine_evidence_text(drug),
            source="retrieval",
            retrieval_rank=rank,
            rerank_rank=None,
            score=None,
        )
        candidates.append(
            RagCandidate(
                drugid=drugid,
                name=drug.name,
                drug=drug,
                is_gold=drugid in gold_drugid_set,
                retrieval_rank=rank,
                retrieval_score=None,
                rerank_rank=None,
                rerank_score=None,
                evidences=[evidence],
            )
        )
    return RagCase(
        patient_id=sample.people.id,
        split=sample.people.part,
        patient=sample.people,
        gold_drugids=gold_drugids,
        candidates=candidates,
    )


def apply_ranked_drugs(
    case: RagCase,
    ranked_drugs: Sequence[rerank_schema.RankedDrug],
) -> RagCase:
    """把 rerank 的药物排序结果补到统一样本上。

    Args:
        case: 当前 RAG 样本。
        ranked_drugs: 精排结果。

    Returns:
        补齐 rerank 字段后的新样本。
    """

    ranked_drug_map: dict[str, rerank_schema.RankedDrug] = {
        item.drugid: item for item in ranked_drugs
    }
    updated_candidates: list[RagCandidate] = []
    for candidate in case.candidates:
        ranked_drug: rerank_schema.RankedDrug | None = ranked_drug_map.get(candidate.drugid)
        updated_candidates.append(
            RagCandidate(
                drugid=candidate.drugid,
                name=candidate.name,
                drug=candidate.drug,
                is_gold=candidate.is_gold,
                retrieval_rank=candidate.retrieval_rank,
                retrieval_score=candidate.retrieval_score,
                rerank_rank=ranked_drug.rank if ranked_drug is not None else candidate.rerank_rank,
                rerank_score=ranked_drug.score
                if ranked_drug is not None
                else candidate.rerank_score,
                evidences=list(candidate.evidences),
            )
        )
    return RagCase(
        patient_id=case.patient_id,
        split=case.split,
        patient=case.patient,
        gold_drugids=list(case.gold_drugids),
        candidates=updated_candidates,
    )


def apply_ranked_evidences(
    case: RagCase,
    ranked_evidences: Sequence[rerank_schema.RankedEvidence],
) -> RagCase:
    """把 rerank 的证据排序结果补到统一样本上。

    Args:
        case: 当前 RAG 样本。
        ranked_evidences: 精排证据结果。

    Returns:
        补齐证据排序字段后的新样本。
    """

    ranked_evidence_map: dict[str, rerank_schema.RankedEvidence] = {
        item.evidence_id: item for item in ranked_evidences
    }
    updated_candidates: list[RagCandidate] = []
    for candidate in case.candidates:
        updated_evidences: list[RagEvidence] = []
        for evidence in candidate.evidences:
            ranked_evidence: rerank_schema.RankedEvidence | None = ranked_evidence_map.get(
                evidence.evidence_id
            )
            updated_evidences.append(
                RagEvidence(
                    evidence_id=evidence.evidence_id,
                    drugid=evidence.drugid,
                    text=ranked_evidence.text if ranked_evidence is not None else evidence.text,
                    source="rerank" if ranked_evidence is not None else evidence.source,
                    retrieval_rank=evidence.retrieval_rank,
                    rerank_rank=(
                        ranked_evidence.rank
                        if ranked_evidence is not None
                        else evidence.rerank_rank
                    ),
                    score=ranked_evidence.score if ranked_evidence is not None else evidence.score,
                )
            )
        updated_candidates.append(
            RagCandidate(
                drugid=candidate.drugid,
                name=candidate.name,
                drug=candidate.drug,
                is_gold=candidate.is_gold,
                retrieval_rank=candidate.retrieval_rank,
                retrieval_score=candidate.retrieval_score,
                rerank_rank=candidate.rerank_rank,
                rerank_score=candidate.rerank_score,
                evidences=updated_evidences,
            )
        )
    return RagCase(
        patient_id=case.patient_id,
        split=case.split,
        patient=case.patient,
        gold_drugids=list(case.gold_drugids),
        candidates=updated_candidates,
    )

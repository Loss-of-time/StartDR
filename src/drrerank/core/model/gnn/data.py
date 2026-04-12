from collections.abc import Callable

import torch

from ...schema import (
    DrugRecCase,
    DrugRecMedicine,
    DrugRecRecord,
)
from .schema import GNNEntity, GNNEvidence, GNNModelInput, GNNNodeType, GNNTrainSample


class SkipTrainSample(Exception):
    pass


def build_gnn_train_sample(
    case: DrugRecCase,
    max_entities: int = 100,
    max_evidences: int = 50,
    train: bool = False,
) -> GNNTrainSample:
    patient = case.patient
    gold_drugids = set(case.gold_drugids)
    entities: list[GNNEntity] = []
    evidences: list[GNNEvidence] = []
    entity_index_by_node_id: dict[str, int] = {}
    evidence_index_by_drugid: dict[str, int] = {}
    ent_to_ev_rows: list[list[float]] = []
    ev_to_ent_rows: list[list[float]] = []

    def get_or_add_entity(
        node_id: str,
        node_type: GNNNodeType,
        text: str,
        label: int,
        drugid: str | None,
    ) -> int | None:
        entity_index = entity_index_by_node_id.get(node_id)
        if entity_index is not None:
            return entity_index
        if len(entities) >= max_entities:
            return None
        entity_index = len(entities)
        entity_index_by_node_id[node_id] = entity_index
        entities.append(
            GNNEntity(
                node_id=node_id,
                node_type=node_type,
                text=text,
                label=label,
                drugid=drugid,
            )
        )
        ent_to_ev_rows.append([0.0] * len(evidences))
        for evidence_row in ev_to_ent_rows:
            evidence_row.append(0.0)
        return entity_index

    def get_or_add_evidence(medicine: DrugRecMedicine) -> int:
        drugid = medicine.drugid
        evidence_index = evidence_index_by_drugid.get(drugid)
        if evidence_index is not None:
            return evidence_index
        evidence_index = len(evidences)
        evidence_index_by_drugid[drugid] = evidence_index
        evidences.append(
            GNNEvidence(
                evidence_id=f"evidence:drug:{drugid}",
                drugid=drugid,
                text=build_evidence_text(medicine),
                label=1 if drugid in gold_drugids else 0,
            )
        )
        for entity_row in ent_to_ev_rows:
            entity_row.append(0.0)
        ev_to_ent_rows.append([0.0] * len(entities))
        return evidence_index

    def link_entity_to_evidence(entity_index: int, evidence_index: int) -> None:
        ent_to_ev_rows[entity_index][evidence_index] = 1.0
        ev_to_ent_rows[evidence_index][entity_index] = 1.0

    def append_entities_by_items[T](
        evidence_index: int,
        items: list[T],
        node_type: GNNNodeType,
        get_text: Callable[[T], str],
        get_node_id: Callable[[T], str],
    ) -> None:
        for item in items:
            if len(entities) >= max_entities:
                break
            text = get_text(item).strip()
            if not text:
                continue
            entity_index = get_or_add_entity(
                get_node_id(item),
                node_type,
                text,
                0,
                None,
            )
            if entity_index is None:
                break
            link_entity_to_evidence(  # TODO 这里原先把 connect 当参数传入，读起来会遮住“给实体和证据连边”这个动作，所以收回到局部作用域里
                entity_index,
                evidence_index,
            )

    ordered_evidence_medicines = get_ordered_evidence_medicines(case)[:max_evidences]
    if train and not any(
        medicine.drugid in gold_drugids
        for medicine in ordered_evidence_medicines
    ):
        raise SkipTrainSample("截断后证据中不含答案药物。")

    for medicine in ordered_evidence_medicines:
        evidence_index = get_or_add_evidence(medicine)
        drug_entity_index = get_or_add_entity(
            node_id=f"drug:{medicine.drugid}",
            node_type="drug",
            text=medicine.name,
            label=1 if medicine.drugid in gold_drugids else 0,
            drugid=medicine.drugid,
        )
        if drug_entity_index is not None:
            link_entity_to_evidence(drug_entity_index, evidence_index)
        append_entities_by_items(
            evidence_index=evidence_index,
            items=medicine.treat,
            node_type="treat",
            get_text=lambda treat: treat.treat or "",
            get_node_id=lambda treat: (
                f"treat:id:{treat.treat_id}"
                if treat.treat_id is not None
                else f"treat:text:{(treat.treat or '').strip()}"
            ),
        )
        append_entities_by_items(
            evidence_index=evidence_index,
            items=medicine.caution,
            node_type="caution",
            get_text=lambda caution: build_caution_text(
                caution.crowd,
                caution.caution_level,
            ),
            get_node_id=lambda caution: (
                f"caution:id:{caution.crowd_id}:{caution.caution_levelid}"
                if caution.caution_levelid is not None
                else f"caution:text:{caution.crowd.strip()}:{(caution.caution_level or '').strip()}"
            ),
        )
        append_entities_by_items(
            evidence_index=evidence_index,
            items=medicine.ingredients,
            node_type="ingredient",
            get_text=lambda ingredient: ingredient.ingredient or "",
            get_node_id=lambda ingredient: (
                f"ingredient:id:{ingredient.ingredient_id}"
                if ingredient.ingredient_id is not None
                else f"ingredient:text:{(ingredient.ingredient or '').strip()}"
            ),
        )
        append_entities_by_items(
            evidence_index=evidence_index,
            items=medicine.interaction,
            node_type="interaction",
            get_text=lambda interaction: interaction.name,
            get_node_id=lambda interaction: (
                f"interaction:id:{interaction.interaction_id}:{interaction.name.strip()}"
            ),
        )
    if train and not any(entity.label == 1 for entity in entities):
        raise SkipTrainSample("截断后答案实体被裁掉。")
    return GNNTrainSample(
        case=case,
        model_input=GNNModelInput(
            patient_text=build_patient_query_text(patient),
            entities=entities,
            evidences=evidences,
            ent_to_ev=normalize_columns(torch.tensor(ent_to_ev_rows, dtype=torch.float32)),
            ev_to_ent=normalize_columns(torch.tensor(ev_to_ent_rows, dtype=torch.float32)),
            entity_labels=torch.tensor(
                [float(entity.label) for entity in entities],
                dtype=torch.float32,
            ),
            evidence_labels=torch.tensor(
                [float(evidence.label) for evidence in evidences],
                dtype=torch.float32,
            ),
            candidate_entity_indices=[
                entity_index_by_node_id.get(f"drug:{candidate.drugid}")
                for candidate in case.candidate_drugs
            ],
        ),
    )


def get_ordered_evidence_medicines(
    case: DrugRecCase,
) -> list[DrugRecMedicine]:
    ordered_medicines: list[DrugRecMedicine] = []
    seen_drugids: set[str] = set()  # 去重
    for medicine in case.patient.on_medicine:
        if medicine.drugid in seen_drugids:
            continue
        ordered_medicines.append(medicine)
        seen_drugids.add(medicine.drugid)
    for candidate in case.candidate_drugs:
        if candidate.drugid in seen_drugids:
            continue
        ordered_medicines.append(candidate.drug)
        seen_drugids.add(candidate.drugid)
    return ordered_medicines  # NOTE 这里order是按加入顺序组织


def build_patient_query_text(patient: DrugRecRecord) -> str:
    return " || ".join(  # NOTE || 作为分隔符
        [
            str(patient.age),
            ",".join(patient.group),
            patient.gender,
            ",".join(patient.diagnosis),
            ",".join(patient.symptom),
            ",".join(patient.antecedents),
            ",".join(medicine.name for medicine in patient.on_medicine),
            ",".join(patient.allergen),
        ]
    )


def build_evidence_text(medicine: DrugRecMedicine) -> str:
    treat_text = (
        ", ".join(treat.treat for treat in medicine.treat if treat.treat is not None) or "None"
    )
    caution_text = (
        ", ".join(
            build_caution_text(caution.crowd, caution.caution_level)
            for caution in medicine.caution
            if caution.crowd.strip()
        )
        or "None"
    )
    ingredient_text = (
        ", ".join(
            ingredient.ingredient
            for ingredient in medicine.ingredients
            if ingredient.ingredient is not None
        )
        or "None"
    )
    interaction_text = (
        ", ".join(
            interaction.name for interaction in medicine.interaction if interaction.name.strip()
        )
        or "None"
    )
    return (
        f"药名:{medicine.name} || "
        f"治疗:{treat_text} || "
        f"禁用:{caution_text} || "
        f"成分:{ingredient_text} || "
        f"相互作用:{interaction_text}"
    )


def build_caution_text(crowd: str, caution_level: str | None) -> str:
    parts = [crowd.strip()]
    if caution_level and caution_level.strip():
        parts.append(caution_level.strip())
    return " ".join(parts).strip()


def normalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    column_sum = torch.sum(matrix, dim=0, keepdim=True)
    column_sum[column_sum == 0] = 1.0
    return matrix / column_sum

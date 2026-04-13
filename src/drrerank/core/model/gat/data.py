from collections.abc import Sequence
from typing import cast

import jieba
import numpy as np
import torch

from ...schema import DrugRecMedicine, DrugRecRecord, TraceDRSample
from .schema import GATEntity, GATEvidence, GATModelSample


class ContinueWithNext(Exception):
    pass


def chinese_tokenizer(text: list[str]) -> list[list[str]]:
    return [jieba.lcut(doc) for doc in text]


def person_to_query(person: DrugRecRecord, delimiter: str) -> str:
    onmedicine = ",".join(item.name for item in person.on_medicine)
    group = ",".join(person.group)
    symptom = ",".join(person.symptom)
    diagnosis = ",".join(person.diagnosis)
    antecedents = ",".join(person.antecedents)
    allergen = ",".join(person.allergen)
    query = f"{person.age} {delimiter} {group} {delimiter} {person.gender} {delimiter} {diagnosis} {delimiter} {symptom} {delimiter} {antecedents} {delimiter} {onmedicine} {delimiter} {allergen}"
    return query


def _deduplicate_medicines_by_drugid(
    medicines: Sequence[DrugRecMedicine],
) -> list[DrugRecMedicine]:
    deduplicated_medicines: list[DrugRecMedicine] = []
    seen_drug_ids: set[str] = set()
    for medicine in medicines:
        if medicine.drugid in seen_drug_ids:
            continue
        seen_drug_ids.add(medicine.drugid)
        deduplicated_medicines.append(medicine)
    return deduplicated_medicines


def build_gat_model_sample(
    sample: TraceDRSample,
    train: bool = False,
    max_entities: int = 50,
    max_evidences: int = 100,
) -> GATModelSample | None:
    """准备单条 GAT 样本。"""
    question = person_to_query(sample.people, "||")
    true_id_list = [item.drugid for item in sample.people.medicine]
    true_id_set = set(true_id_list)

    evidences: list[DrugRecMedicine] = []
    evidences.extend(sample.people.on_medicine)
    evidences.extend(sample.top_k_drugs.values())
    evidences = _deduplicate_medicines_by_drugid(evidences)

    if len(evidences) > max_entities:
        evidences = evidences[:max_entities]

    if train and not any(drug.drugid in true_id_set for drug in evidences):
        return None

    entity_to_id: dict[str, int] = {}
    entity_to_node: dict[str, GATEntity] = {}
    evidence_to_id: dict[str, int] = {}
    id_to_entity = np.empty(max_entities, dtype=object)
    id_to_entity.fill(None)
    id_to_evidence = np.empty(max_evidences, dtype=object)
    id_to_evidence.fill(None)

    entities_list: list[GATEntity] = []
    evidences_list: list[GATEvidence] = []

    ent_to_ev = np.zeros((max_entities, max_evidences), dtype=np.float32)
    ev_to_ent = np.zeros((max_evidences, max_entities), dtype=np.float32)

    entity_labels = np.zeros(max_entities, dtype=np.float32)
    evidence_labels = np.zeros(max_evidences, dtype=np.float32)

    num_entities = 0
    num_evidences = 0

    for drug in evidences:
        drug: DrugRecMedicine
        drug_id: str = drug.drugid
        en_id = entity_to_id.get(drug_id)
        if en_id is None:
            en_id = num_entities
            entity_to_id[drug_id] = en_id
            id_to_entity[en_id] = drug_id
            num_entities += 1

        contain_evidences: list[GATEvidence] = []
        evidence_labels_for_drug: list[str] = []
        for treat in drug.treat:
            if treat.treat is None:
                continue
            evidence_labels_for_drug.append(treat.treat)
        for caution in drug.caution:
            evidence_labels_for_drug.append(caution.crowd)
        for interaction in drug.interaction:
            evidence_labels_for_drug.append(interaction.name)
        for ingredient in drug.ingredients:
            if ingredient.ingredient is None:
                continue
            evidence_labels_for_drug.append(ingredient.ingredient)

        for label in evidence_labels_for_drug:
            ev_id = evidence_to_id.get(label)
            if ev_id is None:
                if num_evidences >= max_evidences:
                    continue
                ev_id = num_evidences
                evidence_to_id[label] = ev_id
                id_to_evidence[ev_id] = label
                num_evidences += 1
                evidences_list.append(GATEvidence(id=ev_id, label=label))

            contain_evidences.append(GATEvidence(id=ev_id, label=label))
            ent_to_ev[en_id, ev_id] = 1
            ev_to_ent[ev_id, en_id] = 1

        entity = entity_to_node.get(drug_id)
        if entity is None:
            entity = GATEntity(
                id=en_id,
                name=drug.name,
                instruction=drug,
                connect_property=contain_evidences,
                is_answer=drug_id in true_id_set,
            )
            entity_to_node[drug_id] = entity
            entities_list.append(entity)
        else:
            entity.connect_property.extend(contain_evidences)
        entity_labels[en_id] = int(drug_id in true_id_set)

    # Convert to tensors
    ent_to_ev = torch.from_numpy(ent_to_ev)
    ev_to_ent = torch.from_numpy(ev_to_ent)
    entity_labels = torch.from_numpy(entity_labels).to(dtype=torch.long)
    evidence_labels = torch.from_numpy(evidence_labels).to(dtype=torch.long)

    # Create masks
    entity_mask = torch.FloatTensor(num_entities * [1] + (max_entities - num_entities) * [0])
    evidence_mask = torch.FloatTensor(num_evidences * [1] + (max_evidences - num_evidences) * [0])

    # Padding
    padding_instruction = DrugRecMedicine(
        CMAN=None,
        caution=[],
        drugid="",
        ingredients=[],
        interaction=[],
        name="",
        treat=[],
    )
    entities_list = entities_list + (max_entities - num_entities) * [
        GATEntity(
            id=0,
            name="",
            instruction=padding_instruction,
            connect_property=[],
            is_answer=False,
        )
    ]
    evidences_list = evidences_list + (max_evidences - num_evidences) * [
        GATEvidence(id=0, label="")
    ]

    if train and not torch.sum(entity_labels):
        return None # 无答案跳过

    # Normalize adjacency matrices
    vec = torch.sum(ent_to_ev, dim=0)
    vec[vec == 0] = 1
    ent_to_ev = ent_to_ev / vec

    vec = torch.sum(ev_to_ent, dim=0)
    vec[vec == 0] = 1
    ev_to_ent = ev_to_ent / vec

    return GATModelSample(
        source_sample=sample,
        question_id=sample.people.id,
        tsf=question,
        entities=entities_list,
        entity_mask=entity_mask,
        evidences=evidences_list,
        evidence_mask=evidence_mask,
        ent_to_ev=ent_to_ev,
        ev_to_ent=ev_to_ent,
        entity_labels=entity_labels,
        evidence_labels=evidence_labels,
        id_to_entity=cast("Sequence[str | None]", id_to_entity),
        id_to_evidence=cast("Sequence[str | None]", id_to_evidence),
        gold_answers=true_id_list,
    )

import numpy as np
import torch

from ...schema import DrugRecMedicine, DrugRecRecord, TraceDRSample
from .schema import TraceDREntity, TraceDREvidence, TraceDRModelSample, TraceDRNodeId


class ContinueWithNext(Exception):
    pass


def build_model_sample(
    sample: TraceDRSample, train: bool = False, max_entities=100, max_evidences: int = 50
) -> TraceDRModelSample | None:
    tsf: str = _person_to_query(sample.people)
    topk_drugs = sample.top_k_drugs.values()
    gold_answers = [medicine.drugid for medicine in sample.people.medicine]
    gold_answer_set = set(gold_answers)
    on_medicine = sample.people.on_medicine

    # evidence 是候选药物 entities 是药物有关属性
    evidences: list[DrugRecMedicine] = list()
    evidences.extend(on_medicine)
    evidences.extend(topk_drugs)
    if len(evidences) > max_evidences - len(on_medicine):
        evidences = evidences[: (max_evidences - len(on_medicine))]

    if train:
        has_answer = any(evidence.drugid in gold_answer_set for evidence in evidences)
        if not has_answer:
            # raise ContinueWithNext("No answers found.")
            return None

    # 从 neo4j 空间到数据集空间的映射
    entity_to_id: dict[TraceDRNodeId, int] = {}
    evidence_to_id: dict[str, int] = {}

    # 这里用连续id代替hash所以是列表 映射的并非是业务空间id而是实体本身
    id_to_entity = np.empty(max_entities, dtype=object)
    id_to_entity.fill(None)
    id_to_evidence = np.empty(max_evidences, dtype=object)
    id_to_evidence.fill(None)

    # 数据集空间存放数据的地方
    entities_list = list()
    evidences_list = list()

    # 邻接矩阵
    ent_to_ev = np.zeros((max_entities, max_evidences), dtype=np.float32)
    ev_to_ent = np.zeros((max_evidences, max_entities), dtype=np.float32)

    # ent 是否属于 gold medicine
    entity_labels = np.zeros(max_entities, dtype=np.float32)
    # ev 是否是 gold medicine
    evidence_labels = np.zeros(max_evidences, dtype=np.float32)

    num_entities = 0
    num_evidences = 0

    for drug in evidences:
        evidence_text = _get_evidence_text(drug)
        drug_entity = TraceDREntity(drug.drugid, drug.name, "药品")
        entities = [
            drug_entity,
        ]  # 这里少的一些字段是被移到其他地方了

        # 将该药物加入 evidence
        g_ev_id = evidence_to_id.get(drug.drugid)
        is_new_evidence = g_ev_id is None
        if g_ev_id is None:
            g_ev_id = num_evidences
            evidence_to_id[drug.drugid] = g_ev_id
            id_to_evidence[g_ev_id] = drug
            num_evidences += 1

        # 将药物自身加入 entity
        if num_entities < (max_entities - 1):
            entity_to_id[drug.drugid] = num_entities
            id_to_entity[num_entities] = drug_entity
            entities_list.append(drug_entity)

            entity_labels[num_entities] = int(drug.drugid in gold_answer_set)

            ent_to_ev[num_entities, g_ev_id] = 1
            ev_to_ent[g_ev_id, num_entities] = 1

            num_entities += 1

        # 添加其他 ent
        # 遍历所有 ent
        new_entities: list[tuple[TraceDRNodeId, TraceDREntity]] = []
        for treat in drug.treat:
            treat_label = treat.treat if treat.treat is not None else "None"
            treat_id = treat.treat_id if treat.treat_id is not None else treat_label
            new_entities.append((treat_label, TraceDREntity(treat_id, treat_label, "治疗")))
        for caution in drug.caution:
            new_entities.append(
                (caution.crowd, TraceDREntity(caution.crowd_id, caution.crowd, "禁用"))
            )
        for ingredient in drug.ingredients:
            ingredient_label = (
                ingredient.ingredient if ingredient.ingredient is not None else "None"
            )
            ingredient_id = (
                ingredient.ingredient_id
                if ingredient.ingredient_id is not None
                else ingredient_label
            )
            new_entities.append(
                (
                    ingredient_label,
                    TraceDREntity(ingredient_id, ingredient_label, "成分"),
                )
            )
        for interaction in drug.interaction:
            new_entities.append(
                (
                    interaction.name,
                    TraceDREntity(interaction.interaction_id, interaction.name, "相互作用"),
                )
            )

        # 将 ent 置入到图中
        for entity_key, entity in new_entities:
            g_ent_id = entity_to_id.get(entity_key)
            if g_ent_id is None:
                if num_entities >= (max_entities - 1):
                    continue
                g_ent_id = num_entities
                id_to_entity[g_ent_id] = entity
                entity_to_id[entity_key] = g_ent_id
                entities_list.append(entity)
                num_entities += 1

            entities.append(entity)
            ent_to_ev[g_ent_id, g_ev_id] = 1
            ev_to_ent[g_ev_id, g_ent_id] = 1

        if is_new_evidence:
            evidences_list.append(
                TraceDREvidence(
                    evidence_text=evidence_text,
                    contain_entities=entities,
                )
            )
        # 判断是否有答案
        evidence_labels[g_ev_id] = int(drug.drugid in gold_answer_set)

    # 转换为 tensor
    ent_to_ev_tensor = torch.from_numpy(ent_to_ev).to_sparse()
    ent_to_ev_tensor.requires_grad = False
    ev_to_ent_tensor = torch.from_numpy(ev_to_ent).to_sparse()
    ev_to_ent_tensor.requires_grad = False
    entity_labels_tensor = torch.from_numpy(entity_labels).to(dtype=torch.long)
    entity_labels_tensor.requires_grad = False
    evidence_labels_tensor = torch.from_numpy(evidence_labels).to(dtype=torch.long)
    evidence_labels_tensor.requires_grad = False

    # 创建 mask
    entity_mask = torch.FloatTensor(num_entities * [1] + (max_entities - num_entities) * [0])
    evidence_mask = torch.FloatTensor(num_evidences * [1] + (max_evidences - num_evidences) * [0])

    # padding
    entities_list = entities_list + (max_entities - num_entities) * [TraceDREntity("", "", "")]
    evidences_list = evidences_list + (max_evidences - num_evidences) * [
        TraceDREvidence(
            evidence_text="",
            contain_entities=[],
        )
    ]

    # 训练时截断后若无答案则跳过
    if train and not torch.sum(entity_labels_tensor):
        # raise ContinueWithNext("Answer pruned via max_entities restriction")
        return None

    # 归一化邻接矩阵
    ent_to_ev_dense = ent_to_ev_tensor.to_dense()
    ent_to_ev_vec = torch.sum(ent_to_ev_dense, dim=0)
    ent_to_ev_vec[ent_to_ev_vec == 0] = 1
    ent_to_ev_dense = ent_to_ev_dense / ent_to_ev_vec

    ev_to_ent_dense = ev_to_ent_tensor.to_dense()
    ev_to_ent_vec = torch.sum(ev_to_ent_dense, dim=0)
    ev_to_ent_vec[ev_to_ent_vec == 0] = 1
    ev_to_ent_dense = ev_to_ent_dense / ev_to_ent_vec

    return TraceDRModelSample(
        question_id=sample.people.id,
        on_medicine=sample.people.on_medicine,
        entities=entities_list,
        entity_mask=entity_mask,
        evidences=evidences_list,
        evidence_mask=evidence_mask,
        ent_to_ev=ent_to_ev_dense,
        ev_to_ent=ev_to_ent_dense,
        entity_labels=entity_labels_tensor,
        evidence_labels=evidence_labels_tensor,
        id_to_entity=id_to_entity,
        id_to_evidence=id_to_evidence,
        tsf=tsf,
        question=tsf,
        gold_answers=gold_answers,
    )

def load_dataset():
    pass


def _person_to_query(person: DrugRecRecord, delimiter: str = " || "):
    onmedicine = ",".join(item.name for item in person.on_medicine)
    group = ",".join(person.group)
    symptom = ",".join(person.symptom)
    diagnosis = ",".join(person.diagnosis)
    antecedents = ",".join(person.antecedents)
    allergen = ",".join(person.allergen)
    query = f"{person.age} {delimiter} {group} {delimiter} {person.gender} {delimiter} {diagnosis} {delimiter} {symptom} {delimiter} {antecedents} {delimiter} {onmedicine} {delimiter} {allergen}"
    return query


def _get_evidence_text(evidence: DrugRecMedicine):
    treatments = [item.treat for item in evidence.treat if item.treat is not None]
    treatments_string = ", ".join(treatments) if treatments else "None"

    caution_values = [
        item.crowd + item.caution_level
        for item in evidence.caution
        if item.crowd is not None and item.caution_level is not None
    ]
    caution_string = ", ".join(caution_values) if caution_values else "None"

    interaction_values = [item.name for item in evidence.interaction if item.name.strip()]
    interaction_string = ", ".join(interaction_values) if interaction_values else "None"

    ingredients_values = [
        item.ingredient
        for item in evidence.ingredients
        if item.ingredient is not None and item.ingredient.strip()
    ]
    ingredients_string = ", ".join(ingredients_values) if ingredients_values else "None"

    name = evidence.name
    query = f"药名:{name} || 治疗:{treatments_string} || 禁用:{caution_string} || 成分:{ingredients_string} || 相互作用:{interaction_string}"
    return query

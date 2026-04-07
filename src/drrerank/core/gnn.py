from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .schema import (
    DrugRecCase,
    DrugRecMedicine,
    DrugRecRecord,
    GNNEntity,
    GNNEvidence,
    GNNModelInput,
    GNNNodeType,
    GNNTrainSample,
)


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


def normalize_rows(
    matrix: Float[torch.Tensor, "row col"],
) -> Float[torch.Tensor, "row col"]:
    if matrix.numel() == 0:
        return matrix
    row_sum = torch.sum(matrix, dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return matrix / row_sum


class FullEncoder(nn.Module):
    def __init__(
        self,
        encoder_model_name: str,
        patient_max_length: int,
        evidence_max_length: int,
        entity_max_length: int,
    ) -> None:
        super().__init__()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(encoder_model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(encoder_model_name)
        self.patient_max_length = patient_max_length
        self.evidence_max_length = evidence_max_length
        self.entity_max_length = entity_max_length
        self.sep_token = self.tokenizer.sep_token or "[SEP]"

    def forward(
        self,
        model_input: GNNModelInput,
        device: torch.device,
    ) -> tuple[
        Float[torch.Tensor, "hidden"],
        Float[torch.Tensor, "entity hidden"],
        Float[torch.Tensor, "evidence hidden"],
    ]:
        patient_vec = self.encode_texts(
            [model_input.patient_text],
            max_length=self.patient_max_length,
            device=device,
        )[0]
        evidence_mat = self.encode_texts(
            [evidence.text for evidence in model_input.evidences],
            max_length=self.evidence_max_length,
            device=device,
        )
        entity_mat = self.encode_texts(
            [f"{entity.text}{self.sep_token}{entity.node_type}" for entity in model_input.entities],
            max_length=self.entity_max_length,
            device=device,
        )
        return patient_vec, entity_mat, evidence_mat

    def encode_texts(
        self,
        texts: list[str],
        max_length: int,
        device: torch.device,
    ) -> Float[torch.Tensor, "item hidden"]:
        if not texts:  # 若一个病人既无候选药业务已用药则此处用 not texts
            hidden_size = int(self.model.config.hidden_size)
            return torch.empty((0, hidden_size), device=device)
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,  # 按 max_length 截断
            padding=True,
            max_length=max_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}  # 转移设备
        outputs = self.model(**batch)
        # NOTE [batch, seq_len] -> [batch, seq_len, hidden]
        mask = batch["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        # NOTE 平均池化
        pooled = torch.sum(outputs.last_hidden_state * mask, dim=1)
        return pooled / torch.clamp(mask.sum(dim=1), min=1.0)


class GNNModel(nn.Module):
    # NOTE 只与 TraceDR 架构等价，实现细节有很大差异
    def __init__(
        self,
        hidden_size: int = 768,
        encoder_model_name: str = "hfl/chinese-roberta-wwm-ext",
        patient_max_length: int = 64,
        max_text_length: int = 256,
        entity_max_length: int = 64,
        message_passing_steps: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_model_name = encoder_model_name
        self.patient_max_length = patient_max_length
        self.max_text_length = max_text_length
        self.entity_max_length = entity_max_length
        self.message_passing_steps = message_passing_steps
        self.dropout = dropout
        self.encoder = FullEncoder(
            encoder_model_name=encoder_model_name,
            patient_max_length=patient_max_length,
            evidence_max_length=max_text_length,
            entity_max_length=entity_max_length,
        )
        # NOTE 一大坨线性层
        self.ev_att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps)]
        )
        self.ev_msg_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps)]
        )
        self.ent_att_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps)]
        )
        self.ent_msg_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(message_passing_steps)]
        )
        # NOTE Bilinear 参数量更大也更容易过拟合
        self.answer_head = nn.Bilinear(hidden_size, hidden_size, 1)
        self.evidence_head = nn.Bilinear(hidden_size, hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        model_inputs: list[GNNModelInput],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        device = next(self.parameters()).device
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for model_input in model_inputs:
            patient_vec, entity_mat, evidence_mat = self.encoder(model_input, device)
            entity_mat, evidence_mat = self.run_message_passing(
                entity_mat=entity_mat,
                evidence_mat=evidence_mat,
                patient_vec=patient_vec,
                ent_to_ev=model_input.ent_to_ev.to(device),
                ev_to_ent=model_input.ev_to_ent.to(device),
            )
            outputs.append(
                (
                    self.answer_head(
                        entity_mat,
                        patient_vec.expand(entity_mat.shape[0], -1),
                    ).squeeze(-1),
                    self.evidence_head(
                        evidence_mat,
                        patient_vec.expand(evidence_mat.shape[0], -1),
                    ).squeeze(-1),
                )
            )
        return outputs  # NOTE 不是求 list 的梯度而是 list 里元素的梯度

    def get_loss(
        self,
        model_inputs: list[GNNModelInput],
        outputs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        losses: list[torch.Tensor] = []
        for model_input, (entity_logits, evidence_logits) in zip(
            model_inputs,
            outputs,
            strict=True,
        ):  # NOTE 因为这里不是单一标准任务，而是“每个样本两个任务、每个任务长度还可变、并且证据分支可能为空”的组合。
            entity_loss = self.loss_fn(
                entity_logits,
                model_input.entity_labels.to(device),
            )
            if evidence_logits.numel() == 0:
                losses.append(entity_loss)
                continue
            evidence_loss = self.loss_fn(
                evidence_logits,
                model_input.evidence_labels.to(device),
            )
            losses.append(0.5 * (entity_loss + evidence_loss))
        return torch.stack(losses).mean()

    def build_checkpoint(self) -> dict[str, object]:
        return {
            "model_name": "gnn",
            "model_state_dict": self.state_dict(),
            "init_kwargs": {
                "hidden_size": self.hidden_size,
                "encoder_model_name": self.encoder_model_name,
                "patient_max_length": self.patient_max_length,
                "max_text_length": self.max_text_length,
                "entity_max_length": self.entity_max_length,
                "message_passing_steps": self.message_passing_steps,
                "dropout": self.dropout,
            },
        }

    def run_message_passing(
        # ANCHOR 以后写中期报告的以latex公式的形式和标准图神经网络过程做对比以加深我理解
        self,
        entity_mat: Float[torch.Tensor, "entity hidden"],
        evidence_mat: Float[torch.Tensor, "evidence hidden"],
        patient_vec: Float[torch.Tensor, "hidden"],
        ent_to_ev: Float[torch.Tensor, "entity evidence"],
        ev_to_ent: Float[torch.Tensor, "evidence entity"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for ev_att_layer, ev_msg_layer, ent_att_layer, ent_msg_layer in zip(
            self.ev_att_layers,
            self.ev_msg_layers,
            self.ent_att_layers,
            self.ent_msg_layers,
            strict=True,
        ):
            evidence_weights = normalize_rows(
                (
                    ev_to_ent
                    * F.softmax(ev_att_layer(evidence_mat) @ patient_vec, dim=0).unsqueeze(-1)
                ).transpose(0, 1)
            )
            entity_mat = F.relu(ev_msg_layer(evidence_weights @ evidence_mat) + entity_mat)
            entity_weights = normalize_rows(
                (
                    ent_to_ev
                    * F.softmax(ent_att_layer(entity_mat) @ patient_vec, dim=0).unsqueeze(-1)
                ).transpose(0, 1)
            )
            evidence_mat = F.relu(ent_msg_layer(entity_weights @ entity_mat) + evidence_mat)
            entity_mat = F.dropout(
                entity_mat,
                p=self.dropout,
                training=self.training,
            )
            evidence_mat = F.dropout(
                evidence_mat,
                p=self.dropout,
                training=self.training,
            )
        return entity_mat, evidence_mat

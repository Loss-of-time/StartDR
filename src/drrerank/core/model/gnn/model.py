from typing import cast

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

from .schema import GNNModelInput


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
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            encoder_model_name,
            use_safetensors=False,
        )
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
        attention_mask = cast(torch.Tensor, batch["attention_mask"])
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
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

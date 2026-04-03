import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import AutoModel, AutoTokenizer

from ...schema.drugrec_task import GNNModelInput


class FullEncoder(nn.Module):
    """编码患者、证据、实体文本。"""

    def __init__(
        self,
        encoder_model_name: str,
        patient_max_length: int,
        evidence_max_length: int,
        entity_max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.model = AutoModel.from_pretrained(encoder_model_name)
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
        """一次性编码患者、实体、证据文本。"""
        patient_vec = self._encode_texts(
            [model_input["patient_text"]],
            max_length=self.patient_max_length,
            device=device,
        )[0]
        evidence_mat = self._encode_texts(
            [evidence["text"] for evidence in model_input["evidences"]],
            max_length=self.evidence_max_length,
            device=device,
        )
        entity_mat = self._encode_texts(
            [
                f"{entity['text']}{self.sep_token}{entity['node_type']}"
                for entity in model_input["entities"]
            ],
            max_length=self.entity_max_length,
            device=device,
        )
        return patient_vec, entity_mat, evidence_mat

    def _encode_texts(
        self,
        texts: list[str],
        max_length: int,
        device: torch.device,
    ) -> Float[torch.Tensor, "item hidden"]:
        """用同一编码器批量编码文本。"""
        if not texts:
            hidden_size = int(self.model.config.hidden_size)
            return torch.empty((0, hidden_size), device=device)
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        pooled = torch.sum(outputs.last_hidden_state * mask, dim=1)
        return pooled / torch.clamp(mask.sum(dim=1), min=1.0)


class GNNModel(nn.Module):
    """作为主模型的 TraceDR 风格实现。"""

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
        self.answer_head = nn.Bilinear(hidden_size, hidden_size, 1)
        self.evidence_head = nn.Bilinear(hidden_size, hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        model_inputs: list[GNNModelInput],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """执行文本编码、图更新和双线性打分。"""
        device = next(self.parameters()).device
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for model_input in model_inputs:
            patient_vec, entity_mat, evidence_mat = self.encoder(model_input, device)
            entity_mat, evidence_mat = self._run_message_passing(
                entity_mat=entity_mat,
                evidence_mat=evidence_mat,
                patient_vec=patient_vec,
                ent_to_ev=model_input["ent_to_ev"].to(device),
                ev_to_ent=model_input["ev_to_ent"].to(device),
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
        return outputs

    def get_loss(
        self,
        model_inputs: list[GNNModelInput],
        outputs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """聚合一批样本的多任务损失。"""
        device = next(self.parameters()).device
        losses: list[torch.Tensor] = []
        for model_input, (entity_logits, evidence_logits) in zip(
            model_inputs,
            outputs,
            strict=True,
        ):
            entity_loss = self.loss_fn(
                entity_logits,
                model_input["entity_labels"].to(device),
            )
            if evidence_logits.numel() == 0:
                losses.append(entity_loss)
                continue
            evidence_loss = self.loss_fn(
                evidence_logits,
                model_input["evidence_labels"].to(device),
            )
            losses.append(0.5 * (entity_loss + evidence_loss))
        return torch.stack(losses).mean()

    def build_checkpoint(self) -> dict[str, object]:
        """导出主模型 checkpoint。"""
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

    def _run_message_passing(
        self,
        entity_mat: Float[torch.Tensor, "entity hidden"],
        evidence_mat: Float[torch.Tensor, "evidence hidden"],
        patient_vec: Float[torch.Tensor, "hidden"],
        ent_to_ev: Float[torch.Tensor, "entity evidence"],
        ev_to_ent: Float[torch.Tensor, "evidence entity"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按 TraceDR 的二部图方式做患者条件消息传递。"""
        for ev_att_layer, ev_msg_layer, ent_att_layer, ent_msg_layer in zip(
            self.ev_att_layers,
            self.ev_msg_layers,
            self.ent_att_layers,
            self.ent_msg_layers,
            strict=True,
        ):
            evidence_weights = _normalize_rows(
                (
                    ev_to_ent
                    * F.softmax(ev_att_layer(evidence_mat) @ patient_vec, dim=0)
                    .unsqueeze(-1)
                ).transpose(0, 1)
            )
            entity_mat = F.relu(ev_msg_layer(evidence_weights @ evidence_mat) + entity_mat)

            entity_weights = _normalize_rows(
                (
                    ent_to_ev
                    * F.softmax(ent_att_layer(entity_mat) @ patient_vec, dim=0)
                    .unsqueeze(-1)
                ).transpose(0, 1)
            )
            evidence_mat = F.relu(
                ent_msg_layer(entity_weights @ entity_mat) + evidence_mat
            )

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


def _normalize_rows(
    matrix: Float[torch.Tensor, "row col"],
) -> Float[torch.Tensor, "row col"]:
    """按行归一化权重矩阵。"""
    if matrix.numel() == 0:
        return matrix
    row_sum = torch.sum(matrix, dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return matrix / row_sum

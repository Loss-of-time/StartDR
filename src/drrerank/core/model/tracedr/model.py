# cspell:words jaxtyping
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
import transformers
from jaxtyping import Float
from torch import Tensor, nn
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .schema import (
    EntityEvidenceMatrix,
    EvidenceEntityMatrix,
    TraceDRModelSample,
)

type TokenEmbeddings = Float[Tensor, "item seq hidden"]
type EncodedTexts = Float[Tensor, "item hidden"]
type SentenceEmbedding = Float[Tensor, "hidden"]
type EvidenceNodeEmbeddings = Float[Tensor, "evidence hidden"]
type EntityNodeEmbeddings = Float[Tensor, "entity hidden"]


@dataclass(slots=True)
class TraceDRForwardResult:
    entity_logits: Float[Tensor, "entity"]
    evidence_logits: Float[Tensor, "evidence"]
    loss: Tensor
    # entity_accuracy: float


class FullEncoder(nn.Module):
    def __init__(
        self,
        emb_dimension: int,
        max_entities: int,
        max_evidences: int,
        max_input_length_sr: int,  # sr: 拼接得到的患者文本
        max_input_length_ev: int,
        max_input_length_ent: int,
        encoder_model_name: str = "hfl/chinese-roberta-wwm-ext",
        encoder_sep_token: str = "[SEP]",
    ) -> None:
        super().__init__()

        if not torch.cuda.is_available():
            raise NotImplementedError("CUDA is not available")

        self.emb_dimension = emb_dimension
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.max_input_length_sr = max_input_length_sr
        self.max_input_length_ev = max_input_length_ev
        self.max_input_length_ent = max_input_length_ent
        self.encoder_model_name = encoder_model_name
        self.tokenizer: PreTrainedTokenizerBase = transformers.AutoTokenizer.from_pretrained(
            encoder_model_name
        )
        self.model: PreTrainedModel = transformers.AutoModel.from_pretrained(
            encoder_model_name,
            use_safetensors=False,
        )
        self.sep_token = encoder_sep_token

        self.device = torch.device("cuda")
        cast(nn.Module, self.model).to(self.device)

    def forward(
        self,
        sample: TraceDRModelSample,
    ) -> tuple[
        SentenceEmbedding,
        EvidenceNodeEmbeddings,
        EntityNodeEmbeddings,
    ]:
        tsf_vec = self.encode(
            [sample.tsf],
            max_input_length=self.max_input_length_sr,
        ).squeeze(0)

        flattened_evidence_texts = [evidence.evidence_text for evidence in sample.evidences]
        evidences_mat = self.encode(
            flattened_evidence_texts,
            max_input_length=self.max_input_length_ev,
        )

        flattened_entity_texts = [
            f"{entity.label}{self.sep_token}{entity.type}" for entity in sample.entities
        ]
        entities_mat = self.encode(
            flattened_entity_texts,
            max_input_length=self.max_input_length_ent,
        )

        return tsf_vec, evidences_mat, entities_mat

    def encode(
        self,
        flattened_input: list[str],
        max_input_length: int,
    ) -> EncodedTexts:

        tokenized_input: BatchEncoding = self.tokenizer(
            flattened_input,
            padding="max_length",  # 固定到当前分支的最大长度，便于批量编码和后续 reshape
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).to(self.device)

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model(**tokenized_input)
        last_hidden_state = outputs.last_hidden_state
        assert last_hidden_state is not None
        lm_encodings: TokenEmbeddings = last_hidden_state.to(self.device)
        attention_mask_tensor = cast(Tensor, tokenized_input["attention_mask"])
        attention_mask: Float[Tensor, "batch seq 1"] = attention_mask_tensor.unsqueeze(dim=2).to(
            lm_encodings.dtype
        )
        pooled_sum: Float[Tensor, "batch hidden"] = torch.sum(lm_encodings * attention_mask, dim=1)
        pooled_count: Float[Tensor, "batch 1"] = torch.sum(attention_mask, dim=1).clamp(min=1.0)
        encodings: EncodedTexts = pooled_sum / pooled_count
        return encodings


class GNNLayer(nn.Module):
    def __init__(self, emb_dimension: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.w_ev_att = nn.Linear(emb_dimension, emb_dimension)
        self.w_ev_ent = nn.Linear(emb_dimension, emb_dimension)
        self.w_ent_att = nn.Linear(emb_dimension, emb_dimension)
        self.w_ent_ev = nn.Linear(emb_dimension, emb_dimension)

    def forward(
        self,
        tsf_vec: SentenceEmbedding,
        evidences_mat: EvidenceNodeEmbeddings,
        entities_mat: EntityNodeEmbeddings,
        ent_to_ev: EntityEvidenceMatrix,
        ev_to_ent: EvidenceEntityMatrix,
    ) -> tuple[EvidenceNodeEmbeddings, EntityNodeEmbeddings]:
        projected_evs: Float[Tensor, "evidence hidden"] = self.w_ev_att(evidences_mat)
        ev_att_scores: Float[Tensor, "evidence 1"] = projected_evs @ tsf_vec.unsqueeze(dim=1)
        ev_att_scores = F.softmax(ev_att_scores, dim=0)
        ev_att_scores = ev_att_scores.clamp(min=1e-30, max=1e20)

        evidence_weights: Float[Tensor, "entity evidence"] = ev_att_scores * ev_to_ent
        evidence_weights = evidence_weights.clamp(min=1e-30, max=1e20)
        evidence_weights = evidence_weights.transpose(0, 1)

        vec: Float[Tensor, "entity 1"] = torch.sum(
            evidence_weights,
            keepdim=True,
            dim=1,
        )
        vec[vec == 0] = 1
        evidence_weights = evidence_weights / vec

        ev_message_ent: Float[Tensor, "entity hidden"] = torch.mm(
            evidence_weights,
            evidences_mat,
        )
        ev_message_ent = self.w_ev_ent(ev_message_ent)

        entities_mat = F.relu(ev_message_ent + entities_mat)

        projected_ents: Float[Tensor, "entity hidden"] = self.w_ent_att(entities_mat)
        ent_att_scores: Float[Tensor, "entity 1"] = projected_ents @ tsf_vec.unsqueeze(dim=1)
        ent_att_scores = F.softmax(ent_att_scores, dim=0)
        ent_att_scores = ent_att_scores.clamp(min=1e-30, max=1e20)

        entity_weights: Float[Tensor, "evidence entity"] = ent_att_scores * ent_to_ev
        entity_weights = entity_weights.clamp(min=1e-30, max=1e20)
        entity_weights = entity_weights.transpose(0, 1)

        vec: Float[Tensor, "evidence 1"] = torch.sum(
            entity_weights,
            keepdim=True,
            dim=1,
        )
        vec[vec == 0] = 1
        entity_weights = entity_weights / vec

        ent_messages_ev: Float[Tensor, "evidence hidden"] = torch.mm(
            entity_weights,
            entities_mat,
        )
        ent_messages_ev = self.w_ent_ev(ent_messages_ev)

        evidences_mat = F.relu(ent_messages_ev + evidences_mat)

        entities_mat = F.dropout(
            entities_mat,
            self.dropout,
            training=self.training,
        )
        evidences_mat = F.dropout(
            evidences_mat,
            self.dropout,
            training=self.training,
        )

        return evidences_mat, entities_mat


class MultitaskBilinearAnswering(nn.Module):
    def __init__(
        self,
        emb_dimension: int = 768,
        max_entities: int = 100,
        max_evidence: int = 50,
        use_evidence_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.emb_dimension = emb_dimension
        self.max_entities = max_entities
        self.max_evidences = max_evidence
        self.use_evidence_supervision = use_evidence_supervision

        self.bilinear_answer = nn.Bilinear(
            emb_dimension, emb_dimension, 1
        )  # 1 是 out_features，表示双线性层输出 1 维
        self.bilinear_evidence = nn.Bilinear(emb_dimension, emb_dimension, 1)

        # 损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _get_masked_loss(
        self,
        logits: Float[Tensor, "item"],
        labels: Tensor,
        mask: Tensor,
    ) -> Tensor:
        raw_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            reduction="none",
        )
        masked_loss = raw_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1.0)

    # def _get_masked_accuracy(
    #     self,
    #     logits: Float[Tensor, "item"],
    #     labels: Tensor,
    #     mask: Tensor,
    # ) -> float:
    #     preds = (torch.sigmoid(logits) > 0.5).float()
    #     correct = ((preds == labels.float()).float() * mask).sum()
    #     total = mask.sum().clamp(min=1.0)
    #     return float((correct / total).item())

    def forward(
        self,
        sample: TraceDRModelSample,
        entity_mat: EntityNodeEmbeddings,
        sr_vec: SentenceEmbedding,
        ev_mat: EvidenceNodeEmbeddings,
    ) -> TraceDRForwardResult:
        """参数：
        - sample: 单个样本
        - entity_mat (num_ent x emb_dim): 实体编码
        - sr_vec (emb_dim): SR 向量
        - ev_mat (num_ev x emb_dim): 证据编码
        """
        sr_vec_expanded: Float[Tensor, "entity hidden"] = sr_vec.unsqueeze(0).expand(
            self.max_entities, -1
        )
        answer_logits: Float[Tensor, "entity"] = self.bilinear_answer(
            entity_mat, sr_vec_expanded
        ).squeeze(-1)
        sr_vec_ev_expanded: Float[Tensor, "evidence hidden"] = sr_vec.unsqueeze(0).expand(
            self.max_evidences, -1
        )
        ev_logits: Float[Tensor, "evidence"] = self.bilinear_evidence(
            ev_mat, sr_vec_ev_expanded
        ).squeeze(-1)
        answer_logits = answer_logits.masked_fill(sample.entity_mask == 0, 0.0)
        ev_logits = ev_logits.masked_fill(sample.evidence_mask == 0, 0.0)

        loss = torch.tensor(0.0, device=entity_mat.device)
        answer_loss = self._get_masked_loss(
            answer_logits.view(-1),
            sample.entity_labels.view(-1),
            sample.entity_mask.view(-1),
        )
        evidence_loss = self._get_masked_loss(
            ev_logits.view(-1),
            sample.evidence_labels.view(-1),
            sample.evidence_mask.view(-1),
        )

        # 目的：在同一训练入口中切换“是否保留证据监督”，避免维护第二套 answering 逻辑。
        if self.use_evidence_supervision:
            ANSWER_WEIGHT = 0.5
            EV_WEIGHT = 1 - ANSWER_WEIGHT
            loss += ANSWER_WEIGHT * answer_loss + EV_WEIGHT * evidence_loss
        else:
            loss += answer_loss
        # entity_accuracy = self._get_masked_accuracy(
        #     answer_logits,
        #     sample.entity_labels,
        #     sample.entity_mask,
        # )

        return TraceDRForwardResult(
            loss=loss,
            entity_logits=answer_logits,
            evidence_logits=ev_logits,
            # entity_accuracy=entity_accuracy,
        )


class HeterogeneousGNN(nn.Module):
    # TODO 以后换用更多不同的图模型
    def __init__(
        self,
        emb_dimension: int = 768,
        num_layers: int = 3,
        dropout: float = 0.0,
        max_entities: int = 100,
        max_evidences: int = 50,
        use_evidence_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.emb_dimension = emb_dimension
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_entities = max_entities
        self.max_evidences = max_evidences

        self.encoder: FullEncoder = FullEncoder(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidences=max_evidences,
            max_input_length_sr=30,
            max_input_length_ev=80,
            max_input_length_ent=60,
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            [GNNLayer(emb_dimension=emb_dimension, dropout=dropout) for _ in range(num_layers)]
        )
        self.answering = MultitaskBilinearAnswering(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidence=max_evidences,
            use_evidence_supervision=use_evidence_supervision,
        )

        self.cuda()

    def forward(
        self,
        sample: TraceDRModelSample,
    ) -> TraceDRForwardResult:
        tsf_vec, evidences_mat, entities_mat = self.encoder.forward(sample)
        for layer in self.layers:
            layer = cast(GNNLayer, layer)
            evidences_mat, entities_mat = layer.forward(
                tsf_vec, evidences_mat, entities_mat, sample.ent_to_ev, sample.ev_to_ent
            )

        res = self.answering(sample, entities_mat, tsf_vec, evidences_mat)
        return res

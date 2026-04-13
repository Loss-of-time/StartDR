from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
import transformers
from jaxtyping import Float
from torch import Tensor, nn
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .schema import EntityEvidenceMatrix, EvidenceEntityMatrix, GATModelSample

type TokenEmbeddings = Float[Tensor, "item seq hidden"]
type EncodedTexts = Float[Tensor, "item hidden"]
type SentenceEmbedding = Float[Tensor, "hidden"]
type EvidenceNodeEmbeddings = Float[Tensor, "evidence hidden"]
type EntityNodeEmbeddings = Float[Tensor, "entity hidden"]


@dataclass(slots=True)
class GATForwardResult:
    entity_logits: Float[Tensor, "entity"]
    loss: Tensor


class FullEncoder(nn.Module):
    def __init__(
        self,
        emb_dimension: int,
        max_entities: int,
        max_evidences: int,
        max_input_length_sr: int,
        max_input_length_ev: int,
        max_input_length_ent: int,
        encoder_model_name: str = "hfl/chinese-roberta-wwm-ext",
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

        self.device = torch.device("cuda")
        cast(nn.Module, self.model).to(self.device)

    def forward(
        self,
        sample: GATModelSample,
    ) -> tuple[
        SentenceEmbedding,
        EvidenceNodeEmbeddings,
        EntityNodeEmbeddings,
    ]:
        tsf_vec = self.encode(
            [sample.tsf],
            max_input_length=self.max_input_length_sr,
        ).squeeze(0)
        evidences_mat = self.encode(
            [evidence.label for evidence in sample.evidences],
            max_input_length=self.max_input_length_ev,
        )
        entities_mat = self.encode(
            [entity.name for entity in sample.entities],
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
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).to(self.device)

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model(**tokenized_input)
        last_hidden_state = outputs.last_hidden_state
        assert last_hidden_state is not None
        lm_encodings: TokenEmbeddings = last_hidden_state.to(self.device)
        attention_mask_tensor = cast(Tensor, tokenized_input["attention_mask"])
        attention_mask: Float[Tensor, "batch seq 1"] = (
            attention_mask_tensor.unsqueeze(dim=2).to(lm_encodings.dtype)
        )
        pooled_sum: Float[Tensor, "batch hidden"] = torch.sum(lm_encodings * attention_mask, dim=1)
        pooled_count: Float[Tensor, "batch 1"] = torch.sum(attention_mask, dim=1).clamp(min=1.0)
        encodings: EncodedTexts = pooled_sum / pooled_count
        return encodings


class GNNLayer(nn.Module):
    def __init__(self, emb_dimension: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.w = nn.Linear(emb_dimension, emb_dimension)
        self.w_att = nn.Linear(emb_dimension, emb_dimension)

    def forward(
        self,
        tsf_vec: SentenceEmbedding,
        evidences_mat: EvidenceNodeEmbeddings,
        entities_mat: EntityNodeEmbeddings,
        ent_to_ev: EntityEvidenceMatrix,
        ev_to_ent: EvidenceEntityMatrix,
    ) -> tuple[EvidenceNodeEmbeddings, EntityNodeEmbeddings]:
        del tsf_vec

        projected_evs: Float[Tensor, "evidence hidden"] = self.w_att(evidences_mat)
        projected_ents: Float[Tensor, "entity hidden"] = self.w_att(entities_mat)

        ev_att_scores: Float[Tensor, "evidence entity"] = projected_evs @ projected_ents.transpose(0, 1)
        ev_valid_mask = ev_to_ent > 0
        ev_masked_scores = ev_att_scores.masked_fill(
            ~ev_valid_mask,
            torch.finfo(ev_att_scores.dtype).min,
        )
        ev_max_scores = ev_masked_scores.max(dim=0, keepdim=True).values
        ev_has_valid_edge = ev_valid_mask.any(dim=0, keepdim=True)
        ev_max_scores = torch.where(ev_has_valid_edge, ev_max_scores, torch.zeros_like(ev_max_scores))
        ev_att_scores = torch.exp(ev_att_scores - ev_max_scores) * ev_to_ent * ev_valid_mask.to(
            ev_att_scores.dtype
        )
        ev_normalizer = ev_att_scores.sum(dim=0, keepdim=True)
        ev_att_scores = torch.where(
            ev_normalizer > 0,
            ev_att_scores / ev_normalizer,
            torch.zeros_like(ev_att_scores),
        )
        ev_messages_ent: Float[Tensor, "entity hidden"] = torch.mm(
            ev_att_scores.transpose(0, 1),
            evidences_mat,
        )
        ev_messages_ent = self.w(ev_messages_ent)
        ent_messages_ent: Float[Tensor, "entity hidden"] = self.w(entities_mat)
        entities_mat = F.relu(ev_messages_ent + ent_messages_ent)

        projected_ents = self.w_att(entities_mat)
        ent_att_scores: Float[Tensor, "entity evidence"] = projected_ents @ projected_evs.transpose(0, 1)
        ent_valid_mask = ent_to_ev > 0
        ent_masked_scores = ent_att_scores.masked_fill(
            ~ent_valid_mask,
            torch.finfo(ent_att_scores.dtype).min,
        )
        ent_max_scores = ent_masked_scores.max(dim=0, keepdim=True).values
        ent_has_valid_edge = ent_valid_mask.any(dim=0, keepdim=True)
        ent_max_scores = torch.where(ent_has_valid_edge, ent_max_scores, torch.zeros_like(ent_max_scores))
        ent_att_scores = torch.exp(ent_att_scores - ent_max_scores) * ent_to_ev * ent_valid_mask.to(
            ent_att_scores.dtype
        )
        ent_normalizer = ent_att_scores.sum(dim=0, keepdim=True)
        ent_att_scores = torch.where(
            ent_normalizer > 0,
            ent_att_scores / ent_normalizer,
            torch.zeros_like(ent_att_scores),
        )
        ent_messages_ev: Float[Tensor, "evidence hidden"] = torch.mm(
            ent_att_scores.transpose(0, 1),
            entities_mat,
        )
        ent_messages_ev = self.w(ent_messages_ev)
        ev_messages_ev: Float[Tensor, "evidence hidden"] = self.w(evidences_mat)
        evidences_mat = F.relu(ent_messages_ev + ev_messages_ev)

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


class BilinearAnswering(nn.Module):
    def __init__(
        self,
        emb_dimension: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.emb_dimension = emb_dimension
        self.dropout = dropout
        self.answer_linear_projection = nn.Linear(
            in_features=emb_dimension,
            out_features=emb_dimension,
        )

    def _get_masked_loss(
        self,
        logits: Float[Tensor, "entity"],
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

    def forward(
        self,
        sample: GATModelSample,
        entity_mat: EntityNodeEmbeddings,
        sr_vec: SentenceEmbedding,
    ) -> GATForwardResult:
        projected_entities: EntityNodeEmbeddings = self.answer_linear_projection(entity_mat)
        projected_entities = F.dropout(
            projected_entities,
            p=self.dropout,
            training=self.training,
        )
        entity_logits: Float[Tensor, "entity"] = torch.matmul(projected_entities, sr_vec)
        entity_logits = entity_logits.masked_fill(sample.entity_mask == 0, 0.0)
        loss = self._get_masked_loss(
            entity_logits,
            sample.entity_labels,
            sample.entity_mask,
        )
        return GATForwardResult(
            entity_logits=entity_logits,
            loss=loss,
        )

class GAT(torch.nn.Module):
    def __init__(
        self,
        emb_dimension: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_entities: int = 50,
        max_evidences: int = 100,
        max_input_length_sr: int = 128,
        max_input_length_ev: int = 64,
        max_input_length_ent: int = 32,
        encoder_model_name: str = "hfl/chinese-roberta-wwm-ext",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.emb_dimension = emb_dimension
        self.dropout = dropout
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.max_input_length_sr = max_input_length_sr
        self.max_input_length_ev = max_input_length_ev
        self.max_input_length_ent = max_input_length_ent
        self.encoder_model_name = encoder_model_name

        self.encoder = FullEncoder(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidences=max_evidences,
            max_input_length_sr=max_input_length_sr,
            max_input_length_ev=max_input_length_ev,
            max_input_length_ent=max_input_length_ent,
            encoder_model_name=encoder_model_name,
        )
        self.answering = BilinearAnswering(
            emb_dimension=emb_dimension,
            dropout=dropout,
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            [GNNLayer(emb_dimension=emb_dimension, dropout=dropout) for _ in range(num_layers)]
        )

        self.cuda()

    def forward(
        self,
        sample: GATModelSample,
    ) -> GATForwardResult:
        tsf_vec, evidences_mat, entities_mat = self.encoder(sample)
        for layer in self.layers:
            layer = cast(GNNLayer, layer)
            evidences_mat, entities_mat = layer(
                tsf_vec,
                evidences_mat,
                entities_mat,
                sample.ent_to_ev,
                sample.ev_to_ent,
            )
        return self.answering(sample, entities_mat, tsf_vec)

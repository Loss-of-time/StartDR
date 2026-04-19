"""4SDrug `main1` 训练版模型。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .schema import (
    FourSDrugDrugTensor,
    FourSDrugForwardResult,
    FourSDrugLogits,
    FourSDrugModelConfig,
    FourSDrugSimilarIndexTensor,
    FourSDrugSymptomTensor,
)


class SymptomAttention(nn.Module):
    """症状集合注意力聚合器。"""

    def __init__(self, embed_dim: int) -> None:
        """初始化症状集合聚合器。

        Args:
            embed_dim: 嵌入维度。
        """

        super().__init__()
        self.aggregation: nn.Linear = nn.Linear(embed_dim, 1)

    def forward(
        self,
        symptom_embeddings: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """聚合症状集合表示。

        Args:
            symptom_embeddings: 症状嵌入张量。
            mask: 有效症状位置掩码。

        Returns:
            聚合后的集合表示。
        """

        attention_logits: Tensor = torch.tanh(
            self.aggregation(symptom_embeddings).squeeze(-1),
        )
        if mask is None:
            attention_weights: Tensor = F.softmax(attention_logits, dim=1)
        else:
            # 目的：对 padding 位归零，并在全空共有症状时安全返回零向量。
            attention_logits = attention_logits.masked_fill(~mask, -1e9)
            attention_weights = F.softmax(attention_logits, dim=1)
            attention_weights = attention_weights * mask.float()
            attention_weights = attention_weights / attention_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1.0)
        return torch.bmm(attention_weights.unsqueeze(1), symptom_embeddings).squeeze(1)


class FourSDrugModel(nn.Module):
    """4SDrug `main1` 最短路径复刻模型。"""

    def __init__(
        self,
        config: FourSDrugModelConfig,
        ddi_adj: Tensor,
    ) -> None:
        """初始化模型。

        Args:
            config: 模型配置。
            ddi_adj: 0-based 药物 DDI 稀疏邻接矩阵。
        """

        super().__init__()
        self.config: FourSDrugModelConfig = config
        self.sym_embeddings: nn.Embedding = nn.Embedding(
            config.symptom_vocab_size + 1,
            config.embed_dim,
        )
        self.drug_embeddings: nn.Embedding = nn.Embedding(
            config.medicine_vocab_size + 1,
            config.embed_dim,
        )
        self.sym_agg: SymptomAttention = SymptomAttention(config.embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.register_buffer("ddi_adj", ddi_adj.coalesce())
        self._init_parameters()

    def _init_parameters(self) -> None:
        """初始化模型参数。"""

        init_range: float = 1.0 / math.sqrt(self.config.embed_dim)
        parameter: nn.Parameter
        for parameter in self.parameters():
            parameter.data.uniform_(-init_range, init_range)

    def _get_all_drug_embeddings(self) -> Tensor:
        """获取并归一化全部药物嵌入。"""

        drug_ids: Tensor = torch.arange(
            1,
            self.config.medicine_vocab_size + 1,
            device=self.drug_embeddings.weight.device,
            dtype=torch.long,
        )
        drug_embeddings: Tensor = self.drug_embeddings(drug_ids)
        return F.normalize(drug_embeddings, p=2, dim=-1)

    def _encode_symptom_set(self, symptoms: FourSDrugSymptomTensor) -> Tensor:
        """编码症状集合。"""

        symptom_embeddings: Tensor = self.sym_embeddings(symptoms.long())
        symptom_embeddings = self.dropout(symptom_embeddings)
        set_embeddings: Tensor = self.sym_agg(symptom_embeddings)
        return F.normalize(set_embeddings, p=2, dim=-1)

    def _build_common_symptom_tensor(
        self,
        symptoms: FourSDrugSymptomTensor,
        similar_indices: FourSDrugSimilarIndexTensor,
    ) -> tuple[FourSDrugSymptomTensor, Tensor]:
        """构造当前样本与相似样本的共有症状张量。"""

        batch_size: int = int(symptoms.shape[0])
        common_symptoms_per_row: list[list[int]] = []
        row_index: int
        for row_index in range(batch_size):
            current_row: list[int] = [int(value) for value in symptoms[row_index].tolist()]
            similar_row: list[int] = [
                int(value) for value in symptoms[int(similar_indices[row_index].item())].tolist()
            ]
            similar_set: set[int] = set(similar_row)
            common_row: list[int] = [value for value in current_row if value in similar_set]
            common_symptoms_per_row.append(common_row)

        max_common_count: int = max((len(row) for row in common_symptoms_per_row), default=0)
        max_common_count = max(max_common_count, 1)
        common_tensor: Tensor = torch.zeros(
            (batch_size, max_common_count),
            dtype=torch.long,
            device=symptoms.device,
        )
        common_mask: Tensor = torch.zeros(
            (batch_size, max_common_count),
            dtype=torch.bool,
            device=symptoms.device,
        )

        current_row: list[int]
        for row_index, current_row in enumerate(common_symptoms_per_row):
            if not current_row:
                continue
            common_tensor[row_index, : len(current_row)] = torch.tensor(
                current_row,
                dtype=torch.long,
                device=symptoms.device,
            )
            common_mask[row_index, : len(current_row)] = True
        return common_tensor, common_mask

    def _compute_ddi_loss(self, probabilities: Tensor) -> Tensor:
        """计算 DDI 正则项。"""

        predicted_binary: Tensor = (probabilities >= self.config.prediction_threshold).float()
        if float(predicted_binary.sum().item()) == 0.0:
            return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)

        ddi_hits: Tensor = torch.sparse.mm(
            self.ddi_adj, predicted_binary.transpose(0, 1)
        ).transpose(
            0,
            1,
        )
        pair_counts: Tensor = (ddi_hits * predicted_binary).sum(dim=1) / 2.0
        return 1e-6 * pair_counts.mean()

    def _compute_augmentation_loss(
        self,
        symptoms: FourSDrugSymptomTensor,
        drugs: FourSDrugDrugTensor,
        similar_indices: FourSDrugSimilarIndexTensor,
        drug_embeddings: Tensor,
    ) -> Tensor:
        """计算集内增强项。"""

        similar_drugs: Tensor = drugs.index_select(0, similar_indices)
        common_drugs: Tensor = drugs * similar_drugs
        if float(common_drugs.sum().item()) == 0.0:
            return torch.zeros((), dtype=drugs.dtype, device=drugs.device)

        common_symptoms: Tensor
        common_mask: Tensor
        common_symptoms, common_mask = self._build_common_symptom_tensor(symptoms, similar_indices)
        common_symptom_embeddings: Tensor = self.sym_embeddings(common_symptoms.long())
        common_set_embeddings: Tensor = self.sym_agg(common_symptom_embeddings, common_mask)
        common_set_embeddings = F.normalize(common_set_embeddings, p=2, dim=-1)
        augmentation_logits: Tensor = torch.matmul(
            common_set_embeddings, drug_embeddings.transpose(0, 1)
        )
        return F.binary_cross_entropy_with_logits(augmentation_logits, common_drugs)

    def _compute_interset_ddi_loss(
        self,
        symptoms: FourSDrugSymptomTensor,
        drugs: FourSDrugDrugTensor,
        similar_indices: FourSDrugSimilarIndexTensor,
        drug_embeddings: Tensor,
    ) -> Tensor:
        """计算跨集 DDI 约束项。"""

        different_drugs: Tensor = torch.abs(drugs - drugs.index_select(0, similar_indices))
        if float(different_drugs.sum().item()) == 0.0:
            return torch.zeros((), dtype=drugs.dtype, device=drugs.device)

        common_symptoms: Tensor
        common_mask: Tensor
        common_symptoms, common_mask = self._build_common_symptom_tensor(symptoms, similar_indices)
        common_symptom_embeddings: Tensor = self.sym_embeddings(common_symptoms.long())
        common_set_embeddings: Tensor = self.sym_agg(common_symptom_embeddings, common_mask)
        common_set_embeddings = F.normalize(common_set_embeddings, p=2, dim=-1)

        different_counts: Tensor = different_drugs.sum(dim=1, keepdim=True).clamp(min=1.0)
        different_drug_embeddings: Tensor = (
            torch.matmul(different_drugs, drug_embeddings) / different_counts
        )
        interaction_scores: Tensor = torch.sigmoid(
            (common_set_embeddings * different_drug_embeddings).sum(dim=1),
        )
        return 1e-4 * interaction_scores.mean()

    def forward(
        self,
        symptoms: FourSDrugSymptomTensor,
        drugs: FourSDrugDrugTensor,
        similar_indices: FourSDrugSimilarIndexTensor,
    ) -> FourSDrugForwardResult:
        """执行单个 batch 的前向计算。"""

        set_embeddings: Tensor = self._encode_symptom_set(symptoms)
        drug_embeddings: Tensor = self._get_all_drug_embeddings()
        logits: Tensor = torch.matmul(set_embeddings, drug_embeddings.transpose(0, 1))
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        probabilities: Tensor = torch.sigmoid(logits)

        ddi_loss: Tensor = self._compute_ddi_loss(probabilities)
        augmentation_loss: Tensor = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if symptoms.shape[0] > 2 and symptoms.shape[1] > 2:
            augmentation_loss = self._compute_augmentation_loss(
                symptoms,
                drugs,
                similar_indices,
                drug_embeddings,
            )
            ddi_loss = ddi_loss + self._compute_interset_ddi_loss(
                symptoms,
                drugs,
                similar_indices,
                drug_embeddings,
            )

        return FourSDrugForwardResult(
            logits=logits,
            probabilities=probabilities,
            ddi_loss=ddi_loss,
            augmentation_loss=augmentation_loss,
        )

    def predict_logits(self, symptoms: FourSDrugSymptomTensor) -> FourSDrugLogits:
        """为评估阶段输出全部药物 logits。"""

        set_embeddings: Tensor = self._encode_symptom_set(symptoms)
        drug_embeddings: Tensor = self._get_all_drug_embeddings()
        logits: Tensor = torch.matmul(set_embeddings, drug_embeddings.transpose(0, 1))
        return torch.clamp(logits, min=-10.0, max=10.0)

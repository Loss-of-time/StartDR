import math
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float

from ...metrics.drugrec import aggregate_drugrec_metrics, get_drugrec_metrics
from ...metrics.gnn_drugrec import aggregate_gnn_metrics, get_gnn_metrics
from ...schema.drugrec_task import (
    DrugRecCase,
    DrugRecCheckpoint,
    GNNEdge,
    GNNGraphSample,
    DrugRecMetrics,
    EvalStepOutput,
    GNNNodeScore,
    GNNRecResult,
    ModelStateDict,
    NumericFeatureStats,
    RankedDrug,
    RankedEvidence,
    TrainStepOutput,
)
from ..drugrec_model import GNNRecModel
from .data_set import (
    GNNGraphSampleBuilder,
    fit_numeric_feature_stats,
)


class GNNModel(GNNRecModel):
    """基于候选药局部图的推荐模型。"""

    DEFAULT_HIDDEN_SIZE = 16
    model_name = "gnn"
    selection_metric = "mrr"

    @classmethod
    def build_for_train(
        cls,
        train_cases: list[DrugRecCase],
        top_k: int,
    ) -> "GNNModel":
        """用训练病例拟合标准化统计量并构建模型。"""
        return cls(
            stats=fit_numeric_feature_stats(train_cases),
            top_k=top_k,
        )

    def __init__(
        self,
        stats: NumericFeatureStats,
        top_k: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        """初始化 GNN 推荐模型的参数。"""
        super().__init__()
        self.stats = stats
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        graph_samples: list[GNNGraphSample],
    ) -> list[torch.Tensor]:
        """对一批图样本输出候选药 logits。"""
        logits_list: list[torch.Tensor] = []
        for graph_sample in graph_samples:
            rows: list[list[float]] = []
            for target in graph_sample.candidate_targets:
                feature = graph_sample.drug_numeric_features[target.drug_node_id]
                if feature.retrieval_score is None:
                    score_z = 0.0
                else:
                    score_z = (
                        math.log1p(feature.retrieval_score)
                        - self.stats["score_log_mean"]
                    ) / self.stats["score_log_std"]
                if feature.retrieval_rank is None:
                    rank_norm = 1.0
                else:
                    rank_norm = feature.retrieval_rank / self.top_k
                rows.append(
                    [
                        score_z,
                        rank_norm,
                        float(feature.is_candidate),
                        float(feature.is_on_medicine),
                    ]
                )
            feature_tensor: Float[torch.Tensor, "..."] = torch.tensor(
                rows,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )
            logits: Float[torch.Tensor, "..."] = self.scorer(
                feature_tensor
            ).squeeze(-1)
            logits_list.append(logits)
        return logits_list

    def predict(self, case: DrugRecCase) -> GNNRecResult:
        """对单个病例输出药物排序结果。"""
        graph_sample = GNNGraphSampleBuilder(case).build()
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward([graph_sample])[0]
        if was_training:
            self.train()
        return self._build_result(case, graph_sample, logits)

    def train_step(
        self,
        cases: list[DrugRecCase],
    ) -> TrainStepOutput:
        """执行一次训练步并返回训练损失。"""
        self.train()
        graph_samples = [
            GNNGraphSampleBuilder(case).build() for case in cases
        ]
        logits_list = self.forward(graph_samples)
        loss_terms = [
            self.loss_fn(
                logits,
                torch.tensor(
                    [
                        float(target.label)
                        for target in graph_sample.candidate_targets
                    ],
                    dtype=torch.float32,
                    device=next(self.parameters()).device,
                ),
            )
            for logits, graph_sample in zip(logits_list, graph_samples, strict=True)
        ]
        loss = torch.stack(loss_terms).mean()
        results = [
            self._build_result(case, graph_sample, logits)
            for case, graph_sample, logits in zip(
                cases,
                graph_samples,
                logits_list,
                strict=True,
            )
        ]
        base_metrics = aggregate_drugrec_metrics(
            [
                get_drugrec_metrics(case, result)
                for case, result in zip(cases, results, strict=True)
            ]
        )
        gnn_metrics = aggregate_gnn_metrics(
            [
                get_gnn_metrics(case, result)
                for case, result in zip(cases, results, strict=True)
            ]
        )
        metrics: DrugRecMetrics = {
            "loss": float(loss.detach().item()),
            **base_metrics,
            **gnn_metrics,
        }
        return {
            "loss": loss,
            "loss_value": float(loss.detach().item()),
            "metrics": metrics,
        }

    def eval_step(
        self,
        cases: list[DrugRecCase],
    ) -> EvalStepOutput:
        """执行一次评测步并返回批量排序结果。"""
        graph_samples = [
            GNNGraphSampleBuilder(case).build() for case in cases
        ]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits_list = self.forward(graph_samples)
        if was_training:
            self.train()
        results = [
            self._build_result(case, graph_sample, logits)
            for case, graph_sample, logits in zip(
                cases,
                graph_samples,
                logits_list,
                strict=True,
            )
        ]
        return {
            "results": results,
        }

    def build_checkpoint(self) -> DrugRecCheckpoint:
        """导出当前模型的 checkpoint。"""
        return {
            "model_name": "gnn",
            "model_state_dict": cast(ModelStateDict, self.state_dict()),
            "init_kwargs": {
                "stats": {
                    "score_log_mean": self.stats["score_log_mean"],
                    "score_log_std": self.stats["score_log_std"],
                },
                "top_k": self.top_k,
                "hidden_size": self.hidden_size,
            },
        }

    def _build_result(
        self,
        case: DrugRecCase,
        graph_sample: GNNGraphSample,
        logits: torch.Tensor,
    ) -> GNNRecResult:
        """把候选药 logits 整理成统一结果。"""
        score_list = torch.sigmoid(logits).detach().cpu().tolist()
        candidate_by_drugid = {
            candidate["drugid"]: candidate
            for candidate in case["candidate_drugs"]
        }
        candidate_score_by_node_id = {
            target.drug_node_id: float(score)
            for target, score in zip(
                graph_sample.candidate_targets,
                score_list,
                strict=True,
            )
        }
        ranked_drugs: list[RankedDrug] = []
        for target, score in zip(
            graph_sample.candidate_targets,
            score_list,
            strict=True,
        ):
            candidate = candidate_by_drugid[target.drugid]
            ranked_drugs.append(
                {
                    "drugid": candidate["drugid"],
                    "score": float(score),
                    "rank": 0,
                    "drug": candidate["drug"],
                    "retrieval_score": candidate["score"],
                    "retrieval_rank": candidate["rank"],
                    "label": target.label,
                }
            )
        ranked_drugs.sort(key=lambda item: item["score"], reverse=True)
        for rank, ranked_drug in enumerate(ranked_drugs, start=1):
            ranked_drug["rank"] = rank
        return {
            "patient_id": case["patient_id"],
            "split": case["split"],
            "model_name": "gnn",
            "node_scores": self._build_node_scores(
                graph_sample,
                candidate_score_by_node_id,
            ),
            "ranked_evidences": self._build_ranked_evidences(
                graph_sample,
                candidate_score_by_node_id,
            ),
            "ranked_drugs": ranked_drugs,
        }

    def _build_node_scores(
        self,
        graph_sample: GNNGraphSample,
        candidate_score_by_node_id: dict[str, float],
    ) -> list[GNNNodeScore]:
        """为图中节点生成可解释分数。"""
        evidence_score_by_node_id = self._get_evidence_score_by_node_id(
            graph_sample.edges,
            candidate_score_by_node_id,
        )
        node_scores: list[GNNNodeScore] = []
        for node in graph_sample.nodes:
            node_scores.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "score": candidate_score_by_node_id.get(
                        node.node_id,
                        evidence_score_by_node_id.get(node.node_id, 0.0),
                    ),
                }
            )
        node_scores.sort(key=lambda item: item["score"], reverse=True)
        return node_scores

    def _build_ranked_evidences(
        self,
        graph_sample: GNNGraphSample,
        candidate_score_by_node_id: dict[str, float],
    ) -> list[RankedEvidence]:
        """根据候选药得分整理证据节点排序。"""
        node_text_by_id = {
            node.node_id: node.text
            for node in graph_sample.nodes
            if node.node_type != "drug"
        }
        if not node_text_by_id:
            return []
        evidence_score_by_node_id = self._get_evidence_score_by_node_id(
            graph_sample.edges,
            candidate_score_by_node_id,
        )
        positive_drug_node_ids = {
            target.drug_node_id
            for target in graph_sample.candidate_targets
            if target.label == 1
        }
        positive_evidence_node_ids = {
            edge.src_node_id
            for edge in graph_sample.edges
            if edge.edge_type.startswith("rev_")
            and edge.dst_node_id in positive_drug_node_ids
        }
        ranked_evidences: list[RankedEvidence] = [
            {
                "evidence_id": node_id,
                "score": evidence_score_by_node_id.get(node_id, 0.0),
                "rank": 0,
                "text": node_text_by_id[node_id],
                "label": 1 if node_id in positive_evidence_node_ids else 0,
            }
            for node_id in node_text_by_id
        ]
        ranked_evidences.sort(key=lambda item: item["score"], reverse=True)
        for rank, ranked_evidence in enumerate(ranked_evidences, start=1):
            ranked_evidence["rank"] = rank
        return ranked_evidences

    def _get_evidence_score_by_node_id(
        self,
        edges: list[GNNEdge],
        candidate_score_by_node_id: dict[str, float],
    ) -> dict[str, float]:
        """把候选药得分传播到其关联证据节点。"""
        evidence_score_by_node_id: dict[str, float] = {}
        for edge in edges:
            if not edge.edge_type.startswith("rev_"):
                continue
            drug_score = candidate_score_by_node_id.get(edge.dst_node_id)
            if drug_score is None:
                continue
            previous_score = evidence_score_by_node_id.get(edge.src_node_id)
            if previous_score is None or drug_score > previous_score:
                evidence_score_by_node_id[edge.src_node_id] = drug_score
        return evidence_score_by_node_id

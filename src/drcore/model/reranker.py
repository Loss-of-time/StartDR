from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.nn as nn

from ..schema.model.gnn_reranker import GNNGraphSample
from ..schema.patient_candidate_set import PatientCandidateSet
from ..schema.reranker import (
    EvalStepOutput,
    RerankerCheckpoint,
    RerankResult,
    TrainStepOutput,
)


class RerankerModule(nn.Module, ABC):
    """供 trainer 调用的精排模型统一接口。"""

    @classmethod
    @abstractmethod
    def build_for_train(
        cls,
        train_samples: list[PatientCandidateSet],
        top_k: int,
    ) -> Self:
        """用训练样本完成模型训练期初始化。"""

    @abstractmethod
    def forward(
        self,
        graph_samples: list[GNNGraphSample],
    ) -> list[torch.Tensor]:
        """返回每条样本候选药的未归一化打分。"""

    @abstractmethod
    def rerank(self, sample: PatientCandidateSet) -> RerankResult:
        """对单条冻结候选集样本执行精排。"""

    @abstractmethod
    def train_step(
        self,
        samples: list[PatientCandidateSet],
    ) -> TrainStepOutput:
        """执行一次训练步并返回 trainer 可消费结果。"""

    @abstractmethod
    def eval_step(
        self,
        samples: list[PatientCandidateSet],
    ) -> EvalStepOutput:
        """执行一次评测步并返回排序结果与指标。"""

    @abstractmethod
    def build_checkpoint(self) -> RerankerCheckpoint:
        """导出当前模型的 checkpoint。"""

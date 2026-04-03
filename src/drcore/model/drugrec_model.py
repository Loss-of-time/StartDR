from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Self

import torch.nn as nn

from ..schema.drugrec_task import (
    DrugRecCase,
    DrugRecCheckpoint,
    DrugRecModelName,
    DrugRecResult,
    EvalStepOutput,
    GNNRecResult,
    TrainStepOutput,
)


class DrugRecModel(nn.Module, ABC):
    model_name: ClassVar[DrugRecModelName]
    selection_metric: ClassVar[str]
    result_kind: ClassVar[Literal["base", "gnn"]] = "base"

    @classmethod
    @abstractmethod
    def build_for_train(
        cls,
        train_cases: list[DrugRecCase],
        top_k: int,
    ) -> Self:
        """用训练病例完成训练期初始化。"""

    @abstractmethod
    def train_step(
        self,
        cases: list[DrugRecCase],
    ) -> TrainStepOutput:
        """执行一次训练步。"""

    @abstractmethod
    def eval_step(
        self,
        cases: list[DrugRecCase],
    ) -> EvalStepOutput:
        """执行一次评测步并返回通用结果。"""

    @abstractmethod
    def predict(
        self,
        case: DrugRecCase,
    ) -> DrugRecResult:
        """对单个病例输出药品排序结果。"""

    @abstractmethod
    def build_checkpoint(self) -> DrugRecCheckpoint:
        """导出当前模型的 checkpoint。"""


class GNNRecModel(DrugRecModel, ABC):
    result_kind: ClassVar[Literal["base", "gnn"]] = "gnn"

    @abstractmethod
    def eval_step(
        self,
        cases: list[DrugRecCase],
    ) -> EvalStepOutput:
        """执行一次评测步并返回 GNN 扩展结果。"""

    @abstractmethod
    def predict(
        self,
        case: DrugRecCase,
    ) -> GNNRecResult:
        """对单个病例输出 GNN 扩展结果。"""

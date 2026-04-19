"""统一训练实验的公共数据结构。"""

from dataclasses import dataclass, field
from typing import Literal

type ExperimentMetricMap = dict[str, float]
type ExperimentSplit = Literal["dev", "test"]


@dataclass(slots=True)
class ComparableMetrics:
    """跨模型可直接对比的排序指标。"""

    p_at_1: float = 0.0
    mrr: float = 0.0
    hit_at_5: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    f1_at_5: float = 0.0

    def to_dict(self) -> ExperimentMetricMap:
        """展开可比较指标。

        Returns:
            以字段名为键的指标字典。
        """

        return {
            "p_at_1": self.p_at_1,
            "mrr": self.mrr,
            "hit_at_5": self.hit_at_5,
            "precision_at_5": self.precision_at_5,
            "recall_at_5": self.recall_at_5,
            "f1_at_5": self.f1_at_5,
        }


@dataclass(slots=True)
class ExperimentEvalResult:
    """单个数据切分上的评测结果。"""

    loss: float
    comparable_metrics: ComparableMetrics
    extra_metrics: ExperimentMetricMap = field(default_factory=dict)

    def to_flat_dict(self) -> ExperimentMetricMap:
        """展开全部评测指标。

        Returns:
            便于日志与落盘的平铺指标字典。
        """

        metric_map: ExperimentMetricMap = {"loss": self.loss}
        metric_map.update(self.comparable_metrics.to_dict())
        metric_map.update(self.extra_metrics)
        return metric_map

    def get_metric_value(self, metric_name: str) -> float:
        """读取原始指标值。

        Args:
            metric_name: 指标名称。

        Returns:
            原始指标值。
        """

        metric_map: ExperimentMetricMap = self.to_flat_dict()
        if metric_name not in metric_map:
            raise ValueError(f"未找到指标 `{metric_name}`。")
        return metric_map[metric_name]

    def get_metric_score(self, metric_name: str) -> float:
        """读取可用于排序的指标分数。

        Args:
            metric_name: 指标名称。

        Returns:
            越大越优的排序分数。
        """

        metric_value: float = self.get_metric_value(metric_name)
        # 目的：统一把损失转成“越大越优”的排序分数，便于同一 runner 处理最佳轮次选择。
        if metric_name == "loss":
            return -metric_value
        return metric_value


@dataclass(slots=True)
class ExperimentEpochResult:
    """单轮训练结果。"""

    epoch: int
    train_loss: float
    dev_metrics: ExperimentEvalResult


@dataclass(slots=True)
class ExperimentReport:
    """单个模型实验的完整报告。"""

    experiment_name: str
    output_name: str
    selection_metric: str
    best_epoch: int
    best_metric_value: float
    best_dev_metrics: ExperimentEvalResult
    test_metrics: ExperimentEvalResult | None
    epochs: list[ExperimentEpochResult]


@dataclass(slots=True)
class TrainingArtifacts:
    """单次训练产物路径。"""

    report_path: str
    checkpoint_path: str


@dataclass(slots=True)
class TrainingRunResult:
    """统一 runner 的返回结果。"""

    report: ExperimentReport
    artifacts: TrainingArtifacts


@dataclass(slots=True)
class CompareSummaryRow:
    """单个模型在对比实验中的汇总行。"""

    model_name: str
    output_name: str
    best_epoch: int
    selection_metric: str
    best_metric_value: float
    compare_split: str
    compare_metric: str
    compare_metric_value: float
    best_dev_metrics: ExperimentMetricMap
    test_metrics: ExperimentMetricMap | None
    report_path: str
    checkpoint_path: str


@dataclass(slots=True)
class CompareReport:
    """多模型对比实验汇总报告。"""

    output_name: str
    compare_metric: str
    rows: list[CompareSummaryRow]

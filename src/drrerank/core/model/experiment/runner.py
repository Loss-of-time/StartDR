"""统一训练实验的 runner。"""

import json
from pathlib import Path
from typing import Protocol

from ...schema import unstructure
from ...setting import DEFAULT_MODEL_OUTPUT_DIR
from .schema import (
    ExperimentEpochResult,
    ExperimentEvalResult,
    ExperimentReport,
    ExperimentSplit,
    TrainingArtifacts,
    TrainingRunResult,
)


class ExperimentConfigLike(Protocol):
    """统一 runner 依赖的最小配置协议。"""

    output_name: str
    epochs: int
    selection_metric: str


class ExperimentAdapter[ConfigT: ExperimentConfigLike, StateT, SnapshotT](Protocol):
    """模型实验适配协议。"""

    experiment_name: str

    def setup(self, config: ConfigT) -> StateT:
        """构造实验状态。

        Args:
            config: 实验配置。

        Returns:
            后续训练与评测复用的状态对象。
        """
        ...

    def train_epoch(self, state: StateT, epoch: int, total_epochs: int) -> float:
        """执行单轮训练。

        Args:
            state: 当前实验状态。
            epoch: 当前轮次，从 1 开始。
            total_epochs: 总轮数。

        Returns:
            当前轮训练损失均值。
        """
        ...

    def evaluate(self, state: StateT, split: ExperimentSplit) -> ExperimentEvalResult:
        """执行指定切分的评测。

        Args:
            state: 当前实验状态。
            split: 待评测切分。

        Returns:
            统一评测结果。
        """
        ...

    def has_split(self, state: StateT, split: ExperimentSplit) -> bool:
        """判断是否存在指定切分。

        Args:
            state: 当前实验状态。
            split: 待查询切分。

        Returns:
            若存在可评测数据则返回真。
        """
        ...

    def capture_snapshot(self, state: StateT) -> SnapshotT:
        """捕获最佳权重快照。

        Args:
            state: 当前实验状态。

        Returns:
            可恢复的权重快照。
        """
        ...

    def restore_snapshot(self, state: StateT, snapshot: SnapshotT) -> None:
        """恢复最佳权重快照。

        Args:
            state: 当前实验状态。
            snapshot: 待恢复的权重快照。
        """
        ...

    def export_checkpoint(self, state: StateT, output_path: Path) -> None:
        """导出最终 checkpoint。

        Args:
            state: 当前实验状态。
            output_path: checkpoint 输出路径。
        """
        ...


def _format_eval_summary(
    selection_metric: str,
    eval_result: ExperimentEvalResult,
) -> str:
    """格式化单次评测日志。

    Args:
        selection_metric: 当前最佳轮次选择指标。
        eval_result: 评测结果。

    Returns:
        单行日志字符串。
    """

    metric_items: list[tuple[str, float]] = [("dev_loss", eval_result.loss)]
    selected_value: float = eval_result.get_metric_value(selection_metric)
    metric_items.append((f"select_{selection_metric}", selected_value))

    for metric_name, metric_value in eval_result.comparable_metrics.to_dict().items():
        if metric_name == selection_metric:
            continue
        metric_items.append((metric_name, metric_value))
    for metric_name, metric_value in eval_result.extra_metrics.items():
        if metric_name == selection_metric:
            continue
        metric_items.append((metric_name, metric_value))

    return " ".join(
        f"{metric_name}={metric_value:.6f}" for metric_name, metric_value in metric_items
    )


def run_training_experiment[ConfigT: ExperimentConfigLike, StateT, SnapshotT](
    config: ConfigT,
    adapter: ExperimentAdapter[ConfigT, StateT, SnapshotT],
) -> TrainingRunResult:
    """执行统一训练与评测流程。

    Args:
        config: 训练配置。
        adapter: 模型适配器。

    Returns:
        落盘后的实验报告与产物路径。
    """

    state: StateT = adapter.setup(config)
    DEFAULT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path: Path = DEFAULT_MODEL_OUTPUT_DIR / f"{config.output_name}.json"
    checkpoint_path: Path = DEFAULT_MODEL_OUTPUT_DIR / f"{config.output_name}.pt"

    best_epoch: int = 0
    best_metric_value: float = 0.0
    best_metric_score: float | None = None
    best_snapshot: SnapshotT | None = None
    epoch_results: list[ExperimentEpochResult] = []

    epoch_index: int
    for epoch_index in range(1, config.epochs + 1):
        train_loss: float = adapter.train_epoch(state, epoch_index, config.epochs)
        dev_metrics: ExperimentEvalResult = adapter.evaluate(state, "dev")
        epoch_results.append(
            ExperimentEpochResult(
                epoch=epoch_index,
                train_loss=train_loss,
                dev_metrics=dev_metrics,
            )
        )
        print(f"epoch={epoch_index} train_loss={train_loss:.6f}")
        print(_format_eval_summary(config.selection_metric, dev_metrics))

        selected_value: float = dev_metrics.get_metric_value(config.selection_metric)
        selected_score: float = dev_metrics.get_metric_score(config.selection_metric)
        if best_metric_score is not None and selected_score <= best_metric_score:
            continue
        best_epoch = epoch_index
        best_metric_value = selected_value
        best_metric_score = selected_score
        # 目的：统一在 runner 内冻结最佳权重，避免每个模型重复实现最优轮次选择。
        best_snapshot = adapter.capture_snapshot(state)

    if best_snapshot is None:
        raise ValueError("训练阶段未产生可恢复的最佳权重。")

    adapter.restore_snapshot(state, best_snapshot)
    adapter.export_checkpoint(state, checkpoint_path)

    test_metrics: ExperimentEvalResult | None = None
    if adapter.has_split(state, "test"):
        test_metrics = adapter.evaluate(state, "test")

    best_dev_metrics: ExperimentEvalResult = epoch_results[best_epoch - 1].dev_metrics
    report: ExperimentReport = ExperimentReport(
        experiment_name=adapter.experiment_name,
        output_name=config.output_name,
        selection_metric=config.selection_metric,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        best_dev_metrics=best_dev_metrics,
        test_metrics=test_metrics,
        epochs=epoch_results,
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"训练完成，最佳模型已写入: {checkpoint_path.resolve()}")
    print(f"训练完成，报告已写入: {report_path.resolve()}")

    return TrainingRunResult(
        report=report,
        artifacts=TrainingArtifacts(
            report_path=str(report_path.resolve()),
            checkpoint_path=str(checkpoint_path.resolve()),
        ),
    )

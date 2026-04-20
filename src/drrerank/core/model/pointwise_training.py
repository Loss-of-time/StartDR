"""GAT 与 TraceDR 共享的点式精排训练骨架。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from .experiment.progress import build_progress

type PointwiseSnapshot = dict[str, Tensor]


class LossResult(Protocol):
    """最小前向结果协议。"""

    loss: Tensor


class TrainEvalModel(Protocol):
    """支持训练态和评测态切换的最小模型协议。"""

    def eval(self) -> object:
        """切换到评测态。"""
        ...

    def train(self, mode: bool = True) -> object:
        """切换到训练态。"""
        ...


class StateDictModel(Protocol):
    """支持 `state_dict` 的最小模型协议。"""

    def state_dict(self) -> dict[str, Tensor]:
        """导出当前权重。"""
        ...

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> object:
        """恢复已有权重。"""
        ...


@dataclass(slots=True)
class PointwiseTrainState[ModelT, SampleT]:
    """点式精排训练状态。"""

    model: ModelT
    optimizer: torch.optim.Optimizer
    train_samples: list[SampleT]
    dev_samples: list[SampleT]
    test_samples: list[SampleT]


def load_pointwise_splits[SampleT](
    train_input: Path,
    dev_input: Path,
    test_input: Path | None,
    train_limit: int | None,
    dev_limit: int | None,
    test_limit: int | None,
    load_samples: Callable[[Path, int | None, bool], list[SampleT]],
    experiment_name: str,
) -> tuple[list[SampleT], list[SampleT], list[SampleT]]:
    """加载训练、验证、测试三个切分。

    Args:
        train_input: 训练集路径。
        dev_input: 验证集路径。
        test_input: 测试集路径。
        train_limit: 训练集数量上限。
        dev_limit: 验证集数量上限。
        test_limit: 测试集数量上限。
        load_samples: 单个切分的样本加载函数。
        experiment_name: 实验名，用于报错信息。

    Returns:
        训练、验证、测试样本列表。
    """

    train_samples: list[SampleT] = load_samples(train_input, train_limit, True)
    dev_samples: list[SampleT] = load_samples(dev_input, dev_limit, False)
    test_samples: list[SampleT] = []
    if test_input is not None:
        test_samples = load_samples(test_input, test_limit, False)
    if not train_samples:
        raise ValueError(f"训练集为空，无法执行 {experiment_name} 训练。")
    if not dev_samples:
        raise ValueError(f"验证集为空，无法执行 {experiment_name} 评估。")
    return train_samples, dev_samples, test_samples


def train_pointwise_epoch[ModelT, SampleT, CudaSampleT, ResultT: LossResult](
    state: PointwiseTrainState[ModelT, SampleT],
    epoch: int,
    total_epochs: int,
    to_cuda: Callable[[SampleT], CudaSampleT],
    run_forward: Callable[[ModelT, CudaSampleT], ResultT],
    after_backward: Callable[[ModelT], None] | None = None,
) -> float:
    """执行单轮点式精排训练。

    Args:
        state: 当前训练状态。
        epoch: 当前轮次，从 1 开始。
        total_epochs: 总轮数。
        to_cuda: 样本上卡函数。
        run_forward: 模型前向函数。
        after_backward: 反向传播后的附加处理。

    Returns:
        当前轮训练损失均值。
    """

    losses: list[float] = []
    total_steps: int = total_epochs * len(state.train_samples)
    with build_progress(
        state.train_samples,
        desc=f"训练 epoch {epoch}/{total_epochs}",
        leave=False,
    ) as progress:
        sample_index: int
        sample: SampleT
        for sample_index, sample in enumerate(progress, start=1):
            cuda_sample: CudaSampleT = to_cuda(sample)
            state.optimizer.zero_grad(set_to_none=True)
            result: ResultT = run_forward(state.model, cuda_sample)
            result.loss.backward()
            if after_backward is not None:
                # 目的：把模型特定的梯度清理逻辑外提成可选钩子，避免每个训练入口重复拼装相同骨架。
                after_backward(state.model)
            state.optimizer.step()

            loss: float = float(result.loss.item())
            global_step: int = (epoch - 1) * len(state.train_samples) + sample_index
            losses.append(loss)
            progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")
    return sum(losses) / len(losses)


def select_split_samples[SampleT](
    dev_samples: list[SampleT],
    test_samples: list[SampleT],
    split: str,
) -> list[SampleT]:
    """按切分名选择样本列表。"""

    if split == "dev":
        return dev_samples
    return test_samples


def evaluate_pointwise_model[
    ModelT: TrainEvalModel,
    SampleT,
    CudaSampleT,
    ResultT: LossResult,
    MetricsT,
    AggregateMetricsT,
](
    model: ModelT,
    samples: list[SampleT],
    to_cuda: Callable[[SampleT], CudaSampleT],
    run_forward: Callable[[ModelT, CudaSampleT], ResultT],
    build_metrics: Callable[[SampleT, ResultT], MetricsT],
    merge_metrics: Callable[[list[float], list[MetricsT]], AggregateMetricsT],
    empty_error_message: str,
) -> AggregateMetricsT:
    """执行点式精排评测。

    Args:
        model: 待评测模型。
        samples: 待评测样本。
        to_cuda: 样本上卡函数。
        run_forward: 模型前向函数。
        build_metrics: 单样本指标构造函数。
        merge_metrics: 聚合损失和指标的函数。
        empty_error_message: 空评测结果时的报错信息。

    Returns:
        聚合后的评测结果。
    """

    model.eval()
    losses: list[float] = []
    metrics_list: list[MetricsT] = []

    with torch.no_grad():
        with build_progress(samples, desc="验证", leave=False) as progress:
            sample: SampleT
            for sample in progress:
                cuda_sample: CudaSampleT = to_cuda(sample)
                result: ResultT = run_forward(model, cuda_sample)
                losses.append(float(result.loss.item()))
                metrics_list.append(build_metrics(sample, result))

    model.train()
    if not metrics_list:
        raise ValueError(empty_error_message)
    return merge_metrics(losses, metrics_list)


def capture_state_dict_snapshot(model: StateDictModel) -> PointwiseSnapshot:
    """捕获当前模型权重快照。"""

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_state_dict_snapshot(
    model: StateDictModel,
    snapshot: PointwiseSnapshot,
) -> None:
    """恢复模型权重快照。"""

    model.load_state_dict(snapshot)


def export_state_dict_checkpoint(
    model: StateDictModel,
    output_path: Path,
) -> None:
    """导出模型 `state_dict` checkpoint。"""

    torch.save(model.state_dict(), output_path)

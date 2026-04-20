"""GAT 精排模型训练流程。"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...tracedr import load_tracedr_samples
from ..experiment.runner import ExperimentAdapter, run_training_experiment
from ..experiment.schema import ComparableMetrics, ExperimentEvalResult
from ..pointwise_training import (
    PointwiseSnapshot,
    PointwiseTrainState,
    capture_state_dict_snapshot,
    evaluate_pointwise_model,
    export_state_dict_checkpoint,
    load_pointwise_splits,
    restore_state_dict_snapshot,
    select_split_samples,
    train_pointwise_epoch,
)
from ..tracedr.metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .data import build_gat_model_sample
from .model import GAT, GATForwardResult
from .schema import GATEntity, GATModelSample

type GATSnapshot = PointwiseSnapshot
type GATTrainState = PointwiseTrainState[GAT, GATModelSample]


@dataclass(slots=True)
class TrainConfig:
    """GAT 训练配置。"""

    train_input: Path
    dev_input: Path
    output_name: str
    epochs: int
    encoder_model_name: str
    test_input: Path | None = None
    train_limit: int | None = None
    dev_limit: int | None = None
    test_limit: int | None = None
    selection_metric: str = "mrr"


@dataclass(slots=True)
class RankedAnswer:
    """排序后的候选药物。"""

    id: str
    label: str
    score: float
    rank: int


def load_samples(
    input_path: Path,
    limit: int | None,
    train: bool = True,
) -> list[GATModelSample]:
    """加载并构造 GAT 样本。

    Args:
        input_path: TraceDR 风格候选集路径。
        limit: 样本数量上限。
        train: 是否按训练模式构造样本。

    Returns:
        可直接送入 GAT 的样本列表。
    """

    raw_samples = load_tracedr_samples(input_path, limit=limit)
    model_samples: list[GATModelSample] = []
    for sample in raw_samples:
        model_sample: GATModelSample | None = build_gat_model_sample(sample, train=train)
        if model_sample is None:
            continue
        entity_count: float = float(model_sample.entity_mask.sum().item())
        evidence_count: float = float(model_sample.evidence_mask.sum().item())
        # 目的：过滤空图样本，避免注意力归一化阶段出现无意义的 NaN 损失。
        if entity_count == 0.0 or evidence_count == 0.0:
            continue
        model_samples.append(model_sample)
    return model_samples


def build_ranked_answers(
    sample: GATModelSample,
    result: GATForwardResult,
) -> list[RankedAnswer]:
    """把模型输出转成排序结果。

    Args:
        sample: 当前样本。
        result: 前向输出。

    Returns:
        排序后的候选药物列表。
    """

    entity_scores: Tensor = torch.sigmoid(result.entity_logits).detach().cpu()
    sorted_indices_tensor: Tensor = torch.argsort(entity_scores, descending=True)
    sorted_indices: list[int] = [int(index) for index in sorted_indices_tensor]
    ranked_answers: list[RankedAnswer] = []

    for entity_index in sorted_indices:
        entity_mask_value: float = float(sample.entity_mask[entity_index].item())
        if entity_mask_value == 0.0:
            continue

        entity: GATEntity = sample.entities[entity_index]
        drug_id: str = entity.instruction.drugid
        if drug_id == "":
            continue

        score: float = float(entity_scores[entity_index].item())
        ranked_answers.append(
            RankedAnswer(
                id=drug_id,
                label=entity.name,
                score=score,
                rank=len(ranked_answers) + 1,
            )
        )
    return ranked_answers


def evaluate_model(
    model: GAT,
    samples: list[GATModelSample],
) -> TraceDRMetrics:
    """执行验证集评估。

    Args:
        model: 待评估模型。
        samples: 验证样本。

    Returns:
        聚合后的验证指标。
    """

    return evaluate_pointwise_model(
        model=model,
        samples=samples,
        to_cuda=lambda sample: sample.to_cuda(),
        run_forward=lambda current_model, cuda_sample: current_model(cuda_sample),
        build_metrics=_build_sample_metrics,
        merge_metrics=_merge_eval_metrics,
        empty_error_message="验证阶段未产生任何指标，请检查 GAT 样本构造流程。",
    )


def _build_sample_metrics(
    sample: GATModelSample,
    result: GATForwardResult,
) -> TraceDRMetrics:
    """构造单个 GAT 样本的评测指标。"""

    ranked_answers: list[RankedAnswer] = build_ranked_answers(sample, result)
    return calculate_metrics(
        question_id=str(sample.question_id),
        answers=ranked_answers,
        gold_answers=sample.gold_answers,
        k=5,
        # 目的：GAT 的 DDI 口径对齐参考实现，只统计候选药物与当前在用药的相互作用风险。
        candidate_drug_map=sample.source_sample.top_k_drugs,
        on_medicines=sample.source_sample.people.on_medicine,
    )


def _merge_eval_metrics(
    losses: list[float],
    metrics_list: list[TraceDRMetrics],
) -> TraceDRMetrics:
    """聚合 GAT 验证阶段的损失与排序指标。"""

    aggregated_metrics: TraceDRMetrics = aggregate_metrics(metrics_list)
    return TraceDRMetrics(
        loss=sum(losses) / len(losses) if losses else 0.0,
        p_at_1=aggregated_metrics.p_at_1,
        mrr=aggregated_metrics.mrr,
        h_at_5=aggregated_metrics.h_at_5,
        answer_presence=aggregated_metrics.answer_presence,
        ddi_rate=aggregated_metrics.ddi_rate,
        jaccard_similarity=aggregated_metrics.jaccard_similarity,
        precision_at_5=aggregated_metrics.precision_at_5,
        recall_at_5=aggregated_metrics.recall_at_5,
        f1_at_5=aggregated_metrics.f1_at_5,
    )


def build_eval_result(metrics: TraceDRMetrics) -> ExperimentEvalResult:
    """把 GAT 指标映射到统一评测结构。

    Args:
        metrics: GAT 原始指标。

    Returns:
        统一评测结果。
    """

    return ExperimentEvalResult(
        loss=metrics.loss,
        comparable_metrics=ComparableMetrics(
            p_at_1=metrics.p_at_1,
            mrr=metrics.mrr,
            hit_at_5=metrics.h_at_5,
            precision_at_5=metrics.precision_at_5,
            recall_at_5=metrics.recall_at_5,
            f1_at_5=metrics.f1_at_5,
        ),
        extra_metrics={
            "answer_presence": metrics.answer_presence,
            "ddi_rate": metrics.ddi_rate,
            "jaccard_similarity": metrics.jaccard_similarity,
        },
    )


def sanitize_gradients(model: GAT) -> None:
    """把反向传播中的非有限梯度归零。

    Args:
        model: 当前训练模型。
    """

    for parameter in model.parameters():
        gradient: Tensor | None = parameter.grad
        if gradient is None:
            continue
        if torch.isfinite(gradient).all():
            continue
        # 目的：当前 GAT 反向在小样本下会产出 NaN 梯度，这里直接清零避免污染参数。
        parameter.grad = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


class GATTrainAdapter(ExperimentAdapter[TrainConfig, GATTrainState, GATSnapshot]):
    """GAT 统一训练适配器。"""

    experiment_name: str = "gat"

    def setup(self, config: TrainConfig) -> GATTrainState:
        """构造 GAT 训练状态。"""

        train_samples: list[GATModelSample]
        dev_samples: list[GATModelSample]
        test_samples: list[GATModelSample]
        train_samples, dev_samples, test_samples = load_pointwise_splits(
            train_input=config.train_input,
            dev_input=config.dev_input,
            test_input=config.test_input,
            train_limit=config.train_limit,
            dev_limit=config.dev_limit,
            test_limit=config.test_limit,
            load_samples=load_samples,
            experiment_name="GAT",
        )

        model: GAT = GAT(encoder_model_name=config.encoder_model_name)
        model.train()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )
        return PointwiseTrainState(
            model=model,
            optimizer=optimizer,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )

    def train_epoch(self, state: GATTrainState, epoch: int, total_epochs: int) -> float:
        """执行单轮 GAT 训练。"""

        return train_pointwise_epoch(
            state=state,
            epoch=epoch,
            total_epochs=total_epochs,
            to_cuda=lambda sample: sample.to_cuda(),
            run_forward=lambda current_model, cuda_sample: current_model(cuda_sample),
            after_backward=sanitize_gradients,
        )

    def evaluate(self, state: GATTrainState, split: str) -> ExperimentEvalResult:
        """执行指定切分的 GAT 评测。"""

        samples: list[GATModelSample] = select_split_samples(
            state.dev_samples,
            state.test_samples,
            split,
        )
        return build_eval_result(evaluate_model(state.model, samples))

    def has_split(self, state: GATTrainState, split: str) -> bool:
        """判断指定切分是否存在。"""

        if split == "test":
            return bool(state.test_samples)
        return True

    def capture_snapshot(self, state: GATTrainState) -> GATSnapshot:
        """捕获当前最佳权重。"""

        return capture_state_dict_snapshot(state.model)

    def restore_snapshot(self, state: GATTrainState, snapshot: GATSnapshot) -> None:
        """恢复最佳权重。"""

        restore_state_dict_snapshot(state.model, snapshot)

    def export_checkpoint(self, state: GATTrainState, output_path: Path) -> None:
        """导出 GAT checkpoint。"""

        # 目的：统一导出最佳轮次权重，保证后续对比实验按同一准则复现。
        export_state_dict_checkpoint(state.model, output_path)


def train(config: TrainConfig) -> None:
    """执行 GAT 训练。"""

    run_training_experiment(config, GATTrainAdapter())

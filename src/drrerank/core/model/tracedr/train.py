"""TraceDR 精排模型训练流程。"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...schema import DrugRecMedicine, RankedEvidence
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
from .data import build_model_sample
from .metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .model import HeterogeneousGNN, TraceDRForwardResult
from .schema import (
    TraceDRAblationConfig,
    TraceDREntity,
    TraceDREvidenceTextMode,
    TraceDRModelSample,
)

type TraceDRSnapshot = PointwiseSnapshot
type TraceDRTrainState = PointwiseTrainState[HeterogeneousGNN, TraceDRModelSample]


@dataclass(slots=True)
class TrainConfig:
    """TraceDR 训练配置。"""

    train_input: Path
    dev_input: Path
    output_name: str
    epochs: int
    test_input: Path | None = None
    train_limit: int | None = None
    dev_limit: int | None = None
    test_limit: int | None = None
    selection_metric: str = "mrr"
    num_layers: int = 3
    use_evidence_supervision: bool = True
    evidence_text_mode: TraceDREvidenceTextMode = "full"
    include_on_medicine: bool = True


@dataclass(slots=True)
class RankedAnswer:
    """排序后的候选药物。"""

    id: str
    label: str
    score: float
    rank: int


def _build_candidate_drug_map(
    sample: TraceDRModelSample,
) -> dict[str, DrugRecMedicine]:
    on_medicine_ids: set[str] = {medicine.drugid for medicine in sample.on_medicine}
    candidate_drug_map: dict[str, DrugRecMedicine] = {}
    evidence: DrugRecMedicine | None
    for evidence in sample.id_to_evidence:
        if evidence is None or evidence.drugid in on_medicine_ids:
            continue
        candidate_drug_map[evidence.drugid] = evidence
    return candidate_drug_map


def load_samples(
    input_path: Path,
    limit: int | None,
    train: bool = True,
    ablation_config: TraceDRAblationConfig | None = None,
) -> list[TraceDRModelSample]:
    """加载并构造 TraceDR 样本。

    Args:
        input_path: 输入 `jsonl` 路径。
        limit: 样本数量上限。
        train: 是否按训练模式构造样本。
        ablation_config: 关键消融配置。

    Returns:
        可直接送入模型的样本列表。
    """

    raw_samples = load_tracedr_samples(input_path, limit=limit)
    return [
        model_sample
        for sample in raw_samples
        if (
            model_sample := build_model_sample(
                sample,
                train=train,
                ablation_config=ablation_config,
            )
        )
        is not None
    ]


def build_ranked_answers(
    sample: TraceDRModelSample,
    result: TraceDRForwardResult,
) -> list[RankedAnswer]:
    """把模型输出转换为排序药物列表。

    Args:
        sample: 当前样本。
        result: 当前前向结果。

    Returns:
        按分数降序排序的药物列表。
    """

    entity_scores: Tensor = torch.sigmoid(result.entity_logits).detach().cpu()
    sorted_indices_tensor: Tensor = torch.argsort(entity_scores, descending=True)
    sorted_indices: list[int] = [int(index) for index in sorted_indices_tensor]

    ranked_answers: list[RankedAnswer] = []
    for entity_index in sorted_indices:
        if sample.entity_mask[entity_index].item() == 0:
            continue

        entity: TraceDREntity | None = sample.id_to_entity[entity_index]
        if entity is None:
            continue
        if entity.type != "药品":
            continue
        if entity.id == "":
            continue

        ranked_answers.append(
            RankedAnswer(
                id=str(entity.id),
                label=entity.label,
                score=float(entity_scores[entity_index].item()),
                rank=len(ranked_answers) + 1,
            )
        )
    return ranked_answers


def build_ranked_evidences(
    sample: TraceDRModelSample,
    result: TraceDRForwardResult,
) -> list[RankedEvidence]:
    """把模型输出转换为排序证据列表。

    Args:
        sample: 当前样本。
        result: 当前前向结果。

    Returns:
        按分数降序排序的证据列表。
    """

    evidence_scores: Tensor = torch.sigmoid(result.evidence_logits).detach().cpu()
    sorted_indices_tensor: Tensor = torch.argsort(evidence_scores, descending=True)
    sorted_indices: list[int] = [int(index) for index in sorted_indices_tensor]

    ranked_evidences: list[RankedEvidence] = []
    for evidence_index in sorted_indices:
        if sample.evidence_mask[evidence_index].item() == 0:
            continue

        evidence: DrugRecMedicine | None = sample.id_to_evidence[evidence_index]
        if evidence is None:
            continue

        evidence_text: str = sample.evidences[evidence_index].evidence_text
        ranked_evidences.append(
            RankedEvidence(
                evidence_id=f"retrieval::{evidence.drugid}",
                score=float(evidence_scores[evidence_index].item()),
                rank=len(ranked_evidences) + 1,
                text=evidence_text,
                label=int(sample.evidence_labels[evidence_index].item()),
            )
        )
    return ranked_evidences


def evaluate_model(
    model: HeterogeneousGNN,
    samples: list[TraceDRModelSample],
) -> TraceDRMetrics:
    """执行 TraceDR 评测。

    Args:
        model: 待评测模型。
        samples: 待评测样本。

    Returns:
        聚合后的 TraceDR 指标。
    """

    return evaluate_pointwise_model(
        model=model,
        samples=samples,
        to_cuda=lambda sample: sample.to_cuda(),
        run_forward=lambda current_model, cuda_sample: current_model(cuda_sample),
        build_metrics=_build_sample_metrics,
        merge_metrics=_merge_eval_metrics,
        empty_error_message="验证阶段未产生任何指标，请检查 samples 或评估流程。",
    )


def _build_sample_metrics(
    sample: TraceDRModelSample,
    result: TraceDRForwardResult,
) -> TraceDRMetrics:
    """构造单个 TraceDR 样本的评测指标。"""

    ranked_answers: list[RankedAnswer] = build_ranked_answers(sample, result)
    return calculate_metrics(
        question_id=str(sample.question_id),
        answers=ranked_answers,
        gold_answers=sample.gold_answers,
        k=5,
        candidate_drug_map=_build_candidate_drug_map(sample),
        on_medicines=sample.on_medicine,
    )


def _merge_eval_metrics(
    losses: list[float],
    metrics_list: list[TraceDRMetrics],
) -> TraceDRMetrics:
    """聚合 TraceDR 验证阶段的损失与排序指标。"""

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
    """把 TraceDR 指标映射到统一评测结构。

    Args:
        metrics: TraceDR 原始指标。

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


class TraceDRTrainAdapter(ExperimentAdapter[TrainConfig, TraceDRTrainState, TraceDRSnapshot]):
    """TraceDR 统一训练适配器。"""

    experiment_name: str = "tracedr"

    def setup(self, config: TrainConfig) -> TraceDRTrainState:
        """构造 TraceDR 训练状态。"""

        # 目的：把关键消融参数固定在 setup 阶段，确保训练/验证/测试共用同一口径。
        ablation_config: TraceDRAblationConfig = TraceDRAblationConfig(
            num_layers=config.num_layers,
            use_evidence_supervision=config.use_evidence_supervision,
            evidence_text_mode=config.evidence_text_mode,
            include_on_medicine=config.include_on_medicine,
        )
        train_samples: list[TraceDRModelSample]
        dev_samples: list[TraceDRModelSample]
        test_samples: list[TraceDRModelSample]
        train_samples, dev_samples, test_samples = load_pointwise_splits(
            train_input=config.train_input,
            dev_input=config.dev_input,
            test_input=config.test_input,
            train_limit=config.train_limit,
            dev_limit=config.dev_limit,
            test_limit=config.test_limit,
            load_samples=lambda input_path, limit, train: load_samples(
                input_path,
                limit,
                train,
                ablation_config=ablation_config,
            ),
            experiment_name="TraceDR",
        )

        model: HeterogeneousGNN = HeterogeneousGNN(
            num_layers=config.num_layers,
            use_evidence_supervision=config.use_evidence_supervision,
        )
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

    def train_epoch(
        self,
        state: TraceDRTrainState,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """执行单轮 TraceDR 训练。"""

        return train_pointwise_epoch(
            state=state,
            epoch=epoch,
            total_epochs=total_epochs,
            to_cuda=lambda sample: sample.to_cuda(),
            run_forward=lambda current_model, cuda_sample: current_model(cuda_sample),
        )

    def evaluate(
        self,
        state: TraceDRTrainState,
        split: str,
    ) -> ExperimentEvalResult:
        """执行指定切分的 TraceDR 评测。"""

        samples: list[TraceDRModelSample] = select_split_samples(
            state.dev_samples,
            state.test_samples,
            split,
        )
        return build_eval_result(evaluate_model(state.model, samples))

    def has_split(
        self,
        state: TraceDRTrainState,
        split: str,
    ) -> bool:
        """判断指定切分是否存在。"""

        if split == "test":
            return bool(state.test_samples)
        return True

    def capture_snapshot(self, state: TraceDRTrainState) -> TraceDRSnapshot:
        """捕获当前最佳权重。"""

        return capture_state_dict_snapshot(state.model)

    def restore_snapshot(
        self,
        state: TraceDRTrainState,
        snapshot: TraceDRSnapshot,
    ) -> None:
        """恢复最佳权重。"""

        restore_state_dict_snapshot(state.model, snapshot)

    def export_checkpoint(self, state: TraceDRTrainState, output_path: Path) -> None:
        """导出 TraceDR checkpoint。"""

        # 目的：统一把最佳轮次权重落盘，便于不同模型按同一方式复现实验。
        export_state_dict_checkpoint(state.model, output_path)


def train(config: TrainConfig) -> None:
    """执行 TraceDR 训练。"""

    run_training_experiment(config, TraceDRTrainAdapter())

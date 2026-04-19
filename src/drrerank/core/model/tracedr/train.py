"""TraceDR 精排模型训练入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from ...schema import DrugRecMedicine
from ...setting import DEFAULT_TRACEDR_DEV_INPUT_PATH, DEFAULT_TRACEDR_TRAIN_INPUT_PATH
from ...tracedr import load_tracedr_samples
from ..experiment.runner import ExperimentAdapter, run_training_experiment
from ..experiment.schema import ComparableMetrics, ExperimentEvalResult
from .data import build_model_sample
from .metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .model import HeterogeneousGNN, TraceDRForwardResult
from .schema import TraceDREntity, TraceDRModelSample

type TraceDRSnapshot = dict[str, Tensor]


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


@dataclass(slots=True)
class RankedAnswer:
    """排序后的候选药物。"""

    id: str
    label: str
    score: float
    rank: int


@dataclass(slots=True)
class TraceDRTrainState:
    """TraceDR 统一 runner 需要的状态。"""

    model: HeterogeneousGNN
    optimizer: torch.optim.Optimizer
    train_samples: list[TraceDRModelSample]
    dev_samples: list[TraceDRModelSample]
    test_samples: list[TraceDRModelSample]


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
) -> list[TraceDRModelSample]:
    """加载并构造 TraceDR 样本。

    Args:
        input_path: 输入 `jsonl` 路径。
        limit: 样本数量上限。
        train: 是否按训练模式构造样本。

    Returns:
        可直接送入模型的样本列表。
    """

    raw_samples = load_tracedr_samples(input_path, limit=limit)
    return [
        model_sample
        for sample in raw_samples
        if (model_sample := build_model_sample(sample, train=train)) is not None
    ]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="训练 TraceDR rerank 模型。")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_TRACEDR_DEV_INPUT_PATH)
    parser.add_argument("--test-input", type=Path, default=None)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--selection-metric", type=str, default="mrr")
    return parser.parse_args()


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

    model.eval()
    losses: list[float] = []
    metrics_list: list[TraceDRMetrics] = []

    with torch.no_grad():
        with tqdm(samples, desc="验证", leave=False) as progress:
            for sample in progress:
                cuda_sample: TraceDRModelSample = sample.to_cuda()
                result: TraceDRForwardResult = model(cuda_sample)

                losses.append(float(result.loss.item()))

                ranked_answers: list[RankedAnswer] = build_ranked_answers(sample, result)
                metrics: TraceDRMetrics = calculate_metrics(
                    question_id=str(sample.question_id),
                    answers=ranked_answers,
                    gold_answers=sample.gold_answers,
                    k=5,
                    candidate_drug_map=_build_candidate_drug_map(sample),
                    on_medicines=sample.on_medicine,
                )
                metrics_list.append(metrics)

    model.train()

    if not metrics_list:
        raise ValueError("验证阶段未产生任何指标，请检查 samples 或评估流程。")

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

        train_samples: list[TraceDRModelSample] = load_samples(
            config.train_input,
            config.train_limit,
            train=True,
        )
        dev_samples: list[TraceDRModelSample] = load_samples(
            config.dev_input,
            config.dev_limit,
            train=False,
        )
        test_samples: list[TraceDRModelSample] = []
        if config.test_input is not None:
            test_samples = load_samples(
                config.test_input,
                config.test_limit,
                train=False,
            )
        if not train_samples:
            raise ValueError("训练集为空，无法执行 TraceDR 训练。")
        if not dev_samples:
            raise ValueError("验证集为空，无法执行 TraceDR 评估。")

        model: HeterogeneousGNN = HeterogeneousGNN()
        model.train()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )
        return TraceDRTrainState(
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

        losses: list[float] = []
        total_steps: int = total_epochs * len(state.train_samples)
        with tqdm(
            state.train_samples,
            desc=f"训练 epoch {epoch}/{total_epochs}",
            leave=False,
        ) as progress:
            sample_idx: int
            sample: TraceDRModelSample
            for sample_idx, sample in enumerate(progress, start=1):
                cuda_sample: TraceDRModelSample = sample.to_cuda()
                state.optimizer.zero_grad(set_to_none=True)
                result: TraceDRForwardResult = state.model(cuda_sample)
                result.loss.backward()
                state.optimizer.step()
                loss: float = float(result.loss.item())
                losses.append(loss)
                global_step: int = (epoch - 1) * len(state.train_samples) + sample_idx
                progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")
        return sum(losses) / len(losses)

    def evaluate(
        self,
        state: TraceDRTrainState,
        split: str,
    ) -> ExperimentEvalResult:
        """执行指定切分的 TraceDR 评测。"""

        samples: list[TraceDRModelSample]
        if split == "dev":
            samples = state.dev_samples
        else:
            samples = state.test_samples
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

        return {
            key: value.detach().cpu().clone() for key, value in state.model.state_dict().items()
        }

    def restore_snapshot(
        self,
        state: TraceDRTrainState,
        snapshot: TraceDRSnapshot,
    ) -> None:
        """恢复最佳权重。"""

        state.model.load_state_dict(snapshot)

    def export_checkpoint(self, state: TraceDRTrainState, output_path: Path) -> None:
        """导出 TraceDR checkpoint。"""

        # 目的：统一把最佳轮次权重落盘，便于不同模型按同一方式复现实验。
        torch.save(state.model.state_dict(), output_path)


def train(config: TrainConfig) -> None:
    """执行 TraceDR 训练。"""

    run_training_experiment(config, TraceDRTrainAdapter())


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    train(
        TrainConfig(
            train_input=args.train_input,
            dev_input=args.dev_input,
            test_input=args.test_input,
            output_name=args.output_name,
            epochs=args.epochs,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
            test_limit=args.test_limit,
            selection_metric=args.selection_metric,
        )
    )


if __name__ == "__main__":
    main()

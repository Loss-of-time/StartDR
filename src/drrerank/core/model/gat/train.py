"""GAT 精排模型训练入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from ...schema import unstructure
from ...setting import (
    DEFAULT_GNN_DEV_INPUT_PATH,
    DEFAULT_GNN_TRAIN_INPUT_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
)
from ...tracedr import load_tracedr_samples
from ..tracedr.metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .data import build_gat_model_sample
from .model import GAT, GATForwardResult
from .schema import GATEntity, GATModelSample


@dataclass(slots=True)
class TrainConfig:
    """GAT 训练配置。

    Attributes:
        train_input: 训练集输入路径。
        dev_input: 验证集输入路径。
        output_name: 输出名称前缀。
        epochs: 训练轮数。
        encoder_model_name: 编码器模型名称。
        train_limit: 训练集样本上限。
        dev_limit: 验证集样本上限。
    """

    train_input: Path
    dev_input: Path
    output_name: str
    epochs: int
    encoder_model_name: str
    train_limit: int | None = None
    dev_limit: int | None = None


@dataclass(slots=True)
class RankedAnswer:
    """排序后的候选药物。"""

    id: str
    label: str
    score: float
    rank: int


@dataclass(slots=True)
class TrainEpochResult:
    """单轮训练报告。"""

    epoch: int
    train_loss: float
    dev_loss: float
    p_at_1: float
    mrr: float
    h_at_5: float
    answer_presence: float
    jaccard: float
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float


@dataclass(slots=True)
class TrainReport:
    """训练总报告。"""

    output_name: str
    epochs: list[TrainEpochResult]


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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="训练 GAT rerank 模型。")
    parser.add_argument(
        "--train-input",
        type=Path,
        default=DEFAULT_GNN_TRAIN_INPUT_PATH,
    )
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=DEFAULT_GNN_DEV_INPUT_PATH,
    )
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--encoder-model-name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    return parser.parse_args()


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

    entity_scores: torch.Tensor = torch.sigmoid(result.entity_logits).detach().cpu()
    sorted_indices_tensor: torch.Tensor = torch.argsort(entity_scores, descending=True)
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

    model.eval()
    losses: list[float] = []
    metrics_list: list[TraceDRMetrics] = []

    with torch.no_grad():
        with tqdm(samples, desc="验证", leave=False) as progress:
            for sample in progress:
                cuda_sample: GATModelSample = sample.to_cuda()
                result: GATForwardResult = model(cuda_sample)
                loss: float = float(result.loss.item())
                ranked_answers: list[RankedAnswer] = build_ranked_answers(sample, result)
                metrics: TraceDRMetrics = calculate_metrics(
                    question_id=str(sample.question_id),
                    answers=ranked_answers,
                    gold_answers=sample.gold_answers,
                    k=5,
                )
                losses.append(loss)
                metrics_list.append(metrics)

    model.train()

    if not metrics_list:
        raise ValueError("验证阶段未产生任何指标，请检查 GAT 样本构造流程。")

    aggregated_metrics: TraceDRMetrics = aggregate_metrics(metrics_list)
    return TraceDRMetrics(
        loss=sum(losses) / len(losses) if losses else 0.0,
        p_at_1=aggregated_metrics.p_at_1,
        mrr=aggregated_metrics.mrr,
        h_at_5=aggregated_metrics.h_at_5,
        answer_presence=aggregated_metrics.answer_presence,
        jaccard=aggregated_metrics.jaccard,
        precision_at_5=aggregated_metrics.precision_at_5,
        recall_at_5=aggregated_metrics.recall_at_5,
        f1_at_5=aggregated_metrics.f1_at_5,
    )


def sanitize_gradients(model: GAT) -> None:
    """把反向传播中的非有限梯度归零。

    Args:
        model: 当前训练模型。
    """

    for parameter in model.parameters():
        gradient: torch.Tensor | None = parameter.grad
        if gradient is None:
            continue
        if torch.isfinite(gradient).all():
            continue
        # 目的：当前 GAT 反向在小样本下会产出 NaN 梯度，这里直接清零避免污染参数。
        parameter.grad = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)


def train(config: TrainConfig) -> None:
    """执行 GAT 训练。

    Args:
        config: 训练配置。
    """

    train_samples: list[GATModelSample] = load_samples(
        config.train_input,
        config.train_limit,
        train=True,
    )
    dev_samples: list[GATModelSample] = load_samples(
        config.dev_input,
        config.dev_limit,
        train=False,
    )
    if not train_samples:
        raise ValueError("训练集为空，无法执行 GAT 训练。")
    if not dev_samples:
        raise ValueError("验证集为空，无法执行 GAT 评估。")

    model: GAT = GAT(encoder_model_name=config.encoder_model_name)
    model.train()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )

    DEFAULT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path: Path = DEFAULT_MODEL_OUTPUT_DIR / f"{config.output_name}.json"
    state_dict_path: Path = DEFAULT_MODEL_OUTPUT_DIR / f"{config.output_name}.pt"

    total_steps: int = config.epochs * len(train_samples)
    epoch_results: list[TrainEpochResult] = []

    for epoch in range(config.epochs):
        losses: list[float] = []
        with tqdm(
            train_samples,
            desc=f"训练 epoch {epoch + 1}/{config.epochs}",
            leave=False,
        ) as progress:
            for sample_idx, sample in enumerate(progress, start=1):
                cuda_sample: GATModelSample = sample.to_cuda()
                optimizer.zero_grad(set_to_none=True)
                result: GATForwardResult = model(cuda_sample)
                result.loss.backward()
                sanitize_gradients(model)
                optimizer.step()

                loss: float = float(result.loss.item())
                global_step: int = epoch * len(train_samples) + sample_idx
                losses.append(loss)
                progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")

        train_loss: float = sum(losses) / len(losses)
        dev_metrics: TraceDRMetrics = evaluate_model(model, dev_samples)
        epoch_results.append(
            TrainEpochResult(
                epoch=epoch + 1,
                train_loss=train_loss,
                dev_loss=dev_metrics.loss,
                p_at_1=dev_metrics.p_at_1,
                mrr=dev_metrics.mrr,
                h_at_5=dev_metrics.h_at_5,
                answer_presence=dev_metrics.answer_presence,
                jaccard=dev_metrics.jaccard,
                precision_at_5=dev_metrics.precision_at_5,
                recall_at_5=dev_metrics.recall_at_5,
                f1_at_5=dev_metrics.f1_at_5,
            )
        )
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f}")
        print(
            " ".join(
                [
                    f"epoch={epoch + 1}",
                    f"dev_loss={dev_metrics.loss:.6f}",
                    f"p_at_1={dev_metrics.p_at_1:.6f}",
                    f"mrr={dev_metrics.mrr:.6f}",
                    f"h_at_5={dev_metrics.h_at_5:.6f}",
                    f"precision_at_5={dev_metrics.precision_at_5:.6f}",
                    f"recall_at_5={dev_metrics.recall_at_5:.6f}",
                    f"f1_at_5={dev_metrics.f1_at_5:.6f}",
                    f"jaccard={dev_metrics.jaccard:.6f}",
                ]
            )
        )

    report: TrainReport = TrainReport(
        output_name=config.output_name,
        epochs=epoch_results,
    )
    # 目的：训练入口同时落盘指标和权重，便于后续直接复现烟测结果。
    torch.save(model.state_dict(), state_dict_path)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"训练完成，模型已写入: {state_dict_path.resolve()}")
    print(f"训练完成，报告已写入: {report_path.resolve()}")


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    train(
        TrainConfig(
            train_input=args.train_input,
            dev_input=args.dev_input,
            output_name=args.output_name,
            epochs=args.epochs,
            encoder_model_name=args.encoder_model_name,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
        )
    )


if __name__ == "__main__":
    main()

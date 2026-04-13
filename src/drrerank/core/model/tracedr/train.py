import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from ...schema import unstructure
from ...setting import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_TRACEDR_DEV_INPUT_PATH,
    DEFAULT_TRACEDR_TRAIN_INPUT_PATH,
)
from ...tracedr import load_tracedr_samples
from .data import build_model_sample
from .metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .model import HeterogeneousGNN, TraceDRForwardResult
from .schema import TraceDREntity, TraceDRModelSample


@dataclass(slots=True)
class TrainConfig:
    train_input: Path
    dev_input: Path
    output_name: str
    epochs: int
    train_limit: int | None = None
    dev_limit: int | None = None


@dataclass(slots=True)
class RankedAnswer:
    id: str
    label: str
    score: float
    rank: int


@dataclass(slots=True)
class TrainEpochResult:
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
    output_name: str
    epochs: list[TrainEpochResult]


def load_samples(
    input_path: Path,
    limit: int | None,
    train: bool = True,
) -> list[TraceDRModelSample]:
    raw_samples = load_tracedr_samples(input_path, limit=limit)
    return [
        model_sample
        for sample in raw_samples
        if (model_sample := build_model_sample(sample, train=train)) is not None
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TraceDR rerank 模型。")
    parser.add_argument(
        "--train-input",
        type=Path,
        default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH,
    )
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=DEFAULT_TRACEDR_DEV_INPUT_PATH,
    )
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    return parser.parse_args()


def build_ranked_answers(
    sample: TraceDRModelSample,
    result: TraceDRForwardResult,
) -> list[RankedAnswer]:
    entity_scores = torch.sigmoid(result.entity_logits).detach().cpu()
    sorted_indices_tensor = torch.argsort(entity_scores, descending=True)
    sorted_indices = [int(index) for index in sorted_indices_tensor]

    ranked_answers: list[RankedAnswer] = []
    for entity_index in sorted_indices:
        if sample.entity_mask[entity_index].item() == 0:
            continue

        entity = sample.id_to_entity[entity_index]
        if entity is None:
            continue
        if not isinstance(entity, TraceDREntity):
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
    model.eval()
    losses: list[float] = []
    metrics_list: list[TraceDRMetrics] = []

    with torch.no_grad():
        with tqdm(samples, desc="验证", leave=False) as progress:
            for sample in progress:
                cuda_sample = sample.to_cuda()
                result = model(cuda_sample)

                losses.append(float(result.loss.item()))

                ranked_answers = build_ranked_answers(sample, result)
                metrics = calculate_metrics(
                    question_id=str(sample.question_id),
                    answers=ranked_answers,
                    gold_answers=sample.gold_answers,
                    k=5,
                )
                metrics_list.append(metrics)

    model.train()

    if not metrics_list:
        raise ValueError("验证阶段未产生任何指标，请检查 samples 或评估流程。")

    aggregated_metrics = aggregate_metrics(metrics_list)
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


def train(config: TrainConfig) -> None:
    train_samples = load_samples(
        config.train_input,
        config.train_limit,
        train=True,
    )
    dev_samples = load_samples(
        config.dev_input,
        config.dev_limit,
        train=False,
    )
    if not train_samples:
        raise ValueError("训练集为空，无法执行训练。")
    if not dev_samples:
        raise ValueError("验证集为空，无法执行评估。")

    model: HeterogeneousGNN = HeterogeneousGNN()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )
    DEFAULT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_MODEL_OUTPUT_DIR / f"{config.output_name}.json"

    total_steps = config.epochs * len(train_samples)
    epoch_results: list[TrainEpochResult] = []

    for epoch in range(config.epochs):
        losses: list[float] = []
        with tqdm(
            train_samples,
            desc=f"训练 epoch {epoch + 1}/{config.epochs}",
            leave=False,
        ) as progress:
            for sample_idx, sample in enumerate(progress, start=1):
                sample = sample.to_cuda()
                optimizer.zero_grad(set_to_none=True)
                result: TraceDRForwardResult = model(sample)
                result.loss.backward()
                optimizer.step()
                loss = float(result.loss.item())
                losses.append(loss)
                global_step = epoch * len(train_samples) + sample_idx
                progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")

        train_loss = sum(losses) / len(losses)
        dev_metrics = evaluate_model(model, dev_samples)
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

    report = TrainReport(
        output_name=config.output_name,
        epochs=epoch_results,
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"训练完成，报告已写入: {report_path.resolve()}")


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            train_input=args.train_input,
            dev_input=args.dev_input,
            output_name=args.output_name,
            epochs=args.epochs,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
        )
    )


if __name__ == "__main__":
    main()

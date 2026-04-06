import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .core.gnn import GNNModel
from .core.metrics import aggregate_gnn_metrics, get_gnn_metrics
from .core.pipeline import load_train_samples
from .core.schema import (
    GNNMetrics,
    GNNRecResult,
    GNNTrainSample,
    RankedDrug,
    RankedEvidence,
    unstructure,
)
from .core.setting import (
    DEFAULT_DEV_INPUT_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_TRAIN_INPUT_PATH,
)


@dataclass(slots=True)
class TrainEpochResult:
    epoch: int
    train_loss: float
    hit: float | None = None
    mrr: float | None = None
    precision_at_5: float | None = None
    recall_at_5: float | None = None
    f1_at_5: float | None = None
    jaccard_at_5: float | None = None
    evidence_mrr: float | None = None
    evidence_hit_at_5: float | None = None


@dataclass(slots=True)
class TrainReport:
    best_epoch: int
    best_metric_value: float
    checkpoint_path: str
    epochs: list[TrainEpochResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TraceDR 风格主模型。")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT_PATH)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
    )
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def iter_batches[T](items: list[T], batch_size: int) -> list[list[T]]:
    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def build_epoch_result(
    epoch: int,
    train_loss: float,
    dev_metrics: GNNMetrics,
) -> TrainEpochResult:
    return TrainEpochResult(
        epoch=epoch,
        train_loss=train_loss,
        hit=dev_metrics.hit,
        mrr=dev_metrics.mrr,
        precision_at_5=dev_metrics.precision_at_5,
        recall_at_5=dev_metrics.recall_at_5,
        f1_at_5=dev_metrics.f1_at_5,
        jaccard_at_5=dev_metrics.jaccard_at_5,
        evidence_mrr=dev_metrics.evidence_mrr,
        evidence_hit_at_5=dev_metrics.evidence_hit_at_5,
    )


def evaluate_model(
    model: GNNModel,
    samples: list[GNNTrainSample],
    batch_size: int,
) -> GNNMetrics:
    metrics_list: list[GNNMetrics] = []
    with torch.no_grad():
        for batch_samples in tqdm(
            iter_batches(samples, batch_size),
            desc="验证",
            leave=False,
        ):
            batch_inputs = [sample.model_input for sample in batch_samples]
            outputs = model(batch_inputs)
            for sample, (entity_logits, evidence_logits) in zip(
                batch_samples,
                outputs,
                strict=True,
            ):
                entity_scores = torch.sigmoid(entity_logits).detach().cpu().tolist()
                evidence_scores = torch.sigmoid(
                    evidence_logits
                ).detach().cpu().tolist()
                ranked_drugs = [
                    RankedDrug(
                        drugid=candidate.drugid,
                        score=float(entity_scores[entity_index]),
                        rank=0,
                        drug=candidate.drug,
                        retrieval_score=candidate.score,
                        retrieval_rank=candidate.rank,
                        label=1 if candidate.is_gold else 0,
                    )
                    for candidate, entity_index in zip(
                        sample.case.candidate_drugs,
                        sample.model_input.candidate_entity_indices,
                        strict=True,
                    )
                ]
                ranked_drugs.sort(key=lambda item: item.score, reverse=True)
                for rank, ranked_drug in enumerate(ranked_drugs, start=1):
                    ranked_drug.rank = rank
                ranked_evidences = [
                    RankedEvidence(
                        evidence_id=evidence.evidence_id,
                        score=float(score),
                        rank=0,
                        text=evidence.text,
                        label=evidence.label,
                    )
                    for evidence, score in zip(
                        sample.model_input.evidences,
                        evidence_scores,
                        strict=True,
                    )
                ]
                ranked_evidences.sort(key=lambda item: item.score, reverse=True)
                for rank, ranked_evidence in enumerate(ranked_evidences, start=1):
                    ranked_evidence.rank = rank
                result = GNNRecResult(
                    patient_id=sample.case.patient_id,
                    split=sample.case.split,
                    ranked_drugs=ranked_drugs,
                    ranked_evidences=ranked_evidences,
                )
                metrics_list.append(get_gnn_metrics(sample.case, result))
    return aggregate_gnn_metrics(metrics_list)


def main() -> None:
    args = parse_args()
    print(f"开始读取训练样本: {args.train_input.resolve()}")
    train_samples = load_train_samples(args.train_input, args.train_limit)
    print(f"训练样本数: {len(train_samples)}")
    print(f"开始读取验证样本: {args.dev_input.resolve()}")
    dev_samples = load_train_samples(args.dev_input, args.dev_limit)
    print(f"验证样本数: {len(dev_samples)}")
    if not train_samples:
        raise ValueError("训练集为空，无法执行训练。")
    if not dev_samples:
        raise ValueError("验证集为空，无法选择最佳 checkpoint。")

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu") if args.device == "auto"
        else torch.device(args.device)
    )
    random_state = random.Random(args.seed)
    model = GNNModel(
        hidden_size=args.hidden_size,
        encoder_model_name=args.encoder_model_name,
        max_text_length=args.max_text_length,
        message_passing_steps=args.message_passing_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    DEFAULT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = DEFAULT_MODEL_OUTPUT_DIR / f"{args.output_name}.pt"
    report_path = DEFAULT_MODEL_OUTPUT_DIR / f"{args.output_name}.json"
    epoch_results: list[TrainEpochResult] = []
    best_epoch = 1
    best_metric_value = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_samples = list(train_samples)
        random_state.shuffle(epoch_samples)
        batch_losses: list[float] = []
        for batch_samples in tqdm(
            iter_batches(epoch_samples, args.batch_size),
            desc=f"训练 epoch {epoch}",
        ):
            model.train()
            optimizer.zero_grad()
            batch_inputs = [sample.model_input for sample in batch_samples]
            outputs = model(batch_inputs)
            loss = model.get_loss(batch_inputs, outputs)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        train_loss = sum(batch_losses) / len(batch_losses)
        model.eval()
        dev_metrics = evaluate_model(model, dev_samples, args.batch_size)
        dev_mrr = dev_metrics.mrr or 0.0
        epoch_results.append(build_epoch_result(epoch, train_loss, dev_metrics))
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"selection_metric=mrr "
            f"selection_value={dev_mrr:.4f}"
        )
        if dev_mrr > best_metric_value:
            best_epoch = epoch
            best_metric_value = dev_mrr
            torch.save(model.build_checkpoint(), checkpoint_path)
            print(f"已更新最佳 checkpoint: {checkpoint_path.resolve()}")

    report = TrainReport(
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        checkpoint_path=str(checkpoint_path.resolve()),
        epochs=epoch_results,
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"训练完成，报告已写入: {report_path.resolve()}")


if __name__ == "__main__":
    main()

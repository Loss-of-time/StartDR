import argparse
import json
import logging
import random
from pathlib import Path
from typing import TypedDict, cast

import torch
from rich.progress import Progress

from .metrics.gnn import aggregate_gnn_metrics, get_gnn_metrics
from .model.gnn.intermediate import load_train_samples
from .model.gnn.model import GNNModel
from .schema.drugrec_task import (
    GNNMetrics,
    GNNRecResult,
    GNNTrainSample,
    RankedDrug,
    RankedEvidence,
)
from .utils.log import get_console, setup_logging
from .utils.paths import OUTPUT_DIR, RESOURCE_DIR

DEFAULT_TRAIN_INPUT = (
    RESOURCE_DIR / "gnn_data" / "pyserini_bm25_top50" / "train"
)
DEFAULT_DEV_INPUT = (
    RESOURCE_DIR / "gnn_data" / "pyserini_bm25_top50" / "dev"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "model"
LOGGER = logging.getLogger(__name__)


class TrainEpochResult(TypedDict, total=False):
    epoch: int
    train_loss: float
    hit: float
    mrr: float
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float
    jaccard_at_5: float
    evidence_mrr: float
    evidence_hit_at_5: float


class TrainReport(TypedDict):
    best_epoch: int
    best_metric_value: float
    checkpoint_path: str
    epochs: list[TrainEpochResult]


def parse_args() -> argparse.Namespace:
    """解析当前主模型训练参数。"""
    parser = argparse.ArgumentParser(
        description="训练 TraceDR 风格主模型。",
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
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
    """按固定 batch size 切分列表。"""
    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def evaluate_model(
    model: GNNModel,
    samples: list[GNNTrainSample],
    batch_size: int,
) -> GNNMetrics:
    """执行一轮开发集评测并聚合指标。"""
    metrics_list: list[GNNMetrics] = []
    with torch.no_grad():
        for batch_samples in iter_batches(samples, batch_size):
            batch_inputs = [sample["model_input"] for sample in batch_samples]
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
                ranked_drugs: list[RankedDrug] = [
                    {
                        "drugid": candidate["drugid"],
                        "score": float(entity_scores[entity_index]),
                        "rank": 0,
                        "drug": candidate["drug"],
                        "retrieval_score": candidate["score"],
                        "retrieval_rank": candidate["rank"],
                        "label": 1 if candidate["is_gold"] else 0,
                    }
                    for candidate, entity_index in zip(
                        sample["case"]["candidate_drugs"],
                        sample["model_input"]["candidate_entity_indices"],
                        strict=True,
                    )
                ]
                ranked_drugs.sort(key=lambda item: item["score"], reverse=True)
                for rank, ranked_drug in enumerate(ranked_drugs, start=1):
                    ranked_drug["rank"] = rank
                ranked_evidences: list[RankedEvidence] = [
                    {
                        "evidence_id": evidence["evidence_id"],
                        "score": float(score),
                        "rank": 0,
                        "text": evidence["text"],
                        "label": evidence["label"],
                    }
                    for evidence, score in zip(
                        sample["model_input"]["evidences"],
                        evidence_scores,
                        strict=True,
                    )
                ]
                ranked_evidences.sort(
                    key=lambda item: item["score"],
                    reverse=True,
                )
                for rank, ranked_evidence in enumerate(
                    ranked_evidences,
                    start=1,
                ):
                    ranked_evidence["rank"] = rank
                result: GNNRecResult = {
                    "patient_id": sample["case"]["patient_id"],
                    "split": sample["case"]["split"],
                    "ranked_drugs": ranked_drugs,
                    "ranked_evidences": ranked_evidences,
                }
                metrics_list.append(
                    get_gnn_metrics(sample["case"], result)
                )
    return cast(GNNMetrics, aggregate_gnn_metrics(metrics_list))


def main() -> None:
    """执行主模型训练。"""
    args = parse_args()
    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取训练中间目录: %s", args.train_input.resolve())
    train_samples = load_train_samples(args.train_input, args.train_limit)
    LOGGER.info("训练样本数: %s", len(train_samples))
    LOGGER.info("开始读取验证中间目录: %s", args.dev_input.resolve())
    dev_samples = load_train_samples(args.dev_input, args.dev_limit)
    LOGGER.info("验证样本数: %s", len(dev_samples))
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = DEFAULT_OUTPUT_DIR / f"{args.output_name}.pt"
    report_path = DEFAULT_OUTPUT_DIR / f"{args.output_name}.json"
    epoch_results: list[TrainEpochResult] = []
    best_epoch = 1
    best_metric_value = -1.0

    with Progress(console=get_console()) as progress:
        task_id = progress.add_task("训练主模型", total=args.epochs)
        for epoch in range(1, args.epochs + 1):
            epoch_samples = list(train_samples)
            random_state.shuffle(epoch_samples)
            batch_losses: list[float] = []
            for batch_samples in iter_batches(epoch_samples, args.batch_size):
                model.train()
                optimizer.zero_grad()
                batch_inputs = [sample["model_input"] for sample in batch_samples]
                outputs = model(batch_inputs)
                loss = model.get_loss(batch_inputs, outputs)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().item()))
            train_loss = sum(batch_losses) / len(batch_losses)
            model.eval()
            dev_metrics = evaluate_model(
                model,
                dev_samples,
                args.batch_size,
            )
            dev_mrr = dev_metrics.get("mrr", 0.0)
            epoch_result: TrainEpochResult = {
                "epoch": epoch,
                "train_loss": train_loss,
                **dev_metrics,
            }
            epoch_results.append(epoch_result)
            LOGGER.info(
                "epoch=%s train_loss=%.6f selection_metric=mrr selection_value=%.4f",
                epoch,
                train_loss,
                dev_mrr,
            )
            if dev_mrr > best_metric_value:
                best_epoch = epoch
                best_metric_value = dev_mrr
                torch.save(model.build_checkpoint(), checkpoint_path)
                LOGGER.info("已更新最佳 checkpoint: %s", checkpoint_path.resolve())
            progress.advance(task_id)

    report: TrainReport = {
        "best_epoch": best_epoch,
        "best_metric_value": best_metric_value,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "epochs": epoch_results,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    LOGGER.info("训练完成，报告已写入: %s", report_path.resolve())


if __name__ == "__main__":
    main()

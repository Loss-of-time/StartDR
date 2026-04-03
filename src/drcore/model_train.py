import argparse
import json
import logging
import random
from pathlib import Path
from typing import TypedDict, cast

import torch
from rich.progress import Progress

from .data.jsonl import load_jsonl
from .metrics.drugrec import aggregate_drugrec_metrics, get_drugrec_metrics
from .metrics.gnn_drugrec import aggregate_gnn_metrics, get_gnn_metrics
from .model import (
    DrugRecCheckpoint,
    DrugRecModel,
    build_model,
    get_model_names,
)
from .schema.drugrec_task import (
    DrugRecCase,
    DrugRecMetrics,
    DrugRecModelName,
    DrugRecResult,
    GNNRecResult,
)
from .utils.log import get_console, setup_logging
from .utils.paths import OUTPUT_DIR, RESOURCE_DIR

DEFAULT_TRAIN_INPUT = (
    RESOURCE_DIR / "patient_candidate" / "pyserini_bm25_top50" / "train.jsonl"
)
DEFAULT_DEV_INPUT = (
    RESOURCE_DIR / "patient_candidate" / "pyserini_bm25_top50" / "dev.jsonl"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "model"
DEFAULT_TOP_K = 50
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
    ddi_rate_at_5: float
    evidence_mrr: float
    evidence_hit_at_5: float


class TrainReportConfig(TypedDict):
    model: DrugRecModelName
    train_input: str
    dev_input: str
    train_case_count: int
    dev_case_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    top_k: int
    seed: int
    device: str
    selection_metric: str


class TrainReport(TypedDict):
    config: TrainReportConfig
    best_epoch: int
    best_metric_value: float
    checkpoint_path: str
    epochs: list[TrainEpochResult]


def parse_args() -> argparse.Namespace:
    """解析推荐模型训练命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练药品推荐模型。",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=get_model_names(),
        default="gnn",
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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


def iter_batches(
    cases: list[DrugRecCase],
    batch_size: int,
) -> list[list[DrugRecCase]]:
    """按固定 batch size 切分病例列表。"""
    return [
        cases[index:index + batch_size]
        for index in range(0, len(cases), batch_size)
    ]


def evaluate_model(
    model: DrugRecModel,
    cases: list[DrugRecCase],
    batch_size: int,
) -> tuple[DrugRecMetrics, list[DrugRecResult]]:
    """执行一轮开发集评测并聚合正式指标。"""
    results: list[DrugRecResult] = []
    with torch.no_grad():
        for batch_cases in iter_batches(cases, batch_size):
            output = model.eval_step(batch_cases)
            results.extend(output["results"])
    base_metrics = aggregate_drugrec_metrics(
        [
            get_drugrec_metrics(case, result)
            for case, result in zip(cases, results, strict=True)
        ]
    )
    if model.result_kind != "gnn":
        return base_metrics, results
    gnn_metrics = aggregate_gnn_metrics(
        [
            get_gnn_metrics(case, cast(GNNRecResult, result))
            for case, result in zip(cases, results, strict=True)
        ]
    )
    return {
        **base_metrics,
        **gnn_metrics,
    }, results


def main() -> None:
    """执行推荐模型训练主流程。"""
    args = parse_args()
    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取训练集: %s", args.train_input.resolve())
    train_cases = load_jsonl(
        path=args.train_input,
        parse_line=lambda row: cast(DrugRecCase, row),
        limit=args.train_limit,
    )
    LOGGER.info("训练病例数: %s", len(train_cases))
    LOGGER.info("开始读取验证集: %s", args.dev_input.resolve())
    dev_cases = load_jsonl(
        path=args.dev_input,
        parse_line=lambda row: cast(DrugRecCase, row),
        limit=args.dev_limit,
    )
    LOGGER.info("验证病例数: %s", len(dev_cases))
    if not train_cases:
        raise ValueError("训练集为空，无法执行训练。")
    if not dev_cases:
        raise ValueError("验证集为空，无法选择最佳 checkpoint。")

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu") if args.device == "auto"
        else torch.device(args.device)
    )

    random_state = random.Random(args.seed)

    model_name: DrugRecModelName = cast(DrugRecModelName, args.model)
    model = build_model(
        name=model_name,
        train_cases=train_cases,
        top_k=args.top_k,
    ).to(device)
    optimizer = torch.optim.Adam(
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
        task_id = progress.add_task("训练模型", total=args.epochs)
        for epoch in range(1, args.epochs + 1):
            epoch_cases = list(train_cases)
            random_state.shuffle(epoch_cases)
            batch_losses: list[float] = []

            for batch_cases in iter_batches(epoch_cases, args.batch_size):
                optimizer.zero_grad()
                output = model.train_step(batch_cases)
                output["loss"].backward()
                optimizer.step()
                batch_losses.append(output["loss_value"])

            train_loss = sum(batch_losses) / len(batch_losses)
            model.eval()
            dev_metrics, _ = evaluate_model(model, dev_cases, args.batch_size)
            epoch_result = cast(
                TrainEpochResult,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **dev_metrics,
                },
            )
            epoch_results.append(epoch_result)
            LOGGER.info(
                "epoch=%s train_loss=%.6f selection_metric=%s selection_value=%.4f",
                epoch,
                train_loss,
                model.selection_metric,
                dev_metrics[model.selection_metric],
            )

            if dev_metrics[model.selection_metric] > best_metric_value:
                best_epoch = epoch
                best_metric_value = dev_metrics[model.selection_metric]
                checkpoint: DrugRecCheckpoint = model.build_checkpoint()
                torch.save(checkpoint, checkpoint_path)
                LOGGER.info("已更新最佳 checkpoint: %s", checkpoint_path.resolve())
            progress.advance(task_id)

    report: TrainReport = {
        "config": {
            "model": model_name,
            "train_input": str(args.train_input.resolve()),
            "dev_input": str(args.dev_input.resolve()),
            "train_case_count": len(train_cases),
            "dev_case_count": len(dev_cases),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": str(device),
            "selection_metric": model.selection_metric,
        },
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

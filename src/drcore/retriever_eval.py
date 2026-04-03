import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from rich.progress import Progress

from .data.jsonl import load_jsonl
from .metrics.retrieval import (
    aggregate_metrics,
    get_gold_ids,
    get_metrics_result,
    get_summary,
)
from .retrieval import build_retriver, get_retriver_names
from .schema import (
    DrugRecRecord,
    MetricsResult,
    RetrievedDrugCandidate,
    Retriver,
    RetriverEvalReport,
)
from .utils.log import get_console, setup_logging
from .utils.paths import OUTPUT_DIR, RESOURCE_DIR

DEFAULT_INPUT = RESOURCE_DIR / "DrugRec0328" / "test.jsonl"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "retriver"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评测检索器在 DrugRec 数据集上的离线召回效果。",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--retriver",
        type=str,
        choices=get_retriver_names(),
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", type=str, required=True)
    return parser.parse_args()


def evaluate_one_batch(
    patient: DrugRecRecord,
    candidates: Sequence[RetrievedDrugCandidate],
) -> MetricsResult:
    gold_ids = get_gold_ids(patient)
    retrieved_ids = [candidate["drugid"] for candidate in candidates]
    return get_metrics_result(gold_ids, retrieved_ids)


def test_retriver(
    retriver: Retriver,
    data: Sequence[DrugRecRecord],
    top_k: int,
) -> RetriverEvalReport:
    candidate_batches: list[list[RetrievedDrugCandidate]] = []
    with Progress(console=get_console()) as progress:
        task_id = progress.add_task("评测检索器", total=len(data))
        for patient in data:
            candidate_batches.append(retriver.retrieve(patient, top_k=top_k))
            progress.advance(task_id)
    metrics_list = [
        evaluate_one_batch(patient, candidates)
        for patient, candidates in zip(data, candidate_batches, strict=True)
    ]
    return {
        "config": {
            "retriver_name": type(retriver).__name__,
            "input_path": "",
            "top_k": top_k,
            "sample_count": len(data),
        },
        "summary": get_summary(metrics_list),
        "metrics": aggregate_metrics(metrics_list),
    }


def build_output_path(output_name: str) -> Path:
    output_file_name = (
        output_name if output_name.endswith(".json") else f"{output_name}.json"
    )
    return DEFAULT_OUTPUT_DIR / output_file_name


def main() -> None:
    args = parse_args()
    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取评测数据: %s", args.input.resolve())
    data = load_jsonl(
        path=args.input,
        parse_line=lambda row: cast(DrugRecRecord, row),
        limit=args.limit,
    )
    LOGGER.info("评测样本数: %s", len(data))
    LOGGER.info("开始构建检索器: %s", args.retriver)
    retriver = build_retriver(args.retriver)
    LOGGER.info("开始离线评测，top_k=%s", args.top_k)
    report = test_retriver(
        retriver=retriver,
        data=data,
        top_k=args.top_k,
    )
    report["config"]["retriver_name"] = args.retriver
    report["config"]["input_path"] = str(args.input.resolve())
    output_path = build_output_path(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    LOGGER.info("评测完成，结果已写入: %s", output_path.resolve())


if __name__ == "__main__":
    main()

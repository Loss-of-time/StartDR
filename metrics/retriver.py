import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rich.progress import Progress

from ..input_process import load_jsonl_limit
from ..retrieval import build_retriver, get_retriver_names
from ..schema import (
    DrugRecMedicine,
    DrugRecRecord,
    MetricsResult,
    RetrievedDrugCandidate,
    Retriver,
    RetriverEvalReport,
    SummaryResult,
)
from ..utils.log import get_console, setup_logging

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "data" / "DrugRec0328" / "test.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "retriver"
LOGGER = logging.getLogger("MINE.metrics.retriver")


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


###############################################################
# 工具函数
###############################################################

def get_gold_medicines(patient: DrugRecRecord) -> list[DrugRecMedicine]:
    return patient["medicine"]


def get_gold_ids(patient: DrugRecRecord) -> set[str]:
    return {medicine["drugid"] for medicine in get_gold_medicines(patient)}


###############################################################
# 指标
###############################################################


def hit_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    return 1.0 if gold_ids.intersection(retrieved_ids) else 0.0


def recall_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    if not gold_ids:
        return 0.0
    return len(gold_ids.intersection(retrieved_ids)) / len(gold_ids)


def mrr_at_k(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> float:
    for index, drugid in enumerate(retrieved_ids, start=1):
        if drugid in gold_ids:
            return 1.0 / index
    return 0.0


###############################################################
# 业务逻辑
###############################################################


def get_metrics_result(
    gold_ids: set[str],
    retrieved_ids: Sequence[str],
) -> MetricsResult:
    return {
        "hit": hit_at_k(gold_ids, retrieved_ids),
        "recall": recall_at_k(gold_ids, retrieved_ids),
        "mrr": mrr_at_k(gold_ids, retrieved_ids),
    }


def evaluate_one_batch(
    patient: DrugRecRecord,
    candidates: Sequence[RetrievedDrugCandidate],
) -> MetricsResult:
    gold_ids = get_gold_ids(patient)
    retrieved_ids = [candidate["drugid"] for candidate in candidates]
    return get_metrics_result(gold_ids, retrieved_ids)


def aggregate_metrics(metrics_list: Sequence[MetricsResult]) -> MetricsResult:
    if not metrics_list:
        return {
            "hit": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
        }
    total = len(metrics_list)
    return {
        "hit": sum(metrics["hit"] for metrics in metrics_list) / total,
        "recall": sum(metrics["recall"] for metrics in metrics_list) / total,
        "mrr": sum(metrics["mrr"] for metrics in metrics_list) / total,
    }


def get_summary(
    metrics_list: Sequence[MetricsResult],
) -> SummaryResult:
    return {
        "patient_count": len(metrics_list),
        "failure_count": sum(1 for metrics in metrics_list if metrics["hit"] == 0.0),
    }


def test_retriver(  # 核心逻辑
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
    summary = get_summary(metrics_list)
    # TODO: 有更多检索器后，由检索器各自负责失败样例输出。
    return {
        "config": {
            "retriver_name": type(retriver).__name__,
            "input_path": "",
            "top_k": top_k,
            "sample_count": len(data),
        },
        "summary": summary,
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
    data = load_jsonl_limit(args.input, args.limit)
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
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOGGER.info("评测完成，结果已写入: %s", output_path.resolve())


if __name__ == "__main__":
    main()

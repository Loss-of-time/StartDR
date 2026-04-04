import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .core.io import load_jsonl
from .core.metrics import (
    aggregate_metrics,
    get_gold_ids,
    get_metrics_result,
    get_summary,
)
from .core.retrieval import build_retriever, get_retriever_names
from .core.schema import (
    DrugRecRecord,
    RetrievedDrugCandidate,
    Retriever,
    RetrieverEvalConfig,
    RetrieverEvalReport,
    structure,
    unstructure,
)
from .core.setting import (
    DEFAULT_RETRIEVER_EVAL_INPUT,
    DEFAULT_RETRIEVER_EVAL_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测检索器在 DrugRec 上的召回效果。")
    parser.add_argument("--input", type=Path, default=DEFAULT_RETRIEVER_EVAL_INPUT)
    parser.add_argument(
        "--retriever",
        type=str,
        choices=get_retriever_names(),
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", type=str, required=True)
    return parser.parse_args()


def test_retriever(
    retriever: Retriever,
    data: list[DrugRecRecord],
    top_k: int,
) -> RetrieverEvalReport:
    candidate_batches: list[list[RetrievedDrugCandidate]] = []
    for patient in tqdm(data, desc="评测检索器"):
        candidate_batches.append(retriever.retrieve(patient, top_k=top_k))
    metrics_list = [
        get_metrics_result(
            get_gold_ids(patient),
            [candidate.drugid for candidate in candidates],
        )
        for patient, candidates in zip(data, candidate_batches, strict=True)
    ]
    return RetrieverEvalReport(
        config=RetrieverEvalConfig(
            retriever_name=type(retriever).__name__,
            input_path="",
            top_k=top_k,
            sample_count=len(data),
        ),
        summary=get_summary(metrics_list),
        metrics=aggregate_metrics(metrics_list),
    )


def main() -> None:
    args = parse_args()
    print(f"开始读取评测数据: {args.input.resolve()}")
    data = load_jsonl(
        path=args.input,
        parse_line=lambda row: structure(row, DrugRecRecord),
        limit=args.limit,
    )
    print(f"评测样本数: {len(data)}")
    print(f"开始构建检索器: {args.retriever}")
    retriever = build_retriever(args.retriever)
    print(f"开始离线评测，top_k={args.top_k}")
    report = test_retriever(
        retriever=retriever,
        data=data,
        top_k=args.top_k,
    )
    report.config.retriever_name = args.retriever
    report.config.input_path = str(args.input.resolve())
    output_file_name = (
        args.output_name
        if args.output_name.endswith(".json")
        else f"{args.output_name}.json"
    )
    output_path = DEFAULT_RETRIEVER_EVAL_OUTPUT_DIR / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(report), file, ensure_ascii=False, indent=2)
    print(f"评测完成，结果已写入: {output_path.resolve()}")


if __name__ == "__main__":
    main()

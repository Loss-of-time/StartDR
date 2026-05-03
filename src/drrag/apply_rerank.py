"""把精排结果合并到 RAG 样本的命令行入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from drrerank.core.schema import RankedCase
from drrerank.core.schema import structure as structure_ranked_case

from .core.adapters import apply_ranked_drugs, apply_ranked_evidences
from .core.generation import load_rag_cases
from .core.io import load_jsonl, write_jsonl
from .core.prompt import freeze_case_candidates
from .core.schema import RagCase, RagInputFormat, unstructure
from .core.setting import DEFAULT_RAG_CASE_OUTPUT_DIR, DEFAULT_TRACEDR_SAMPLE_INPUT


@dataclass(slots=True)
class ApplyRerankConfig:
    """RAG 合并精排结果配置。"""

    input_path: Path
    input_format: RagInputFormat
    ranked_input_path: Path
    output_path: Path
    top_k: int | None
    limit: int | None


def build_default_output_path(input_path: Path, ranked_input_path: Path) -> Path:
    """构造默认输出路径。

    Args:
        input_path: RAG 输入路径。
        ranked_input_path: 精排结果路径。

    Returns:
        默认输出文件路径。
    """

    return DEFAULT_RAG_CASE_OUTPUT_DIR / f"{input_path.stem}__{ranked_input_path.stem}.jsonl"


def load_ranked_cases(input_path: Path) -> dict[str, RankedCase]:
    """加载病例级精排结果。

    Args:
        input_path: 精排结果路径。

    Returns:
        以 patient_id 为键的精排结果映射。
    """

    ranked_cases: list[RankedCase] = load_jsonl(
        path=input_path,
        parse_line=lambda row: structure_ranked_case(row, RankedCase),
    )
    return {ranked_case.patient_id: ranked_case for ranked_case in ranked_cases}


def apply_rerank(config: ApplyRerankConfig) -> Path:
    """把精排排序结果补到 RAG 样本。

    Args:
        config: 合并配置。

    Returns:
        输出文件路径。
    """

    rag_cases: list[RagCase] = load_rag_cases(
        input_path=config.input_path,
        input_format=config.input_format,
        limit=config.limit,
    )
    ranked_case_map: dict[str, RankedCase] = load_ranked_cases(config.ranked_input_path)
    merged_cases: list[RagCase] = []
    rag_case: RagCase
    for rag_case in rag_cases:
        ranked_case: RankedCase | None = ranked_case_map.get(rag_case.patient_id)
        if ranked_case is None:
            merged_cases.append(freeze_case_candidates(rag_case, config.top_k))
            continue
        # 目的：把 TraceDR 的药物与证据排序一并补齐到统一 RAG 样本，保持在线与离线口径一致。
        merged_case: RagCase = apply_ranked_drugs(rag_case, ranked_case.ranked_drugs)
        merged_case = apply_ranked_evidences(merged_case, ranked_case.ranked_evidences)
        merged_cases.append(freeze_case_candidates(merged_case, config.top_k))
    write_jsonl(
        path=config.output_path,
        rows=merged_cases,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"RAG 精排合并完成: {config.output_path.resolve()} rows={len(merged_cases)}")
    return config.output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="把病例级精排结果补到 RAG 样本。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACEDR_SAMPLE_INPUT)
    parser.add_argument(
        "--input-format", choices=["tracedr_sample", "rag_case"], default="tracedr_sample"
    )
    parser.add_argument("--ranked-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    input_format: RagInputFormat = cast(RagInputFormat, args.input_format)
    output_path: Path = (
        args.output
        if args.output is not None
        else build_default_output_path(args.input, args.ranked_input)
    )
    apply_rerank(
        ApplyRerankConfig(
            input_path=args.input,
            input_format=input_format,
            ranked_input_path=args.ranked_input,
            output_path=output_path,
            top_k=args.top_k,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()

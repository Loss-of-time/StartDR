"""RAG 样本导出命令行入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .core.adapters import build_rag_case
from .core.io import load_jsonl, write_jsonl
from .core.schema import RagCase, TraceDRSample, structure, unstructure
from .core.setting import DEFAULT_RAG_CASE_OUTPUT_DIR, DEFAULT_TRACEDR_SAMPLE_INPUT


@dataclass(slots=True)
class ExportCasesConfig:
    """RAG 样本导出配置。"""

    input_path: Path
    output_path: Path
    top_k: int | None
    limit: int | None = None


def build_default_output_path(input_path: Path, top_k: int | None) -> Path:
    """构造默认输出路径。"""

    suffix: str = "all" if top_k is None else f"top{top_k}"
    return DEFAULT_RAG_CASE_OUTPUT_DIR / f"{input_path.stem}__{suffix}.jsonl"


def export_rag_cases(config: ExportCasesConfig) -> Path:
    """把 TraceDR 风格候选集导出为 RAG 统一样本。

    Args:
        config: 导出配置。

    Returns:
        输出文件路径。
    """

    print(f"开始读取候选集: {config.input_path.resolve()}")
    tracedr_samples: list[TraceDRSample] = load_jsonl(
        path=config.input_path,
        parse_line=lambda row: structure(row, TraceDRSample),
        limit=config.limit,
    )
    rag_cases: list[RagCase] = [
        build_rag_case(sample, candidate_limit=config.top_k) for sample in tracedr_samples
    ]
    # 目的：把 retrieval 的通用产物冻结成 RAG 统一契约，后续 prompt 与成本分析直接复用。
    write_jsonl(
        path=config.output_path,
        rows=rag_cases,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"RAG 样本写出完成: {config.output_path.resolve()} rows={len(rag_cases)}")
    return config.output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="把 TraceDR 风格候选集导出为 RAG 统一样本。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACEDR_SAMPLE_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    output_path: Path = (
        args.output
        if args.output is not None
        else build_default_output_path(args.input, args.top_k)
    )
    export_rag_cases(
        ExportCasesConfig(
            input_path=args.input,
            output_path=output_path,
            top_k=args.top_k,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()

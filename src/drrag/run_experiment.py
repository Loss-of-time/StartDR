"""RAG 实验编排命令行入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from drrerank.tracedr_export_rank import resolve_checkpoint_path

from .core.experiment import (
    DEFAULT_EXPERIMENT_TOP_KS,
    DEFAULT_EXPERIMENT_VARIANTS,
    RagExperimentConfig,
    run_rag_experiment,
)
from .core.schema import RagTask
from .core.setting import (
    DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    DEFAULT_SILICONFLOW_MAX_TOKENS,
    DEFAULT_SILICONFLOW_MODEL,
    DEFAULT_SILICONFLOW_TEMPERATURE,
    DEFAULT_SILICONFLOW_TIMEOUT_SECONDS,
    DEFAULT_TRACEDR_SAMPLE_INPUT,
)


@dataclass(slots=True)
class ParsedArgs:
    """解析后的命令行参数。"""

    input_path: Path
    checkpoint_path: Path
    model_name: str
    task: RagTask
    top_ks: tuple[int, ...]
    variants: tuple[str, ...]
    max_evidences_per_candidate: int
    max_tokens: int
    temperature: float
    timeout_seconds: int
    limit: int | None
    overwrite: bool


def parse_top_ks(raw_value: str) -> tuple[int, ...]:
    """解析 `top-k` 列表。"""

    top_ks: list[int] = []
    raw_item: str
    for raw_item in raw_value.split(","):
        normalized_item: str = raw_item.strip()
        if normalized_item == "":
            continue
        top_ks.append(int(normalized_item))
    if len(top_ks) == 0:
        raise ValueError("至少需要一个 top-k。")
    return tuple(top_ks)


def parse_variants(raw_value: str) -> tuple[str, ...]:
    """解析实验输入方案。"""

    variants: list[str] = []
    raw_item: str
    for raw_item in raw_value.split(","):
        normalized_item: str = raw_item.strip()
        if normalized_item == "":
            continue
        variants.append(normalized_item)
    if len(variants) == 0:
        raise ValueError("至少需要一个实验输入方案。")
    return tuple(variants)


def parse_args() -> ParsedArgs:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="按固定口径批量运行 RAG 实验并汇总结果。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACEDR_SAMPLE_INPUT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_SILICONFLOW_MODEL)
    parser.add_argument("--task", choices=["recommend", "explain"], default="recommend")
    parser.add_argument(
        "--top-ks",
        type=str,
        default=",".join(str(item) for item in DEFAULT_EXPERIMENT_TOP_KS),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_EXPERIMENT_VARIANTS),
    )
    parser.add_argument(
        "--max-evidences-per-candidate",
        type=int,
        default=DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_SILICONFLOW_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SILICONFLOW_TEMPERATURE)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_SILICONFLOW_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    return ParsedArgs(
        input_path=args.input,
        checkpoint_path=resolve_checkpoint_path(args.checkpoint),
        model_name=args.model,
        task=args.task,
        top_ks=parse_top_ks(args.top_ks),
        variants=parse_variants(args.variants),
        max_evidences_per_candidate=args.max_evidences_per_candidate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        overwrite=args.overwrite,
    )


def main() -> None:
    """命令行入口。"""

    args: ParsedArgs = parse_args()
    artifacts = run_rag_experiment(
        RagExperimentConfig(
            input_path=args.input_path,
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            task=args.task,
            top_ks=args.top_ks,
            variants=args.variants,
            max_evidences_per_candidate=args.max_evidences_per_candidate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    )
    print(f"实验完成，结果表: {artifacts.table_md_path.resolve()}")
    print(f"案例分析: {artifacts.case_analysis_md_path.resolve()}")


if __name__ == "__main__":
    main()

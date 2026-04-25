"""药品可解释生成结果离线评估命令行入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from .core.generation import (
    build_default_eval_output_path,
    evaluate_generation_record,
    summarize_generation_evaluations,
    unstructure_generation_evaluations,
)
from .core.io import load_jsonl, write_json
from .core.schema import RagGenerationRecord, structure, unstructure


@dataclass(slots=True)
class EvaluateGenerationConfig:
    """生成结果评估配置。"""

    input_path: Path
    output_path: Path
    limit: int | None


def evaluate_generation(config: EvaluateGenerationConfig) -> Path:
    """执行整份生成结果的离线评估。"""

    generation_records: list[RagGenerationRecord] = load_jsonl(
        path=config.input_path,
        parse_line=lambda row: structure(row, RagGenerationRecord),
        limit=config.limit,
    )
    evaluations = [evaluate_generation_record(record) for record in generation_records]
    summary = summarize_generation_evaluations(evaluations)
    error_counter: dict[str, int] = {}
    for evaluation in evaluations:
        for error in evaluation.validation_errors:
            error_counter[error] = error_counter.get(error, 0) + 1
    sample_count: int = len(evaluations)
    structured_legal_count: int = sum(
        1 for evaluation in evaluations if not evaluation.validation_errors
    )
    field_complete_count: int = sum(1 for evaluation in evaluations if evaluation.field_complete)
    selection_alignment_count: int = sum(
        1 for evaluation in evaluations if evaluation.selection_alignment_valid
    )
    candidate_refs_valid_count: int = sum(
        1 for evaluation in evaluations if evaluation.candidate_refs_valid
    )
    evidence_refs_valid_count: int = sum(
        1 for evaluation in evaluations if evaluation.evidence_refs_valid
    )
    reason_fields_valid_count: int = sum(
        1 for evaluation in evaluations if evaluation.reason_fields_valid
    )
    rate_summary: dict[str, float] = {
        "success_rate": summary.success_count / sample_count if sample_count > 0 else 0.0,
        "structured_answer_rate": (
            summary.structured_answer_count / sample_count if sample_count > 0 else 0.0
        ),
        "structure_legal_rate": structured_legal_count / sample_count if sample_count > 0 else 0.0,
        "field_complete_rate": field_complete_count / sample_count if sample_count > 0 else 0.0,
        "selection_alignment_rate": (
            selection_alignment_count / sample_count if sample_count > 0 else 0.0
        ),
        "candidate_refs_valid_rate": (
            candidate_refs_valid_count / sample_count if sample_count > 0 else 0.0
        ),
        "evidence_refs_valid_rate": (
            evidence_refs_valid_count / sample_count if sample_count > 0 else 0.0
        ),
        "reason_fields_valid_rate": (
            reason_fields_valid_count / sample_count if sample_count > 0 else 0.0
        ),
        "hit_rate": summary.hit_count / sample_count if sample_count > 0 else 0.0,
        "exact_match_rate": summary.exact_match_count / sample_count if sample_count > 0 else 0.0,
        "average_precision": summary.average_precision,
        "average_recall": summary.average_recall,
        "average_f1": summary.average_f1,
    }
    report: dict[str, object] = {
        "input_path": str(config.input_path),
        "summary": unstructure(summary),
        "rate_summary": rate_summary,
        "error_counter": error_counter,
        # 目的：只写预览，避免逐样本全量评估结果在默认产物里膨胀。
        "sample_preview": unstructure_generation_evaluations(evaluations),
    }
    write_json(config.output_path, report)
    print(
        "评估完成: "
        f"samples={summary.sample_count} valid={summary.fully_valid_count} "
        f"hit={summary.hit_count} exact_match={summary.exact_match_count}"
    )
    print(f"报告路径: {config.output_path.resolve()}")
    return config.output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="离线评估硅基流动药品可解释生成结果。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    output_path: Path = (
        args.output if args.output is not None else build_default_eval_output_path(args.input)
    )
    evaluate_generation(
        EvaluateGenerationConfig(
            input_path=args.input,
            output_path=output_path,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()

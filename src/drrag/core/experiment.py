"""RAG 实验编排与结果汇总。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drrerank.core.model.tracedr.schema import TraceDRAblationConfig
from drrerank.core.setting import DEFAULT_RERANK_OUTPUT_DIR
from drrerank.tracedr_export_rank import ExportRankConfig, export_ranked_cases

from ..apply_rerank import ApplyRerankConfig, apply_rerank
from ..evaluate_generation import EvaluateGenerationConfig, evaluate_generation
from ..export_cases import ExportCasesConfig, export_rag_cases
from ..generate_siliconflow import GenerateConfig, generate_records
from .generation import (
    build_default_eval_output_path,
    build_default_generation_output_path,
    evaluate_generation_record,
)
from .io import load_jsonl
from .schema import RagCase, RagGenerationRecord, RagInputFormat, RagTask, structure
from .setting import DEFAULT_RAG_CASE_OUTPUT_DIR

type RagExperimentVariant = str

DEFAULT_EXPERIMENT_TOP_KS = (10, 20, 50)
DEFAULT_EXPERIMENT_VARIANTS = ("retrieval_direct", "tracedr_rerank")


@dataclass(slots=True)
class RagExperimentConfig:
    """RAG 实验配置。"""

    input_path: Path
    checkpoint_path: Path
    model_name: str
    task: RagTask
    top_ks: tuple[int, ...]
    variants: tuple[RagExperimentVariant, ...]
    max_evidences_per_candidate: int
    max_tokens: int
    temperature: float
    timeout_seconds: int
    limit: int | None
    overwrite: bool


@dataclass(slots=True)
class RagExperimentRunResult:
    """单组实验结果。"""

    variant: RagExperimentVariant
    top_k: int
    input_format: RagInputFormat
    case_path: Path
    generation_path: Path
    eval_path: Path
    sample_count: int
    rate_summary: dict[str, float]


@dataclass(slots=True)
class RagExperimentArtifacts:
    """整次实验产物。"""

    ranked_path: Path | None
    run_results: list[RagExperimentRunResult]
    table_json_path: Path
    table_md_path: Path
    case_analysis_md_path: Path


def build_experiment_case_path(
    input_path: Path,
    variant: RagExperimentVariant,
    top_k: int,
) -> Path:
    """构造实验病例文件路径。"""

    return DEFAULT_RAG_CASE_OUTPUT_DIR / f"{input_path.stem}__{variant}__top{top_k}.jsonl"


def load_eval_report(path: Path) -> dict[str, Any]:
    """读取评估报告。"""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def format_percentage(value: float) -> str:
    """格式化百分比。"""

    return f"{value * 100:.2f}%"


def build_result_table_json(run_results: list[RagExperimentRunResult]) -> list[dict[str, object]]:
    """构造结果表 JSON。"""

    rows: list[dict[str, object]] = []
    run_result: RagExperimentRunResult
    for run_result in run_results:
        rows.append(
            {
                "variant": run_result.variant,
                "top_k": run_result.top_k,
                "sample_count": run_result.sample_count,
                "structure_legal_rate": run_result.rate_summary["structure_legal_rate"],
                "field_complete_rate": run_result.rate_summary["field_complete_rate"],
                "hit_rate": run_result.rate_summary["hit_rate"],
                "exact_match_rate": run_result.rate_summary["exact_match_rate"],
                "average_precision": run_result.rate_summary["average_precision"],
                "average_recall": run_result.rate_summary["average_recall"],
                "average_f1": run_result.rate_summary["average_f1"],
                "candidate_refs_valid_rate": run_result.rate_summary["candidate_refs_valid_rate"],
                "evidence_refs_valid_rate": run_result.rate_summary["evidence_refs_valid_rate"],
            }
        )
    return rows


def build_result_table_markdown(run_results: list[RagExperimentRunResult]) -> str:
    """构造结果表 Markdown。"""

    header: str = (
        "| 输入方案 | Top-K | 样本数 | 结构合法率 | 字段完整率 | 命中率 | 精确匹配率 | "
        "平均 Precision | 平均 Recall | 平均 F1 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines: list[str] = [header]
    run_result: RagExperimentRunResult
    for run_result in run_results:
        lines.append(
            "| "
            f"{run_result.variant} | "
            f"{run_result.top_k} | "
            f"{run_result.sample_count} | "
            f"{format_percentage(run_result.rate_summary['structure_legal_rate'])} | "
            f"{format_percentage(run_result.rate_summary['field_complete_rate'])} | "
            f"{format_percentage(run_result.rate_summary['hit_rate'])} | "
            f"{format_percentage(run_result.rate_summary['exact_match_rate'])} | "
            f"{run_result.rate_summary['average_precision']:.4f} | "
            f"{run_result.rate_summary['average_recall']:.4f} | "
            f"{run_result.rate_summary['average_f1']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def select_case_analysis_records(
    run_results: list[RagExperimentRunResult],
) -> list[tuple[RagExperimentRunResult, RagCase, RagGenerationRecord]]:
    """挑选用于案例分析的记录。"""

    selected_records: list[tuple[RagExperimentRunResult, RagCase, RagGenerationRecord]] = []
    seen_patient_ids: set[tuple[str, int, str]] = set()
    run_result: RagExperimentRunResult
    for run_result in run_results:
        case_map: dict[str, RagCase] = {
            case.patient_id: case
            for case in load_jsonl(
                path=run_result.case_path,
                parse_line=lambda row: structure(row, RagCase),
            )
        }
        generation_records: list[RagGenerationRecord] = load_jsonl(
            path=run_result.generation_path,
            parse_line=lambda row: structure(row, RagGenerationRecord),
        )
        generation_record: RagGenerationRecord
        for generation_record in generation_records:
            evaluation = evaluate_generation_record(generation_record)
            selection_key: tuple[str, int, str] = (
                generation_record.patient_id,
                run_result.top_k,
                run_result.variant,
            )
            if selection_key in seen_patient_ids:
                continue
            if not generation_record.success or generation_record.parsed_answer is None:
                continue
            if len(evaluation.validation_errors) > 0:
                continue
            case: RagCase | None = case_map.get(generation_record.patient_id)
            if case is None:
                continue
            selected_records.append((run_result, case, generation_record))
            seen_patient_ids.add(selection_key)
            if len(selected_records) >= 6:
                return selected_records
    return selected_records


def build_case_analysis_markdown(run_results: list[RagExperimentRunResult]) -> str:
    """构造案例分析 Markdown。"""

    selected_records: list[tuple[RagExperimentRunResult, RagCase, RagGenerationRecord]] = (
        select_case_analysis_records(run_results)
    )
    lines: list[str] = [
        "# RAG 案例分析",
        "",
        "以下内容来自当前小样本验证，可直接作为论文案例分析草稿。",
        "",
    ]
    run_result: RagExperimentRunResult
    case: RagCase
    generation_record: RagGenerationRecord
    for index, (run_result, case, generation_record) in enumerate(selected_records, start=1):
        patient = case.patient
        parsed_answer = generation_record.parsed_answer
        if parsed_answer is None:
            continue
        predicted_drugids: list[str] = parsed_answer.selected_drugids
        gold_set: set[str] = set(case.gold_drugids)
        hit_drugids: list[str] = [drugid for drugid in predicted_drugids if drugid in gold_set]
        lines.extend(
            [
                f"## 案例 {index}",
                f"- 实验组：`{run_result.variant}` / `top{run_result.top_k}`",
                f"- patient_id：`{case.patient_id}`",
                f"- 患者摘要：{patient.age} 岁，{patient.gender}；诊断={', '.join(patient.diagnosis) if patient.diagnosis else 'None'}；症状={', '.join(patient.symptom) if patient.symptom else 'None'}；当前用药={', '.join(item.name for item in patient.on_medicine) if patient.on_medicine else 'None'}",
                f"- 预测药物：{', '.join(predicted_drugids) if predicted_drugids else 'None'}",
                f"- 命中 gold：{', '.join(hit_drugids) if hit_drugids else 'None'}",
            ]
        )
        item_lines: list[str] = []
        item_index: int
        for item_index, item in enumerate(parsed_answer.items, start=1):
            item_lines.append(
                f"- 解释 {item_index}：drugid={item.drugid}；证据={', '.join(item.evidence_ids)}；理由={item.reason}"
            )
        lines.extend(item_lines)
        lines.append("")
    if len(selected_records) == 0:
        lines.extend(["- 当前小样本验证未筛出可用案例，请增大 `--limit` 后重跑。", ""])
    return "\n".join(lines)


def build_experiment_report_paths(
    input_path: Path,
    model_name: str,
    task: RagTask,
) -> tuple[Path, Path, Path]:
    """构造实验汇总产物路径。"""

    generation_stub: Path = build_default_generation_output_path(input_path, model_name, task)
    eval_stub: Path = build_default_eval_output_path(generation_stub)
    table_json_path: Path = eval_stub.with_name(f"{eval_stub.stem}__comparison_table.json")
    table_md_path: Path = eval_stub.with_name(f"{eval_stub.stem}__comparison_table.md")
    case_analysis_md_path: Path = eval_stub.with_name(f"{eval_stub.stem}__case_analysis.md")
    return table_json_path, table_md_path, case_analysis_md_path


def run_single_variant(
    config: RagExperimentConfig,
    variant: RagExperimentVariant,
    top_k: int,
    ranked_path: Path | None,
) -> RagExperimentRunResult:
    """运行单组实验。"""

    case_path: Path = build_experiment_case_path(config.input_path, variant, top_k)
    input_format: RagInputFormat = "rag_case"
    if variant == "retrieval_direct":
        export_rag_cases(
            ExportCasesConfig(
                input_path=config.input_path,
                output_path=case_path,
                top_k=top_k,
                limit=config.limit,
            )
        )
    elif variant == "tracedr_rerank":
        if ranked_path is None:
            raise ValueError("tracedr_rerank 方案缺少 ranked_path。")
        apply_rerank(
            ApplyRerankConfig(
                input_path=config.input_path,
                input_format="tracedr_sample",
                ranked_input_path=ranked_path,
                output_path=case_path,
                top_k=top_k,
                limit=config.limit,
            )
        )
    else:
        raise ValueError(f"未知实验输入方案: {variant}")

    generation_path: Path = build_default_generation_output_path(
        case_path,
        config.model_name,
        config.task,
        top_k=top_k,
        max_evidences_per_candidate=config.max_evidences_per_candidate,
    )
    generate_records(
        GenerateConfig(
            input_path=case_path,
            output_path=generation_path,
            input_format=input_format,
            model_name=config.model_name,
            task=config.task,
            top_k=top_k,
            max_evidences_per_candidate=config.max_evidences_per_candidate,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            limit=config.limit,
            overwrite=config.overwrite,
        )
    )
    eval_path: Path = build_default_eval_output_path(generation_path)
    evaluate_generation(
        EvaluateGenerationConfig(
            input_path=generation_path,
            output_path=eval_path,
            limit=None,
        )
    )
    eval_report: dict[str, Any] = load_eval_report(eval_path)
    return RagExperimentRunResult(
        variant=variant,
        top_k=top_k,
        input_format=input_format,
        case_path=case_path,
        generation_path=generation_path,
        eval_path=eval_path,
        sample_count=int(eval_report["summary"]["sample_count"]),
        rate_summary=dict(eval_report["rate_summary"]),
    )


def run_rag_experiment(config: RagExperimentConfig) -> RagExperimentArtifacts:
    """运行整次 RAG 实验。"""

    ranked_path: Path | None = None
    if "tracedr_rerank" in config.variants:
        ranked_path = DEFAULT_RERANK_OUTPUT_DIR / (
            f"{config.input_path.stem}__{config.checkpoint_path.stem}.jsonl"
        )
        export_ranked_cases(
            ExportRankConfig(
                input_path=config.input_path,
                checkpoint_path=config.checkpoint_path,
                output_path=ranked_path,
                limit=config.limit,
                ablation_config=TraceDRAblationConfig(),
            )
        )

    run_results: list[RagExperimentRunResult] = []
    variant: RagExperimentVariant
    top_k: int
    for variant in config.variants:
        for top_k in config.top_ks:
            # 目的：逐组落盘，确保每个输入方案与 top-k 组合都有独立可复现产物。
            run_results.append(run_single_variant(config, variant, top_k, ranked_path))

    table_json_path, table_md_path, case_analysis_md_path = build_experiment_report_paths(
        config.input_path,
        config.model_name,
        config.task,
    )
    table_rows: list[dict[str, object]] = build_result_table_json(run_results)
    table_json_path.parent.mkdir(parents=True, exist_ok=True)
    table_json_path.write_text(
        json.dumps(table_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    table_md_path.write_text(
        build_result_table_markdown(run_results), encoding="utf-8", newline="\n"
    )
    case_analysis_md_path.write_text(
        build_case_analysis_markdown(run_results),
        encoding="utf-8",
        newline="\n",
    )
    return RagExperimentArtifacts(
        ranked_path=ranked_path,
        run_results=run_results,
        table_json_path=table_json_path,
        table_md_path=table_md_path,
        case_analysis_md_path=case_analysis_md_path,
    )

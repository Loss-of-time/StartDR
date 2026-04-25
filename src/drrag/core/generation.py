"""RAG 在线生成与评估的共享工具。"""

import json
from pathlib import Path
from typing import cast

from .adapters import build_rag_case
from .io import load_jsonl
from .prompt import build_prompt, select_candidates, select_evidences
from .schema import (
    RagCase,
    RagGeneratedAnswer,
    RagGenerationEvalRecord,
    RagGenerationEvalSummary,
    RagGenerationRecord,
    RagInputFormat,
    RagRequest,
    TraceDRSample,
    structure,
    unstructure,
)
from .setting import DEFAULT_RAG_EVAL_OUTPUT_DIR, DEFAULT_RAG_GENERATION_OUTPUT_DIR


def sanitize_model_name(model_name: str) -> str:
    """把模型名转成稳定的文件名片段。"""

    # 目的：把 API 模型名稳定映射到本地输出文件名，避免路径分隔符污染目录结构。
    return model_name.replace("/", "__").replace("-", "_").replace(".", "_")


def build_default_generation_output_path(
    input_path: Path,
    model_name: str,
    task: str,
    top_k: int | None = None,
    max_evidences_per_candidate: int | None = None,
) -> Path:
    """构造默认生成输出路径。"""

    model_slug: str = sanitize_model_name(model_name)
    filename_parts: list[str] = [input_path.stem, model_slug, task]
    if top_k is not None:
        # 目的：把会改变候选可见范围的关键参数编码进文件名，避免不同实验配置互相覆盖。
        filename_parts.append(f"top{top_k}")
    if max_evidences_per_candidate is not None:
        filename_parts.append(f"ev{max_evidences_per_candidate}")
    return DEFAULT_RAG_GENERATION_OUTPUT_DIR / ("__".join(filename_parts) + ".jsonl")


def build_default_eval_output_path(input_path: Path) -> Path:
    """构造默认评估输出路径。"""

    return DEFAULT_RAG_EVAL_OUTPUT_DIR / f"{input_path.stem}__eval.json"


def load_rag_cases(
    input_path: Path,
    input_format: RagInputFormat,
    limit: int | None,
) -> list[RagCase]:
    """按输入格式加载统一 RAG 样本。"""

    if input_format == "rag_case":
        return load_jsonl(
            path=input_path,
            parse_line=lambda row: structure(row, RagCase),
            limit=limit,
        )
    tracedr_samples: list[TraceDRSample] = load_jsonl(
        path=input_path,
        parse_line=lambda row: structure(row, TraceDRSample),
        limit=limit,
    )
    return [build_rag_case(sample) for sample in tracedr_samples]


def build_visible_candidate_drugids(request: RagRequest) -> list[str]:
    """提取当前请求可见的候选药物列表。"""

    return [candidate.drugid for candidate in select_candidates(request)]


def build_visible_evidence_ids(request: RagRequest) -> list[str]:
    """提取当前请求可见的证据编号列表。"""

    evidence_ids: list[str] = []
    for candidate in select_candidates(request):
        for evidence in select_evidences(candidate, request.max_evidences_per_candidate):
            evidence_ids.append(evidence.evidence_id)
    return evidence_ids


def extract_json_object_text(response_content: str) -> str:
    """从模型输出中截取 JSON 对象文本。"""

    start_index: int = response_content.find("{")
    end_index: int = response_content.rfind("}")
    if start_index < 0 or end_index < start_index:
        return response_content
    return response_content[start_index : end_index + 1]


def parse_generated_answer(response_content: str) -> RagGeneratedAnswer:
    """把模型文本结果解析成结构化推荐对象。"""

    payload_text: str = extract_json_object_text(response_content)
    payload: dict[str, object] = cast(dict[str, object], json.loads(payload_text))
    return structure(payload, RagGeneratedAnswer)


def validate_generation_record(record: RagGenerationRecord) -> list[str]:
    """校验单样本生成结果是否满足结构与引用约束。"""

    errors: list[str] = []
    if not record.success:
        errors.append("generation_failed")
        return errors
    if record.parsed_answer is None:
        errors.append("missing_structured_answer")
        return errors

    parsed_answer: RagGeneratedAnswer = record.parsed_answer
    visible_candidate_set: set[str] = set(record.visible_candidate_drugids)
    visible_evidence_set: set[str] = set(record.visible_evidence_ids)
    item_drugids: list[str] = [item.drugid for item in parsed_answer.items]

    if parsed_answer.selected_drugids != item_drugids:
        errors.append("selected_drugids_and_items_are_not_aligned")

    for drugid in parsed_answer.selected_drugids:
        if drugid not in visible_candidate_set:
            errors.append(f"unknown_candidate:{drugid}")

    for item in parsed_answer.items:
        if item.drugid not in visible_candidate_set:
            errors.append(f"unknown_item_candidate:{item.drugid}")
        if item.reason.strip() == "":
            errors.append(f"empty_reason:{item.drugid}")
        if len(item.evidence_ids) == 0:
            errors.append(f"empty_evidence_ids:{item.drugid}")
        for evidence_id in item.evidence_ids:
            if evidence_id not in visible_evidence_set:
                errors.append(f"unknown_evidence:{item.drugid}:{evidence_id}")
    return errors


def evaluate_generation_record(record: RagGenerationRecord) -> RagGenerationEvalRecord:
    """评估单样本生成结果。"""

    validation_errors: list[str] = validate_generation_record(record)
    parsed_answer: RagGeneratedAnswer | None = record.parsed_answer
    predicted_drugids: list[str] = (
        list(dict.fromkeys(parsed_answer.selected_drugids)) if parsed_answer is not None else []
    )
    predicted_set: set[str] = set(predicted_drugids)
    gold_set: set[str] = set(record.gold_drugids)
    overlap_count: int = len(predicted_set & gold_set)
    precision: float = overlap_count / len(predicted_set) if predicted_set else 0.0
    recall: float = overlap_count / len(gold_set) if gold_set else 0.0
    f1: float = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    selection_alignment_valid: bool = (
        "selected_drugids_and_items_are_not_aligned" not in validation_errors
    )
    candidate_refs_valid: bool = not any(
        error.startswith("unknown_candidate:") or error.startswith("unknown_item_candidate:")
        for error in validation_errors
    )
    evidence_refs_valid: bool = not any(
        error.startswith("unknown_evidence:") or error.startswith("empty_evidence_ids:")
        for error in validation_errors
    )
    reason_fields_valid: bool = not any(
        error.startswith("empty_reason:") for error in validation_errors
    )
    evidence_fields_present: bool = not any(
        error.startswith("empty_evidence_ids:") for error in validation_errors
    )
    return RagGenerationEvalRecord(
        patient_id=record.patient_id,
        success=record.success,
        has_structured_answer=parsed_answer is not None,
        selection_alignment_valid=selection_alignment_valid,
        candidate_refs_valid=candidate_refs_valid,
        evidence_refs_valid=evidence_refs_valid,
        reason_fields_valid=reason_fields_valid,
        field_complete=(
            parsed_answer is not None
            and selection_alignment_valid
            and reason_fields_valid
            and evidence_fields_present
        ),
        selected_count=len(predicted_drugids),
        gold_count=len(record.gold_drugids),
        hit=overlap_count > 0,
        exact_match=predicted_set == gold_set,
        precision=precision,
        recall=recall,
        f1=f1,
        validation_errors=validation_errors,
    )


def summarize_generation_evaluations(
    evaluations: list[RagGenerationEvalRecord],
) -> RagGenerationEvalSummary:
    """汇总整份生成评估结果。"""

    sample_count: int = len(evaluations)
    success_count: int = sum(1 for item in evaluations if item.success)
    structured_answer_count: int = sum(1 for item in evaluations if item.has_structured_answer)
    fully_valid_count: int = sum(1 for item in evaluations if len(item.validation_errors) == 0)
    hit_count: int = sum(1 for item in evaluations if item.hit)
    exact_match_count: int = sum(1 for item in evaluations if item.exact_match)
    average_precision: float = (
        sum(item.precision for item in evaluations) / sample_count if sample_count > 0 else 0.0
    )
    average_recall: float = (
        sum(item.recall for item in evaluations) / sample_count if sample_count > 0 else 0.0
    )
    average_f1: float = (
        sum(item.f1 for item in evaluations) / sample_count if sample_count > 0 else 0.0
    )
    return RagGenerationEvalSummary(
        sample_count=sample_count,
        success_count=success_count,
        structured_answer_count=structured_answer_count,
        fully_valid_count=fully_valid_count,
        hit_count=hit_count,
        exact_match_count=exact_match_count,
        average_precision=average_precision,
        average_recall=average_recall,
        average_f1=average_f1,
    )


def unstructure_generation_evaluations(
    evaluations: list[RagGenerationEvalRecord],
    preview_limit: int = 20,
) -> list[object]:
    """把评估对象转换成适合写盘的预览列表。"""

    return [unstructure(item) for item in evaluations[: min(preview_limit, len(evaluations))]]


def build_prompt_for_request(request: RagRequest) -> object:
    """导出统一 prompt 结果供外层调用。"""

    return build_prompt(request)

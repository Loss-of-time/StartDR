"""硅基流动药品可解释生成命令行入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError

from .core.generation import (
    build_default_generation_output_path,
    build_prompt_for_request,
    build_visible_candidate_drugids,
    build_visible_evidence_ids,
    load_rag_cases,
    parse_generated_answer,
)
from .core.io import load_jsonl, write_jsonl
from .core.schema import (
    PromptBuildResult,
    RagCase,
    RagGenerationRecord,
    RagInputFormat,
    RagRequest,
    RagTask,
    structure,
    unstructure,
)
from .core.setting import (
    DEFAULT_RAG_GENERATION_LIMIT,
    DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    DEFAULT_RAG_TOP_K,
    DEFAULT_SILICONFLOW_MAX_TOKENS,
    DEFAULT_SILICONFLOW_MODEL,
    DEFAULT_SILICONFLOW_TEMPERATURE,
    DEFAULT_SILICONFLOW_TIMEOUT_SECONDS,
    DEFAULT_TRACEDR_SAMPLE_INPUT,
)
from .core.siliconflow import SiliconFlowCompletion, request_json_completion


@dataclass(slots=True)
class GenerateConfig:
    """硅基流动生成配置。"""

    input_path: Path
    output_path: Path
    input_format: RagInputFormat
    model_name: str
    task: RagTask
    top_k: int
    max_evidences_per_candidate: int
    max_tokens: int
    temperature: float
    timeout_seconds: int
    limit: int | None
    overwrite: bool


def build_record_identity(record: RagGenerationRecord) -> tuple[str, str, str, int, int]:
    """构造生成记录的同配置身份键。"""

    return (
        record.patient_id,
        record.model_name,
        record.task,
        record.top_k,
        record.max_evidences_per_candidate,
    )


def build_request_identity(config: GenerateConfig, case: RagCase) -> tuple[str, str, str, int, int]:
    """构造当前请求的同配置身份键。"""

    return (
        case.patient_id,
        config.model_name,
        config.task,
        config.top_k,
        config.max_evidences_per_candidate,
    )


def load_existing_records(output_path: Path) -> list[RagGenerationRecord]:
    """读取已存在的生成记录。"""

    if not output_path.exists():
        return []
    return load_jsonl(
        path=output_path,
        parse_line=lambda row: structure(row, RagGenerationRecord),
    )


def build_generation_record(
    request: RagRequest,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> RagGenerationRecord:
    """执行单样本硅基流动生成。"""

    prompt: PromptBuildResult = cast(PromptBuildResult, build_prompt_for_request(request))
    visible_candidate_drugids: list[str] = build_visible_candidate_drugids(request)
    visible_evidence_ids: list[str] = build_visible_evidence_ids(request)
    completion: SiliconFlowCompletion | None = None
    raw_response: dict[str, object] | None = None
    response_content: str | None = None
    usage = None
    finish_reason: str | None = None
    trace_id: str | None = None
    parsed_answer = None
    error_message: str | None = None
    # 目的：把外部 HTTP 与 JSON 解析失败压缩成单条样本级记录，避免重复请求导致无意义放量。
    try:
        completion = request_json_completion(
            model_name=model_name,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        raw_response = completion.raw_response
        response_content = completion.response_content
        usage = completion.usage
        finish_reason = completion.finish_reason
        trace_id = completion.trace_id
        parsed_answer = parse_generated_answer(completion.response_content)
    except HTTPError as error:
        error_message = f"HTTP {error.code}"
        trace_id = error.headers.get("x-siliconcloud-trace-id")
        raw_response = {"error_body": error.read().decode("utf-8")}
    except URLError as error:
        error_message = str(error.reason)
    except json.JSONDecodeError as error:
        error_message = f"JSONDecodeError: {error.msg}"

    return RagGenerationRecord(
        patient_id=request.case.patient_id,
        task=request.task,
        model_name=model_name,
        top_k=request.top_k,
        max_evidences_per_candidate=request.max_evidences_per_candidate,
        success=completion is not None and parsed_answer is not None,
        finish_reason=finish_reason,
        trace_id=trace_id,
        error_message=error_message,
        visible_candidate_drugids=visible_candidate_drugids,
        visible_evidence_ids=visible_evidence_ids,
        gold_drugids=list(request.case.gold_drugids),
        prompt=prompt,
        raw_response=raw_response,
        response_content=response_content,
        usage=usage,
        parsed_answer=parsed_answer,
    )


def generate_records(config: GenerateConfig) -> Path:
    """执行整批硅基流动生成。"""

    existing_records: list[RagGenerationRecord] = (
        [] if config.overwrite else load_existing_records(config.output_path)
    )
    successful_identities: set[tuple[str, str, str, int, int]] = {
        build_record_identity(record) for record in existing_records if record.success
    }
    rag_cases: list[RagCase] = load_rag_cases(
        input_path=config.input_path,
        input_format=config.input_format,
        limit=config.limit,
    )
    new_records: list[RagGenerationRecord] = []
    preserved_records: list[RagGenerationRecord] = []
    record: RagGenerationRecord
    for record in existing_records:
        record_identity: tuple[str, str, str, int, int] = build_record_identity(record)
        if record_identity in successful_identities:
            preserved_records.append(record)
    for case in rag_cases:
        request_identity: tuple[str, str, str, int, int] = build_request_identity(config, case)
        if request_identity in successful_identities:
            print(f"跳过已完成样本: patient_id={case.patient_id}")
            continue
        request: RagRequest = RagRequest(
            case=case,
            task=config.task,
            top_k=config.top_k,
            max_evidences_per_candidate=config.max_evidences_per_candidate,
        )
        record: RagGenerationRecord = build_generation_record(
            request=request,
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )
        new_records.append(record)
        print(
            "生成完成: "
            f"patient_id={record.patient_id} success={record.success} "
            f"trace_id={record.trace_id if record.trace_id is not None else 'None'}"
        )
    # 目的：仅保留旧的成功记录，失败记录在重跑成功后由新记录替换，避免输出文件残留重复样本。
    all_records: list[RagGenerationRecord] = preserved_records + new_records
    write_jsonl(
        path=config.output_path,
        rows=all_records,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"写出完成: {config.output_path.resolve()} rows={len(all_records)}")
    return config.output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="使用硅基流动 JSON Mode 生成药品可解释推荐结果。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACEDR_SAMPLE_INPUT)
    parser.add_argument(
        "--input-format", choices=["tracedr_sample", "rag_case"], default="tracedr_sample"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_SILICONFLOW_MODEL)
    parser.add_argument("--task", choices=["recommend", "explain"], default="recommend")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RAG_TOP_K)
    parser.add_argument(
        "--max-evidences-per-candidate",
        type=int,
        default=DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_SILICONFLOW_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SILICONFLOW_TEMPERATURE)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_SILICONFLOW_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=DEFAULT_RAG_GENERATION_LIMIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    input_format: RagInputFormat = cast(RagInputFormat, args.input_format)
    task: RagTask = cast(RagTask, args.task)
    output_path: Path = (
        args.output
        if args.output is not None
        else build_default_generation_output_path(
            args.input,
            args.model,
            task,
            top_k=args.top_k,
            max_evidences_per_candidate=args.max_evidences_per_candidate,
        )
    )
    generate_records(
        GenerateConfig(
            input_path=args.input,
            output_path=output_path,
            input_format=input_format,
            model_name=args.model,
            task=task,
            top_k=args.top_k,
            max_evidences_per_candidate=args.max_evidences_per_candidate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()

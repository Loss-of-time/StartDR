"""TraceDR 精排结果导出命令行入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .core.io import write_jsonl
from .core.model.tracedr.data import build_model_sample
from .core.model.tracedr.model import HeterogeneousGNN
from .core.model.tracedr.train import RankedAnswer, build_ranked_answers
from .core.schema import DrugRecMedicine, RankedCase, RankedDrug, TraceDRSample, unstructure
from .core.setting import DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_RERANK_OUTPUT_DIR
from .core.tracedr import load_tracedr_samples


@dataclass(slots=True)
class ExportRankConfig:
    """TraceDR 精排结果导出配置。"""

    input_path: Path
    checkpoint_path: Path
    output_path: Path
    limit: int | None


def build_default_output_path(input_path: Path, checkpoint_path: Path) -> Path:
    """构造默认输出路径。

    Args:
        input_path: 待导出样本路径。
        checkpoint_path: TraceDR checkpoint 路径。

    Returns:
        默认输出文件路径。
    """

    return DEFAULT_RERANK_OUTPUT_DIR / f"{input_path.stem}__{checkpoint_path.stem}.jsonl"


def resolve_checkpoint_path(raw_path: Path) -> Path:
    """解析 checkpoint 输入路径。

    Args:
        raw_path: CLI 传入的 checkpoint 路径。

    Returns:
        可实际读取的 checkpoint 路径。
    """

    if raw_path.is_absolute():
        return raw_path
    if raw_path.exists():
        return raw_path
    # 目的：优先兼容用户直接传 `resource/model/...`，仅在本地不存在时再回退到默认模型目录。
    return DEFAULT_MODEL_OUTPUT_DIR / raw_path


def load_model(checkpoint_path: Path) -> HeterogeneousGNN:
    """加载 TraceDR checkpoint。

    Args:
        checkpoint_path: checkpoint 路径。

    Returns:
        已加载权重的 TraceDR 模型。
    """

    model: HeterogeneousGNN = HeterogeneousGNN()
    state_dict: dict[str, torch.Tensor] = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_ranked_drugs(
    sample: TraceDRSample,
    ranked_answers: list[RankedAnswer],
) -> list[RankedDrug]:
    """把 TraceDR 排序答案转成标准精排导出结构。

    Args:
        sample: 原始 TraceDR 样本。
        ranked_answers: 模型排序结果。

    Returns:
        仅包含候选药物空间的排序结果。
    """

    candidate_drug_map: dict[str, DrugRecMedicine] = sample.top_k_drugs
    retrieval_rank_map: dict[str, int] = {
        drugid: rank for rank, drugid in enumerate(sample.top_k_drugs.keys(), start=1)
    }
    gold_drugid_set: set[str] = {drug.drugid for drug in sample.people.medicine}
    ranked_drugs: list[RankedDrug] = []
    ranked_answer: RankedAnswer
    for ranked_answer in ranked_answers:
        if ranked_answer.id not in candidate_drug_map:
            continue
        drugid: str = ranked_answer.id
        ranked_drugs.append(
            RankedDrug(
                drugid=drugid,
                score=ranked_answer.score,
                rank=len(ranked_drugs) + 1,
                drug=sample.top_k_drugs[drugid],
                label=int(drugid in gold_drugid_set),
                retrieval_score=None,
                retrieval_rank=retrieval_rank_map[drugid],
            )
        )
    return ranked_drugs


def build_ranked_case(
    model: HeterogeneousGNN,
    sample: TraceDRSample,
) -> RankedCase:
    """对单个病例执行 TraceDR 精排导出。

    Args:
        model: 已加载权重的 TraceDR 模型。
        sample: 单个 TraceDR 样本。

    Returns:
        单病例精排结果。
    """

    model_sample = build_model_sample(sample, train=False)
    if model_sample is None:
        return RankedCase(
            patient_id=sample.people.id,
            split=sample.people.part,
            ranked_drugs=[],
        )
    with torch.no_grad():
        result = model(model_sample.to_cuda())
    ranked_answers: list[RankedAnswer] = build_ranked_answers(model_sample, result)
    return RankedCase(
        patient_id=sample.people.id,
        split=sample.people.part,
        ranked_drugs=build_ranked_drugs(sample, ranked_answers),
    )


def export_ranked_cases(config: ExportRankConfig) -> Path:
    """导出整份 TraceDR 精排结果。

    Args:
        config: 导出配置。

    Returns:
        输出文件路径。
    """

    model: HeterogeneousGNN = load_model(config.checkpoint_path)
    samples: list[TraceDRSample] = load_tracedr_samples(config.input_path, limit=config.limit)
    ranked_cases: list[RankedCase] = [build_ranked_case(model, sample) for sample in samples]
    # 目的：把 TraceDR 的病例级排序结果冻结成 jsonl 交接件，供 drrag 直接复用。
    write_jsonl(
        path=config.output_path,
        rows=ranked_cases,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"精排结果写出完成: {config.output_path.resolve()} rows={len(ranked_cases)}")
    return config.output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="使用 TraceDR checkpoint 导出病例级精排结果。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    checkpoint_path: Path = resolve_checkpoint_path(args.checkpoint)
    output_path: Path = (
        args.output
        if args.output is not None
        else build_default_output_path(args.input, checkpoint_path)
    )
    export_ranked_cases(
        ExportRankConfig(
            input_path=args.input,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()

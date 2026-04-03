import argparse
import json
import logging
from pathlib import Path
from typing import cast

from rich.progress import Progress

from .data.jsonl import load_jsonl
from .data.pkl import write_pickle
from .model.gnn.data_set import build_gnn_train_sample
from .schema.drugrec import DatasetSplit
from .schema.drugrec_task import (
    DrugRecCase,
    GNNIntermediateMeta,
    GNNTrainSample,
)
from .schema.patient_candidate_set import PatientCandidateSet
from .utils.log import get_console, setup_logging
from .utils.paths import RESOURCE_DIR

DEFAULT_INPUT_DIR = (
    RESOURCE_DIR / "patient_candidate" / "pyserini_bm25_top50"
)
DEFAULT_OUTPUT_ROOT = RESOURCE_DIR / "gnn_intermediate"
DEFAULT_CHUNK_SIZE = 64
LOGGER = logging.getLogger(__name__)


def build_drugrec_case(sample: PatientCandidateSet) -> DrugRecCase:
    """从冻结候选集样本提取训练所需病例字段。"""
    return {
        "patient_id": sample["patient_id"],
        "split": sample["split"],
        "patient": sample["patient"],
        "gold_drugids": sample["gold_drugids"],
        "candidate_drugs": sample["candidate_drugs"],
    }


def build_gnn_train_samples(
    cases: list[DrugRecCase],
) -> list[GNNTrainSample]:
    """批量构建 GNN 中间样本。"""
    return [build_gnn_train_sample(case) for case in cases]


def split_chunks(
    cases: list[DrugRecCase],
    chunk_size: int,
) -> list[list[DrugRecCase]]:
    """按固定 chunk size 切分病例列表。"""
    return [
        cases[index:index + chunk_size]
        for index in range(0, len(cases), chunk_size)
    ]


def write_meta(
    output_dir: Path,
    meta: GNNIntermediateMeta,
) -> None:
    """写出 GNN 中间文件元信息。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析 GNN 中间文件构建参数。"""
    parser = argparse.ArgumentParser(
        description="构建 GNN 训练中间文件。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """执行 GNN 中间文件构建流程。"""
    args = parse_args()
    input_path = args.input_dir / f"{args.split}.jsonl"
    output_dir = DEFAULT_OUTPUT_ROOT / args.input_dir.name / args.split

    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取冻结候选集: %s", input_path.resolve())
    patient_candidate_sets = load_jsonl(
        path=input_path,
        parse_line=lambda row: cast(PatientCandidateSet, row),
        limit=args.limit,
    )
    LOGGER.info("冻结候选集样本数: %s", len(patient_candidate_sets))
    cases = [
        build_drugrec_case(sample)
        for sample in patient_candidate_sets
    ]
    case_chunks = split_chunks(cases, args.chunk_size)
    LOGGER.info(
        "开始构建 GNN 中间样本: chunk_size=%s slots=%s",
        args.chunk_size,
        len(case_chunks),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    slot_names: list[str] = []
    with Progress(console=get_console()) as progress:
        task_id = progress.add_task("构建 GNN 中间文件", total=len(case_chunks))
        chunk_results = (
            build_gnn_train_samples(case_chunk)
            for case_chunk in case_chunks
        )
        for slot_index, gnn_train_samples in enumerate(chunk_results):
            slot_name = f"slot_{slot_index:05d}.pkl"
            slot_names.append(slot_name)
            write_pickle(output_dir / slot_name, gnn_train_samples)
            progress.advance(task_id)

    meta: GNNIntermediateMeta = {
        "split": cast(DatasetSplit, args.split),
        "sample_count": len(cases),
        "chunk_size": args.chunk_size,
        "source_path": str(input_path.resolve()),
        "slot_names": slot_names,
    }
    write_meta(output_dir, meta)
    LOGGER.info("写出完成: %s", output_dir.resolve())


if __name__ == "__main__":
    main()

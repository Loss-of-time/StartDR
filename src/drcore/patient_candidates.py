import argparse
import json
import logging
from pathlib import Path
from typing import cast

from rich.progress import Progress

from .data.jsonl import load_jsonl, write_jsonl
from .retrieval.registry import build_retriver
from .schema import (
    CandidateDrug,
    DatasetSplit,
    DrugRecMedicine,
    DrugRecRecord,
    PatientCandidateRetriever,
    PatientCandidateSet,
    PatientCandidateTopK,
    Retriver,
)
from .utils.kg import list_full_drug_details
from .utils.log import get_console, setup_logging
from .utils.paths import RESOURCE_DIR

DEFAULT_OUTPUT_DIR = (
    RESOURCE_DIR / "patient_candidate"
)
DEFAULT_INPUT_DIR = RESOURCE_DIR / "DrugRec0328"
DEFAULT_RETRIEVER_NAME: PatientCandidateRetriever = "pyserini_bm25"
DEFAULT_TOP_K: PatientCandidateTopK = 50
LOGGER = logging.getLogger(__name__)


def build_patient_candidate_set(
    patient: DrugRecRecord,
    split: DatasetSplit,
    retriever_name: PatientCandidateRetriever,
    top_k: PatientCandidateTopK,
    retriever: Retriver,
    drug_detail_map: dict[str, DrugRecMedicine],
) -> PatientCandidateSet:
    """生成单个患者的患者候选集样本。"""
    diagnosis = [item.strip() for item in patient["diagnosis"] if item.strip()]
    symptom = [item.strip() for item in patient["symptom"] if item.strip()]
    gold_drugids = list(
        dict.fromkeys(medicine["drugid"] for medicine in patient["medicine"])
    )
    gold_drugid_set = set(gold_drugids)
    retrieved = retriever.retrieve(patient, top_k=top_k)
    candidate_drugs: list[CandidateDrug] = []

    for rank, candidate in enumerate(retrieved, start=1):
        drugid = candidate["drugid"]
        candidate_drugs.append(
            {
                "drugid": drugid,
                "rank": rank,
                "score": candidate["score"],
                "drug": drug_detail_map[drugid],
                "is_gold": drugid in gold_drugid_set,
            }
        )

    sample: PatientCandidateSet = {
        "patient_id": patient["id"],
        "split": split,
        "retriever": retriever_name,
        "top_k": top_k,
        "retrieval_query": " ".join([*diagnosis, *symptom]),
        "patient": patient,
        "gold_drugids": gold_drugids,
        "candidate_drugs": candidate_drugs,
    }
    # _validate_patient_candidate_set(sample)
    return sample


def build_patient_candidate_sets(
    patients: list[DrugRecRecord],
    split: DatasetSplit,
    retriever_name: PatientCandidateRetriever,
    top_k: PatientCandidateTopK,
    retriever: Retriver,
    drug_detail_map: dict[str, DrugRecMedicine],
) -> list[PatientCandidateSet]:
    """批量生成一个 split 的患者候选集样本。"""
    samples: list[PatientCandidateSet] = []
    with Progress(console=get_console()) as progress:
        task_id = progress.add_task("生成患者候选集", total=len(patients))
        for patient in patients:
            samples.append(
                build_patient_candidate_set(
                    patient=patient,
                    split=split,
                    retriever_name=retriever_name,
                    top_k=top_k,
                    retriever=retriever,
                    drug_detail_map=drug_detail_map,
                )
            )
            progress.advance(task_id)
    return samples


# def _validate_patient_candidate_set(sample: PatientCandidateSet) -> None:
#     """校验单条患者候选集样本，不通过直接报错。"""
#     ...
###############################################################
# 命令行入口
###############################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成冻结后的患者候选集样本。",
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
    parser.add_argument(
        "--retriver",
        type=str,
        choices=["bm25", "pyserini_bm25"],
        default=DEFAULT_RETRIEVER_NAME,
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split
    retriever_name: PatientCandidateRetriever = args.retriver
    input_path = args.input_dir / f"{split}.jsonl"
    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"{retriever_name}_top{DEFAULT_TOP_K}"
        / f"{split}.jsonl"
    )

    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取患者数据: %s", input_path.resolve())
    patients = load_jsonl(
        path=input_path,
        parse_line=lambda row: cast(DrugRecRecord, row),
        limit=args.limit,
    )
    LOGGER.info("患者样本数: %s", len(patients))
    LOGGER.info("开始构建检索器: %s", retriever_name)
    retriever = build_retriver(retriever_name)
    LOGGER.info("开始加载全量药品详情")
    drug_detail_map = {
        detail["drugid"]: detail
        for detail in list_full_drug_details()
    }
    LOGGER.info("开始生成患者候选集，top_k=%s", DEFAULT_TOP_K)
    samples = build_patient_candidate_sets(
        patients=patients,
        split=split,
        retriever_name=retriever_name,
        top_k=DEFAULT_TOP_K,
        retriever=retriever,
        drug_detail_map=drug_detail_map,
    )
    write_jsonl(
        path=output_path,
        rows=samples,
        serialize_row=lambda row: json.dumps(row, ensure_ascii=False),
    )
    LOGGER.info("写出完成: %s", output_path.resolve())


__all__ = [
    "build_patient_candidate_set",
    "build_patient_candidate_sets",
]


if __name__ == "__main__":
    main()

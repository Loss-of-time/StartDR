import argparse
import json
import logging
from pathlib import Path

from rich.progress import Progress

from .input_process import load_jsonl_limit
from .kg import list_full_drug_details
from .retrieval import build_retriver
from .schema import (
    CandidateDrug,
    DatasetSplit,
    DrugRecMedicine,
    DrugRecRecord,
    PatientCandidateRetriever,
    PatientCandidateSet,
    PatientCandidateTopK,
    RetrievedDrugCandidate,
    Retriver,
)
from .utils.log import get_console, setup_logging

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "data" / "patient_candidate"
)
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data" / "DrugRec0328"
DEFAULT_RETRIEVER_NAME: PatientCandidateRetriever = "pyserini_bm25"
DEFAULT_TOP_K: PatientCandidateTopK = 50
LOGGER = logging.getLogger("MINE.patient_candidates")


def build_retrieval_query(patient: DrugRecRecord) -> str:
    """按固定规则生成落盘用检索 query。"""
    diagnosis = [item.strip() for item in patient["diagnosis"] if item.strip()]
    symptom = [item.strip() for item in patient["symptom"] if item.strip()]
    return " ".join([*diagnosis, *symptom])


def build_patient_candidate_set(
    patient: DrugRecRecord,
    split: DatasetSplit,
    retriever_name: PatientCandidateRetriever,
    top_k: PatientCandidateTopK,
    retriever: Retriver,
    drug_detail_map: dict[str, DrugRecMedicine],
) -> PatientCandidateSet:
    """生成单个患者的患者候选集样本。"""
    retrieval_query = build_retrieval_query(patient)
    gold_drugids = _get_gold_drugids(patient)
    retrieved = retriever.retrieve(patient, top_k=top_k)
    sample: PatientCandidateSet = {
        "patient_id": patient["id"],
        "split": split,
        "retriever": retriever_name,
        "top_k": top_k,
        "retrieval_query": retrieval_query,
        "patient": patient,
        "gold_drugids": gold_drugids,
        "candidate_drugs": _build_candidate_drugs(
            retrieved=retrieved,
            drug_detail_map=drug_detail_map,
            gold_drugids=set(gold_drugids),
        ),
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


###############################################################
# 工具函数
###############################################################


def write_patient_candidate_sets(
    samples: list[PatientCandidateSet],
    output_path: Path,
) -> None:
    """按 jsonl 写出患者候选集样本。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False))
            file.write("\n")


def _get_gold_drugids(patient: DrugRecRecord) -> list[str]:
    """返回当前患者的金标准药品 ID 列表。"""
    return list(
        dict.fromkeys(medicine["drugid"] for medicine in patient["medicine"])
    )


def _build_candidate_drugs(
    retrieved: list[RetrievedDrugCandidate],
    drug_detail_map: dict[str, DrugRecMedicine],
    gold_drugids: set[str],
) -> list[CandidateDrug]:
    """把召回结果补全成冻结候选药列表。"""
    candidate_drugs: list[CandidateDrug] = []

    for rank, candidate in enumerate(retrieved, start=1):
        drugid = candidate["drugid"]
        candidate_drugs.append(
            {
                "drugid": drugid,
                "rank": rank,
                "score": candidate["score"],
                "drug": drug_detail_map[drugid],
                "is_gold": drugid in gold_drugids,
            }
        )

    return candidate_drugs


# def _validate_patient_candidate_set(sample: PatientCandidateSet) -> None:
#     """校验单条患者候选集样本，不通过直接报错。"""
#     ...


def _build_output_path(
    retriever_name: PatientCandidateRetriever,
    top_k: PatientCandidateTopK,
    split: DatasetSplit,
) -> Path:
    """构造患者候选集输出路径。"""
    return (
        DEFAULT_OUTPUT_DIR / f"{retriever_name}_top{top_k}" / f"{split}.jsonl"
    )


def _build_drug_detail_map() -> dict[str, DrugRecMedicine]:
    return {detail["drugid"]: detail for detail in list_full_drug_details()}


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
    output_path = _build_output_path(
        retriever_name=retriever_name,
        top_k=DEFAULT_TOP_K,
        split=split,
    )

    log_path = setup_logging()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("开始读取患者数据: %s", input_path.resolve())
    patients = load_jsonl_limit(input_path, args.limit)
    LOGGER.info("患者样本数: %s", len(patients))
    LOGGER.info("开始构建检索器: %s", retriever_name)
    retriever = build_retriver(retriever_name)
    LOGGER.info("开始加载全量药品详情")
    drug_detail_map = _build_drug_detail_map()
    LOGGER.info("开始生成患者候选集，top_k=%s", DEFAULT_TOP_K)
    samples = build_patient_candidate_sets(
        patients=patients,
        split=split,
        retriever_name=retriever_name,
        top_k=DEFAULT_TOP_K,
        retriever=retriever,
        drug_detail_map=drug_detail_map,
    )
    write_patient_candidate_sets(samples, output_path)
    LOGGER.info("写出完成: %s", output_path.resolve())


__all__ = [
    "build_patient_candidate_set",
    "build_patient_candidate_sets",
    "build_retrieval_query",
    "write_patient_candidate_sets",
]


if __name__ == "__main__":
    main()

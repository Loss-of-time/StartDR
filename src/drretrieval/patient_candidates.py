import argparse
import json
from pathlib import Path
from typing import cast

from tqdm import tqdm

from .core.io import load_jsonl, write_jsonl
from .core.kg import list_full_drug_details
from .core.retrieval import (
    build_retriever,
    get_retriever_names,
)
from .core.schema import (
    DatasetSplit,
    DrugRecMedicine,
    DrugRecRecord,
    PatientCandidateRetriever,
    Retriever,
    TraceDRSample,
    structure,
    unstructure,
)
from .core.setting import (
    DEFAULT_PATIENT_CANDIDATE_OUTPUT_DIR,
    DEFAULT_PATIENT_INPUT_DIR,
    DEFAULT_RETRIEVER_NAME,
    DEFAULT_TOP_K,
)


def build_tracedr_sample(
    patient: DrugRecRecord,
    top_k: int,
    retriever: Retriever,
    drug_detail_map: dict[str, DrugRecMedicine],
) -> TraceDRSample:
    retrieved = retriever.retrieve(patient, top_k=top_k)
    top_k_drugs: dict[str, DrugRecMedicine] = {}
    for candidate in retrieved:
        drugid = candidate.drugid
        if drugid in top_k_drugs:
            continue
        top_k_drugs[drugid] = drug_detail_map[drugid]
    return TraceDRSample(
        people=patient,
        top_k_drugs=top_k_drugs,
    )


def build_tracedr_samples(
    patients: list[DrugRecRecord],
    top_k: int,
    retriever: Retriever,
    drug_detail_map: dict[str, DrugRecMedicine],
) -> list[TraceDRSample]:
    return [
        build_tracedr_sample(
            patient=patient,
            top_k=top_k,
            retriever=retriever,
            drug_detail_map=drug_detail_map,
        )
        for patient in tqdm(patients, desc="生成候选集")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成冻结后的患者候选集样本。")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_PATIENT_INPUT_DIR)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        required=True,
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=get_retriever_names(),
        default=DEFAULT_RETRIEVER_NAME,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = cast(DatasetSplit, args.split)
    retriever_name = cast(PatientCandidateRetriever, args.retriever)
    input_path = args.input_dir / f"{split}.jsonl"
    output_path = (
        DEFAULT_PATIENT_CANDIDATE_OUTPUT_DIR
        / f"{retriever_name}_top{args.top_k}"
        / f"{split}.jsonl"
    )
    print(f"开始读取患者数据: {input_path.resolve()}")
    patients = load_jsonl(
        path=input_path,
        parse_line=lambda row: structure(row, DrugRecRecord),
        limit=args.limit,
    )
    print(f"患者样本数: {len(patients)}")
    print(f"开始构建检索器: {retriever_name}")
    retriever = build_retriever(retriever_name)
    print("开始加载全量药品详情")
    drug_detail_map = {
        detail.drugid: detail
        for detail in list_full_drug_details()
    }
    print(f"开始生成候选集，top_k={args.top_k}")
    samples = build_tracedr_samples(
        patients=patients,
        top_k=args.top_k,
        retriever=retriever,
        drug_detail_map=drug_detail_map,
    )
    write_jsonl(
        path=output_path,
        rows=samples,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"写出完成: {output_path.resolve()}")


if __name__ == "__main__":
    main()

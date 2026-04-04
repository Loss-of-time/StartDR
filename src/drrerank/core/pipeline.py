import json
from pathlib import Path
from typing import cast

from .gnn import build_gnn_train_sample
from .io import load_jsonl, load_pickle, write_pickle
from .schema import (
    DatasetSplit,
    DrugRecCase,
    GNNIntermediateMeta,
    GNNTrainSample,
    PatientCandidateSet,
    structure,
    unstructure,
)
from .setting import DEFAULT_DATA_FILE


def build_drugrec_case(sample: PatientCandidateSet) -> DrugRecCase:
    return DrugRecCase(
        patient_id=sample.patient_id,
        split=sample.split,
        patient=sample.patient,
        gold_drugids=sample.gold_drugids,
        candidate_drugs=sample.candidate_drugs,
    )


def build_gnn_train_samples(
    cases: list[DrugRecCase],
) -> list[GNNTrainSample]:
    return [build_gnn_train_sample(case) for case in cases]


def write_meta(
    output_dir: Path,
    meta: GNNIntermediateMeta,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(unstructure(meta), file, ensure_ascii=False, indent=2)


def write_train_samples(
    output_dir: Path,
    input_path: Path,
    split: DatasetSplit,
    samples: list[GNNTrainSample],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_path in output_dir.glob("*.pkl"):
        data_path.unlink()
    write_pickle(output_dir / DEFAULT_DATA_FILE, samples)
    write_meta(
        output_dir,
        GNNIntermediateMeta(
            split=split,
            sample_count=len(samples),
            source_path=str(input_path.resolve()),
            data_file=DEFAULT_DATA_FILE,
        ),
    )


def load_candidate_sets(
    input_path: Path,
    limit: int | None = None,
) -> list[PatientCandidateSet]:
    return load_jsonl(
        path=input_path,
        parse_line=lambda row: structure(row, PatientCandidateSet),
        limit=limit,
    )


def load_train_samples(
    input_dir: Path,
    limit: int | None = None,
) -> list[GNNTrainSample]:
    meta_path = input_dir / "meta.json"
    with meta_path.open(encoding="utf-8") as file:
        meta = cast(GNNIntermediateMeta, structure(json.load(file), GNNIntermediateMeta))
    data_file = meta.data_file
    samples = cast(
        list[GNNTrainSample],
        structure(load_pickle(input_dir / data_file), list[GNNTrainSample]),
    )
    if limit is not None:
        return samples[:limit]
    return samples

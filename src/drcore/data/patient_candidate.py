import json
from pathlib import Path
from typing import cast

from ..schema import PatientCandidateSet
from .jsonl import load_jsonl, write_jsonl


def load_patient_candidate_sets(
    path: Path,
    limit: int | None = None,
) -> list[PatientCandidateSet]:
    return load_jsonl(
        path=path,
        parse_line=lambda row: cast(PatientCandidateSet, row),
        limit=limit,
    )


def write_patient_candidate_sets(
    path: Path,
    samples: list[PatientCandidateSet],
) -> None:
    write_jsonl(
        path=path,
        rows=samples,
        serialize_row=lambda row: json.dumps(row, ensure_ascii=False),
    )

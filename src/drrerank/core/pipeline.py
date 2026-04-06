from pathlib import Path

from .gnn import build_gnn_train_sample
from .schema import GNNTrainSample
from .tracedr import load_tracedr_cases


def load_train_samples(
    input_path: Path,
    limit: int | None = None,
) -> list[GNNTrainSample]:
    return [
        build_gnn_train_sample(case)
        for case in load_tracedr_cases(input_path, limit)
    ]

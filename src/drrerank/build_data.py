import argparse
from pathlib import Path
from typing import cast

from tqdm import tqdm

from .core.pipeline import (
    build_drugrec_case,
    build_gnn_train_samples,
    load_candidate_sets,
    write_train_samples,
)
from .core.schema import DatasetSplit
from .core.setting import (
    DEFAULT_GNN_DATA_INPUT_DIR,
    DEFAULT_GNN_DATA_OUTPUT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 GNN 训练中间文件。")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_GNN_DATA_INPUT_DIR)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = cast(DatasetSplit, args.split)
    input_path = args.input_dir / f"{split}.jsonl"
    output_dir = DEFAULT_GNN_DATA_OUTPUT_ROOT / args.input_dir.name / split
    print(f"开始读取冻结候选集: {input_path.resolve()}")
    patient_candidate_sets = load_candidate_sets(input_path, args.limit)
    print(f"冻结候选集样本数: {len(patient_candidate_sets)}")
    cases = [
        build_drugrec_case(sample)
        for sample in tqdm(patient_candidate_sets, desc="整理病例")
    ]
    print("开始构建 GNN 中间样本")
    samples = build_gnn_train_samples(cases)
    write_train_samples(output_dir, input_path, split, samples)
    print(f"写出完成: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import cast

from .core.io import write_jsonl
from .core.schema import DatasetSplit, unstructure
from .core.setting import DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT, RESOURCE_DIR
from .core.tracedr import (
    build_tracedr_sample_from_raw,
    load_raw_tracedr_dataset,
)

DEFAULT_TRACEDR_INPUT_DIR = RESOURCE_DIR / "DrugRec0716_from_traceDR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 TraceDR 原始 pkl 规范化为 TraceDR 风格 jsonl。"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_TRACEDR_INPUT_DIR)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        required=True,
    )
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = cast(DatasetSplit, args.split)
    input_path = args.input_dir / f"{split}.pkl"
    print(f"开始读取 TraceDR 候选集: {input_path.resolve()}")
    raw_dataset = load_raw_tracedr_dataset(input_path)
    if args.limit is None:
        selected_items = list(raw_dataset.items())
    else:
        selected_items = list(raw_dataset.items())[:args.limit]
    if not selected_items:
        raise ValueError("输入数据为空，无法执行转换。")
    inferred_top_k = max(len(sample["top_k_drugs"]) for _, sample in selected_items)
    output_name = args.output_name or f"tracedr_top{inferred_top_k}"
    output_path = DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT / output_name / f"{split}.jsonl"
    print(f"样本数: {len(selected_items)}")
    print(f"推断 top_k: {inferred_top_k}")
    samples = [
        build_tracedr_sample_from_raw(raw_sample, split)
        for _, raw_sample in selected_items
    ]
    write_jsonl(
        path=output_path,
        rows=samples,
        serialize_row=lambda row: json.dumps(unstructure(row), ensure_ascii=False),
    )
    print(f"写出完成: {output_path.resolve()}")


if __name__ == "__main__":
    main()

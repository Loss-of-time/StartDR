"""TraceDR 原始 `pkl` 候选集导入命令行入口。"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .core.io import write_jsonl
from .core.schema import DatasetSplit, TraceDRSample, unstructure
from .core.setting import DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT, RESOURCE_DIR
from .core.tracedr import (
    RawTraceDRDataset,
    RawTraceDRRecord,
    build_tracedr_sample_from_raw,
    load_raw_tracedr_dataset,
)

DEFAULT_TRACEDR_PKL_ROOT = RESOURCE_DIR / "DrugRec0716_from_traceDR"


@dataclass(slots=True)
class ImportTraceDRConfig:
    """TraceDR `pkl` 导入配置。"""

    split: DatasetSplit
    input_root: Path = DEFAULT_TRACEDR_PKL_ROOT
    output_root: Path = DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT / "tracedr_top50"


def build_input_path(input_root: Path, split: DatasetSplit) -> Path:
    """构造单个 split 的输入路径。"""

    return input_root / f"{split}.pkl"


def build_output_path(output_root: Path, split: DatasetSplit) -> Path:
    """构造单个 split 的输出路径。"""

    return output_root / f"{split}.jsonl"


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    """构造样本排序键。"""

    if sample_id.isdigit():
        return int(sample_id), sample_id
    return 10**18, sample_id


def convert_raw_dataset(
    raw_dataset: RawTraceDRDataset,
    split: DatasetSplit,
) -> list[TraceDRSample]:
    """把 TraceDR 原始映射转换成标准 `jsonl` 样本列表。"""

    sample_items: list[tuple[str, RawTraceDRRecord]] = sorted(
        ((str(sample_id), raw_sample) for sample_id, raw_sample in raw_dataset.items()),
        key=lambda item: _sample_sort_key(item[0]),
    )
    samples: list[TraceDRSample] = []
    raw_sample: RawTraceDRRecord
    for _, raw_sample in sample_items:
        samples.append(build_tracedr_sample_from_raw(raw_sample, split))
    return samples


def import_tracedr_samples(config: ImportTraceDRConfig) -> Path:
    """导入单个 split 的 TraceDR `pkl` 候选集。"""

    input_path: Path = build_input_path(config.input_root, config.split)
    output_path: Path = build_output_path(config.output_root, config.split)
    print(f"开始读取 TraceDR pkl: {input_path.resolve()}")
    raw_dataset: RawTraceDRDataset = load_raw_tracedr_dataset(input_path)
    samples: list[TraceDRSample] = convert_raw_dataset(raw_dataset, config.split)
    # 目的：统一输出到项目约定的 `jsonl` 结构，保证检索侧与导入侧产物完全同构。
    write_jsonl(
        output_path,
        samples,
        serialize_row=lambda sample: json.dumps(unstructure(sample), ensure_ascii=False),
    )
    print(f"写出完成: {output_path.resolve()} rows={len(samples)}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="把 TraceDR 原始 pkl 候选集转换成项目统一的 jsonl。"
    )
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_TRACEDR_PKL_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT / "tracedr_top50",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    import_tracedr_samples(
        ImportTraceDRConfig(
            split=args.split,
            input_root=args.input_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()

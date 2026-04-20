"""4SDrug 离线导出命令行入口。"""

import argparse
from pathlib import Path
from typing import cast

from .core.model.foursdrug.export import FourSDrugExportConfig, export_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="从 TraceDR 风格 jsonl 导出 4SDrug 所需离线文件。",
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    export_dataset(
        FourSDrugExportConfig(
            train_input=args.train_input,
            dev_input=args.dev_input,
            test_input=args.test_input,
            output_dir=args.output_dir,
            batch_sizes=cast(list[int], args.batch_sizes),
        )
    )


if __name__ == "__main__":
    main()

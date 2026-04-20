"""KGDNet 训练命令行入口。"""

import argparse
from pathlib import Path

from .core.model.kgd.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="训练 KGD rerank 模型。")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--selection-metric", type=str, default="mrr")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    train(
        TrainConfig(
            input_dir=args.input_dir,
            output_name=args.output_name,
            epochs=args.epochs,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
            test_limit=args.test_limit,
            selection_metric=args.selection_metric,
        )
    )


if __name__ == "__main__":
    main()

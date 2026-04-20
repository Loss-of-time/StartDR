"""TraceDR 训练命令行入口。"""

import argparse
from pathlib import Path

from .core.model.tracedr.train import TrainConfig, train
from .core.setting import DEFAULT_TRACEDR_DEV_INPUT_PATH, DEFAULT_TRACEDR_TRAIN_INPUT_PATH


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="训练 TraceDR rerank 模型。"
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_TRACEDR_DEV_INPUT_PATH)
    parser.add_argument("--test-input", type=Path, default=None)
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
            train_input=args.train_input,
            dev_input=args.dev_input,
            test_input=args.test_input,
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

"""4SDrug 训练命令行入口。"""

import argparse
from pathlib import Path

from .core.model.foursdrug.schema import FourSDrugTrainConfig
from .core.model.foursdrug.train import train


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="训练 4SDrug `main1` 模型。"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-threshold", type=float, default=0.8)
    parser.add_argument("--selection-metric", type=str, default="ja")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args: argparse.Namespace = parse_args()
    train(
        FourSDrugTrainConfig(
            input_dir=args.input_dir,
            output_name=args.output_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
            test_limit=args.test_limit,
            eval_threshold=args.eval_threshold,
            selection_metric=args.selection_metric,
        ),
    )


if __name__ == "__main__":
    main()

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
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--disable-evidence-supervision",
        action="store_true",
        help="仅保留 answer_loss，去掉 evidence_loss。",
    )
    parser.add_argument(
        "--evidence-text-mode",
        type=str,
        choices=("full", "name_only"),
        default="full",
        help="证据文本模式：full=完整属性串，name_only=仅药名。",
    )
    parser.add_argument(
        "--exclude-on-medicine",
        action="store_true",
        help="构图时移除当前在用药，仅保留 top_k_drugs。",
    )
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
            # 目的：把关键消融参数显式写入训练配置，避免实验只靠 output_name 约定。
            num_layers=args.num_layers,
            use_evidence_supervision=not args.disable_evidence_supervision,
            evidence_text_mode=args.evidence_text_mode,
            include_on_medicine=not args.exclude_on_medicine,
        )
    )


if __name__ == "__main__":
    main()

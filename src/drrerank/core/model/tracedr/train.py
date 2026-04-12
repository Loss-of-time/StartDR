import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from ...setting import DEFAULT_TRACEDR_TRAIN_INPUT_PATH
from ...tracedr import load_tracedr_samples
from .data import build_model_sample
from .model import HeterogeneousGNN, TraceDRForwardResult


@dataclass(slots=True)
class TrainConfig:
    train_input: Path
    epochs: int
    limit: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TraceDR rerank 模型。")
    parser.add_argument(
        "--train-input",
        type=Path,
        default=DEFAULT_TRACEDR_TRAIN_INPUT_PATH,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    return parser.parse_args()


def train(config: TrainConfig) -> None:
    samples = load_tracedr_samples(
        config.train_input,
        limit=config.limit,
    )
    samples = [
        model_sample
        for sample in samples
        if (model_sample := build_model_sample(sample, train=True)) is not None
    ]

    model : HeterogeneousGNN = HeterogeneousGNN()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )

    total_steps = config.epochs * len(samples)

    for epoch in range(config.epochs):
        losses : list[float] = []
        with tqdm(
            samples,
            desc=f"训练 epoch {epoch + 1}/{config.epochs}",
            leave=False,
        ) as progress:
            for sample_idx, sample in enumerate(progress, start=1):
                sample = sample.to_cuda()
                optimizer.zero_grad(set_to_none=True)
                result : TraceDRForwardResult = model(sample)
                result.loss.backward()
                optimizer.step()
                loss = float(result.loss.item())
                losses.append(loss)
                global_step = epoch * len(samples) + sample_idx
                progress.set_postfix_str(
                    f"step={global_step}/{total_steps} loss={loss:.6f}"
                )

        ave_loss = sum(losses) / len(losses)
        print(f"epoch={epoch + 1} train_loss={ave_loss:.6f}")


def main() -> None:
    args = parse_args()
    train(
        TrainConfig(
            train_input=args.train_input,
            epochs=args.epochs,
            limit=args.train_limit,
        )
    )


if __name__ == "__main__":
    main()

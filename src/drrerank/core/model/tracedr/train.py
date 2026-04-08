import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from tqdm import tqdm

from ...metrics import get_mrr, get_precision_at_k
from ...setting import (
    DEFAULT_DEV_INPUT_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_TRAIN_INPUT_PATH,
)
from ...tracedr import load_tracedr_samples
from .data import ContinueWithNext, build_model_sample
from .model import HeterogeneousGNN, TraceDRForwardResult
from .schema import TraceDRModelSample


@dataclass(slots=True)
class TraceDREvalResult:
    loss: float
    entity_accuracy: float
    p_at_1: float
    mrr: float
    evidence_mrr: float
    evidence_hit_at_5: float


@dataclass(slots=True)
class TraceDRTrainEpochResult:
    epoch: int
    train_loss: float
    dev_loss: float
    dev_entity_accuracy: float
    dev_p_at_1: float
    dev_mrr: float
    dev_evidence_mrr: float
    dev_evidence_hit_at_5: float


@dataclass(slots=True)
class TraceDRTrainReport:
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    checkpoint_path: str
    epochs: list[TraceDRTrainEpochResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TraceDR 模型。")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT_PATH)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT_PATH)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--emb-dimension", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-entities", type=int, default=100)
    parser.add_argument("--max-evidences", type=int, default=50)
    return parser.parse_args()


def iter_batches[T](items: list[T], batch_size: int) -> list[list[T]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_samples(
    input_path: Path,
    limit: int | None,
    *,
    train: bool,
    max_entities: int,
    max_evidences: int,
) -> list[TraceDRModelSample]:
    tracedr_samples = load_tracedr_samples(input_path, limit=limit)
    model_samples: list[TraceDRModelSample] = []
    skipped_sample_count = 0
    for tracedr_sample in tracedr_samples:
        try:
            model_samples.append(
                build_model_sample(
                    tracedr_sample,
                    train=train,
                    max_entities=max_entities,
                    max_evidences=max_evidences,
                )
            )
        except ContinueWithNext:
            skipped_sample_count += 1
    if skipped_sample_count:
        print(f"跳过样本数: {skipped_sample_count}")
    return model_samples


def move_sample_to_device(
    sample: TraceDRModelSample,
    device: torch.device,
) -> TraceDRModelSample:
    return replace(
        sample,
        entity_mask=sample.entity_mask.to(device),
        evidence_mask=sample.evidence_mask.to(device),
        ent_to_ev=sample.ent_to_ev.to(device),
        ev_to_ent=sample.ev_to_ent.to(device),
        entity_labels=sample.entity_labels.to(device),
        evidence_labels=sample.evidence_labels.to(device),
    )


def rank_drugids(
    sample: TraceDRModelSample,
    entity_logits: torch.Tensor,
) -> list[str]:
    score_map: dict[str, float] = {}
    valid_entity_count = int(sample.entity_mask.sum().item())
    for entity_index in range(valid_entity_count):
        entity = sample.entities[entity_index]
        if entity.type != "药品" or entity.id == "":
            continue
        drugid = str(entity.id)
        score = float(entity_logits[entity_index].detach().cpu().item())
        previous_score = score_map.get(drugid)
        if previous_score is None or score > previous_score:
            score_map[drugid] = score
    ranked_items = sorted(
        score_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [drugid for drugid, _ in ranked_items]


def get_evidence_metrics(
    sample: TraceDRModelSample,
    evidence_logits: torch.Tensor,
) -> tuple[float, float]:
    valid_evidence_count = int(sample.evidence_mask.sum().item())
    scored_labels = [
        (
            int(sample.evidence_labels[evidence_index].item()),
            float(evidence_logits[evidence_index].detach().cpu().item()),
        )
        for evidence_index in range(valid_evidence_count)
    ]
    scored_labels.sort(key=lambda item: item[1], reverse=True)

    evidence_mrr = 0.0
    for rank, (label, _) in enumerate(scored_labels, start=1):
        if label == 1:
            evidence_mrr = 1.0 / rank
            break

    evidence_hit_at_5 = (
        1.0 if any(label == 1 for label, _ in scored_labels[:5]) else 0.0
    )
    return evidence_mrr, evidence_hit_at_5


def evaluate_model(
    model: HeterogeneousGNN,
    samples: list[TraceDRModelSample],
    batch_size: int,
    device: torch.device,
) -> TraceDREvalResult:
    losses: list[float] = []
    accuracies: list[float] = []
    p_at_1_scores: list[float] = []
    mrr_scores: list[float] = []
    evidence_mrr_scores: list[float] = []
    evidence_hit_at_5_scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch_samples in tqdm(
            iter_batches(samples, batch_size),
            desc="验证",
            leave=False,
        ):
            for raw_sample in batch_samples:
                sample = move_sample_to_device(raw_sample, device)
                result = model(sample)
                ranked_drugids = rank_drugids(sample, result.entity_logits)
                gold_drugids = set(sample.gold_answers)
                evidence_mrr, evidence_hit_at_5 = get_evidence_metrics(
                    sample,
                    result.evidence_logits,
                )

                losses.append(float(result.loss.detach().item()))
                accuracies.append(result.entity_accuracy)
                p_at_1_scores.append(
                    get_precision_at_k(gold_drugids, ranked_drugids[:1])
                )
                mrr_scores.append(get_mrr(gold_drugids, ranked_drugids))
                evidence_mrr_scores.append(evidence_mrr)
                evidence_hit_at_5_scores.append(evidence_hit_at_5)

    return TraceDREvalResult(
        loss=mean(losses),
        entity_accuracy=mean(accuracies),
        p_at_1=mean(p_at_1_scores),
        mrr=mean(mrr_scores),
        evidence_mrr=mean(evidence_mrr_scores),
        evidence_hit_at_5=mean(evidence_hit_at_5_scores),
    )


def train_batch(
    model: HeterogeneousGNN,
    batch_samples: list[TraceDRModelSample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    batch_results: list[TraceDRForwardResult] = []
    for raw_sample in batch_samples:
        sample = move_sample_to_device(raw_sample, device)
        batch_results.append(model(sample))

    batch_loss = torch.stack([result.loss for result in batch_results]).mean()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(batch_loss.detach().item())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TraceDR 当前实现只支持 CUDA。")

    set_seed(args.seed)
    random_state = random.Random(args.seed)

    print(f"开始读取训练样本: {args.train_input.resolve()}")
    train_samples = load_model_samples(
        args.train_input,
        args.train_limit,
        train=True,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
    )
    print(f"训练样本数: {len(train_samples)}")

    print(f"开始读取验证样本: {args.dev_input.resolve()}")
    dev_samples = load_model_samples(
        args.dev_input,
        args.dev_limit,
        train=False,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
    )
    print(f"验证样本数: {len(dev_samples)}")

    if not train_samples:
        raise ValueError("训练集为空，无法执行训练。")
    if not dev_samples:
        raise ValueError("验证集为空，无法选择最佳 checkpoint。")

    model = HeterogeneousGNN(
        emb_dimension=args.emb_dimension,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_entities=args.max_entities,
        max_evidences=args.max_evidences,
    )
    device = model.encoder.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = DEFAULT_MODEL_OUTPUT_DIR / "tracedr"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.output_name}.pt"
    report_path = output_dir / f"{args.output_name}.json"

    epoch_results: list[TraceDRTrainEpochResult] = []
    best_epoch = 1
    best_metric_value = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_samples = list(train_samples)
        random_state.shuffle(epoch_samples)
        model.train()

        batch_losses: list[float] = []
        for batch_samples in tqdm(
            iter_batches(epoch_samples, args.batch_size),
            desc=f"训练 epoch {epoch}",
        ):
            batch_loss = train_batch(
                model,
                batch_samples,
                optimizer,
                device,
                args.max_grad_norm,
            )
            batch_losses.append(batch_loss)

        train_loss = mean(batch_losses)
        dev_result = evaluate_model(
            model,
            dev_samples,
            args.batch_size,
            device,
        )
        epoch_results.append(
            TraceDRTrainEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                dev_loss=dev_result.loss,
                dev_entity_accuracy=dev_result.entity_accuracy,
                dev_p_at_1=dev_result.p_at_1,
                dev_mrr=dev_result.mrr,
                dev_evidence_mrr=dev_result.evidence_mrr,
                dev_evidence_hit_at_5=dev_result.evidence_hit_at_5,
            )
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"dev_loss={dev_result.loss:.6f} "
            f"dev_p_at_1={dev_result.p_at_1:.4f} "
            f"dev_mrr={dev_result.mrr:.4f}"
        )

        if dev_result.p_at_1 > best_metric_value:
            best_epoch = epoch
            best_metric_value = dev_result.p_at_1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "selection_metric": "p_at_1",
                    "selection_value": dev_result.p_at_1,
                },
                checkpoint_path,
            )
            print(f"已更新最佳 checkpoint: {checkpoint_path.resolve()}")

    report = TraceDRTrainReport(
        best_epoch=best_epoch,
        best_metric_name="p_at_1",
        best_metric_value=best_metric_value,
        checkpoint_path=str(checkpoint_path.resolve()),
        epochs=epoch_results,
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(report), file, ensure_ascii=False, indent=2)
    print(f"训练完成，报告已写入: {report_path.resolve()}")


if __name__ == "__main__":
    main()

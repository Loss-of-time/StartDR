"""4SDrug `main1` 训练入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import dill
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import Tensor
from tqdm import tqdm

from ...io import load_pickle
from ..experiment.runner import ExperimentAdapter, run_training_experiment
from ..experiment.schema import ComparableMetrics, ExperimentEvalResult
from .export import build_batched_training_data
from .metrics import aggregate_metrics, calculate_metrics
from .model import FourSDrugModel
from .schema import (
    FourSDrugForwardResult,
    FourSDrugInputPaths,
    FourSDrugLoadedData,
    FourSDrugMetrics,
    FourSDrugModelConfig,
    FourSDrugTrainBatch,
    FourSDrugTrainConfig,
    FourSDrugVocabulary,
    FourSDrugVocFile,
)

type FourSDrugSnapshot = dict[str, Tensor]


@dataclass(slots=True)
class FourSDrugTrainState:
    """4SDrug 统一 runner 需要的状态。"""

    config: FourSDrugTrainConfig
    model: FourSDrugModel
    optimizer: torch.optim.Optimizer
    loaded_data: FourSDrugLoadedData
    device: torch.device


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


def build_input_paths(input_dir: Path, batch_size: int) -> FourSDrugInputPaths:
    """构造 4SDrug 输入路径集合。"""

    return FourSDrugInputPaths(
        input_dir=input_dir,
        voc_final=input_dir / "voc_final.pkl",
        data_train=input_dir / "data_train.pkl",
        data_eval=input_dir / "data_eval.pkl",
        data_test=input_dir / "data_test.pkl",
        ddi_A_final=input_dir / "ddi_A_final.pkl",
        sym_train=input_dir / f"sym_train_{batch_size}.pkl",
        drug_train=input_dir / f"drug_train_{batch_size}.pkl",
    )


def load_vocabulary(voc_path: Path) -> FourSDrugVocFile:
    """加载 4SDrug 词表。"""

    with voc_path.open("rb") as file:
        raw_voc: dict[str, object] = cast(dict[str, object], dill.load(file))

    def build_vocab_entry(name: str) -> FourSDrugVocabulary:
        raw_entry: dict[str, object] = cast(dict[str, object], raw_voc[name])
        return FourSDrugVocabulary(
            word2idx=cast(dict[str, int], raw_entry["word2idx"]),
            idx2word=cast(list[str], raw_entry["idx2word"]),
        )

    return FourSDrugVocFile(
        sym_voc=build_vocab_entry("sym_voc"),
        diag_voc=build_vocab_entry("diag_voc"),
        med_voc=build_vocab_entry("med_voc"),
    )


def load_split_rows(split_path: Path, limit: int | None) -> list[list[list[int]]]:
    """加载单个 split 的索引病例。"""

    with split_path.open("rb") as file:
        rows: list[list[list[int]]] = cast(list[list[list[int]]], dill.load(file))
    if limit is None:
        return rows
    return rows[:limit]


def build_similarity_indices(symptom_batches: list[list[list[int]]]) -> list[list[int]]:
    """按 batch 内症状重叠数构造最相似样本索引。"""

    similarity_indices: list[list[int]] = []
    symptom_batch: list[list[int]]
    for symptom_batch in symptom_batches:
        if not symptom_batch:
            similarity_indices.append([])
            continue
        if len(symptom_batch[0]) <= 2:
            similarity_indices.append(list(range(len(symptom_batch))))
            continue

        current_batch_indices: list[int] = []
        batch_sets: list[set[int]] = [set(symptoms) for symptoms in symptom_batch]
        current_index: int
        for current_index, current_set in enumerate(batch_sets):
            best_index: int = current_index
            best_overlap: int = -1
            compare_index: int
            compare_set: set[int]
            for compare_index, compare_set in enumerate(batch_sets):
                if current_index == compare_index:
                    continue
                overlap: int = len(current_set.intersection(compare_set))
                if overlap > best_overlap:
                    best_index = compare_index
                    best_overlap = overlap
            current_batch_indices.append(best_index)
        similarity_indices.append(current_batch_indices)
    return similarity_indices


def build_train_batches(
    input_paths: FourSDrugInputPaths,
    train_rows: list[list[list[int]]],
    medicine_vocab_size: int,
    device: torch.device,
    batch_size: int,
    train_limit: int | None,
) -> list[FourSDrugTrainBatch]:
    """构造 4SDrug 训练 batch。"""

    symptom_batches: list[list[list[int]]]
    drug_batches: list[list[np.ndarray]]
    if train_limit is None and input_paths.sym_train.exists() and input_paths.drug_train.exists():
        with input_paths.sym_train.open("rb") as file:
            symptom_batches = cast(list[list[list[int]]], dill.load(file))
        with input_paths.drug_train.open("rb") as file:
            drug_batches = cast(list[list[np.ndarray]], dill.load(file))
    else:
        rebuilt_batches = build_batched_training_data(train_rows, batch_size, medicine_vocab_size)
        symptom_batches = rebuilt_batches.sym_train
        drug_batches = rebuilt_batches.drug_train

    similarity_indices: list[list[int]] = build_similarity_indices(symptom_batches)
    train_batches: list[FourSDrugTrainBatch] = []
    symptom_batch: list[list[int]]
    drug_batch: list[np.ndarray]
    similar_batch: list[int]
    for symptom_batch, drug_batch, similar_batch in zip(
        symptom_batches,
        drug_batches,
        similarity_indices,
        strict=True,
    ):
        symptom_tensor: Tensor = torch.tensor(
            np.asarray(symptom_batch, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        drug_tensor: Tensor = torch.tensor(
            np.asarray(drug_batch, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        similar_tensor: Tensor = torch.tensor(
            np.asarray(similar_batch, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        train_batches.append(
            FourSDrugTrainBatch(
                symptoms=symptom_tensor,
                drugs=drug_tensor,
                similar_indices=similar_tensor,
            ),
        )
    return train_batches


def load_data(config: FourSDrugTrainConfig, device: torch.device) -> FourSDrugLoadedData:
    """加载 4SDrug 训练与验证数据。"""

    input_paths: FourSDrugInputPaths = build_input_paths(config.input_dir, config.batch_size)
    vocabulary: FourSDrugVocFile = load_vocabulary(input_paths.voc_final)
    train_rows: list[list[list[int]]] = load_split_rows(input_paths.data_train, config.train_limit)
    dev_rows: list[list[list[int]]] = load_split_rows(input_paths.data_eval, config.dev_limit)
    test_rows: list[list[list[int]]] = []
    if input_paths.data_test.exists():
        test_rows = load_split_rows(input_paths.data_test, config.test_limit)
    ddi_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.ddi_A_final))
    train_batches: list[FourSDrugTrainBatch] = build_train_batches(
        input_paths=input_paths,
        train_rows=train_rows,
        medicine_vocab_size=len(vocabulary.med_voc.idx2word),
        device=device,
        batch_size=config.batch_size,
        train_limit=config.train_limit,
    )
    return FourSDrugLoadedData(
        train_batches=train_batches,
        dev_rows=dev_rows,
        test_rows=test_rows,
        ddi_adj=ddi_adj,
        vocabulary=vocabulary,
    )


def build_ddi_tensor(ddi_adj: csr_matrix, device: torch.device) -> Tensor:
    """把 1-based DDI 邻接矩阵转成 0-based torch 稀疏张量。"""

    trimmed_ddi_adj: csr_matrix = ddi_adj[1:, 1:].tocsr()
    ddi_coo = trimmed_ddi_adj.tocoo()
    indices: Tensor = torch.tensor(
        np.vstack((ddi_coo.row, ddi_coo.col)),
        dtype=torch.long,
        device=device,
    )
    values: Tensor = torch.tensor(
        ddi_coo.data.astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=trimmed_ddi_adj.shape,
        device=device,
    ).coalesce()


def compute_loss(
    result: FourSDrugForwardResult,
    drugs: Tensor,
    alpha: float,
    beta: float,
) -> Tensor:
    """计算 4SDrug 训练损失。"""

    prediction_loss: Tensor = F.binary_cross_entropy_with_logits(result.logits, drugs)
    probabilities: Tensor = result.probabilities.clamp(min=1e-8, max=1.0)
    # 目的：保留原始 4SDrug 中对高置信预测施加熵约束的主设计。
    entropy: Tensor = -(probabilities * torch.log(probabilities)).mean()
    return (
        prediction_loss + 0.5 * entropy + alpha * result.augmentation_loss + beta * result.ddi_loss
    )


def evaluate_model(
    model: FourSDrugModel,
    rows: list[list[list[int]]],
    ddi_adj: csr_matrix,
    threshold: float,
    device: torch.device,
) -> FourSDrugMetrics:
    """执行验证集评估。"""

    model.eval()
    metrics_list: list[FourSDrugMetrics] = []
    with torch.no_grad():
        row: list[list[int]]
        for row in tqdm(rows, desc="验证", leave=False):
            symptoms: Tensor = torch.tensor([row[0]], dtype=torch.long, device=device)
            target: Tensor = torch.zeros(
                (1, model.config.medicine_vocab_size),
                dtype=torch.float32,
                device=device,
            )
            medicine_ids: list[int] = list(dict.fromkeys(row[2]))
            if medicine_ids:
                target[0, np.asarray(medicine_ids, dtype=np.int64) - 1] = 1.0
            logits: Tensor = model.predict_logits(symptoms)
            probabilities: Tensor = torch.sigmoid(logits)
            loss: float = float(
                F.binary_cross_entropy_with_logits(logits, target).item(),
            )
            metrics_list.append(
                calculate_metrics(
                    probabilities=probabilities[0].detach().cpu().numpy(),
                    gold_drug_ids=medicine_ids,
                    ddi_adj=ddi_adj,
                    threshold=threshold,
                    loss=loss,
                ),
            )
    model.train()
    return aggregate_metrics(metrics_list)


def build_eval_result(metrics: FourSDrugMetrics) -> ExperimentEvalResult:
    """把 4SDrug 指标映射到统一评测结构。

    Args:
        metrics: 4SDrug 原始指标。

    Returns:
        统一评测结果。
    """

    return ExperimentEvalResult(
        loss=metrics.loss,
        comparable_metrics=ComparableMetrics(
            p_at_1=metrics.p_at_1,
            mrr=metrics.mrr,
            hit_at_5=metrics.hit_at_5,
            precision_at_5=metrics.precision_at_5,
            recall_at_5=metrics.recall_at_5,
            f1_at_5=metrics.f1_at_5,
        ),
        extra_metrics={
            "ja": metrics.ja,
            "jaccard_similarity": metrics.ja,
            "prauc": metrics.prauc,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "avg_drugs": metrics.avg_drugs,
            "ddi_rate": metrics.ddi_rate,
        },
    )


class FourSDrugTrainAdapter(
    ExperimentAdapter[FourSDrugTrainConfig, FourSDrugTrainState, FourSDrugSnapshot]
):
    """4SDrug 统一训练适配器。"""

    experiment_name: str = "foursdrug"

    def setup(self, config: FourSDrugTrainConfig) -> FourSDrugTrainState:
        """构造 4SDrug 训练状态。"""

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_data: FourSDrugLoadedData = load_data(config, device)
        if not loaded_data.train_batches:
            raise ValueError("训练集为空，无法执行 4SDrug 训练。")
        if not loaded_data.dev_rows:
            raise ValueError("验证集为空，无法执行 4SDrug 评估。")

        model: FourSDrugModel = FourSDrugModel(
            config=FourSDrugModelConfig(
                symptom_vocab_size=len(loaded_data.vocabulary.sym_voc.idx2word),
                medicine_vocab_size=len(loaded_data.vocabulary.med_voc.idx2word),
                embed_dim=config.embed_dim,
            ),
            ddi_adj=build_ddi_tensor(loaded_data.ddi_adj, device),
        ).to(device)
        optimizer: torch.optim.Optimizer = torch.optim.RAdam(model.parameters(), lr=config.lr)
        return FourSDrugTrainState(
            config=config,
            model=model,
            optimizer=optimizer,
            loaded_data=loaded_data,
            device=device,
        )

    def train_epoch(
        self,
        state: FourSDrugTrainState,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """执行单轮 4SDrug 训练。"""

        state.model.train()
        batch_losses: list[float] = []
        total_steps: int = total_epochs * len(state.loaded_data.train_batches)
        progress = tqdm(
            state.loaded_data.train_batches,
            desc=f"训练 epoch {epoch}/{total_epochs}",
            leave=False,
        )
        batch_index: int
        batch: FourSDrugTrainBatch
        for batch_index, batch in enumerate(progress, start=1):
            state.optimizer.zero_grad(set_to_none=True)
            result: FourSDrugForwardResult = state.model(
                symptoms=batch.symptoms,
                drugs=batch.drugs,
                similar_indices=batch.similar_indices,
            )
            loss_tensor: Tensor = compute_loss(
                result=result,
                drugs=batch.drugs,
                alpha=state.config.alpha,
                beta=state.config.beta,
            )
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
            state.optimizer.step()

            loss: float = float(loss_tensor.item())
            batch_losses.append(loss)
            global_step: int = (epoch - 1) * len(state.loaded_data.train_batches) + batch_index
            progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")
        progress.close()
        return sum(batch_losses) / len(batch_losses)

    def evaluate(self, state: FourSDrugTrainState, split: str) -> ExperimentEvalResult:
        """执行指定切分的 4SDrug 评测。"""

        rows: list[list[list[int]]]
        if split == "dev":
            rows = state.loaded_data.dev_rows
        else:
            rows = state.loaded_data.test_rows
        metrics: FourSDrugMetrics = evaluate_model(
            model=state.model,
            rows=rows,
            ddi_adj=state.loaded_data.ddi_adj,
            threshold=state.config.eval_threshold,
            device=state.device,
        )
        return build_eval_result(metrics)

    def has_split(self, state: FourSDrugTrainState, split: str) -> bool:
        """判断指定切分是否存在。"""

        if split == "test":
            return bool(state.loaded_data.test_rows)
        return True

    def capture_snapshot(self, state: FourSDrugTrainState) -> FourSDrugSnapshot:
        """捕获当前最佳权重。"""

        return {
            key: value.detach().cpu().clone() for key, value in state.model.state_dict().items()
        }

    def restore_snapshot(
        self,
        state: FourSDrugTrainState,
        snapshot: FourSDrugSnapshot,
    ) -> None:
        """恢复最佳权重。"""

        state.model.load_state_dict(snapshot)

    def export_checkpoint(self, state: FourSDrugTrainState, output_path: Path) -> None:
        """导出 4SDrug checkpoint。"""

        # 目的：统一导出最佳轮次权重，便于与其他 rerank 模型做同口径对比。
        torch.save(state.model.state_dict(), output_path)


def train(config: FourSDrugTrainConfig) -> None:
    """执行 4SDrug 训练。"""

    run_training_experiment(config, FourSDrugTrainAdapter())


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

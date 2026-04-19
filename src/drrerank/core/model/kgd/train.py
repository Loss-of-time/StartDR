"""KGDNet 精排模型训练入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import dill
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from ...io import load_pickle, load_pickle_rows
from ..experiment.runner import ExperimentAdapter, run_training_experiment
from ..experiment.schema import ComparableMetrics, ExperimentEvalResult
from ..tracedr.metrics import TraceDRMetrics, aggregate_metrics, calculate_metrics
from .common import KGDInputPaths, KGDVocabulary, KGDVocFile, sparse_matrix_to_edge_index
from .model import KGDNet
from .runtime import (
    build_ddi_kg,
    build_ehr_kgs,
    build_global_clinical_edges,
    build_patient_info,
)
from .schema import KGDForwardResult, KGDModelConfig

type KGDSnapshot = dict[str, Tensor]


@dataclass(slots=True)
class TrainConfig:
    """KGD 训练配置。

    Attributes:
        input_dir: KGD 离线导出目录。
        output_name: 输出名称前缀。
        epochs: 训练轮数。
        train_limit: 训练集样本上限。
        dev_limit: 验证集样本上限。
        test_limit: 测试集样本上限。
        selection_metric: 最佳轮次选择指标。
    """

    input_dir: Path
    output_name: str
    epochs: int
    train_limit: int | None = None
    dev_limit: int | None = None
    test_limit: int | None = None
    selection_metric: str = "mrr"


@dataclass(slots=True)
class RankedAnswer:
    """排序后的候选药物。"""

    id: str
    label: str
    score: float
    rank: int


@dataclass(slots=True)
class TrainEpochResult:
    """单轮训练报告。"""

    epoch: int
    train_loss: float
    dev_loss: float
    p_at_1: float
    mrr: float
    h_at_5: float
    answer_presence: float
    ddi_rate: float
    jaccard_similarity: float
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float


@dataclass(slots=True)
class TrainReport:
    """训练总报告。"""

    output_name: str
    epochs: list[TrainEpochResult]


@dataclass(slots=True)
class KGDSharedContext:
    """KGD 训练共享上下文。"""

    clinical_edges: dict[tuple[str, str, str], dict[str, int | Tensor]]
    ddi_graph: Data
    ddi_adj: csr_matrix
    medicine_vocab: KGDVocabulary
    num_clinical_nodes: int
    num_med_nodes: int


@dataclass(slots=True)
class KGDTrainSample:
    """KGD 单个病人训练样本。"""

    question_id: str
    patient_graphs: list[list[Data]]
    label_tensor: Tensor
    label_mask: Tensor
    gold_answers: list[str]


@dataclass(slots=True)
class KGDTrainState:
    """KGD 统一 runner 需要的状态。"""

    model: KGDNet
    optimizer: torch.optim.Optimizer
    ddi_graph: Data
    ddi_adj: csr_matrix
    medicine_vocab: KGDVocabulary
    train_samples: list[KGDTrainSample]
    dev_samples: list[KGDTrainSample]
    test_samples: list[KGDTrainSample]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="训练 KGD rerank 模型。")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--selection-metric", type=str, default="mrr")
    return parser.parse_args()


def build_input_paths(input_dir: Path) -> KGDInputPaths:
    """构造 KGD 离线路径集合。

    Args:
        input_dir: 离线导出目录。

    Returns:
        路径对象集合。
    """

    return KGDInputPaths(
        input_dir=input_dir,
        voc_final=input_dir / "voc_final.pkl",
        data_train=input_dir / "data_train.pkl",
        data_eval=input_dir / "data_eval.pkl",
        data_test=input_dir / "data_test.pkl",
        diag_adj=input_dir / "diag_adj.pkl",
        proc_adj=input_dir / "proc_adj.pkl",
        diag_proc_adj=input_dir / "diag_proc_adj.pkl",
        proc_diag_adj=input_dir / "proc_diag_adj.pkl",
        prescriptions_adj=input_dir / "prescriptions_adj.pkl",
        ddi_A_final=input_dir / "ddi_A_final.pkl",
    )


def load_vocabulary(voc_path: Path) -> KGDVocFile:
    """加载 KGD 词表。

    Args:
        voc_path: 词表文件路径。

    Returns:
        恢复后的词表对象。
    """

    with voc_path.open("rb") as file:
        raw_voc: dict[str, object] = cast(dict[str, object], dill.load(file))

    def build_vocab_entry(name: str) -> KGDVocabulary:
        raw_entry: dict[str, object] = cast(dict[str, object], raw_voc[name])
        return KGDVocabulary(
            word2idx=cast(dict[str, int], raw_entry["word2idx"]),
            idx2word=cast(list[str], raw_entry["idx2word"]),
        )

    return KGDVocFile(
        sym_voc=build_vocab_entry("sym_voc"),
        diag_voc=build_vocab_entry("diag_voc"),
        med_voc=build_vocab_entry("med_voc"),
    )


def load_split_records(
    split_path: Path,
    limit: int | None,
) -> list[list[list[int]]]:
    """加载单个 split 的索引病例。

    Args:
        split_path: split 文件路径。
        limit: 样本数量上限。

    Returns:
        索引化病例列表。
    """

    # 目的：兼容旧版整表 `list` 文件与新版逐条 pickle 流，避免导出阶段先把整份 split 堆进内存。
    return load_pickle_rows(split_path, limit)


def build_shared_context(
    input_dir: Path,
    device: torch.device,
) -> KGDSharedContext:
    """构造训练与验证共享的图上下文。

    Args:
        input_dir: KGD 离线导出目录。
        device: 目标设备。

    Returns:
        共享上下文。
    """

    input_paths: KGDInputPaths = build_input_paths(input_dir)
    vocabulary: KGDVocFile = load_vocabulary(input_paths.voc_final)

    diag_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.diag_adj))
    proc_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.proc_adj))
    diag_proc_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.diag_proc_adj))
    proc_diag_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.proc_diag_adj))
    ddi_adj: csr_matrix = cast(csr_matrix, load_pickle(input_paths.ddi_A_final))

    num_diag_nodes: int = len(vocabulary.diag_voc.idx2word)
    num_proc_nodes: int = len(vocabulary.sym_voc.idx2word)
    num_clinical_nodes: int = num_diag_nodes + num_proc_nodes
    num_med_nodes: int = len(vocabulary.med_voc.idx2word)

    clinical_edges: dict[tuple[str, str, str], dict[str, int | Tensor]] = (
        build_global_clinical_edges(
            diag_adj=diag_adj,
            proc_adj=proc_adj,
            diag_proc_adj=diag_proc_adj,
            proc_diag_adj=proc_diag_adj,
            num_diag_nodes=num_diag_nodes,
            device=device,
        )
    )
    ddi_edge_index: Tensor = sparse_matrix_to_edge_index(ddi_adj, device)
    ddi_graph: Data = build_ddi_kg(
        ddi_edge_index=ddi_edge_index,
        num_med_nodes=num_med_nodes,
        device=device,
    )
    return KGDSharedContext(
        clinical_edges=clinical_edges,
        ddi_graph=ddi_graph,
        ddi_adj=ddi_adj,
        medicine_vocab=vocabulary.med_voc,
        num_clinical_nodes=num_clinical_nodes,
        num_med_nodes=num_med_nodes,
    )


def build_label_tensor(
    medicine_ids: list[int],
    output_dim: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """构造 KGD 多标签监督张量。

    Args:
        medicine_ids: 当前病人的金标药物索引。
        output_dim: 模型输出维度。
        device: 目标设备。

    Returns:
        标签张量与损失掩码。
    """

    label_tensor: Tensor = torch.zeros((1, output_dim), dtype=torch.float32, device=device)
    label_mask: Tensor = torch.ones((1, output_dim), dtype=torch.float32, device=device)
    # 目的：屏蔽 0 号 padding 位，保持 1-based 药物索引语义。
    label_mask[:, 0] = 0.0
    for medicine_id in dict.fromkeys(medicine_ids):
        label_tensor[0, medicine_id] = 1.0
    return label_tensor, label_mask


def build_gold_answers(
    medicine_ids: list[int],
    medicine_vocab: KGDVocabulary,
) -> list[str]:
    """把药物索引恢复成药物编号。

    Args:
        medicine_ids: 当前病人的金标药物索引。
        medicine_vocab: 药物词表。

    Returns:
        去重后的金标药物编号列表。
    """

    gold_answers: list[str] = []
    medicine_count: int = len(medicine_vocab.idx2word)
    for medicine_id in dict.fromkeys(medicine_ids):
        if 0 < medicine_id <= medicine_count:
            gold_answers.append(medicine_vocab.idx2word[medicine_id - 1])
    return gold_answers


def build_dataset(
    records: list[list[list[int]]],
    context: KGDSharedContext,
    device: torch.device,
) -> list[KGDTrainSample]:
    """把索引病例构造成可训练样本。

    Args:
        records: 索引病例列表。
        context: 共享上下文。
        device: 目标设备。

    Returns:
        KGD 训练样本列表。
    """

    patient_info = build_patient_info(
        ehr_records=records,
        num_clinical_nodes=context.num_clinical_nodes,
        num_med_nodes=context.num_med_nodes,
        device=device,
    )
    ehr_graphs: list[list[list[Data]]] = build_ehr_kgs(
        patient_info=patient_info,
        clinical_edges=context.clinical_edges,
        num_clinical_nodes=context.num_clinical_nodes,
        num_med_nodes=context.num_med_nodes,
        device=device,
    )

    samples: list[KGDTrainSample] = []
    output_dim: int = context.num_med_nodes + 1
    for sample_index, (record, patient_graphs) in enumerate(
        zip(records, ehr_graphs, strict=True),
        start=1,
    ):
        medicine_ids: list[int] = record[2]
        label_tensor: Tensor
        label_mask: Tensor
        label_tensor, label_mask = build_label_tensor(
            medicine_ids=medicine_ids,
            output_dim=output_dim,
            device=device,
        )
        gold_answers: list[str] = build_gold_answers(medicine_ids, context.medicine_vocab)
        samples.append(
            KGDTrainSample(
                question_id=str(sample_index),
                patient_graphs=patient_graphs,
                label_tensor=label_tensor,
                label_mask=label_mask,
                gold_answers=gold_answers,
            )
        )
    return samples


def compute_loss(
    result: KGDForwardResult,
    sample: KGDTrainSample,
) -> Tensor:
    """计算 KGD 多标签损失。

    Args:
        result: 模型前向结果。
        sample: 当前训练样本。

    Returns:
        标量损失。
    """

    raw_loss: Tensor = F.binary_cross_entropy_with_logits(
        result.logits,
        sample.label_tensor,
        reduction="none",
    )
    masked_loss: Tensor = raw_loss * sample.label_mask
    return masked_loss.sum() / sample.label_mask.sum().clamp(min=1.0)


def build_ranked_answers(
    sample: KGDTrainSample,
    result: KGDForwardResult,
    medicine_vocab: KGDVocabulary,
) -> list[RankedAnswer]:
    """把模型输出转成排序结果。

    Args:
        sample: 当前样本。
        result: 前向输出。
        medicine_vocab: 药物词表。

    Returns:
        排序后的候选药物列表。
    """

    del sample
    medicine_count: int = len(medicine_vocab.idx2word)
    final_scores: Tensor = result.probabilities[-1].detach().cpu()
    sorted_indices_tensor: Tensor = torch.argsort(final_scores, descending=True)
    sorted_indices: list[int] = [int(index) for index in sorted_indices_tensor]
    ranked_answers: list[RankedAnswer] = []

    for medicine_index in sorted_indices:
        if medicine_index == 0 or medicine_index > medicine_count:
            continue
        drug_id: str = medicine_vocab.idx2word[medicine_index - 1]
        ranked_answers.append(
            RankedAnswer(
                id=drug_id,
                label=drug_id,
                score=float(final_scores[medicine_index].item()),
                rank=len(ranked_answers) + 1,
            )
        )
    return ranked_answers


def evaluate_model(
    model: KGDNet,
    samples: list[KGDTrainSample],
    ddi_graph: Data,
    ddi_adj: csr_matrix,
    medicine_vocab: KGDVocabulary,
) -> TraceDRMetrics:
    """执行验证集评估。

    Args:
        model: 待评估模型。
        samples: 验证样本。
        ddi_graph: 全局 DDI 图。
        medicine_vocab: 药物词表。

    Returns:
        聚合后的验证指标。
    """

    model.eval()
    losses: list[float] = []
    metrics_list: list[TraceDRMetrics] = []

    with torch.no_grad():
        with tqdm(samples, desc="验证", leave=False) as progress:
            for sample in progress:
                result: KGDForwardResult = model(sample.patient_graphs, ddi_graph)
                loss: float = float(compute_loss(result, sample).item())
                ranked_answers: list[RankedAnswer] = build_ranked_answers(
                    sample=sample,
                    result=result,
                    medicine_vocab=medicine_vocab,
                )
                metrics: TraceDRMetrics = calculate_metrics(
                    question_id=sample.question_id,
                    answers=ranked_answers,
                    gold_answers=sample.gold_answers,
                    k=5,
                    ddi_adj=ddi_adj,
                    drugid_to_index=medicine_vocab.word2idx,
                )
                losses.append(loss)
                metrics_list.append(metrics)

    model.train()

    if not metrics_list:
        raise ValueError("验证阶段未产生任何指标，请检查 KGD 样本构造流程。")

    aggregated_metrics: TraceDRMetrics = aggregate_metrics(metrics_list)
    return TraceDRMetrics(
        loss=sum(losses) / len(losses) if losses else 0.0,
        p_at_1=aggregated_metrics.p_at_1,
        mrr=aggregated_metrics.mrr,
        h_at_5=aggregated_metrics.h_at_5,
        answer_presence=aggregated_metrics.answer_presence,
        ddi_rate=aggregated_metrics.ddi_rate,
        jaccard_similarity=aggregated_metrics.jaccard_similarity,
        precision_at_5=aggregated_metrics.precision_at_5,
        recall_at_5=aggregated_metrics.recall_at_5,
        f1_at_5=aggregated_metrics.f1_at_5,
    )


def build_eval_result(metrics: TraceDRMetrics) -> ExperimentEvalResult:
    """把 KGD 指标映射到统一评测结构。

    Args:
        metrics: KGD 原始指标。

    Returns:
        统一评测结果。
    """

    return ExperimentEvalResult(
        loss=metrics.loss,
        comparable_metrics=ComparableMetrics(
            p_at_1=metrics.p_at_1,
            mrr=metrics.mrr,
            hit_at_5=metrics.h_at_5,
            precision_at_5=metrics.precision_at_5,
            recall_at_5=metrics.recall_at_5,
            f1_at_5=metrics.f1_at_5,
        ),
        extra_metrics={
            "answer_presence": metrics.answer_presence,
            "ddi_rate": metrics.ddi_rate,
            "jaccard_similarity": metrics.jaccard_similarity,
        },
    )


class KGDTrainAdapter(ExperimentAdapter[TrainConfig, KGDTrainState, KGDSnapshot]):
    """KGD 统一训练适配器。"""

    experiment_name: str = "kgd"

    def setup(self, config: TrainConfig) -> KGDTrainState:
        """构造 KGD 训练状态。"""

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_paths: KGDInputPaths = build_input_paths(config.input_dir)
        context: KGDSharedContext = build_shared_context(config.input_dir, device)
        train_records: list[list[list[int]]] = load_split_records(
            input_paths.data_train,
            config.train_limit,
        )
        dev_records: list[list[list[int]]] = load_split_records(
            input_paths.data_eval,
            config.dev_limit,
        )
        train_samples: list[KGDTrainSample] = build_dataset(train_records, context, device)
        dev_samples: list[KGDTrainSample] = build_dataset(dev_records, context, device)
        test_samples: list[KGDTrainSample] = []
        if input_paths.data_test.exists():
            test_records: list[list[list[int]]] = load_split_records(
                input_paths.data_test,
                config.test_limit,
            )
            test_samples = build_dataset(test_records, context, device)
        if not train_samples:
            raise ValueError("训练集为空，无法执行 KGD 训练。")
        if not dev_samples:
            raise ValueError("验证集为空，无法执行 KGD 评估。")

        model_config: KGDModelConfig = KGDModelConfig(
            clinical_vocab_size=context.num_clinical_nodes,
            medicine_vocab_size=context.num_med_nodes,
        )
        model: KGDNet = KGDNet(model_config).to(device)
        model.train()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )
        return KGDTrainState(
            model=model,
            optimizer=optimizer,
            ddi_graph=context.ddi_graph,
            ddi_adj=context.ddi_adj,
            medicine_vocab=context.medicine_vocab,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
        )

    def train_epoch(self, state: KGDTrainState, epoch: int, total_epochs: int) -> float:
        """执行单轮 KGD 训练。"""

        losses: list[float] = []
        total_steps: int = total_epochs * len(state.train_samples)
        with tqdm(
            state.train_samples,
            desc=f"训练 epoch {epoch}/{total_epochs}",
            leave=False,
        ) as progress:
            sample_idx: int
            sample: KGDTrainSample
            for sample_idx, sample in enumerate(progress, start=1):
                state.optimizer.zero_grad(set_to_none=True)
                result: KGDForwardResult = state.model(sample.patient_graphs, state.ddi_graph)
                loss_tensor: Tensor = compute_loss(result, sample)
                loss_tensor.backward()
                state.optimizer.step()

                loss: float = float(loss_tensor.item())
                global_step: int = (epoch - 1) * len(state.train_samples) + sample_idx
                losses.append(loss)
                progress.set_postfix_str(f"step={global_step}/{total_steps} loss={loss:.6f}")
        return sum(losses) / len(losses)

    def evaluate(self, state: KGDTrainState, split: str) -> ExperimentEvalResult:
        """执行指定切分的 KGD 评测。"""

        samples: list[KGDTrainSample]
        if split == "dev":
            samples = state.dev_samples
        else:
            samples = state.test_samples
        metrics: TraceDRMetrics = evaluate_model(
            model=state.model,
            samples=samples,
            ddi_graph=state.ddi_graph,
            ddi_adj=state.ddi_adj,
            medicine_vocab=state.medicine_vocab,
        )
        return build_eval_result(metrics)

    def has_split(self, state: KGDTrainState, split: str) -> bool:
        """判断指定切分是否存在。"""

        if split == "test":
            return bool(state.test_samples)
        return True

    def capture_snapshot(self, state: KGDTrainState) -> KGDSnapshot:
        """捕获当前最佳权重。"""

        return {
            key: value.detach().cpu().clone() for key, value in state.model.state_dict().items()
        }

    def restore_snapshot(self, state: KGDTrainState, snapshot: KGDSnapshot) -> None:
        """恢复最佳权重。"""

        state.model.load_state_dict(snapshot)

    def export_checkpoint(self, state: KGDTrainState, output_path: Path) -> None:
        """导出 KGD checkpoint。"""

        # 目的：统一导出最佳轮次权重，保证 KGD 可直接参与同口径对比实验。
        torch.save(state.model.state_dict(), output_path)


def train(config: TrainConfig) -> None:
    """执行 KGD 训练。

    Args:
        config: 训练配置。
    """

    run_training_experiment(config, KGDTrainAdapter())


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

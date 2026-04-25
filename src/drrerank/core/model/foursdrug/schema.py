"""4SDrug 模型结构定义。"""

from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Bool, Float, Int
from scipy.sparse import csr_matrix
from torch import Tensor

from .common import FourSDrugIndexedRow

type FourSDrugLogits = Float[Tensor, "batch medicine"]
type FourSDrugProbabilities = Float[Tensor, "batch medicine"]
type FourSDrugSymptomTensor = Int[Tensor, "batch symptom"]
type FourSDrugDrugTensor = Float[Tensor, "batch medicine"]
type FourSDrugCandidateMaskTensor = Bool[Tensor, "batch medicine"]
type FourSDrugSimilarIndexTensor = Int[Tensor, "batch"]


@dataclass(slots=True)
class FourSDrugModelConfig:
    """4SDrug 模型配置。"""

    symptom_vocab_size: int
    medicine_vocab_size: int
    embed_dim: int = 64
    dropout: float = 0.4
    prediction_threshold: float = 0.5


@dataclass(slots=True)
class FourSDrugForwardResult:
    """4SDrug 单个 batch 的前向结果。"""

    logits: FourSDrugLogits
    probabilities: FourSDrugProbabilities
    ddi_loss: Float[Tensor, ""]
    augmentation_loss: Float[Tensor, ""]


@dataclass(slots=True)
class FourSDrugTrainConfig:
    """4SDrug 训练配置。"""

    input_dir: Path
    output_name: str
    epochs: int
    batch_size: int = 16
    embed_dim: int = 64
    lr: float = 5e-3
    alpha: float = 0.5
    beta: float = 1.0
    train_limit: int | None = None
    dev_limit: int | None = None
    test_limit: int | None = None
    eval_threshold: float = 0.8
    selection_metric: str = "ja"


@dataclass(slots=True)
class FourSDrugInputPaths:
    """4SDrug 训练输入路径集合。"""

    input_dir: Path
    voc_final: Path
    data_train: Path
    data_eval: Path
    data_test: Path
    ddi_A_final: Path
    sym_train: Path
    drug_train: Path
    candidate_train: Path


@dataclass(slots=True)
class FourSDrugTrainBatch:
    """4SDrug 单个训练 batch。"""

    symptoms: FourSDrugSymptomTensor
    drugs: FourSDrugDrugTensor
    candidate_mask: FourSDrugCandidateMaskTensor
    similar_indices: FourSDrugSimilarIndexTensor


@dataclass(slots=True)
class FourSDrugVocabulary:
    """4SDrug 词表对象。"""

    word2idx: dict[str, int]
    idx2word: list[str]


@dataclass(slots=True)
class FourSDrugVocFile:
    """4SDrug 词表文件结构。"""

    sym_voc: FourSDrugVocabulary
    diag_voc: FourSDrugVocabulary
    med_voc: FourSDrugVocabulary


@dataclass(slots=True)
class FourSDrugMetrics:
    """4SDrug 单样本或聚合指标。"""

    loss: float = 0.0
    ja: float = 0.0
    prauc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_drugs: float = 0.0
    ddi_rate: float = 0.0
    p_at_1: float = 0.0
    mrr: float = 0.0
    hit_at_5: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    f1_at_5: float = 0.0


@dataclass(slots=True)
class FourSDrugEpochResult:
    """4SDrug 单轮训练报告。"""

    epoch: int
    train_loss: float
    dev_metrics: FourSDrugMetrics


@dataclass(slots=True)
class FourSDrugTrainReport:
    """4SDrug 训练总报告。"""

    output_name: str
    epochs: list[FourSDrugEpochResult]


@dataclass(slots=True)
class FourSDrugLoadedData:
    """4SDrug 训练阶段加载后的数据集合。"""

    train_batches: list[FourSDrugTrainBatch]
    dev_rows: list[FourSDrugIndexedRow]
    test_rows: list[FourSDrugIndexedRow]
    ddi_adj: csr_matrix
    vocabulary: FourSDrugVocFile

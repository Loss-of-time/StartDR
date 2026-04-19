"""4SDrug 数据处理公共结构。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

type FourSDrugUpperEdge = tuple[int, int]
type FourSDrugDrugMultiHot = npt.NDArray[np.bool_]


@dataclass(slots=True)
class FourSDrugSourceCase:
    """从单条 TraceDR 样本抽取出的 4SDrug 病例字段。"""

    symptoms: list[str]
    diagnosis: list[str]
    medicines: list[str]


@dataclass(slots=True)
class FourSDrugStringCases:
    """4SDrug 字符串病例集合。"""

    symptoms: list[list[str]]
    diagnosis: list[list[str]]
    medicines: list[list[str]]


@dataclass(slots=True)
class FourSDrugIndexedCases:
    """4SDrug 索引化病例集合。"""

    symptoms: list[list[int]]
    diagnosis: list[list[int]]
    medicines: list[list[int]]


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
class FourSDrugBatchData:
    """单个 batch size 下的 4SDrug 训练分桶数据。"""

    sym_train: list[list[list[int]]]
    drug_train: list[list[FourSDrugDrugMultiHot]]


@dataclass(slots=True)
class FourSDrugExportArtifacts:
    """4SDrug 离线导出阶段写盘前的全部产物。"""

    voc_final: FourSDrugVocFile
    data_train: list[list[list[int]]]
    data_eval: list[list[list[int]]]
    data_test: list[list[list[int]]]
    ddi_A_final: csr_matrix
    sym_sets: list[list[int]]
    drug_multihots: csr_matrix
    batch_data: dict[int, FourSDrugBatchData]


@dataclass(slots=True)
class FourSDrugOutputPaths:
    """4SDrug 离线导出输出路径集合。"""

    output_dir: Path
    voc_final: Path
    data_train: Path
    data_eval: Path
    data_test: Path
    ddi_A_final: Path
    sym_sets: Path
    drug_multihots: Path
    sym_train: dict[int, Path]
    drug_train: dict[int, Path]


def build_symmetric_adj(size: int, upper_edges: set[FourSDrugUpperEdge]) -> csr_matrix:
    """构造对称稀疏矩阵。

    Args:
        size: 邻接矩阵边长。
        upper_edges: 只保留上三角方向的无向边集合。

    Returns:
        对称 CSR 稀疏矩阵。
    """

    if not upper_edges:
        return csr_matrix((size, size), dtype=np.uint8)
    edge_count: int = len(upper_edges)
    upper_rows: npt.NDArray[np.int64] = np.fromiter(
        (row for row, _ in upper_edges),
        dtype=np.int64,
        count=edge_count,
    )
    upper_cols: npt.NDArray[np.int64] = np.fromiter(
        (col for _, col in upper_edges),
        dtype=np.int64,
        count=edge_count,
    )
    rows: npt.NDArray[np.int64] = np.empty(edge_count * 2, dtype=np.int64)
    cols: npt.NDArray[np.int64] = np.empty(edge_count * 2, dtype=np.int64)
    rows[:edge_count] = upper_rows
    rows[edge_count:] = upper_cols
    cols[:edge_count] = upper_cols
    cols[edge_count:] = upper_rows
    data: npt.NDArray[np.uint8] = np.ones(edge_count * 2, dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(size, size), dtype=np.uint8)


def normalize_upper_edge(left: int, right: int) -> FourSDrugUpperEdge:
    """统一无向边方向。

    Args:
        left: 左侧节点索引。
        right: 右侧节点索引。

    Returns:
        规范化后的上三角边。
    """

    if left < right:
        return left, right
    return right, left

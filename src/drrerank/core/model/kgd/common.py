"""KGD 公共结构与工具函数。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data


@dataclass(slots=True)
class KGDIndexedCases:
    """索引化后的病例集合。"""
    symptoms: list[list[int]]
    diagnosis: list[list[int]]
    medicines: list[list[int]]


@dataclass(slots=True)
class KGDStringCases:
    """原始字符串病例集合。"""
    symptoms: list[list[str]]
    diagnosis: list[list[str]]
    medicines: list[list[str]]


@dataclass(slots=True)
class KGDSourceCase:
    """从单条 TraceDR 样本抽取出的病例字段。"""
    symptoms: list[str]
    diagnosis: list[str]
    medicines: list[str]


@dataclass(slots=True)
class KGDVocabulary:
    """KGD 词表对象。"""
    word2idx: dict[str, int]
    idx2word: list[str]


@dataclass(slots=True)
class KGDVocFile:
    """KGD 词表文件结构。"""
    sym_voc: KGDVocabulary
    diag_voc: KGDVocabulary
    med_voc: KGDVocabulary


@dataclass(slots=True)
class KGDExportArtifacts:
    """离线导出阶段写盘前的全部产物。"""
    voc_final: KGDVocFile
    data_train: list[list[list[int]]]
    data_eval: list[list[list[int]]]
    data_test: list[list[list[int]]]
    diag_adj: csr_matrix
    proc_adj: csr_matrix
    diag_proc_adj: csr_matrix
    proc_diag_adj: csr_matrix
    prescriptions_adj: csr_matrix
    ddi_A_final: csr_matrix


@dataclass(slots=True)
class KGDOutputPaths:
    """离线导出输出路径集合。"""
    output_dir: Path
    voc_final: Path
    data_train: Path
    data_eval: Path
    data_test: Path
    diag_adj: Path
    proc_adj: Path
    diag_proc_adj: Path
    proc_diag_adj: Path
    prescriptions_adj: Path
    ddi_A_final: Path


@dataclass(slots=True)
class KGDInputPaths:
    """运行时构图输入路径集合。"""
    input_dir: Path
    voc_final: Path
    data_train: Path
    data_eval: Path
    data_test: Path
    diag_adj: Path
    proc_adj: Path
    diag_proc_adj: Path
    proc_diag_adj: Path
    prescriptions_adj: Path
    ddi_A_final: Path


@dataclass(slots=True)
class KGDRuntimeArtifacts:
    """运行时构图阶段加载后的全部产物。"""
    ehr_records: list[list[list[int]]]
    diag_adj: csr_matrix
    proc_adj: csr_matrix
    diag_proc_adj: csr_matrix
    proc_diag_adj: csr_matrix
    prescriptions_adj: csr_matrix
    ddi_adj: csr_matrix
    voc: KGDVocFile


type KGDRelationKey = tuple[str, str, str]
type KGDRelationPayload = dict[str, int | torch.Tensor]
type KGDAdmissionInfo = dict[KGDRelationKey, KGDRelationPayload | None]
type KGDPatientInfo = list[list[KGDAdmissionInfo]]
type KGDEHRGraphs = list[list[list[Data]]]


def build_symmetric_adj(size: int, upper_edges: set[tuple[int, int]]) -> csr_matrix:
    """构造对称稀疏矩阵。"""
    if not upper_edges:
        return csr_matrix((size, size), dtype=np.uint8)
    edge_count = len(upper_edges)
    upper_rows = np.fromiter(
        (row for row, _ in upper_edges),
        dtype=np.int64,
        count=edge_count,
    )
    upper_cols = np.fromiter(
        (col for _, col in upper_edges),
        dtype=np.int64,
        count=edge_count,
    )
    rows = np.empty(edge_count * 2, dtype=np.int64)
    cols = np.empty(edge_count * 2, dtype=np.int64)
    rows[:edge_count] = upper_rows
    rows[edge_count:] = upper_cols
    cols[:edge_count] = upper_cols
    cols[edge_count:] = upper_rows
    data = np.ones(edge_count * 2, dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(size, size), dtype=np.uint8)


def normalize_upper_edge(left: int, right: int) -> tuple[int, int]:
    """统一无向边方向。"""
    if left < right:
        return left, right
    return right, left


def sparse_matrix_to_edge_index(
    matrix: csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    """把稀疏矩阵转成 PyG edge_index。"""
    coo_matrix = matrix.tocoo()
    if coo_matrix.nnz == 0:
        return empty_edge_index(device)
    return torch.tensor(
        np.vstack((coo_matrix.row, coo_matrix.col)),
        dtype=torch.int64,
        device=device,
    )


def empty_edge_index(device: torch.device) -> torch.Tensor:
    """构造空 edge_index。"""
    return torch.empty((2, 0), dtype=torch.int64, device=device)


def empty_edge_ids(device: torch.device) -> torch.Tensor:
    """构造空 relation id。"""
    return torch.empty((0,), dtype=torch.int64, device=device)


def build_relation_payload(
    code: int,
    edges: torch.Tensor,
    edge_id: torch.Tensor,
) -> KGDRelationPayload:
    """构造 relation payload。"""
    return {
        "code": code,
        "edges": edges,
        "edge_id": edge_id,
    }

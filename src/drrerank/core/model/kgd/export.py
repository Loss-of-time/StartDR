"""KGD 离线导出入口。"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from pickle import dump as pickle_dump
from typing import cast

import dill
import numpy as np
from scipy.sparse import csr_matrix

from ...io import write_pickle_row_stream
from .common import (
    KGDOutputPaths,
    KGDSourceCase,
    KGDVocabulary,
    KGDVocFile,
    build_symmetric_adj,
    normalize_upper_edge,
)

# misc/KGDNet从patient_candidate生成数据文档.md
# misc/KGDNet实现文档.md
# misc/TraceDR-main/TraceDR-model/baseline/data_process/KGDNet_dataprocess.py


def update_vocab(vocab: dict[str, int], row: list[str]) -> None:
    """按单行数据增量更新 1-based 词表。

    Args:
        vocab: 待更新词表。
        row: 当前字段行。
    """

    value: str
    for value in row:
        if value not in vocab:
            # 目的：保留 KGDNet 的 1-based 词表约定，同时避免整份 split 一次性驻留内存。
            vocab[value] = len(vocab) + 1


def parse_kgd_source_case(row: dict[str, object]) -> KGDSourceCase:
    """从单条 TraceDR 样本提取 KGD 所需字段。

    Args:
        row: 原始 `jsonl` 行对象。

    Returns:
        KGD 导出所需的最小病例对象。
    """

    people = cast(dict[str, object], row["people"])
    raw_medicines = cast(list[dict[str, object]], people["medicine"])
    return KGDSourceCase(
        symptoms=list(dict.fromkeys(cast(list[str], people["symptom"]))),
        diagnosis=list(dict.fromkeys(cast(list[str], people["diagnosis"]))),
        medicines=list(
            dict.fromkeys(str(raw_medicine["drugid"]) for raw_medicine in raw_medicines)
        ),
    )


def iter_kgd_source_cases(input_path: Path) -> Iterator[KGDSourceCase]:
    """流式遍历单个 split 的 KGD 字符串病例。

    Args:
        input_path: TraceDR 风格 `jsonl` 路径。

    Yields:
        单条 KGD 病例。
    """

    with input_path.open(encoding="utf-8") as file:
        line: str
        for line in file:
            yield parse_kgd_source_case(cast(dict[str, object], json.loads(line)))


def index_row(row: list[str], vocab: dict[str, int]) -> list[int]:
    """按词表编码单行字符串字段。

    Args:
        row: 字符串字段行。
        vocab: 目标词表。

    Returns:
        编码后的索引行。
    """

    return [vocab[value] for value in row]


def build_vocab_file_entry(vocab: dict[str, int]) -> KGDVocabulary:
    """把词表转换为可写盘对象。

    Args:
        vocab: 1-based 词表。

    Returns:
        可直接写入 `voc_final.pkl` 的词表对象。
    """

    idx2word: list[str] = [""] * len(vocab)
    word: str
    index: int
    for word, index in vocab.items():
        idx2word[index - 1] = word
    return KGDVocabulary(word2idx=vocab, idx2word=idx2word)


def indexed_case_to_row(source_case: KGDSourceCase, voc_final: KGDVocFile) -> list[list[int]]:
    """把单条病例编码成 KGD 原始 `pkl` 行结构。

    Args:
        source_case: 原始字符串病例。
        voc_final: 已构造完成的词表文件对象。

    Returns:
        `[symptoms, diagnosis, medicines]` 形式的编码样本。
    """

    return [
        index_row(source_case.symptoms, voc_final.sym_voc.word2idx),
        index_row(source_case.diagnosis, voc_final.diag_voc.word2idx),
        index_row(source_case.medicines, voc_final.med_voc.word2idx),
    ]


def build_output_paths(output_dir: Path) -> KGDOutputPaths:
    """构造 KGD 导出输出路径集合。

    Args:
        output_dir: 输出目录。

    Returns:
        输出路径对象集合。
    """

    return KGDOutputPaths(
        output_dir=output_dir,
        voc_final=output_dir / "voc_final.pkl",
        data_train=output_dir / "data_train.pkl",
        data_eval=output_dir / "data_eval.pkl",
        data_test=output_dir / "data_test.pkl",
        diag_adj=output_dir / "diag_adj.pkl",
        proc_adj=output_dir / "proc_adj.pkl",
        diag_proc_adj=output_dir / "diag_proc_adj.pkl",
        proc_diag_adj=output_dir / "proc_diag_adj.pkl",
        prescriptions_adj=output_dir / "prescriptions_adj.pkl",
        ddi_A_final=output_dir / "ddi_A_final.pkl",
    )


def build_bipartite_adj(
    row_count: int,
    col_count: int,
    edges: set[tuple[int, int]],
) -> csr_matrix:
    """构造二部稀疏矩阵。"""

    if not edges:
        return csr_matrix((row_count, col_count), dtype=np.uint8)
    edge_count = len(edges)
    rows = np.fromiter((row for row, _ in edges), dtype=np.int64, count=edge_count)
    cols = np.fromiter((col for _, col in edges), dtype=np.int64, count=edge_count)
    data = np.ones(edge_count, dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(row_count, col_count), dtype=np.uint8)


def update_edge_sets(
    row: list[list[int]],
    diag_upper_edges: set[tuple[int, int]],
    symptom_upper_edges: set[tuple[int, int]],
    diag_symptom_edges: set[tuple[int, int]],
) -> None:
    """按单条编码病例增量更新共现边集合。

    Args:
        row: 单条编码病例。
        diag_upper_edges: 诊断共现边集合。
        symptom_upper_edges: 症状共现边集合。
        diag_symptom_edges: 诊断-症状二部边集合。
    """

    symptoms: list[int] = row[0]
    diagnosis: list[int] = row[1]

    diagnosis_count: int = len(diagnosis)
    left_index: int
    right_index: int
    for left_index in range(diagnosis_count):
        left = diagnosis[left_index]
        for right_index in range(left_index + 1, diagnosis_count):
            # 目的：统一共现边方向，保持最终稀疏矩阵的二值语义。
            diag_upper_edges.add(normalize_upper_edge(left, diagnosis[right_index]))

    symptom_count: int = len(symptoms)
    for left_index in range(symptom_count):
        left = symptoms[left_index]
        for right_index in range(left_index + 1, symptom_count):
            # 目的：统一共现边方向，保持最终稀疏矩阵的二值语义。
            symptom_upper_edges.add(normalize_upper_edge(left, symptoms[right_index]))

    diagnosis_index: int
    symptom_index: int
    for diagnosis_index in diagnosis:
        for symptom_index in symptoms:
            diag_symptom_edges.add((diagnosis_index, symptom_index))


def build_adjacency_artifacts(
    symptom_vocab: dict[str, int],
    diagnosis_vocab: dict[str, int],
    medicine_vocab: dict[str, int],
    diag_upper_edges: set[tuple[int, int]],
    symptom_upper_edges: set[tuple[int, int]],
    diag_symptom_edges: set[tuple[int, int]],
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """根据累计边集合构造病例共现邻接矩阵。

    Args:
        symptom_vocab: 症状词表。
        diagnosis_vocab: 诊断词表。
        medicine_vocab: 药物词表。
        diag_upper_edges: 诊断共现边集合。
        symptom_upper_edges: 症状共现边集合。
        diag_symptom_edges: 诊断-症状边集合。

    Returns:
        KGD 训练需要的五个共现矩阵。
    """

    diag_adj = build_symmetric_adj(len(diagnosis_vocab) + 1, diag_upper_edges)
    proc_adj = build_symmetric_adj(len(symptom_vocab) + 1, symptom_upper_edges)
    diag_proc_adj = build_bipartite_adj(
        len(diagnosis_vocab) + 1,
        len(symptom_vocab) + 1,
        diag_symptom_edges,
    )
    proc_diag_adj = diag_proc_adj.transpose().tocsr()
    # 目的：当前训练与运行时都不消费药物共现图，这里直接写空矩阵占位，避免药物对组合边炸内存。
    prescriptions_adj = csr_matrix(
        (len(medicine_vocab) + 1, len(medicine_vocab) + 1),
        dtype=np.uint8,
    )
    return diag_adj, proc_adj, diag_proc_adj, proc_diag_adj, prescriptions_adj


def build_ddi_adj(medicine_vocab: dict[str, int]) -> csr_matrix:
    """构造空的 DDI 邻接矩阵。

    Args:
        medicine_vocab: 1-based 药物词表。

    Returns:
        与 `med_voc` 对齐的 1-based 空 DDI 邻接矩阵。
    """

    medicine_count: int = len(medicine_vocab) + 1
    # 目的：当前实验默认禁用 DDI，导出阶段直接写空矩阵以消除 Neo4j 依赖。
    return csr_matrix((medicine_count, medicine_count), dtype=np.uint8)


def build_vocabulary_from_inputs(
    train_input: Path,
    dev_input: Path,
    test_input: Path,
) -> tuple[KGDVocFile, tuple[int, int, int]]:
    """按多份输入流式构造词表。

    Args:
        train_input: 训练集路径。
        dev_input: 验证集路径。
        test_input: 测试集路径。

    Returns:
        词表文件对象，以及各 split 的样本数。
    """

    symptom_vocab: dict[str, int] = {}
    diagnosis_vocab: dict[str, int] = {}
    medicine_vocab: dict[str, int] = {}
    split_counts: list[int] = []
    input_path: Path
    for input_path in [train_input, dev_input, test_input]:
        sample_count: int = 0
        source_case: KGDSourceCase
        for source_case in iter_kgd_source_cases(input_path):
            update_vocab(symptom_vocab, source_case.symptoms)
            update_vocab(diagnosis_vocab, source_case.diagnosis)
            update_vocab(medicine_vocab, source_case.medicines)
            sample_count += 1
        split_counts.append(sample_count)

    return (
        KGDVocFile(
            sym_voc=build_vocab_file_entry(symptom_vocab),
            diag_voc=build_vocab_file_entry(diagnosis_vocab),
            med_voc=build_vocab_file_entry(medicine_vocab),
        ),
        (split_counts[0], split_counts[1], split_counts[2]),
    )


def write_split_rows_and_collect_edges(
    input_path: Path,
    output_path: Path,
    voc_final: KGDVocFile,
    diag_upper_edges: set[tuple[int, int]],
    symptom_upper_edges: set[tuple[int, int]],
    diag_symptom_edges: set[tuple[int, int]],
) -> int:
    """写出单个 split 编码结果，并同步累计图边。

    Args:
        input_path: 原始 `jsonl` 路径。
        output_path: 对应 `pkl` 输出路径。
        voc_final: 全局词表文件对象。
        diag_upper_edges: 诊断共现边集合。
        symptom_upper_edges: 症状共现边集合。
        diag_symptom_edges: 诊断-症状边集合。

    Returns:
        当前 split 的样本数。
    """

    def iter_indexed_rows() -> Iterator[list[list[int]]]:
        source_case: KGDSourceCase
        for source_case in iter_kgd_source_cases(input_path):
            indexed_row: list[list[int]] = indexed_case_to_row(source_case, voc_final)
            update_edge_sets(
                indexed_row,
                diag_upper_edges,
                symptom_upper_edges,
                diag_symptom_edges,
            )
            yield indexed_row

    return write_pickle_row_stream(output_path, iter_indexed_rows())


def save_kgd_export_artifacts(
    output_dir: Path,
    voc_final: KGDVocFile,
    diag_adj: csr_matrix,
    proc_adj: csr_matrix,
    diag_proc_adj: csr_matrix,
    proc_diag_adj: csr_matrix,
    prescriptions_adj: csr_matrix,
    ddi_adj: csr_matrix,
) -> KGDOutputPaths:
    """保存 KGD 导出产物。

    Args:
        output_dir: 输出目录。
        voc_final: 词表对象。
        diag_adj: 诊断共现矩阵。
        proc_adj: 症状共现矩阵。
        diag_proc_adj: 诊断到症状二部矩阵。
        proc_diag_adj: 症状到诊断二部矩阵。
        prescriptions_adj: 药物共现矩阵。
        ddi_adj: DDI 矩阵。

    Returns:
        输出路径对象集合。
    """

    # 目的：先创建目录与路径协议，再按最终文件分别写盘。
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: KGDOutputPaths = build_output_paths(output_dir)

    with output_paths.voc_final.open("wb") as file:
        dill.dump(asdict(voc_final), file, protocol=HIGHEST_PROTOCOL)
    with output_paths.diag_adj.open("wb") as file:
        pickle_dump(diag_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.proc_adj.open("wb") as file:
        pickle_dump(proc_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.diag_proc_adj.open("wb") as file:
        pickle_dump(diag_proc_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.proc_diag_adj.open("wb") as file:
        pickle_dump(proc_diag_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.prescriptions_adj.open("wb") as file:
        pickle_dump(prescriptions_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.ddi_A_final.open("wb") as file:
        pickle_dump(ddi_adj, file, protocol=HIGHEST_PROTOCOL)

    return output_paths


def main() -> None:
    """离线导出命令行入口。"""

    parser = argparse.ArgumentParser(description="从 TraceDR 风格 jsonl 导出 KGDNet 离线文件。")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    print("开始流式构造 KGD 词表")
    voc_final, split_counts = build_vocabulary_from_inputs(
        args.train_input,
        args.dev_input,
        args.test_input,
    )
    print(
        f"词表构造完成: train={split_counts[0]}, dev={split_counts[1]}, test={split_counts[2]}",
    )

    output_paths: KGDOutputPaths = build_output_paths(args.output_dir)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)

    diag_upper_edges: set[tuple[int, int]] = set()
    symptom_upper_edges: set[tuple[int, int]] = set()
    diag_symptom_edges: set[tuple[int, int]] = set()

    print(f"开始写出 train 编码结果: {output_paths.data_train.resolve()}")
    write_split_rows_and_collect_edges(
        args.train_input,
        output_paths.data_train,
        voc_final,
        diag_upper_edges,
        symptom_upper_edges,
        diag_symptom_edges,
    )
    print(f"开始写出 dev 编码结果: {output_paths.data_eval.resolve()}")
    write_split_rows_and_collect_edges(
        args.dev_input,
        output_paths.data_eval,
        voc_final,
        diag_upper_edges,
        symptom_upper_edges,
        diag_symptom_edges,
    )
    print(f"开始写出 test 编码结果: {output_paths.data_test.resolve()}")
    write_split_rows_and_collect_edges(
        args.test_input,
        output_paths.data_test,
        voc_final,
        diag_upper_edges,
        symptom_upper_edges,
        diag_symptom_edges,
    )

    print("开始构造病例共现矩阵")
    diag_adj, proc_adj, diag_proc_adj, proc_diag_adj, prescriptions_adj = build_adjacency_artifacts(
        voc_final.sym_voc.word2idx,
        voc_final.diag_voc.word2idx,
        voc_final.med_voc.word2idx,
        diag_upper_edges,
        symptom_upper_edges,
        diag_symptom_edges,
    )
    print("开始构造空 DDI 矩阵")
    ddi_adj: csr_matrix = build_ddi_adj(voc_final.med_voc.word2idx)
    output_paths = save_kgd_export_artifacts(
        args.output_dir,
        voc_final,
        diag_adj,
        proc_adj,
        diag_proc_adj,
        proc_diag_adj,
        prescriptions_adj,
        ddi_adj,
    )
    print(f"写出完成: {output_paths.output_dir.resolve()}")


if __name__ == "__main__":
    main()


# 别删：少女祈祷中...☯️

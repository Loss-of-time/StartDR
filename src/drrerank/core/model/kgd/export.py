"""KGD 离线导出入口。"""

import argparse
from dataclasses import asdict
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from pickle import dump as pickle_dump
from typing import cast

import dill
import numpy as np
from scipy.sparse import csr_matrix

from drretrieval.core.kg import get_driver

from ...io import load_jsonl
from .common import (
    KGDExportArtifacts,
    KGDIndexedCases,
    KGDOutputPaths,
    KGDSourceCase,
    KGDStringCases,
    KGDVocabulary,
    KGDVocFile,
    build_symmetric_adj,
    normalize_upper_edge,
)

# misc/KGDNet从patient_candidate生成数据文档.md
# misc/KGDNet实现文档.md
# misc/TraceDR-main/TraceDR-model/baseline/data_process/KGDNet_dataprocess.py

LIST_DDI_EDGE_QUERY = """
MATCH (source:`药品`)-[:相互作用]->(:`相互作用`)-[:相互作用]->(:`知识组`)-[:相互作用]->(target:`药品`)
WHERE elementId(source) <> elementId(target)
WITH
    toString(toInteger(last(split(elementId(source), ":")))) AS source_drugid,
    toString(toInteger(last(split(elementId(target), ":")))) AS target_drugid
WHERE source_drugid IN $source_drugids
RETURN DISTINCT source_drugid, target_drugid
"""
KGD_DDI_QUERY_BATCH_SIZE = 512


def build_vocab(rows: list[list[str]]) -> dict[str, int]:
    """构造 1-based 词表。"""

    vocab: dict[str, int] = {}
    for row in rows:
        for value in row:
            if value not in vocab:
                # 目的：保留 KGDNet 的 1-based 词表约定。
                vocab[value] = len(vocab) + 1
    return vocab


def index_rows(
    rows: list[list[str]],
    vocab: dict[str, int],
) -> list[list[int]]:
    """按既有词表将文本行转成索引行。"""

    return [[vocab[value] for value in row] for row in rows]


def load_kgd_string_cases(input_path: Path) -> KGDStringCases:
    """读取单个 split 的 KGD 字符串病例。"""

    # 目的：只抽取 KGD 真正需要的三个字段，避免构造额外候选药对象。
    def parse_case(row: dict[str, object]) -> KGDSourceCase:
        people = cast(dict[str, object], row["people"])
        raw_medicines = cast(list[dict[str, object]], people["medicine"])
        return KGDSourceCase(
            symptoms=list(dict.fromkeys(cast(list[str], people["symptom"]))),
            diagnosis=list(dict.fromkeys(cast(list[str], people["diagnosis"]))),
            medicines=list(
                dict.fromkeys(str(raw_medicine["drugid"]) for raw_medicine in raw_medicines)
            ),
        )

    parsed_cases = load_jsonl(input_path, parse_case)
    return KGDStringCases(
        symptoms=[case.symptoms for case in parsed_cases],
        diagnosis=[case.diagnosis for case in parsed_cases],
        medicines=[case.medicines for case in parsed_cases],
    )


def merge_string_cases(*datasets: KGDStringCases) -> KGDStringCases:
    """合并多个 split 的字符串病例。"""

    merged_symptoms: list[list[str]] = []
    merged_diagnosis: list[list[str]] = []
    merged_medicines: list[list[str]] = []
    for dataset in datasets:
        merged_symptoms.extend(dataset.symptoms)
        merged_diagnosis.extend(dataset.diagnosis)
        merged_medicines.extend(dataset.medicines)
    return KGDStringCases(
        symptoms=merged_symptoms,
        diagnosis=merged_diagnosis,
        medicines=merged_medicines,
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


def build_adj(
    indexed_cases: KGDIndexedCases,
    symptom_vocab: dict[str, int],
    diagnosis_vocab: dict[str, int],
    medicine_vocab: dict[str, int],
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """构造病例共现邻接矩阵。"""

    diag_upper_edges: set[tuple[int, int]] = set()
    symptom_upper_edges: set[tuple[int, int]] = set()
    medicine_upper_edges: set[tuple[int, int]] = set()
    diag_symptom_edges: set[tuple[int, int]] = set()

    for diagnosis_indices, symptom_indices, medicine_indices in zip(
        indexed_cases.diagnosis,
        indexed_cases.symptoms,
        indexed_cases.medicines,
        strict=True,
    ):
        diagnosis_count = len(diagnosis_indices)
        for left_index in range(diagnosis_count):
            left = diagnosis_indices[left_index]
            for right_index in range(left_index + 1, diagnosis_count):
                # 目的：统一共现边方向，保持最终稀疏矩阵的二值语义。
                diag_upper_edges.add(
                    normalize_upper_edge(left, diagnosis_indices[right_index]),
                )

        symptom_count = len(symptom_indices)
        for left_index in range(symptom_count):
            left = symptom_indices[left_index]
            for right_index in range(left_index + 1, symptom_count):
                # 目的：统一共现边方向，保持最终稀疏矩阵的二值语义。
                symptom_upper_edges.add(
                    normalize_upper_edge(left, symptom_indices[right_index]),
                )

        for diagnosis_index in diagnosis_indices:
            for symptom_index in symptom_indices:
                diag_symptom_edges.add((diagnosis_index, symptom_index))

        medicine_count = len(medicine_indices)
        for left_index in range(medicine_count):
            left = medicine_indices[left_index]
            for right_index in range(left_index + 1, medicine_count):
                # 目的：统一共现边方向，保持最终稀疏矩阵的二值语义。
                medicine_upper_edges.add(
                    normalize_upper_edge(left, medicine_indices[right_index]),
                )

    diag_adj = build_symmetric_adj(len(diagnosis_vocab) + 1, diag_upper_edges)
    proc_adj = build_symmetric_adj(len(symptom_vocab) + 1, symptom_upper_edges)
    diag_proc_adj = build_bipartite_adj(
        len(diagnosis_vocab) + 1,
        len(symptom_vocab) + 1,
        diag_symptom_edges,
    )
    proc_diag_adj = diag_proc_adj.transpose().tocsr()
    prescriptions_adj = build_symmetric_adj(len(medicine_vocab) + 1, medicine_upper_edges)
    return diag_adj, proc_adj, diag_proc_adj, proc_diag_adj, prescriptions_adj


def build_ddi_adj(medicine_vocab: dict[str, int]) -> csr_matrix:
    """查询并构造 DDI 邻接矩阵。"""

    medicine_count = len(medicine_vocab) + 1
    if medicine_count == 1:
        return csr_matrix((medicine_count, medicine_count), dtype=np.uint8)

    medicine_drugids = list(medicine_vocab)
    upper_edges: set[tuple[int, int]] = set()
    add_edge = upper_edges.add

    with get_driver() as driver, driver.session() as session:
        for batch_start in range(0, len(medicine_drugids), KGD_DDI_QUERY_BATCH_SIZE):
            source_drugids = medicine_drugids[
                batch_start : batch_start + KGD_DDI_QUERY_BATCH_SIZE
            ]
            result = session.run(LIST_DDI_EDGE_QUERY, source_drugids=source_drugids)
            for record in result:
                row = cast(dict[str, str], record.data())
                source = medicine_vocab[row["source_drugid"]]
                target = medicine_vocab.get(row["target_drugid"])
                if target is None or source == target:
                    continue
                # 目的：统一 DDI 无向边方向，保持最终邻接矩阵的二值语义。
                add_edge(normalize_upper_edge(source, target))

    return build_symmetric_adj(medicine_count, upper_edges)


def main() -> None:
    """离线导出命令行入口。"""

    # 目的：把离线导出主干独立成单文件入口，和运行时构图彻底分开。
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="从 TraceDR 风格 jsonl 导出 KGDNet 离线文件。")
        parser.add_argument("--train-input", type=Path, required=True)
        parser.add_argument("--dev-input", type=Path, required=True)
        parser.add_argument("--test-input", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        return parser.parse_args()

    def build_kgd_export_artifacts(
        train_cases: KGDStringCases,
        dev_cases: KGDStringCases,
        test_cases: KGDStringCases,
    ) -> KGDExportArtifacts:
        # 目的：把离线产物构造编排保留在导出入口内部，避免再次扩散模块级 API。
        def build_vocab_file_entry(vocab: dict[str, int]) -> KGDVocabulary:
            idx2word = [""] * len(vocab)
            for word, index in vocab.items():
                idx2word[index - 1] = word
            return KGDVocabulary(word2idx=vocab, idx2word=idx2word)

        def index_case_rows(
            string_cases: KGDStringCases,
            symptom_vocab: dict[str, int],
            diagnosis_vocab: dict[str, int],
            medicine_vocab: dict[str, int],
        ) -> KGDIndexedCases:
            return KGDIndexedCases(
                symptoms=index_rows(string_cases.symptoms, symptom_vocab),
                diagnosis=index_rows(string_cases.diagnosis, diagnosis_vocab),
                medicines=index_rows(string_cases.medicines, medicine_vocab),
            )

        def indexed_cases_to_rows(indexed_cases: KGDIndexedCases) -> list[list[list[int]]]:
            rows: list[list[list[int]]] = []
            for symptoms, diagnosis, medicines in zip(
                indexed_cases.symptoms,
                indexed_cases.diagnosis,
                indexed_cases.medicines,
                strict=True,
            ):
                rows.append([symptoms, diagnosis, medicines])
            return rows

        all_string_cases = merge_string_cases(train_cases, dev_cases, test_cases)
        symptom_vocab = build_vocab(all_string_cases.symptoms)
        diagnosis_vocab = build_vocab(all_string_cases.diagnosis)
        medicine_vocab = build_vocab(all_string_cases.medicines)

        train_indexed_cases = index_case_rows(
            train_cases,
            symptom_vocab,
            diagnosis_vocab,
            medicine_vocab,
        )
        dev_indexed_cases = index_case_rows(
            dev_cases,
            symptom_vocab,
            diagnosis_vocab,
            medicine_vocab,
        )
        test_indexed_cases = index_case_rows(
            test_cases,
            symptom_vocab,
            diagnosis_vocab,
            medicine_vocab,
        )
        all_indexed_cases = index_case_rows(
            all_string_cases,
            symptom_vocab,
            diagnosis_vocab,
            medicine_vocab,
        )

        diag_adj, proc_adj, diag_proc_adj, proc_diag_adj, prescriptions_adj = build_adj(
            all_indexed_cases,
            symptom_vocab,
            diagnosis_vocab,
            medicine_vocab,
        )
        ddi_adj = build_ddi_adj(medicine_vocab)
        voc_final = KGDVocFile(
            sym_voc=build_vocab_file_entry(symptom_vocab),
            diag_voc=build_vocab_file_entry(diagnosis_vocab),
            med_voc=build_vocab_file_entry(medicine_vocab),
        )
        return KGDExportArtifacts(
            voc_final=voc_final,
            data_train=indexed_cases_to_rows(train_indexed_cases),
            data_eval=indexed_cases_to_rows(dev_indexed_cases),
            data_test=indexed_cases_to_rows(test_indexed_cases),
            diag_adj=diag_adj,
            proc_adj=proc_adj,
            diag_proc_adj=diag_proc_adj,
            proc_diag_adj=proc_diag_adj,
            prescriptions_adj=prescriptions_adj,
            ddi_A_final=ddi_adj,
        )

    def save_kgd_export_artifacts(
        output_dir: Path,
        artifacts: KGDExportArtifacts,
    ) -> KGDOutputPaths:
        # 目的：把离线产物写盘流程留在导出入口文件，保持目录协议集中维护。
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = KGDOutputPaths(
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

        with output_paths.voc_final.open("wb") as file:
            dill.dump(asdict(artifacts.voc_final), file, protocol=HIGHEST_PROTOCOL)
        with output_paths.data_train.open("wb") as file:
            dill.dump(artifacts.data_train, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.data_eval.open("wb") as file:
            dill.dump(artifacts.data_eval, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.data_test.open("wb") as file:
            dill.dump(artifacts.data_test, file, protocol=HIGHEST_PROTOCOL)

        with output_paths.diag_adj.open("wb") as file:
            pickle_dump(artifacts.diag_adj, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.proc_adj.open("wb") as file:
            pickle_dump(artifacts.proc_adj, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.diag_proc_adj.open("wb") as file:
            pickle_dump(artifacts.diag_proc_adj, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.proc_diag_adj.open("wb") as file:
            pickle_dump(artifacts.proc_diag_adj, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.prescriptions_adj.open("wb") as file:
            pickle_dump(artifacts.prescriptions_adj, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.ddi_A_final.open("wb") as file:
            pickle_dump(artifacts.ddi_A_final, file, protocol=HIGHEST_PROTOCOL)

        return output_paths

    args = parse_args()
    print(f"开始读取 train 数据: {args.train_input.resolve()}")
    train_cases = load_kgd_string_cases(args.train_input)
    print(f"开始读取 dev 数据: {args.dev_input.resolve()}")
    dev_cases = load_kgd_string_cases(args.dev_input)
    print(f"开始读取 test 数据: {args.test_input.resolve()}")
    test_cases = load_kgd_string_cases(args.test_input)
    print(
        "读取完成: "
        f"train={len(train_cases.symptoms)}, "
        f"dev={len(dev_cases.symptoms)}, "
        f"test={len(test_cases.symptoms)}",
    )
    print("开始构造 KGDNet 离线产物")
    artifacts = build_kgd_export_artifacts(train_cases, dev_cases, test_cases)
    output_paths = save_kgd_export_artifacts(args.output_dir, artifacts)
    print(f"写出完成: {output_paths.output_dir.resolve()}")


if __name__ == "__main__":
    main()


# 别删：少女祈祷中...☯️

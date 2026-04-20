"""4SDrug 离线数据导出流程。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from pickle import HIGHEST_PROTOCOL

import dill
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from ...schema import TraceDRSample
from ...tracedr import load_tracedr_samples
from .common import (
    FourSDrugBatchData,
    FourSDrugExportArtifacts,
    FourSDrugIndexedCases,
    FourSDrugOutputPaths,
    FourSDrugSourceCase,
    FourSDrugStringCases,
    FourSDrugVocabulary,
    FourSDrugVocFile,
)

# misc/4sdrug实现与复现指导.md
# misc/TraceDR-main/TraceDR-model/baseline/4sdrug/utils/dataset2.py
# misc/TraceDR-main/TraceDR-model/baseline/data_process/Drugrec_data_process.py


@dataclass(slots=True)
class FourSDrugExportConfig:
    """4SDrug 离线导出配置。"""

    train_input: Path
    dev_input: Path
    test_input: Path
    output_dir: Path
    batch_sizes: list[int]


def build_vocab(rows: list[list[str]]) -> dict[str, int]:
    """构造 1-based 词表。

    Args:
        rows: 待编码字段的二维字符串列表。

    Returns:
        从 1 开始编号的词表。
    """

    unique_values: set[str] = set()
    row: list[str]
    for row in rows:
        unique_values.update(row)

    vocabulary: dict[str, int] = {}
    index: int
    value: str
    for index, value in enumerate(sorted(unique_values), start=1):
        # 目的：保持 4SDrug 原始预处理阶段的 1-based 编号与排序建表习惯。
        vocabulary[value] = index
    return vocabulary


def index_rows(
    rows: list[list[str]],
    vocabulary: dict[str, int],
) -> list[list[int]]:
    """按既有词表将文本行转成索引行。

    Args:
        rows: 字符串行。
        vocabulary: 词表。

    Returns:
        对应的索引行。
    """

    indexed_rows: list[list[int]] = []
    row: list[str]
    for row in rows:
        indexed_rows.append([vocabulary[value] for value in row])
    return indexed_rows


def load_4sdrug_string_cases(input_path: Path) -> FourSDrugStringCases:
    """读取单个 split 的 4SDrug 字符串病例。

    Args:
        input_path: TraceDR 风格 `jsonl` 路径。

    Returns:
        只保留 4SDrug 需要字段的病例集合。
    """

    samples = load_tracedr_samples(input_path)
    source_cases: list[FourSDrugSourceCase] = []
    source_case: FourSDrugSourceCase
    sample: TraceDRSample
    for sample in samples:
        # 目的：严格对齐 4SDrug 原始直接推荐口径，只使用金标准药物集合。
        source_case = FourSDrugSourceCase(
            symptoms=list(dict.fromkeys(sample.people.symptom)),
            diagnosis=list(dict.fromkeys(sample.people.diagnosis)),
            medicines=list(
                dict.fromkeys(medicine.drugid for medicine in sample.people.medicine),
            ),
        )
        source_cases.append(source_case)
    return FourSDrugStringCases(
        symptoms=[case.symptoms for case in source_cases],
        diagnosis=[case.diagnosis for case in source_cases],
        medicines=[case.medicines for case in source_cases],
    )


def merge_string_cases(*datasets: FourSDrugStringCases) -> FourSDrugStringCases:
    """合并多个 split 的字符串病例。

    Args:
        *datasets: 待合并的字符串病例集合。

    Returns:
        合并后的字符串病例集合。
    """

    merged_symptoms: list[list[str]] = []
    merged_diagnosis: list[list[str]] = []
    merged_medicines: list[list[str]] = []
    dataset: FourSDrugStringCases
    for dataset in datasets:
        merged_symptoms.extend(dataset.symptoms)
        merged_diagnosis.extend(dataset.diagnosis)
        merged_medicines.extend(dataset.medicines)
    return FourSDrugStringCases(
        symptoms=merged_symptoms,
        diagnosis=merged_diagnosis,
        medicines=merged_medicines,
    )


def build_vocab_file_entry(vocabulary: dict[str, int]) -> FourSDrugVocabulary:
    """构造词表文件条目。

    Args:
        vocabulary: 1-based 词表。

    Returns:
        可直接写入 `voc_final.pkl` 的词表对象。
    """

    idx2word: list[str] = [""] * len(vocabulary)
    word: str
    index: int
    for word, index in vocabulary.items():
        idx2word[index - 1] = word
    return FourSDrugVocabulary(word2idx=vocabulary, idx2word=idx2word)


def index_case_rows(
    string_cases: FourSDrugStringCases,
    symptom_vocab: dict[str, int],
    diagnosis_vocab: dict[str, int],
    medicine_vocab: dict[str, int],
) -> FourSDrugIndexedCases:
    """把字符串病例按词表转成索引病例。

    Args:
        string_cases: 字符串病例集合。
        symptom_vocab: 症状词表。
        diagnosis_vocab: 诊断词表。
        medicine_vocab: 药物词表。

    Returns:
        索引化后的病例集合。
    """

    return FourSDrugIndexedCases(
        symptoms=index_rows(string_cases.symptoms, symptom_vocab),
        diagnosis=index_rows(string_cases.diagnosis, diagnosis_vocab),
        medicines=index_rows(string_cases.medicines, medicine_vocab),
    )


def indexed_cases_to_rows(indexed_cases: FourSDrugIndexedCases) -> list[list[list[int]]]:
    """把索引病例转成 4SDrug 原始 `pkl` 行结构。

    Args:
        indexed_cases: 索引化病例集合。

    Returns:
        `[symptoms, diagnosis, medicines]` 形式的样本列表。
    """

    rows: list[list[list[int]]] = []
    symptoms: list[int]
    diagnosis: list[int]
    medicines: list[int]
    for symptoms, diagnosis, medicines in zip(
        indexed_cases.symptoms,
        indexed_cases.diagnosis,
        indexed_cases.medicines,
        strict=True,
    ):
        rows.append([symptoms, diagnosis, medicines])
    return rows


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


def build_drug_multihots(
    train_rows: list[list[list[int]]],
    medicine_vocab_size: int,
) -> csr_matrix:
    """构造训练集药物 multi-hot 稀疏矩阵。

    Args:
        train_rows: `data_train.pkl` 行结构。
        medicine_vocab_size: 药物词表大小，不含 0 占位。

    Returns:
        训练集药物 multi-hot CSR 稀疏矩阵。
    """

    row_indices: list[int] = []
    col_indices: list[int] = []
    sample_index: int
    row: list[list[int]]
    medicine_id: int

    for sample_index, row in enumerate(train_rows):
        for medicine_id in row[2]:
            row_indices.append(sample_index)
            # 目的：保留样本中的 1-based 药物 id，同时把 multi-hot 列索引转换成 0-based。
            col_indices.append(medicine_id - 1)

    data: npt.NDArray[np.int8] = np.ones(len(row_indices), dtype=np.int8)
    return csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(train_rows), medicine_vocab_size),
        dtype=np.int8,
    )


def build_batched_training_data(
    train_rows: list[list[list[int]]],
    batch_size: int,
    medicine_vocab_size: int,
) -> FourSDrugBatchData:
    """按症状长度分桶并切分训练 batch。

    Args:
        train_rows: `data_train.pkl` 行结构。
        batch_size: 导出的 batch size。
        medicine_vocab_size: 药物词表大小，不含 0 占位。

    Returns:
        4SDrug 训练脚本可直接读取的分桶 batch 数据。
    """

    sym_groups: dict[int, list[list[int]]] = {}
    drug_groups: dict[int, list[npt.NDArray[np.bool_]]] = {}
    row: list[list[int]]
    symptoms: list[int]
    medicines: list[int]
    symptom_count: int
    drug_multihot: npt.NDArray[np.bool_]

    for row in train_rows:
        symptoms = row[0]
        medicines = row[2]
        symptom_count = len(symptoms)
        if symptom_count not in sym_groups:
            sym_groups[symptom_count] = []
            drug_groups[symptom_count] = []

        drug_multihot = np.zeros(medicine_vocab_size, dtype=np.bool_)
        if medicines:
            drug_multihot[np.asarray(medicines, dtype=np.int64) - 1] = True
        sym_groups[symptom_count].append(symptoms)
        drug_groups[symptom_count].append(drug_multihot)

    sym_train: list[list[list[int]]] = []
    drug_train: list[list[npt.NDArray[np.bool_]]] = []
    symptom_length: int
    sym_group: list[list[int]]
    batch_start: int
    batch_end: int

    for symptom_length in sorted(sym_groups):
        sym_group = sym_groups[symptom_length]
        for batch_start in range(0, len(sym_group), batch_size):
            batch_end = batch_start + batch_size
            sym_train.append(sym_group[batch_start:batch_end])
            drug_train.append(drug_groups[symptom_length][batch_start:batch_end])

    return FourSDrugBatchData(sym_train=sym_train, drug_train=drug_train)


def build_4sdrug_export_artifacts(
    train_cases: FourSDrugStringCases,
    dev_cases: FourSDrugStringCases,
    test_cases: FourSDrugStringCases,
    batch_sizes: list[int],
) -> FourSDrugExportArtifacts:
    """构造 4SDrug 全部离线产物。

    Args:
        train_cases: 训练集字符串病例。
        dev_cases: 验证集字符串病例。
        test_cases: 测试集字符串病例。
        batch_sizes: 需要导出的 batch size 列表。

    Returns:
        写盘前的全部 4SDrug 产物。
    """

    all_string_cases: FourSDrugStringCases = merge_string_cases(train_cases, dev_cases, test_cases)
    symptom_vocab: dict[str, int] = build_vocab(all_string_cases.symptoms)
    diagnosis_vocab: dict[str, int] = build_vocab(all_string_cases.diagnosis)
    medicine_vocab: dict[str, int] = build_vocab(all_string_cases.medicines)

    train_indexed_cases: FourSDrugIndexedCases = index_case_rows(
        train_cases,
        symptom_vocab,
        diagnosis_vocab,
        medicine_vocab,
    )
    dev_indexed_cases: FourSDrugIndexedCases = index_case_rows(
        dev_cases,
        symptom_vocab,
        diagnosis_vocab,
        medicine_vocab,
    )
    test_indexed_cases: FourSDrugIndexedCases = index_case_rows(
        test_cases,
        symptom_vocab,
        diagnosis_vocab,
        medicine_vocab,
    )

    data_train: list[list[list[int]]] = indexed_cases_to_rows(train_indexed_cases)
    data_eval: list[list[list[int]]] = indexed_cases_to_rows(dev_indexed_cases)
    data_test: list[list[list[int]]] = indexed_cases_to_rows(test_indexed_cases)
    sym_sets: list[list[int]] = [row[0] for row in data_train]
    drug_multihots: csr_matrix = build_drug_multihots(data_train, len(medicine_vocab))

    unique_batch_sizes: list[int] = sorted(set(batch_sizes))
    batch_data: dict[int, FourSDrugBatchData] = {}
    batch_size: int
    for batch_size in unique_batch_sizes:
        batch_data[batch_size] = build_batched_training_data(
            data_train,
            batch_size,
            len(medicine_vocab),
        )

    voc_final: FourSDrugVocFile = FourSDrugVocFile(
        sym_voc=build_vocab_file_entry(symptom_vocab),
        diag_voc=build_vocab_file_entry(diagnosis_vocab),
        med_voc=build_vocab_file_entry(medicine_vocab),
    )
    ddi_A_final: csr_matrix = build_ddi_adj(medicine_vocab)
    return FourSDrugExportArtifacts(
        voc_final=voc_final,
        data_train=data_train,
        data_eval=data_eval,
        data_test=data_test,
        ddi_A_final=ddi_A_final,
        sym_sets=sym_sets,
        drug_multihots=drug_multihots,
        batch_data=batch_data,
    )


def save_4sdrug_export_artifacts(
    output_dir: Path,
    artifacts: FourSDrugExportArtifacts,
) -> FourSDrugOutputPaths:
    """把 4SDrug 离线产物写入输出目录。

    Args:
        output_dir: 输出目录。
        artifacts: 待写出的 4SDrug 产物。

    Returns:
        实际写出的路径集合。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    sym_train_paths: dict[int, Path] = {}
    drug_train_paths: dict[int, Path] = {}
    batch_size: int

    for batch_size in artifacts.batch_data:
        sym_train_paths[batch_size] = output_dir / f"sym_train_{batch_size}.pkl"
        drug_train_paths[batch_size] = output_dir / f"drug_train_{batch_size}.pkl"

    output_paths: FourSDrugOutputPaths = FourSDrugOutputPaths(
        output_dir=output_dir,
        voc_final=output_dir / "voc_final.pkl",
        data_train=output_dir / "data_train.pkl",
        data_eval=output_dir / "data_eval.pkl",
        data_test=output_dir / "data_test.pkl",
        ddi_A_final=output_dir / "ddi_A_final.pkl",
        sym_sets=output_dir / "sym_sets.pkl",
        drug_multihots=output_dir / "drug_multihots.pkl",
        sym_train=sym_train_paths,
        drug_train=drug_train_paths,
    )

    with output_paths.voc_final.open("wb") as file:
        dill.dump(asdict(artifacts.voc_final), file, protocol=HIGHEST_PROTOCOL)
    with output_paths.data_train.open("wb") as file:
        dill.dump(artifacts.data_train, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.data_eval.open("wb") as file:
        dill.dump(artifacts.data_eval, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.data_test.open("wb") as file:
        dill.dump(artifacts.data_test, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.ddi_A_final.open("wb") as file:
        dill.dump(artifacts.ddi_A_final, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.sym_sets.open("wb") as file:
        dill.dump(artifacts.sym_sets, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.drug_multihots.open("wb") as file:
        dill.dump(artifacts.drug_multihots, file, protocol=HIGHEST_PROTOCOL)

    batch_data: FourSDrugBatchData
    for batch_size, batch_data in artifacts.batch_data.items():
        with output_paths.sym_train[batch_size].open("wb") as file:
            dill.dump(batch_data.sym_train, file, protocol=HIGHEST_PROTOCOL)
        with output_paths.drug_train[batch_size].open("wb") as file:
            dill.dump(batch_data.drug_train, file, protocol=HIGHEST_PROTOCOL)

    return output_paths


def export_dataset(config: FourSDrugExportConfig) -> FourSDrugOutputPaths:
    """执行 4SDrug 离线导出。"""

    print(f"开始读取 train 数据: {config.train_input.resolve()}")
    train_cases: FourSDrugStringCases = load_4sdrug_string_cases(config.train_input)
    print(f"开始读取 dev 数据: {config.dev_input.resolve()}")
    dev_cases: FourSDrugStringCases = load_4sdrug_string_cases(config.dev_input)
    print(f"开始读取 test 数据: {config.test_input.resolve()}")
    test_cases: FourSDrugStringCases = load_4sdrug_string_cases(config.test_input)
    print(
        "读取完成: "
        f"train={len(train_cases.symptoms)}, "
        f"dev={len(dev_cases.symptoms)}, "
        f"test={len(test_cases.symptoms)}",
    )
    print(f"开始构造 4SDrug 离线产物，batch_sizes={sorted(set(config.batch_sizes))}")
    artifacts: FourSDrugExportArtifacts = build_4sdrug_export_artifacts(
        train_cases,
        dev_cases,
        test_cases,
        config.batch_sizes,
    )
    output_paths: FourSDrugOutputPaths = save_4sdrug_export_artifacts(
        config.output_dir,
        artifacts,
    )
    print(f"写出完成: {output_paths.output_dir.resolve()}")
    return output_paths

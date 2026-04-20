"""4SDrug 离线数据导出流程。"""

import json
import shutil
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from typing import cast

import dill
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from ...io import (
    PickleRowStreamWriter,
    iter_pickle_rows,
    open_pickle_row_stream_writer,
    write_pickle_row_stream,
)
from .common import (
    FourSDrugBatchData,
    FourSDrugDrugMultiHot,
    FourSDrugOutputPaths,
    FourSDrugSourceCase,
    FourSDrugVocabulary,
    FourSDrugVocFile,
)

# misc/4sdrug实现与复现指导.md
# misc/TraceDR-main/TraceDR-model/baseline/4sdrug/utils/dataset2.py
# misc/TraceDR-main/TraceDR-model/baseline/data_process/Drugrec_data_process.py

type FourSDrugIndexedRow = list[list[int]]
type FourSDrugBucketRow = list[list[int]]


@dataclass(slots=True)
class FourSDrugExportConfig:
    """4SDrug 离线导出配置。"""

    train_input: Path
    dev_input: Path
    test_input: Path
    output_dir: Path
    batch_sizes: list[int]


@dataclass(slots=True)
class _FourSDrugBucketStreamWriter:
    """管理训练分桶阶段的流式写盘状态。"""

    stack: ExitStack
    temp_dir: Path
    sym_sets_writer: PickleRowStreamWriter
    bucket_paths: dict[int, Path]
    bucket_writers: dict[int, PickleRowStreamWriter]

    @classmethod
    def create(
        cls,
        stack: ExitStack,
        temp_dir: Path,
        sym_sets_path: Path,
    ) -> "_FourSDrugBucketStreamWriter":
        """创建训练分桶阶段需要的多路流式写盘器。"""

        temp_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            stack=stack,
            temp_dir=temp_dir,
            sym_sets_writer=stack.enter_context(open_pickle_row_stream_writer(sym_sets_path)),
            bucket_paths={},
            bucket_writers={},
        )

    def write_case(self, symptoms: list[int], medicines: list[int]) -> None:
        """写入单条训练病例到症状集合流与分桶流。"""

        self.sym_sets_writer.write(symptoms)
        self._get_bucket_writer(len(symptoms)).write([symptoms, medicines])

    def _get_bucket_writer(self, symptom_count: int) -> PickleRowStreamWriter:
        if symptom_count not in self.bucket_writers:
            bucket_path: Path = self.temp_dir / f"symptom_count_{symptom_count}.pkl"
            self.bucket_paths[symptom_count] = bucket_path
            self.bucket_writers[symptom_count] = self.stack.enter_context(
                open_pickle_row_stream_writer(bucket_path),
            )
        return self.bucket_writers[symptom_count]


@dataclass(slots=True)
class _FourSDrugBatchStreamWriter:
    """管理单个 batch size 的批训练缓存写盘。"""

    batch_size: int
    medicine_vocab_size: int
    sym_writer: PickleRowStreamWriter
    drug_writer: PickleRowStreamWriter
    symptom_batch: list[list[int]]
    drug_batch: list[FourSDrugDrugMultiHot]

    @classmethod
    def create(
        cls,
        stack: ExitStack,
        output_paths: FourSDrugOutputPaths,
        batch_size: int,
        medicine_vocab_size: int,
    ) -> "_FourSDrugBatchStreamWriter":
        """创建单个 batch size 的批量写盘器。"""

        return cls(
            batch_size=batch_size,
            medicine_vocab_size=medicine_vocab_size,
            sym_writer=stack.enter_context(
                open_pickle_row_stream_writer(output_paths.sym_train[batch_size]),
            ),
            drug_writer=stack.enter_context(
                open_pickle_row_stream_writer(output_paths.drug_train[batch_size]),
            ),
            symptom_batch=[],
            drug_batch=[],
        )

    def append_case(self, symptoms: list[int], medicines: list[int]) -> None:
        """追加单条病例，达到 batch 阈值时立即落盘。"""

        self.symptom_batch.append(symptoms)
        self.drug_batch.append(build_drug_multihot(medicines, self.medicine_vocab_size))
        if len(self.symptom_batch) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """把当前缓存 batch 写入输出流。"""

        if not self.symptom_batch:
            return
        self.sym_writer.write(self.symptom_batch)
        self.drug_writer.write(self.drug_batch)
        self.symptom_batch = []
        self.drug_batch = []


def update_vocab(unique_values: set[str], row: list[str]) -> None:
    """按单行字段增量更新词表值集合。

    Args:
        unique_values: 待更新的去重值集合。
        row: 当前字段行。
    """

    unique_values.update(row)


def parse_4sdrug_source_case(row: dict[str, object]) -> FourSDrugSourceCase:
    """从单条 TraceDR 样本提取 4SDrug 所需字段。

    Args:
        row: 原始 `jsonl` 行对象。

    Returns:
        4SDrug 导出所需的最小病例对象。
    """

    people: dict[str, object] = cast(dict[str, object], row["people"])
    raw_medicines: list[dict[str, object]] = cast(list[dict[str, object]], people["medicine"])
    return FourSDrugSourceCase(
        symptoms=list(dict.fromkeys(cast(list[str], people["symptom"]))),
        diagnosis=list(dict.fromkeys(cast(list[str], people["diagnosis"]))),
        medicines=list(
            dict.fromkeys(str(raw_medicine["drugid"]) for raw_medicine in raw_medicines)
        ),
    )


def iter_4sdrug_source_cases(input_path: Path) -> Iterator[FourSDrugSourceCase]:
    """流式遍历单个 split 的 4SDrug 字符串病例。

    Args:
        input_path: TraceDR 风格 `jsonl` 路径。

    Yields:
        单条 4SDrug 病例。
    """

    with input_path.open(encoding="utf-8") as file:
        line: str
        for line in file:
            yield parse_4sdrug_source_case(cast(dict[str, object], json.loads(line)))


def build_vocab(unique_values: set[str]) -> dict[str, int]:
    """根据去重值集合构造 1-based 词表。

    Args:
        unique_values: 已去重的字段值集合。

    Returns:
        从 1 开始编号的词表。
    """

    vocabulary: dict[str, int] = {}
    index: int
    value: str
    for index, value in enumerate(sorted(unique_values), start=1):
        # 目的：保持 4SDrug 原始预处理阶段的稳定排序与 1-based 编号习惯。
        vocabulary[value] = index
    return vocabulary


def index_row(
    row: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """按既有词表将文本行转成索引行。

    Args:
        row: 字符串行。
        vocabulary: 词表。

    Returns:
        对应的索引行。
    """

    return [vocabulary[value] for value in row]


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


def indexed_case_to_row(
    source_case: FourSDrugSourceCase,
    voc_final: FourSDrugVocFile,
) -> list[list[int]]:
    """把单条病例编码成 4SDrug 原始 `pkl` 行结构。

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


def build_drug_multihot(
    medicines: list[int],
    medicine_vocab_size: int,
) -> npt.NDArray[np.bool_]:
    """把药物集合转换成单条 multi-hot 向量。"""

    drug_multihot: npt.NDArray[np.bool_] = np.zeros(medicine_vocab_size, dtype=np.bool_)
    if medicines:
        drug_multihot[np.asarray(medicines, dtype=np.int64) - 1] = True
    return drug_multihot


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

        drug_multihot = build_drug_multihot(medicines, medicine_vocab_size)
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


def build_output_paths(
    output_dir: Path,
    batch_sizes: list[int],
) -> FourSDrugOutputPaths:
    """构造 4SDrug 导出输出路径集合。"""

    sym_train_paths: dict[int, Path] = {}
    drug_train_paths: dict[int, Path] = {}
    batch_size: int
    for batch_size in sorted(set(batch_sizes)):
        sym_train_paths[batch_size] = output_dir / f"sym_train_{batch_size}.pkl"
        drug_train_paths[batch_size] = output_dir / f"drug_train_{batch_size}.pkl"

    return FourSDrugOutputPaths(
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


def build_vocabulary_from_inputs(
    train_input: Path,
    dev_input: Path,
    test_input: Path,
) -> tuple[FourSDrugVocFile, tuple[int, int, int]]:
    """按多份输入流式构造词表。

    Args:
        train_input: 训练集路径。
        dev_input: 验证集路径。
        test_input: 测试集路径。

    Returns:
        词表文件对象，以及各 split 的样本数。
    """

    symptom_values: set[str] = set()
    diagnosis_values: set[str] = set()
    medicine_values: set[str] = set()
    split_counts: list[int] = []
    input_path: Path
    for input_path in [train_input, dev_input, test_input]:
        sample_count: int = 0
        source_case: FourSDrugSourceCase
        for source_case in iter_4sdrug_source_cases(input_path):
            update_vocab(symptom_values, source_case.symptoms)
            update_vocab(diagnosis_values, source_case.diagnosis)
            update_vocab(medicine_values, source_case.medicines)
            sample_count += 1
        split_counts.append(sample_count)

    return (
        FourSDrugVocFile(
            sym_voc=build_vocab_file_entry(build_vocab(symptom_values)),
            diag_voc=build_vocab_file_entry(build_vocab(diagnosis_values)),
            med_voc=build_vocab_file_entry(build_vocab(medicine_values)),
        ),
        (split_counts[0], split_counts[1], split_counts[2]),
    )


def write_split_rows(
    input_path: Path,
    output_path: Path,
    voc_final: FourSDrugVocFile,
) -> int:
    """写出单个 split 的编码结果。"""

    def iter_indexed_rows() -> Iterator[list[list[int]]]:
        source_case: FourSDrugSourceCase
        for source_case in iter_4sdrug_source_cases(input_path):
            yield indexed_case_to_row(source_case, voc_final)

    return write_pickle_row_stream(output_path, iter_indexed_rows())


def iter_bucket_cases(bucket_path: Path) -> Iterator[tuple[list[int], list[int]]]:
    """流式读取单个症状长度桶中的训练病例。"""

    bucket_row: FourSDrugBucketRow
    for bucket_row in iter_pickle_rows(bucket_path):
        yield bucket_row[0], bucket_row[1]


def build_train_bucket_paths(
    temp_dir: Path,
    train_rows_path: Path,
    sym_sets_path: Path,
    medicine_vocab_size: int,
) -> tuple[csr_matrix, dict[int, Path]]:
    """从 `data_train.pkl` 流式生成训练侧衍生文件的中间桶。"""

    bucket_paths: dict[int, Path] = {}
    indices: list[int] = []
    indptr: list[int] = [0]
    row_count: int = 0

    with ExitStack() as stack:
        # 目的：把多路桶文件的打开/关闭从导出循环里剥离，主流程只保留病例分桶语义。
        bucket_stream_writer: _FourSDrugBucketStreamWriter = _FourSDrugBucketStreamWriter.create(
            stack=stack,
            temp_dir=temp_dir,
            sym_sets_path=sym_sets_path,
        )
        bucket_paths = bucket_stream_writer.bucket_paths

        row: FourSDrugIndexedRow
        for row in iter_pickle_rows(train_rows_path):
            symptoms: list[int] = row[0]
            medicines: list[int] = row[2]
            bucket_stream_writer.write_case(symptoms, medicines)

            medicine_id: int
            for medicine_id in medicines:
                indices.append(medicine_id - 1)
            indptr.append(len(indices))
            row_count += 1

    data: npt.NDArray[np.int8] = np.ones(len(indices), dtype=np.int8)
    return (
        csr_matrix(
            (
                data,
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(row_count, medicine_vocab_size),
            dtype=np.int8,
        ),
        bucket_paths,
    )


def write_batched_training_files(
    output_paths: FourSDrugOutputPaths,
    bucket_paths: dict[int, Path],
    batch_sizes: list[int],
    medicine_vocab_size: int,
) -> None:
    """根据训练桶文件流式写出批训练缓存。"""

    batch_size: int
    for batch_size in sorted(set(batch_sizes)):
        with ExitStack() as stack:
            # 目的：把 batch 缓冲与落盘细节封装起来，避免业务循环直接操作流式 writer。
            batch_writer: _FourSDrugBatchStreamWriter = _FourSDrugBatchStreamWriter.create(
                stack=stack,
                output_paths=output_paths,
                batch_size=batch_size,
                medicine_vocab_size=medicine_vocab_size,
            )

            symptom_count: int
            for symptom_count in sorted(bucket_paths):
                symptoms: list[int]
                medicines: list[int]
                for symptoms, medicines in iter_bucket_cases(bucket_paths[symptom_count]):
                    batch_writer.append_case(symptoms, medicines)
                # 目的：保持不同症状长度之间的 batch 不混桶，复用原始 4SDrug 分桶语义。
                batch_writer.flush()


def save_static_artifacts(
    output_paths: FourSDrugOutputPaths,
    voc_final: FourSDrugVocFile,
    ddi_adj: csr_matrix,
    drug_multihots: csr_matrix,
) -> None:
    """保存 4SDrug 静态导出产物。"""

    with output_paths.voc_final.open("wb") as file:
        dill.dump(asdict(voc_final), file, protocol=HIGHEST_PROTOCOL)
    with output_paths.ddi_A_final.open("wb") as file:
        dill.dump(ddi_adj, file, protocol=HIGHEST_PROTOCOL)
    with output_paths.drug_multihots.open("wb") as file:
        dill.dump(drug_multihots, file, protocol=HIGHEST_PROTOCOL)


def export_dataset(config: FourSDrugExportConfig) -> FourSDrugOutputPaths:
    """执行 4SDrug 离线导出。"""

    unique_batch_sizes: list[int] = sorted(set(config.batch_sizes))
    output_paths: FourSDrugOutputPaths = build_output_paths(config.output_dir, unique_batch_sizes)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("开始流式构造 4SDrug 词表")
    voc_final, split_counts = build_vocabulary_from_inputs(
        config.train_input,
        config.dev_input,
        config.test_input,
    )
    print(
        f"词表构造完成: train={split_counts[0]}, dev={split_counts[1]}, test={split_counts[2]}",
    )

    print(f"开始写出 train 编码结果: {output_paths.data_train.resolve()}")
    write_split_rows(config.train_input, output_paths.data_train, voc_final)
    print(f"开始写出 dev 编码结果: {output_paths.data_eval.resolve()}")
    write_split_rows(config.dev_input, output_paths.data_eval, voc_final)
    print(f"开始写出 test 编码结果: {output_paths.data_test.resolve()}")
    write_split_rows(config.test_input, output_paths.data_test, voc_final)

    print(f"开始构造训练侧衍生产物，batch_sizes={unique_batch_sizes}")
    temp_dir: Path = output_paths.output_dir / ".foursdrug_export_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    # 目的：以磁盘分桶替代整表驻留内存，压低本地导出峰值。
    drug_multihots, bucket_paths = build_train_bucket_paths(
        temp_dir=temp_dir,
        train_rows_path=output_paths.data_train,
        sym_sets_path=output_paths.sym_sets,
        medicine_vocab_size=len(voc_final.med_voc.idx2word),
    )
    write_batched_training_files(
        output_paths=output_paths,
        bucket_paths=bucket_paths,
        batch_sizes=unique_batch_sizes,
        medicine_vocab_size=len(voc_final.med_voc.idx2word),
    )
    shutil.rmtree(temp_dir)

    print("开始构造空 DDI 矩阵")
    ddi_adj: csr_matrix = build_ddi_adj(voc_final.med_voc.word2idx)
    save_static_artifacts(output_paths, voc_final, ddi_adj, drug_multihots)
    print(f"写出完成: {output_paths.output_dir.resolve()}")
    return output_paths

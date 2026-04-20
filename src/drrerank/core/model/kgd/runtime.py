"""KGD 运行时构图入口。"""

from dataclasses import asdict
from pathlib import Path
from typing import cast

import dill
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

from ...io import load_pickle, load_pickle_rows
from .common import (
    KGDAdmissionInfo,
    KGDEHRGraphs,
    KGDInputPaths,
    KGDPatientInfo,
    KGDRelationKey,
    KGDRelationPayload,
    KGDRuntimeArtifacts,
    KGDVocabulary,
    KGDVocFile,
    build_relation_payload,
    empty_edge_ids,
    empty_edge_index,
    sparse_matrix_to_edge_index,
)


def get_ehr_data(
    device: torch.device,
    input_dir: Path,
) -> tuple[KGDPatientInfo, KGDEHRGraphs, Data, csr_matrix, dict[str, object], int, int]:
    """从离线产物重建 KGDNet 运行时输入。"""

    # 目的：把运行时构图主干独立成单文件入口，和离线导出彻底分开。
    runtime_artifacts = load_kgd_runtime_artifacts(input_dir)
    num_diag_nodes = len(runtime_artifacts.voc.diag_voc.idx2word)
    num_proc_nodes = len(runtime_artifacts.voc.sym_voc.idx2word)
    num_clinical_nodes = num_diag_nodes + num_proc_nodes
    num_med_nodes = len(runtime_artifacts.voc.med_voc.idx2word)

    clinical_edges = build_global_clinical_edges(
        diag_adj=runtime_artifacts.diag_adj,
        proc_adj=runtime_artifacts.proc_adj,
        diag_proc_adj=runtime_artifacts.diag_proc_adj,
        proc_diag_adj=runtime_artifacts.proc_diag_adj,
        num_diag_nodes=num_diag_nodes,
        device=device,
    )
    ddi_edge_index = sparse_matrix_to_edge_index(runtime_artifacts.ddi_adj, device)
    patient_info = build_patient_info(
        ehr_records=runtime_artifacts.ehr_records,
        num_clinical_nodes=num_clinical_nodes,
        num_med_nodes=num_med_nodes,
        device=device,
    )
    ddi_kg = build_ddi_kg(
        ddi_edge_index=ddi_edge_index,
        num_med_nodes=num_med_nodes,
        device=device,
    )
    ehr_kgs = build_ehr_kgs(
        patient_info=patient_info,
        clinical_edges=clinical_edges,
        num_clinical_nodes=num_clinical_nodes,
        num_med_nodes=num_med_nodes,
        device=device,
    )
    return (
        patient_info,
        ehr_kgs,
        ddi_kg,
        runtime_artifacts.ddi_adj,
        cast(dict[str, object], asdict(runtime_artifacts.voc)),
        num_clinical_nodes,
        num_med_nodes,
    )


def load_kgd_runtime_artifacts(input_dir: Path) -> KGDRuntimeArtifacts:
    """加载运行时构图所需的全部离线产物。"""

    input_paths = KGDInputPaths(
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
    with input_paths.voc_final.open("rb") as file:
        raw_voc = cast(dict[str, object], dill.load(file))
    data_train: list[list[list[int]]] = load_pickle_rows(input_paths.data_train)
    data_test: list[list[list[int]]] = load_pickle_rows(input_paths.data_test)
    data_eval: list[list[list[int]]] = load_pickle_rows(input_paths.data_eval)
    return KGDRuntimeArtifacts(
        ehr_records=data_train + data_test + data_eval,
        diag_adj=cast(csr_matrix, load_pickle(input_paths.diag_adj)),
        proc_adj=cast(csr_matrix, load_pickle(input_paths.proc_adj)),
        diag_proc_adj=cast(csr_matrix, load_pickle(input_paths.diag_proc_adj)),
        proc_diag_adj=cast(csr_matrix, load_pickle(input_paths.proc_diag_adj)),
        prescriptions_adj=cast(csr_matrix, load_pickle(input_paths.prescriptions_adj)),
        ddi_adj=cast(csr_matrix, load_pickle(input_paths.ddi_A_final)),
        voc=build_runtime_voc(raw_voc),
    )


def build_runtime_voc(raw_voc: dict[str, object]) -> KGDVocFile:
    """恢复运行时词表对象。"""

    def build_vocab_entry(name: str) -> KGDVocabulary:
        raw_entry = cast(dict[str, object], raw_voc[name])
        return KGDVocabulary(
            word2idx=cast(dict[str, int], raw_entry["word2idx"]),
            idx2word=cast(list[str], raw_entry["idx2word"]),
        )

    return KGDVocFile(
        sym_voc=build_vocab_entry("sym_voc"),
        diag_voc=build_vocab_entry("diag_voc"),
        med_voc=build_vocab_entry("med_voc"),
    )


def build_global_clinical_edges(
    diag_adj: csr_matrix,
    proc_adj: csr_matrix,
    diag_proc_adj: csr_matrix,
    proc_diag_adj: csr_matrix,
    num_diag_nodes: int,
    device: torch.device,
) -> dict[KGDRelationKey, KGDRelationPayload]:
    """构造全局临床边。"""

    diag_diag_edges = sparse_matrix_to_edge_index(diag_adj, device)
    proc_proc_edges = sparse_matrix_to_edge_index(proc_adj, device)
    proc_proc_edges = proc_proc_edges + num_diag_nodes

    diag_proc_edges = sparse_matrix_to_edge_index(diag_proc_adj, device)
    if diag_proc_edges.numel() > 0:
        diag_proc_edges[1] = diag_proc_edges[1] + num_diag_nodes

    proc_diag_edges = sparse_matrix_to_edge_index(proc_diag_adj, device)
    if proc_diag_edges.numel() > 0:
        proc_diag_edges[0] = proc_diag_edges[0] + num_diag_nodes

    return {
        ("diagnoses", "identified_with", "diagnoses"): build_relation_payload(
            code=0,
            edges=diag_diag_edges,
            edge_id=torch.full(
                (diag_diag_edges.size(1),),
                0,
                dtype=torch.int64,
                device=device,
            ),
        ),
        ("procedure", "performed_with", "procedure"): build_relation_payload(
            code=1,
            edges=proc_proc_edges,
            edge_id=torch.full(
                (proc_proc_edges.size(1),),
                1,
                dtype=torch.int64,
                device=device,
            ),
        ),
        ("diagnoses", "given_with", "procedure"): build_relation_payload(
            code=2,
            edges=diag_proc_edges,
            edge_id=torch.full(
                (diag_proc_edges.size(1),),
                2,
                dtype=torch.int64,
                device=device,
            ),
        ),
        ("procedure", "given_with", "diagnoses"): build_relation_payload(
            code=3,
            edges=proc_diag_edges,
            edge_id=torch.full(
                (proc_diag_edges.size(1),),
                3,
                dtype=torch.int64,
                device=device,
            ),
        ),
    }


def build_patient_info(
    ehr_records: list[list[list[int]]],
    num_clinical_nodes: int,
    num_med_nodes: int,
    device: torch.device,
) -> KGDPatientInfo:
    """构造 patient_info。"""

    patient_info: KGDPatientInfo = []
    for subject in ehr_records:
        admission_info: list[KGDAdmissionInfo] = []
        for admission in [subject]:
            diagnoses = admission[1]
            procedures = admission[0]
            medicines = admission[2]
            admission_info.append(
                {
                    ("patient", "prescribed_to", "medicine"): build_patient_relation(
                        code=3,
                        source_node=num_med_nodes,
                        target_nodes=medicines,
                        device=device,
                    ),
                    ("patient", "diagnosed_with", "diagnosis"): build_patient_relation(
                        code=4,
                        source_node=num_clinical_nodes,
                        target_nodes=diagnoses,
                        device=device,
                    ),
                    ("patient", "had_procedure", "procedure"): build_patient_relation(
                        code=5,
                        source_node=num_clinical_nodes,
                        target_nodes=procedures,
                        device=device,
                    ),
                }
            )
        patient_info.append(admission_info)
    return patient_info


def build_patient_relation(
    code: int,
    source_node: int,
    target_nodes: list[int] | None,
    device: torch.device,
) -> KGDRelationPayload | None:
    """构造单条 patient relation。"""

    if target_nodes is None:
        return None
    relation_edges = torch.stack(
        (
            torch.full((len(target_nodes),), source_node, dtype=torch.int64, device=device),
            torch.tensor(target_nodes, dtype=torch.int64, device=device),
        )
    )
    relation_edge_ids = torch.full(
        (len(target_nodes),),
        code,
        dtype=torch.int64,
        device=device,
    )
    return build_relation_payload(code, relation_edges, relation_edge_ids)


def build_ddi_kg(
    ddi_edge_index: torch.Tensor,
    num_med_nodes: int,
    device: torch.device,
) -> Data:
    """构造全局 DDI 图。"""

    del num_med_nodes
    return Data(
        edge_index=ddi_edge_index,
        edge_type=torch.zeros(
            (ddi_edge_index.size(1),),
            dtype=torch.int64,
            device=device,
        ),
    )


def build_ehr_kgs(
    patient_info: KGDPatientInfo,
    clinical_edges: dict[KGDRelationKey, KGDRelationPayload],
    num_clinical_nodes: int,
    num_med_nodes: int,
    device: torch.device,
) -> KGDEHRGraphs:
    """构造 admission 级图集合。"""

    ehr_kgs: KGDEHRGraphs = []
    for patient in patient_info:
        patient_kgs: list[list[Data]] = []
        for admission in patient:
            patient_kgs.append(
                [
                    build_clinical_kg(
                        admission=admission,
                        clinical_edges=clinical_edges,
                        num_clinical_nodes=num_clinical_nodes,
                        device=device,
                    ),
                    build_medical_kg(
                        admission=admission,
                        num_med_nodes=num_med_nodes,
                        device=device,
                    ),
                ]
            )
        ehr_kgs.append(patient_kgs)
    return ehr_kgs


def build_clinical_kg(
    admission: KGDAdmissionInfo,
    clinical_edges: dict[KGDRelationKey, KGDRelationPayload],
    num_clinical_nodes: int,
    device: torch.device,
) -> Data:
    """构造单次临床图。"""

    del num_clinical_nodes
    admission_edges = empty_edge_index(device)
    admission_edge_ids = empty_edge_ids(device)

    diagnosed_relation = admission[("patient", "diagnosed_with", "diagnosis")]
    if diagnosed_relation is not None:
        diagnosed_edges = cast(torch.Tensor, diagnosed_relation["edges"])
        diagnosed_edge_ids = cast(torch.Tensor, diagnosed_relation["edge_id"])
        admission_edges = torch.cat(
            (
                admission_edges,
                diagnosed_edges,
                torch.flip(diagnosed_edges, dims=[0]),
            ),
            dim=1,
        )
        admission_edge_ids = torch.cat(
            (
                admission_edge_ids,
                diagnosed_edge_ids,
                torch.full(
                    (diagnosed_edge_ids.size(0),),
                    6,
                    dtype=torch.int64,
                    device=device,
                ),
            ),
            dim=0,
        )

        diagnoses = diagnosed_edges[1]
        diag_diag_edges_filter = filter_edges_by_source(
            cast(
                torch.Tensor,
                clinical_edges[("diagnoses", "identified_with", "diagnoses")]["edges"],
            ),
            diagnoses,
        )
        diag_proc_edges_filter = filter_edges_by_source(
            cast(
                torch.Tensor,
                clinical_edges[("diagnoses", "given_with", "procedure")]["edges"],
            ),
            diagnoses,
        )
        admission_edges = torch.cat(
            (
                admission_edges,
                diag_diag_edges_filter,
                diag_proc_edges_filter,
                torch.flip(diag_diag_edges_filter, dims=[0]),
                torch.flip(diag_proc_edges_filter, dims=[0]),
            ),
            dim=1,
        )
        admission_edge_ids = torch.cat(
            (
                admission_edge_ids,
                torch.full(
                    (diag_diag_edges_filter.size(1),),
                    0,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (diag_proc_edges_filter.size(1),),
                    2,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (diag_diag_edges_filter.size(1),),
                    8,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (diag_proc_edges_filter.size(1),),
                    9,
                    dtype=torch.int64,
                    device=device,
                ),
            ),
            dim=0,
        )

    procedure_relation = admission[("patient", "had_procedure", "procedure")]
    if procedure_relation is not None:
        procedure_edges = cast(torch.Tensor, procedure_relation["edges"])
        procedure_edge_ids = cast(torch.Tensor, procedure_relation["edge_id"])
        admission_edges = torch.cat(
            (
                admission_edges,
                procedure_edges,
                torch.flip(procedure_edges, dims=[0]),
            ),
            dim=1,
        )
        admission_edge_ids = torch.cat(
            (
                admission_edge_ids,
                procedure_edge_ids,
                torch.full(
                    (procedure_edge_ids.size(0),),
                    7,
                    dtype=torch.int64,
                    device=device,
                ),
            ),
            dim=0,
        )

        procedures = procedure_edges[1]
        proc_proc_edges_filter = filter_edges_by_source(
            cast(
                torch.Tensor,
                clinical_edges[("procedure", "performed_with", "procedure")]["edges"],
            ),
            procedures,
        )
        proc_diag_edges_filter = filter_edges_by_source(
            cast(
                torch.Tensor,
                clinical_edges[("procedure", "given_with", "diagnoses")]["edges"],
            ),
            procedures,
        )
        admission_edges = torch.cat(
            (
                admission_edges,
                proc_proc_edges_filter,
                proc_diag_edges_filter,
                torch.flip(proc_proc_edges_filter, dims=[0]),
                torch.flip(proc_diag_edges_filter, dims=[0]),
            ),
            dim=1,
        )
        admission_edge_ids = torch.cat(
            (
                admission_edge_ids,
                torch.full(
                    (proc_proc_edges_filter.size(1),),
                    1,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (proc_diag_edges_filter.size(1),),
                    3,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (proc_proc_edges_filter.size(1),),
                    10,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (proc_diag_edges_filter.size(1),),
                    11,
                    dtype=torch.int64,
                    device=device,
                ),
            ),
            dim=0,
        )

    return Data(
        edge_index=admission_edges,
        edge_type=admission_edge_ids,
    )


def build_medical_kg(
    admission: KGDAdmissionInfo,
    num_med_nodes: int,
    device: torch.device,
) -> Data:
    """构造单次药物图。"""

    del num_med_nodes
    admission_edges = empty_edge_index(device)
    admission_edge_ids = empty_edge_ids(device)
    admission_edge_weights = empty_edge_ids(device)

    prescribed_relation = admission[("patient", "prescribed_to", "medicine")]
    if prescribed_relation is not None:
        prescribed_edges = cast(torch.Tensor, prescribed_relation["edges"])
        prescribed_edge_ids = cast(torch.Tensor, prescribed_relation["edge_id"])
        admission_edges = torch.cat(
            (
                admission_edges,
                prescribed_edges,
                torch.flip(prescribed_edges, dims=[0]),
            ),
            dim=1,
        )
        admission_edge_ids = torch.cat(
            (
                admission_edge_ids,
                torch.full(
                    (prescribed_edge_ids.size(0),),
                    1,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (prescribed_edge_ids.size(0),),
                    2,
                    dtype=torch.int64,
                    device=device,
                ),
            ),
            dim=0,
        )
        admission_edge_weights = torch.cat(
            (
                admission_edge_weights,
                torch.ones((prescribed_edge_ids.size(0),), dtype=torch.int64, device=device),
                torch.ones((prescribed_edge_ids.size(0),), dtype=torch.int64, device=device),
            ),
            dim=0,
        )

    return Data(
        edge_index=admission_edges,
        edge_type=admission_edge_ids,
        edge_weights=admission_edge_weights,
    )


def filter_edges_by_source(
    edge_index: torch.Tensor,
    source_nodes: torch.Tensor,
) -> torch.Tensor:
    """按 admission 过滤全局边。"""

    if edge_index.numel() == 0 or source_nodes.numel() == 0:
        return edge_index[:, :0]
    return edge_index[:, torch.isin(edge_index[0], source_nodes)]

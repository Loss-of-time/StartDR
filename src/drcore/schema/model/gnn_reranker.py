from dataclasses import dataclass
from typing import Literal, TypedDict

from ..drugrec import DatasetSplit

type NodeId = str
type DrugId = str

GNNNodeType = Literal[
    "drug",
    "treat",
    "caution",
    "ingredient",
    "interaction",
]

GNNEdgeType = Literal[
    "drug_has_treat",
    "drug_has_caution",
    "drug_has_ingredient",
    "drug_has_interaction",
    "rev_drug_has_treat",
    "rev_drug_has_caution",
    "rev_drug_has_ingredient",
    "rev_drug_has_interaction",
]


class NumericFeatureStats(TypedDict):
    score_log_mean: float
    score_log_std: float


@dataclass(slots=True) # TODO slots = True 又是什么
class GNNNode:
    node_id: NodeId
    node_type: GNNNodeType
    text: str # 暂时无用


@dataclass(slots=True)
class DrugNodeNumericFeature:
    retrieval_score: float | None
    retrieval_rank: int | None
    is_candidate: int
    is_on_medicine: int


@dataclass(slots=True)
class GNNEdge:
    edge_type: GNNEdgeType
    src_node_id: NodeId
    dst_node_id: NodeId


@dataclass(slots=True)
class GNNCandidateTarget:
    drug_node_id: NodeId
    drugid: DrugId
    label: int


@dataclass(slots=True)
class GNNGraphSample:
    patient_id: str
    split: DatasetSplit
    patient_text: str
    gold_drugids: list[DrugId]
    nodes: list[GNNNode]
    edges: list[GNNEdge]
    # key 固定为 drug 节点的 node_id，格式为 drug:{drugid}
    drug_numeric_features: dict[NodeId, DrugNodeNumericFeature]
    # 顺序固定对齐 candidate_drugs 原始顺序
    candidate_targets: list[GNNCandidateTarget]


__all__ = [
    "DrugNodeNumericFeature",
    "GNNCandidateTarget",
    "GNNEdge",
    "GNNEdgeType",
    "GNNGraphSample",
    "GNNNode",
    "GNNNodeType",
    "NumericFeatureStats",
]

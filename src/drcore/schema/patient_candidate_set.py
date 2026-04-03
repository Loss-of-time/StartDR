from typing import Literal, TypeAlias, TypedDict

from .drugrec import DatasetSplit, DrugRecMedicine, DrugRecRecord

"""
``resource/patient_candidate_sets/`` 中患者候选集样本的 ``TypedDict`` 定义。

- 每条样本保存完整患者信息、金标准药物 ID，以及冻结后的候选药详情
- 目标是让后续 TraceDR 主线与各类 baseline 在脱离 Neo4j 的情况下复现实验输入
"""

NullableScore: TypeAlias = float | None
PatientCandidateRetriever: TypeAlias = Literal["bm25", "pyserini_bm25"]
PatientCandidateTopK: TypeAlias = Literal[50]


class CandidateDrug(TypedDict):
    drugid: str
    rank: int
    score: NullableScore
    drug: DrugRecMedicine
    is_gold: bool


class PatientCandidateSet(TypedDict):
    patient_id: str
    split: DatasetSplit # 属于哪个划分集合
    retriever: PatientCandidateRetriever
    top_k: PatientCandidateTopK
    retrieval_query: str
    patient: DrugRecRecord
    gold_drugids: list[str]
    candidate_drugs: list[CandidateDrug]


__all__ = [
    "CandidateDrug",
    "NullableScore",
    "PatientCandidateRetriever",
    "PatientCandidateSet",
    "PatientCandidateTopK",
]

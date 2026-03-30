from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict

from .drugrec import DrugRecMedicine, DrugRecRecord


class RetrievedDrugCandidate(TypedDict):
    drugid: str
    score: float
    medicine: NotRequired[DrugRecMedicine]
    metadata: NotRequired[dict[str, Any]]


class Retriver(ABC):
    @abstractmethod
    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        raise NotImplementedError

    def batch_retrieve(
        self,
        patients: Sequence[DrugRecRecord],
        top_k: int = 10,
    ) -> list[list[RetrievedDrugCandidate]]:
        retrieve = self.retrieve
        return [retrieve(patient, top_k=top_k) for patient in patients]

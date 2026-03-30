from ..schema import Retriver
from .bm25 import BM25Retriver
from .dense import DenseRetriver
from .pyserini_bm25 import PyseriniBM25Retriver


def get_retriver_names() -> list[str]:
    return ["bm25", "pyserini_bm25", "dense", "dual_tower"]


def build_retriver(name: str) -> Retriver:
    if name == "bm25":
        return BM25Retriver()
    if name == "pyserini_bm25":
        return PyseriniBM25Retriver()
    if name == "dense":
        return DenseRetriver()
    raise ValueError(name)

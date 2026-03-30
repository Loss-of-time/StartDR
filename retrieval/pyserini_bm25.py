import logging
from pathlib import Path

from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

from ..kg import list_simple_drug_details
from ..schema import (
    DrugRecRecord,
    RetrievedDrugCandidate,
    Retriver,
    SimpleDrugDetailRecord,
)

LOGGER = logging.getLogger("MINE.retrieval.pyserini_bm25")
INDEX_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cache"
    / "pyserini_bm25_zh"
)


def build_document(record: SimpleDrugDetailRecord) -> str:
    treatments = record["treatments"]
    cautions = record["cautions"]
    ingredients = record["ingredients"]
    return (
        f"治疗:{', '.join(treatments) if treatments else 'None'}"
        f" || 禁用:{', '.join(cautions) if cautions else 'None'}"
        f" || 成分:{', '.join(ingredients) if ingredients else 'None'}"
    )


def build_query(patient: DrugRecRecord) -> str:
    diagnosis = [item.strip() for item in patient["diagnosis"] if item.strip()]
    symptom = [item.strip() for item in patient["symptom"] if item.strip()]
    return " ".join([*diagnosis, *symptom])


def _has_index(index_dir: Path) -> bool:
    return index_dir.exists() and any(index_dir.iterdir())


def _build_index(
    index_dir: Path,
    data: list[SimpleDrugDetailRecord],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    indexer = LuceneIndexer(
        args=["-index", str(index_dir), "-language", "zh"],
        threads=8,
    )
    batch: list[dict[str, str]] = []

    for record in data:
        batch.append(
            {
                "id": str(record["drugid"]),
                "contents": build_document(record),
            }
        )
        if len(batch) >= 1000:
            indexer.add_batch_dict(batch)
            batch.clear()

    if batch:
        indexer.add_batch_dict(batch)
    indexer.close()


class PyseriniBM25Retriver(Retriver):
    def __init__(
        self,
        data: list[SimpleDrugDetailRecord] | None = None,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        source_data = list_simple_drug_details() if data is None else data
        if not _has_index(index_dir):
            LOGGER.info("Pyserini 索引不存在，开始构建: %s", index_dir.resolve())
            _build_index(index_dir, source_data)

        self.drug_name_by_id = {
            str(record["drugid"]): record["name"] or "None"
            for record in source_data
        }
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_language("zh")
        self.searcher.set_bm25(k1=1.5, b=0.75)

    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        query = build_query(patient)
        hits = self.searcher.search(query, k=top_k)
        return [
            {
                "drugid": hit.docid,
                "score": float(hit.score),
            }
            for hit in hits
        ]

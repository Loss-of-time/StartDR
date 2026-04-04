import logging
import warnings
from functools import lru_cache
from pathlib import Path

import jieba
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from rank_bm25 import BM25Okapi
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .kg import list_full_drug_details
from .schema import (
    DrugRecMedicine,
    DrugRecRecord,
    RetrievedDrugCandidate,
    Retriever,
)
from .setting import (
    CACHE_DIR,
    DENSE_MODEL_ID,
    PY_SERINI_INDEX_DIR,
    QUERY_INSTRUCTION,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
jieba.setLogLevel(logging.ERROR)

type TokenizedCorpusWithDrugIds = tuple[list[list[str]], list[str]]
type TokenIds = Int[Tensor, "batch seq"]
type AttentionMask = Int[Tensor, "batch seq"]
type TokenEmbeddings = Float[Tensor, "batch seq hidden"]
type SentenceEmbeddings = Float[Tensor, "batch hidden"]
type DrugEmbeddings = Float[Tensor, "drug hidden"]
type EmbeddingMatrix = Float[Tensor, "item hidden"]


def build_query_text(patient: DrugRecRecord) -> str:
    diagnosis = [item.strip() for item in patient.diagnosis if item.strip()]
    symptom = [item.strip() for item in patient.symptom if item.strip()]
    return " ".join([*diagnosis, *symptom])


def get_corpus(
    drugs: list[DrugRecMedicine],
) -> TokenizedCorpusWithDrugIds:
    tokenized_corpus: list[list[str]] = []
    drug_ids: list[str] = []
    lcut = jieba.lcut
    for drug in drugs:
        treatments = [
            row.treat
            for row in drug.treat
            if row.treat is not None
        ]
        cautions = [
            f"{row.crowd}{row.caution_level}"
            if row.caution_level
            else row.crowd
            for row in drug.caution
        ]
        ingredients = [
            row.ingredient
            for row in drug.ingredients
            if row.ingredient is not None
        ]
        document = (
            f"治疗:{', '.join(treatments) if treatments else 'None'}"
            f" || 禁用:{', '.join(cautions) if cautions else 'None'}"
            f" || 成分:{', '.join(ingredients) if ingredients else 'None'}"
        )
        tokenized_corpus.append(lcut(document))
        drug_ids.append(drug.drugid)
    return tokenized_corpus, drug_ids


def get_query_tokens(patient: DrugRecRecord) -> list[str]:
    return jieba.lcut(build_query_text(patient))


def get_drug_docs(drugs: list[DrugRecMedicine]) -> list[str]:
    docs: list[str] = []
    append = docs.append
    for drug in drugs:
        treatments = (
            "、".join(
                row.treat
                for row in drug.treat
                if row.treat is not None
            )
            if drug.treat
            else "None"
        )
        append(
            f"药品:{drug.name or 'None'}"
            f" || 治疗:{treatments}"
        )
    return docs


def build_pyserini_document(drug: DrugRecMedicine) -> str:
    treatments = [
        row.treat
        for row in drug.treat
        if row.treat is not None
    ]
    cautions = [
        f"{row.crowd}{row.caution_level}"
        if row.caution_level
        else row.crowd
        for row in drug.caution
    ]
    ingredients = [
        row.ingredient
        for row in drug.ingredients
        if row.ingredient is not None
    ]
    return (
        f"治疗:{', '.join(treatments) if treatments else 'None'}"
        f" || 禁用:{', '.join(cautions) if cautions else 'None'}"
        f" || 成分:{', '.join(ingredients) if ingredients else 'None'}"
    )


def normalize_embedding_matrix(
    embeddings: EmbeddingMatrix,
) -> EmbeddingMatrix:
    return F.normalize(embeddings, p=2, dim=1)


def get_dense_embedding_cache_path() -> Path:
    model_name = DENSE_MODEL_ID.replace("/", "__").replace("-", "_")
    return CACHE_DIR / f"retrieval__dense__drug_embeddings__{model_name}.pt"


@lru_cache(maxsize=1)
def get_dense_encoder() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_ID)
    model = AutoModel.from_pretrained(DENSE_MODEL_ID)
    return tokenizer, model


def has_pyserini_index(index_dir: Path) -> bool:
    return index_dir.exists() and any(index_dir.iterdir())


def build_pyserini_index(
    index_dir: Path,
    drugs: list[DrugRecMedicine],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    indexer = LuceneIndexer(
        args=["-index", str(index_dir), "-language", "zh"],
        threads=8,
    )
    batch: list[dict[str, str]] = []
    for drug in tqdm(drugs, desc="构建 Pyserini 索引"):
        batch.append(
            {
                "id": drug.drugid,
                "contents": build_pyserini_document(drug),
            }
        )
        if len(batch) >= 1000:
            indexer.add_batch_dict(batch)
            batch.clear()
    if batch:
        indexer.add_batch_dict(batch)
    indexer.close()


class BM25Retriever(Retriever):
    def __init__(
        self,
        drugs: list[DrugRecMedicine] | None = None,
    ) -> None:
        source_drugs = list_full_drug_details() if drugs is None else drugs
        self.drug_ids = [drug.drugid for drug in source_drugs]
        self.corpus, self.drug_ids = get_corpus(source_drugs)
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        query = get_query_tokens(patient)
        if not query or top_k <= 0:
            return []
        scores = self.bm25.get_scores(query)
        limit = min(top_k, len(self.drug_ids))
        top_indices = scores.argpartition(-limit)[-limit:]
        ranked_indices = top_indices[scores[top_indices].argsort()[::-1]]
        return [
            RetrievedDrugCandidate(
                drugid=self.drug_ids[index],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]


class DenseRetriever(Retriever):
    def __init__(
        self,
        drugs: list[DrugRecMedicine] | None = None,
    ) -> None:
        source_drugs = list_full_drug_details() if drugs is None else drugs
        self.drug_ids = [drug.drugid for drug in source_drugs]
        self.drug_docs = get_drug_docs(source_drugs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = get_dense_encoder()
        self.model.to(self.device)  # type: ignore[call-arg]
        self.model.eval()
        self.drug_embeddings = self.get_drug_embeddings()
        self.normalized_drug_embeddings = normalize_embedding_matrix(
            self.drug_embeddings.to(self.device)
        )

    def cls_pool(
        self,
        last_hidden_state: TokenEmbeddings,
    ) -> SentenceEmbeddings:
        return last_hidden_state[:, 0]

    def encode_texts(
        self,
        texts: list[str],
    ) -> SentenceEmbeddings:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch)
        return self.cls_pool(outputs.last_hidden_state)

    def get_drug_embeddings(self) -> DrugEmbeddings:
        cache_path = get_dense_embedding_cache_path()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return torch.load(cache_path)
        embeddings: list[SentenceEmbeddings] = []
        batch_size = 32
        for start in tqdm(
            range(0, len(self.drug_docs), batch_size),
            desc="编码药品向量",
        ):
            batch_docs = self.drug_docs[start:start + batch_size]
            embeddings.append(self.encode_texts(batch_docs).cpu())
        drug_embeddings = torch.cat(embeddings, dim=0)
        torch.save(drug_embeddings, cache_path)
        return drug_embeddings

    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 50,
    ) -> list[RetrievedDrugCandidate]:
        query = build_query_text(patient)
        if not query or top_k <= 0:
            return []
        query_embedding = normalize_embedding_matrix(
            self.encode_texts([f"{QUERY_INSTRUCTION}{query}"])
        )[0]
        scores = self.normalized_drug_embeddings @ query_embedding
        limit = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(
            scores,
            k=limit,
            largest=True,
            sorted=True,
        )
        return [
            RetrievedDrugCandidate(
                drugid=self.drug_ids[index],
                score=float(score),
            )
            for index, score in zip(
                top_indices.tolist(),
                top_scores.tolist(),
                strict=True,
            )
        ]


class PyseriniBM25Retriever(Retriever):
    def __init__(
        self,
        drugs: list[DrugRecMedicine] | None = None,
        index_dir: Path = PY_SERINI_INDEX_DIR,
    ) -> None:
        source_drugs = list_full_drug_details() if drugs is None else drugs
        if not has_pyserini_index(index_dir):
            print(f"Pyserini 索引不存在，开始构建: {index_dir.resolve()}")
            build_pyserini_index(index_dir, source_drugs)
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_language("zh")
        self.searcher.set_bm25(k1=1.5, b=0.75)

    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        hits = self.searcher.search(build_query_text(patient), k=top_k)
        return [
            RetrievedDrugCandidate(
                drugid=hit.docid,
                score=float(hit.score),
            )
            for hit in hits
        ]


def get_retriever_names() -> list[str]:
    return ["bm25", "pyserini_bm25", "dense"]


def build_retriever(name: str) -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    if name == "pyserini_bm25":
        return PyseriniBM25Retriever()
    if name == "dense":
        return DenseRetriever()
    raise ValueError(f"不支持的检索器: {name}")

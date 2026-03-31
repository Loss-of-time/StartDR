import logging
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from rich.progress import Progress
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..constant import CACHE_DIR
from ..kg import list_full_drug_details
from ..schema import (
    DrugRecMedicine,
    DrugRecRecord,
    RetrievedDrugCandidate,
    Retriver,
)
from ..utils.log import get_console, setup_logging

LOGGER = logging.getLogger("MINE.retrieval.dense")

# 512的文档长度够用
# MODEL_ID = "sentence-transformers/distiluse-base-multilingual-cased-v2"
MODEL_ID = "DMetaSoul/sbert-chinese-general-v2"  # 检索 embedding 模型

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章：" # 检索用 instruction

# 这个只是中文基座模型不是专门的双塔检索模型，在TraceDR里用于为后面的异构图检索提供向量化数据
# MODEL_ID = "hfl/chinese-roberta-wwm-ext"  # roberta 输出的 embbding 是 768维的

###############################################################
# jaxtyping 类型标注
###############################################################

TokenIds = Int[Tensor, "batch seq"]  # 这一批的数量 token序列长度
AttentionMask = Int[Tensor, "batch seq"]
TokenEmbeddings = Float[Tensor, "batch seq hidden"]  # . . 编码器隐藏维度
SentenceEmbeddings = Float[Tensor, "batch hidden"]
DrugEmbeddings = Float[Tensor, "drug hidden"]  # 药品条目数 .
EmbeddingMatrix = Float[Tensor, "item hidden"]  # 有多少向量 .


def get_dense_embedding_cache_path() -> Path:
    model_name = MODEL_ID.replace("/", "__").replace("-", "_")
    return CACHE_DIR / f"retrieval__dense__drug_embeddings__{model_name}.pt"


@lru_cache(maxsize=1)  # 缓存结果，模型只获取一次
def get_dense_encoder() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    return tokenizer, model


def get_query(patient: DrugRecRecord) -> str:
    diagnosis = [item.strip() for item in patient["diagnosis"] if item.strip()]
    symptom = [item.strip() for item in patient["symptom"] if item.strip()]
    return " ".join([*diagnosis, *symptom])


def get_drug_docs(drugs: list[DrugRecMedicine]) -> list[str]:
    docs: list[str] = []
    append = docs.append

    for drug in drugs:
        treatments = (
            "、".join(
                row["treat"]
                for row in drug["treat"]
                if row["treat"] is not None
            )
            if drug["treat"]
            else "None"
        )
        cautions = (  # noqa: F841
            "、".join(
                (
                    f"{row['crowd']}{row['caution_level']}"
                    if row["caution_level"]
                    else row["crowd"]
                )
                for row in drug["caution"]
            )
            if drug["caution"]
            else "None"
        )
        ingredients = (  # noqa: F841
            "、".join(
                row["ingredient"]
                for row in drug["ingredients"]
                if row["ingredient"] is not None
            )
            if drug["ingredients"]
            else "None"
        )

        append(
            f"药品:{drug['name'] or 'None'}"
            # f" || 批准文号:{drug['CMAN'] or 'None'}" # 几乎是噪声
            f" || 治疗:{treatments}"
            # f" || 慎用:{cautions}" # 暂时去除 节约文本长度
            # f" || 成分:{ingredients}" # 暂时去除
        )

    return docs


def normalize_embedding_matrix(
    embeddings: EmbeddingMatrix,
) -> EmbeddingMatrix:
    return F.normalize(
        embeddings,
        p=2,  # L2 范数
        dim=1,  # 在第二个维度（hidden）上进行
    )


class DenseRetriver(Retriver):
    def __init__(
        self,
        drugs: list[DrugRecMedicine] | None = None,
    ) -> None:
        source_drugs = list_full_drug_details() if drugs is None else drugs
        self.drug_name_by_id = {
            drug["drugid"]: drug["name"] or "None"
            for drug in source_drugs
        }
        self.drug_ids = [drug["drugid"] for drug in source_drugs]
        self.drug_docs = get_drug_docs(source_drugs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer, self.model = get_dense_encoder()
        self.model.to(self.device)  # type: ignore
        self.model.eval()
        self._drug_embeddings: DrugEmbeddings = self.get_drug_embeddings()
        self._normalized_drug_embeddings: DrugEmbeddings = (
            normalize_embedding_matrix(
                self._drug_embeddings.to(self.device)
            )
        )

    def cls_pool(  # 首向量池化
        self,
        last_hidden_state: TokenEmbeddings,
    ) -> SentenceEmbeddings:
        return last_hidden_state[:, 0]

    def encode_texts(
        self,
        texts: list[str],
    ) -> SentenceEmbeddings:
        # batch同时包含 input_id 和 attention_mask
        batch = self.tokenizer(
            texts,
            return_tensors="pt",  # 输出类型，不加输出python list
            truncation=True,  # 超时截断
            padding=True,  # 批量输入时需要补齐
            max_length=512,  # 最大长度 bge 限制
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():  # 关闭梯度计算
            outputs = self.model(**batch)  # 前项传播
            # 输出包含 last_hidden_state，即每个 token 对应的向量
        return self.cls_pool(outputs.last_hidden_state)

    def get_drug_embeddings(
        self,
    ) -> DrugEmbeddings:  # 提前将所以药物转化成 embeddings
        cache_path = get_dense_embedding_cache_path()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():  # 若存在已有缓存则读取已有的
            self._drug_embeddings = torch.load(cache_path)
            return self._drug_embeddings

        batch_size = 32 # 别爆显存

        embeddings: list[SentenceEmbeddings] = []
        total = (len(self.drug_docs) + batch_size - 1) // batch_size
        with Progress(console=get_console()) as progress:
            task_id = progress.add_task("编码药品向量", total=total)
            for start in range(0, len(self.drug_docs), batch_size):
                batch_docs = self.drug_docs[start : start + batch_size]
                batch_embeddings = self.encode_texts(batch_docs).cpu() # 防止显存越用越多
                embeddings.append(batch_embeddings)
                progress.advance(task_id)

        self._drug_embeddings = torch.cat(embeddings, dim=0)  # 在 CPU 上拼接

        # 药品原始向量缓存保持在 CPU，避免占用显存
        torch.save(self._drug_embeddings, cache_path)
        return self._drug_embeddings # 归一化在加载时运行

    def retrieve(
        self, patient: DrugRecRecord, top_k: int = 50
    ) -> list[RetrievedDrugCandidate]:
        query = get_query(patient)

        if not query or top_k <= 0:  # 空查询截断
            return []

        query_inputs = [f"{QUERY_INSTRUCTION}{query}"]
        query_embedding = normalize_embedding_matrix(  # 转化为单位向量
            self.encode_texts(query_inputs), # 转换为向量编码
        )[0]  # 返回一个列表，由于只有一个元素所以直接取出来

        # @ 是矩阵乘法
        scores = self._normalized_drug_embeddings @ query_embedding

        limit = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(
            scores,
            k=limit,
            largest=True,  # 取最大，False取最小
            sorted=True,  # 返回是否是经过排序的
        )

        return [
            {
                "drugid": self.drug_ids[index],
                "score": float(score),
            }
            for index, score in zip(
                top_indices.tolist(),
                top_scores.tolist(),
                strict=True,  # 强制检查两个列表是否一样长
            )
        ]


if __name__ == "__main__":
    log_path = setup_logging()
    cache_path = get_dense_embedding_cache_path()
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("当前编码模型: %s", MODEL_ID)
    LOGGER.info("目标缓存文件: %s", cache_path.resolve())

    if cache_path.exists():
        cache_path.unlink()
        LOGGER.info("已删除旧缓存，开始重新生成药品向量。")
    else:
        LOGGER.info("未发现旧缓存，开始生成药品向量。")

    retriver = DenseRetriver()
    LOGGER.info(
        "药品向量缓存生成完成，共 %s 条，向量维度 %s。",
        retriver._drug_embeddings.shape[0],
        retriver._drug_embeddings.shape[1],
    )
    LOGGER.info("缓存已写入: %s", cache_path.resolve())

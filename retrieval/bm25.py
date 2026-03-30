# 思路：获取所有药物的治疗文本，组合成一条检索语料，再把病人症状组合成查询。
import logging

# from ..schema impo
import warnings

import jieba
from rank_bm25 import BM25Okapi

from ..kg import list_simple_drug_details
from ..schema import (
    DrugRecRecord,
    RetrievedDrugCandidate,
    Retriver,
    SimpleDrugDetailRecord,
    TokenizedCorpusWithDrugIds,
)
from ..utils.log import setup_logging

# 患者只取 diagnosis symptom
# 药品取 treatments cautions ingredients 治疗:... || 禁用:... || 成分:...
# 用 jieba.lcut(...) 分词

# 1. 忽略 Python 库警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. 关闭 jieba 的加载提示
jieba.setLogLevel(logging.ERROR)

LOGGER = logging.getLogger("MINE.retrieval.bm25")


def get_corpus(
    data: list[SimpleDrugDetailRecord],
) -> TokenizedCorpusWithDrugIds:
    tokenized_corpus: list[list[str]] = []
    drug_ids: list[int] = []
    lcut = jieba.lcut

    for record in data:
        treatments = record["treatments"]
        cautions = record["cautions"]
        ingredients = record["ingredients"]
        document = (
            f"治疗:{', '.join(treatments) if treatments else 'None'}"
            f" || 禁用:{', '.join(cautions) if cautions else 'None'}"
            f" || 成分:{', '.join(ingredients) if ingredients else 'None'}"
        )
        tokenized_corpus.append(lcut(document))
        drug_ids.append(record["drugid"])

    return tokenized_corpus, drug_ids


def get_query(patient: DrugRecRecord) -> list[str]:
    diagnosis = [item.strip() for item in patient["diagnosis"] if item.strip()]
    symptom = [item.strip() for item in patient["symptom"] if item.strip()]
    return jieba.lcut(" ".join([*diagnosis, *symptom]))


class BM25Retriver(Retriver):
    def __init__(
        self,
        data: list[SimpleDrugDetailRecord] | None = None,
    ) -> None:
        source_data = list_simple_drug_details() if data is None else data
        self.drug_name_by_id = {
            str(record["drugid"]): record["name"] or "None"
            for record in source_data
        }
        self.corpus, self.drug_ids = get_corpus(source_data) # 构成药品查询的数据库
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(
        self,
        patient: DrugRecRecord,
        top_k: int = 10,
    ) -> list[RetrievedDrugCandidate]:
        query = get_query(patient)
        scores = self.bm25.get_scores(query) # 核心
        limit = min(top_k, len(self.drug_ids))
        top_indices = scores.argpartition(-limit)[-limit:]
        ranked_indices = top_indices[scores[top_indices].argsort()[::-1]]
        return [
            {
                "drugid": str(self.drug_ids[index]),
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]


if __name__ == "__main__":
    from ..input_process import get_patients

    log_path = setup_logging()
    bm25 = BM25Retriver()
    patients = get_patients(10)
    ans = bm25.batch_retrieve(patients)

    lines: list[str] = []
    for patient, candidates in zip(patients, ans, strict=True):
        diagnosis = "、".join(
            item.strip() for item in patient["diagnosis"] if item.strip()
        )
        symptom = "、".join(
            item.strip() for item in patient["symptom"] if item.strip()
        )
        medicines = "、".join(
            f"{medicine['name']}({medicine['drugid']})"
            for medicine in patient["medicine"]
        )

        lines.append("=" * 80)
        lines.append(f"患者ID: {patient['id']}")
        lines.append(f"诊断: {diagnosis or 'None'}")
        lines.append(f"症状: {symptom or 'None'}")
        lines.append(f"真实用药: {medicines or 'None'}")
        lines.append("BM25召回:")

        for rank, candidate in enumerate(candidates, start=1):
            drugid = candidate["drugid"]
            lines.append(
                f"  {rank:>2}. "
                f"{bm25.drug_name_by_id.get(drugid, '未知药品')}({drugid}) "
                f" score={candidate['score']:.4f}"
            )

        if not candidates:
            lines.append("  无结果")

    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("\n%s", "\n".join(lines))

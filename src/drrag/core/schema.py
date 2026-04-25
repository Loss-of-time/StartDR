"""RAG 项目的统一数据结构。"""

from dataclasses import dataclass
from typing import Any, Literal, cast

from cattrs import Converter

type DatasetSplit = Literal["train", "dev", "test"]
type RagInputFormat = Literal["tracedr_sample", "rag_case"]
type RagTask = Literal["recommend", "explain"]
type RagEvidenceSource = Literal["retrieval", "rerank", "gold"]


@dataclass(slots=True)
class DrugCaution:
    """药品禁忌信息。"""

    caution_level: str | None
    caution_levelid: int | None
    crowd: str
    crowd_id: int


@dataclass(slots=True)
class DrugIngredient:
    """药品成分信息。"""

    ingredient_id: int | None
    ingredient: str | None


@dataclass(slots=True)
class DrugInteraction:
    """药品相互作用信息。"""

    interaction_id: int
    name: str


@dataclass(slots=True)
class DrugTreat:
    """药品治疗信息。"""

    treat: str | None
    treat_id: int | None


@dataclass(slots=True)
class DrugRecMedicine:
    """DrugRec 药品记录。"""

    CMAN: str | None
    caution: list[DrugCaution]
    drugid: str
    ingredients: list[DrugIngredient]
    interaction: list[DrugInteraction]
    name: str
    treat: list[DrugTreat]


@dataclass(slots=True)
class DrugRecRecord:
    """DrugRec 患者记录。"""

    age: int
    allergen: list[str]
    antecedents: list[str]
    diagnosis: list[str]
    gender: str
    group: list[str]
    id: str
    medicine: list[DrugRecMedicine]
    on_medicine: list[DrugRecMedicine]
    part: DatasetSplit
    symptom: list[str]
    conflict: list[DrugRecMedicine] | None = None
    medicine_num: int | None = None


@dataclass(slots=True)
class TraceDRSample:
    """TraceDR 风格候选集样本。"""

    people: DrugRecRecord
    top_k_drugs: dict[str, DrugRecMedicine]


@dataclass(slots=True)
class RagEvidence:
    """RAG 使用的证据单元。"""

    evidence_id: str
    drugid: str
    text: str
    source: RagEvidenceSource
    retrieval_rank: int | None
    rerank_rank: int | None
    score: float | None


@dataclass(slots=True)
class RagCandidate:
    """RAG 使用的候选药物。"""

    drugid: str
    name: str
    drug: DrugRecMedicine
    is_gold: bool
    retrieval_rank: int
    retrieval_score: float | None
    rerank_rank: int | None
    rerank_score: float | None
    evidences: list[RagEvidence]


@dataclass(slots=True)
class RagCase:
    """RAG 统一输入样本。"""

    patient_id: str
    split: DatasetSplit
    patient: DrugRecRecord
    gold_drugids: list[str]
    candidates: list[RagCandidate]


@dataclass(slots=True)
class RagRequest:
    """单次 RAG 调用请求。"""

    case: RagCase
    task: RagTask
    top_k: int
    max_evidences_per_candidate: int


@dataclass(slots=True)
class PromptBuildResult:
    """Prompt 构造结果。"""

    system_prompt: str
    user_prompt: str
    prompt_text: str
    candidate_count: int
    evidence_count: int
    input_token_estimate: int


@dataclass(slots=True)
class RagGeneratedItem:
    """单个药物的可解释生成结果。"""

    drugid: str
    reason: str
    evidence_ids: list[str]


@dataclass(slots=True)
class RagGeneratedAnswer:
    """模型输出的结构化推荐结果。"""

    selected_drugids: list[str]
    items: list[RagGeneratedItem]


@dataclass(slots=True)
class RagGenerationUsage:
    """单次生成的 token 用量。"""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(slots=True)
class RagGenerationRecord:
    """单样本硅基流动生成记录。"""

    patient_id: str
    task: RagTask
    model_name: str
    top_k: int
    max_evidences_per_candidate: int
    success: bool
    finish_reason: str | None
    trace_id: str | None
    error_message: str | None
    visible_candidate_drugids: list[str]
    visible_evidence_ids: list[str]
    gold_drugids: list[str]
    prompt: PromptBuildResult
    raw_response: dict[str, Any] | None
    response_content: str | None
    usage: RagGenerationUsage | None
    parsed_answer: RagGeneratedAnswer | None


@dataclass(slots=True)
class RagGenerationEvalRecord:
    """单样本生成评估结果。"""

    patient_id: str
    success: bool
    has_structured_answer: bool
    selection_alignment_valid: bool
    candidate_refs_valid: bool
    evidence_refs_valid: bool
    reason_fields_valid: bool
    field_complete: bool
    selected_count: int
    gold_count: int
    hit: bool
    exact_match: bool
    precision: float
    recall: float
    f1: float
    validation_errors: list[str]


@dataclass(slots=True)
class RagGenerationEvalSummary:
    """整份生成结果的离线评估摘要。"""

    sample_count: int
    success_count: int
    structured_answer_count: int
    fully_valid_count: int
    hit_count: int
    exact_match_count: int
    average_precision: float
    average_recall: float
    average_f1: float


_converter = Converter()


def structure(data: object, target: object) -> Any:
    """把原始对象结构化为目标类型。

    Args:
        data: 原始对象。
        target: 目标类型。

    Returns:
        结构化后的对象。
    """

    return _converter.structure(data, cast(Any, target))


def unstructure(data: object) -> object:
    """把结构化对象反序列化为原始对象。

    Args:
        data: 待反序列化对象。

    Returns:
        基础 Python 对象。
    """

    return cast(object, _converter.unstructure(data))

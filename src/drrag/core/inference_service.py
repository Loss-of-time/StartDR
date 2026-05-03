"""在线推荐推理服务。"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from drrerank.core import schema as rerank_schema
from drrerank.core.model.tracedr.model import HeterogeneousGNN
from drrerank.core.model.tracedr.schema import TraceDRAblationConfig
from drrerank.core.schema import RankedCase
from drrerank.core.schema import TraceDRSample as RerankTraceDRSample
from drrerank.tracedr_export_rank import build_ranked_case, load_model, resolve_checkpoint_path
from drretrieval.core.kg import list_full_drug_details
from drretrieval.core.retrieval import PyseriniBM25Retriever, build_query_text
from drretrieval.core.schema import DrugRecMedicine as RetrievalDrugRecMedicine
from drretrieval.core.schema import DrugRecRecord as RetrievalDrugRecRecord
from drretrieval.core.schema import RetrievedDrugCandidate, Retriever, TraceDRSample
from drretrieval.core.schema import unstructure as unstructure_retrieval

from ..generate_siliconflow import build_generation_record
from .adapters import (
    apply_ranked_drugs,
    apply_ranked_evidences,
    build_patient_query,
    build_rag_case,
    from_retrieval_sample,
)
from .generation import validate_generation_record
from .prompt import freeze_case_candidates, select_candidates, select_evidences
from .schema import (
    DrugCaution as RagDrugCaution,
)
from .schema import (
    DrugIngredient as RagDrugIngredient,
)
from .schema import (
    DrugInteraction as RagDrugInteraction,
)
from .schema import (
    DrugRecMedicine as RagDrugRecMedicine,
)
from .schema import (
    DrugTreat as RagDrugTreat,
)
from .schema import (
    RagCandidate,
    RagEvidence,
    RagGeneratedItem,
    RagGenerationRecord,
    RagRequest,
)
from .setting import (
    DEFAULT_API_ALLOWED_ORIGINS,
    DEFAULT_API_CHECKPOINT_PATH,
    DEFAULT_API_DISPLAY_TOP_K,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_API_RETRIEVAL_TOP_K,
    DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE,
    DEFAULT_SILICONFLOW_MAX_TOKENS,
    DEFAULT_SILICONFLOW_MODEL,
    DEFAULT_SILICONFLOW_TEMPERATURE,
    DEFAULT_SILICONFLOW_TIMEOUT_SECONDS,
)

type RankCaseCallable = Callable[[RerankTraceDRSample], RankedCase]
type GenerationRecordBuilder = Callable[
    [RagRequest, str, int, float, int],
    RagGenerationRecord,
]

ONLINE_DATASET_SPLIT = "test"
DEFAULT_RECOMMEND_TASK = "recommend"
DEFAULT_RETRIEVER_NAME = "pyserini_bm25"


@dataclass(slots=True)
class OnlineInferenceConfig:
    """在线推荐服务配置。"""

    checkpoint_path: Path = DEFAULT_API_CHECKPOINT_PATH
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    allowed_origins: tuple[str, ...] = DEFAULT_API_ALLOWED_ORIGINS
    retriever_name: str = DEFAULT_RETRIEVER_NAME
    retrieval_top_k: int = DEFAULT_API_RETRIEVAL_TOP_K
    display_top_k: int = DEFAULT_API_DISPLAY_TOP_K
    rag_model_name: str = DEFAULT_SILICONFLOW_MODEL
    rag_task: str = DEFAULT_RECOMMEND_TASK
    max_evidences_per_candidate: int = DEFAULT_RAG_MAX_EVIDENCES_PER_CANDIDATE
    rag_max_tokens: int = DEFAULT_SILICONFLOW_MAX_TOKENS
    rag_temperature: float = DEFAULT_SILICONFLOW_TEMPERATURE
    rag_timeout_seconds: int = DEFAULT_SILICONFLOW_TIMEOUT_SECONDS


@dataclass(slots=True)
class OnlinePatientPayload:
    """在线推荐请求中的病例输入。"""

    age: int
    gender: str
    group: list[str]
    diagnosis: list[str]
    symptom: list[str]
    antecedents: list[str]
    allergen: list[str]
    on_medicine_drugids: list[str]


@dataclass(slots=True)
class SearchableDrugRecord:
    """药品检索索引项。"""

    drug: RetrievalDrugRecMedicine
    treat_summary: str
    normalized_name: str
    normalized_cman: str
    same_name_count: int


@dataclass(slots=True)
class DrugSearchHit:
    """药品搜索命中项。"""

    drugid: str
    name: str
    CMAN: str | None
    treat_summary: str
    same_name_count: int


@dataclass(slots=True)
class ResolvedMedicineView:
    """归一化病例中的在用药视图。"""

    drugid: str
    name: str
    CMAN: str | None


@dataclass(slots=True)
class NormalizedPatientView:
    """归一化病例视图。"""

    patient_id: str
    age: int
    gender: str
    group: list[str]
    diagnosis: list[str]
    symptom: list[str]
    antecedents: list[str]
    allergen: list[str]
    on_medicines: list[ResolvedMedicineView]
    retrieval_query: str
    trace_query: str


@dataclass(slots=True)
class CandidateView:
    """候选药物展示视图。"""

    drugid: str
    name: str
    CMAN: str | None
    treat_summary: str
    retrieval_rank: int
    retrieval_score: float | None
    rerank_rank: int | None
    rerank_score: float | None


@dataclass(slots=True)
class RetrievalStageView:
    """检索阶段返回体。"""

    retrieved_candidate_count: int
    display_candidate_count: int
    candidates: list[CandidateView]


@dataclass(slots=True)
class RerankStageView:
    """精排阶段返回体。"""

    ranked_candidate_count: int
    display_candidate_count: int
    candidates: list[CandidateView]


@dataclass(slots=True)
class RecommendationItemView:
    """单条推荐解释。"""

    drugid: str
    reason: str
    evidence_ids: list[str]


@dataclass(slots=True)
class RecommendationView:
    """推荐结果返回体。"""

    success: bool
    finish_reason: str | None
    error_message: str | None
    trace_id: str | None
    validation_errors: list[str]
    selected_drugids: list[str]
    items: list[RecommendationItemView]


@dataclass(slots=True)
class EvidenceView:
    """证据展开视图。"""

    evidence_id: str
    drugid: str
    drug_name: str
    source: str
    text: str
    retrieval_rank: int | None
    rerank_rank: int | None
    score: float | None
    graph_relations: list["GraphRelationView"]


@dataclass(slots=True)
class GraphRelationView:
    """证据底层图谱关系视图。"""

    subject_id: str
    subject_label: str
    predicate: str
    object_id: str
    object_label: str


@dataclass(slots=True)
class RiskConflictView:
    """推荐药与当前在用药的冲突提示。"""

    on_medicine_drugid: str
    on_medicine_name: str
    interaction_names: list[str]
    matched_ingredients: list[str]


@dataclass(slots=True)
class RiskDrugView:
    """单药风险摘要。"""

    drugid: str
    name: str
    cautions: list[str]
    interactions: list[str]
    on_medicine_conflicts: list[RiskConflictView]


@dataclass(slots=True)
class PipelineMetaView:
    """推理链路元信息。"""

    retriever_name: str
    checkpoint_name: str
    rag_model_name: str
    retrieval_top_k: int
    display_top_k: int
    max_evidences_per_candidate: int


@dataclass(slots=True)
class OnlineRecommendationResponse:
    """在线推荐接口返回体。"""

    normalized_patient: NormalizedPatientView
    retrieval: RetrievalStageView
    rerank: RerankStageView
    recommendation: RecommendationView
    evidence_map: dict[str, EvidenceView]
    risk_summary: list[RiskDrugView]
    pipeline_meta: PipelineMetaView
    timings_ms: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的字典。"""

        payload: dict[str, object] = asdict(self)
        return payload


@dataclass(slots=True)
class TraceDRRankingRuntime:
    """TraceDR 精排运行时。"""

    model: HeterogeneousGNN
    ablation_config: TraceDRAblationConfig

    def rank(self, sample: RerankTraceDRSample) -> RankedCase:
        """对单个在线病例执行精排。"""

        ranked_case: RankedCase = build_ranked_case(
            self.model,
            sample,
            self.ablation_config,
        )
        return ranked_case


def build_treat_summary(
    drug: RetrievalDrugRecMedicine | RagDrugRecMedicine,
    max_items: int = 3,
) -> str:
    """构造治疗摘要。

    Args:
        drug: 待摘要的药物。
        max_items: 最多保留的治疗条目数。

    Returns:
        适合搜索列表展示的治疗摘要。
    """

    treat_values: list[str] = []
    seen_treats: set[str] = set()
    treat_name: str | None
    for treat_name in (item.treat for item in drug.treat):
        if treat_name is None:
            continue
        normalized_treat: str = treat_name.strip()
        if normalized_treat == "" or normalized_treat in seen_treats:
            continue
        seen_treats.add(normalized_treat)
        treat_values.append(normalized_treat)
        if len(treat_values) >= max_items:
            break
    if treat_values:
        return "、".join(treat_values)
    return "None"


def normalize_text(value: str) -> str:
    """规范化搜索文本。"""

    normalized_value: str = value.strip().lower()
    return normalized_value


def normalize_text_list(values: Sequence[str]) -> list[str]:
    """规范化文本数组。"""

    normalized_values: list[str] = []
    raw_value: str
    for raw_value in values:
        normalized_value: str = raw_value.strip()
        if normalized_value == "":
            continue
        normalized_values.append(normalized_value)
    return normalized_values


def build_candidate_view(
    candidate: RagCandidate,
    retrieval_score_map: dict[str, float],
) -> CandidateView:
    """构造候选药物展示对象。"""

    candidate_view: CandidateView = CandidateView(
        drugid=candidate.drugid,
        name=candidate.name,
        CMAN=candidate.drug.CMAN,
        treat_summary=build_treat_summary(candidate.drug),
        retrieval_rank=candidate.retrieval_rank,
        retrieval_score=retrieval_score_map.get(candidate.drugid),
        rerank_rank=candidate.rerank_rank,
        rerank_score=candidate.rerank_score,
    )
    return candidate_view


class OnlineInferenceService:
    """在线推荐推理服务。"""

    def __init__(
        self,
        config: OnlineInferenceConfig,
        searchable_drugs: list[SearchableDrugRecord],
        drugs_by_id: dict[str, RetrievalDrugRecMedicine],
        retriever: Retriever,
        rank_case: RankCaseCallable,
        build_generation: GenerationRecordBuilder,
    ) -> None:
        """构造在线推荐服务。

        Args:
            config: 运行配置。
            searchable_drugs: 搜索索引项。
            drugs_by_id: 药物字典。
            retriever: 检索器。
            rank_case: 精排函数。
            build_generation: RAG 调用函数。
        """

        self.config = config
        self.searchable_drugs = searchable_drugs
        self.drugs_by_id = drugs_by_id
        self.retriever = retriever
        self.rank_case = rank_case
        self.build_generation = build_generation

    @classmethod
    def build(
        cls,
        config: OnlineInferenceConfig,
    ) -> "OnlineInferenceService":
        """按默认运行时依赖构造服务。"""

        raw_drugs: list[RetrievalDrugRecMedicine] = list_full_drug_details()
        name_counter: dict[str, int] = {}
        drug: RetrievalDrugRecMedicine
        for drug in raw_drugs:
            normalized_name: str = normalize_text(drug.name)
            name_counter[normalized_name] = name_counter.get(normalized_name, 0) + 1
        searchable_drugs: list[SearchableDrugRecord] = []
        drugs_by_id: dict[str, RetrievalDrugRecMedicine] = {}
        for drug in raw_drugs:
            normalized_name = normalize_text(drug.name)
            normalized_cman: str = normalize_text(drug.CMAN or "")
            searchable_drugs.append(
                SearchableDrugRecord(
                    drug=drug,
                    treat_summary=build_treat_summary(drug),
                    normalized_name=normalized_name,
                    normalized_cman=normalized_cman,
                    same_name_count=name_counter[normalized_name],
                )
            )
            drugs_by_id[drug.drugid] = drug
        retriever: Retriever = PyseriniBM25Retriever(drugs=raw_drugs)
        ablation_config: TraceDRAblationConfig = TraceDRAblationConfig()
        checkpoint_path: Path = resolve_checkpoint_path(config.checkpoint_path)
        model: object = load_model(checkpoint_path, ablation_config)
        ranking_runtime: TraceDRRankingRuntime = TraceDRRankingRuntime(
            model=model,
            ablation_config=ablation_config,
        )
        service: OnlineInferenceService = cls(
            config=config,
            searchable_drugs=searchable_drugs,
            drugs_by_id=drugs_by_id,
            retriever=retriever,
            rank_case=ranking_runtime.rank,
            build_generation=build_generation_record,
        )
        return service

    def build_health_payload(self) -> dict[str, object]:
        """构造健康检查返回体。"""

        payload: dict[str, object] = {
            "status": "ok",
            "retriever_name": self.config.retriever_name,
            "checkpoint_name": resolve_checkpoint_path(self.config.checkpoint_path).name,
            "rag_model_name": self.config.rag_model_name,
            "retrieval_top_k": self.config.retrieval_top_k,
            "display_top_k": self.config.display_top_k,
            "max_evidences_per_candidate": self.config.max_evidences_per_candidate,
        }
        return payload

    def search_drugs(self, query: str, limit: int) -> list[DrugSearchHit]:
        """按药名或批准文号搜索药品。"""

        normalized_query: str = normalize_text(query)
        if normalized_query == "":
            return []
        ranked_rows: list[tuple[int, int, str, str, SearchableDrugRecord]] = []
        row: SearchableDrugRecord
        for row in self.searchable_drugs:
            category: tuple[int, int] | None = None
            if normalized_query == row.normalized_name:
                category = (0, 0)
            elif normalized_query == row.normalized_cman and row.normalized_cman != "":
                category = (0, 1)
            elif row.normalized_name.startswith(normalized_query):
                category = (1, 0)
            elif row.normalized_cman.startswith(normalized_query) and row.normalized_cman != "":
                category = (1, 1)
            elif normalized_query in row.normalized_name:
                category = (2, 0)
            elif normalized_query in row.normalized_cman and row.normalized_cman != "":
                category = (2, 1)
            if category is None:
                continue
            ranked_rows.append(
                (
                    category[0],
                    category[1],
                    row.drug.name,
                    row.drug.CMAN or "",
                    row,
                )
            )
        ranked_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4].drug.drugid))
        hits: list[DrugSearchHit] = []
        for _, _, _, _, row in ranked_rows[:limit]:
            hits.append(
                DrugSearchHit(
                    drugid=row.drug.drugid,
                    name=row.drug.name,
                    CMAN=row.drug.CMAN,
                    treat_summary=row.treat_summary,
                    same_name_count=row.same_name_count,
                )
            )
        return hits

    def build_online_patient(self, payload: OnlinePatientPayload) -> RetrievalDrugRecRecord:
        """把在线请求转换为内部病例对象。"""

        resolved_on_medicines: list[RetrievalDrugRecMedicine] = self.resolve_on_medicines(
            payload.on_medicine_drugids
        )
        patient_id: str = f"api-{uuid4().hex[:12]}"
        patient: RetrievalDrugRecRecord = RetrievalDrugRecRecord(
            age=payload.age,
            allergen=normalize_text_list(payload.allergen),
            antecedents=normalize_text_list(payload.antecedents),
            diagnosis=normalize_text_list(payload.diagnosis),
            gender=payload.gender.strip(),
            group=normalize_text_list(payload.group),
            id=patient_id,
            medicine=[],
            on_medicine=resolved_on_medicines,
            part=ONLINE_DATASET_SPLIT,
            symptom=normalize_text_list(payload.symptom),
        )
        return patient

    def resolve_on_medicines(
        self,
        on_medicine_drugids: Sequence[str],
    ) -> list[RetrievalDrugRecMedicine]:
        """把前端选定的 drugid 解析为完整药物对象。"""

        resolved_medicines: list[RetrievalDrugRecMedicine] = []
        unique_drugids: list[str] = list(dict.fromkeys(on_medicine_drugids))
        drugid: str
        for drugid in unique_drugids:
            medicine: RetrievalDrugRecMedicine | None = self.drugs_by_id.get(drugid)
            if medicine is None:
                raise ValueError(f"未知的 on_medicine drugid: {drugid}")
            resolved_medicines.append(medicine)
        return resolved_medicines

    def build_retrieval_sample(
        self,
        patient: RetrievalDrugRecRecord,
        retrieved_candidates: Sequence[RetrievedDrugCandidate],
    ) -> TraceDRSample:
        """把在线检索结果规整为 TraceDR 候选集。"""

        top_k_drugs: dict[str, RetrievalDrugRecMedicine] = {}
        retrieved_candidate: RetrievedDrugCandidate
        for retrieved_candidate in retrieved_candidates:
            drug: RetrievalDrugRecMedicine | None = self.drugs_by_id.get(retrieved_candidate.drugid)
            if drug is None:
                continue
            top_k_drugs[retrieved_candidate.drugid] = drug
        tracedr_sample: TraceDRSample = TraceDRSample(
            people=patient,
            top_k_drugs=top_k_drugs,
        )
        return tracedr_sample

    def convert_to_rerank_sample(self, sample: TraceDRSample) -> RerankTraceDRSample:
        """把检索侧样本转换为精排侧样本。"""

        rerank_sample: RerankTraceDRSample = rerank_schema.structure(
            unstructure_retrieval(sample),
            RerankTraceDRSample,
        )
        return rerank_sample

    def build_graph_relations(
        self,
        drug: RagDrugRecMedicine,
    ) -> list[GraphRelationView]:
        """把单药证据展开为结构化图谱关系。"""

        relations: list[GraphRelationView] = []
        subject_id: str = f"drug:{drug.drugid}"
        subject_label: str = drug.name

        treat_item: RagDrugTreat
        for treat_item in drug.treat:
            if treat_item.treat is None:
                continue
            object_label: str = treat_item.treat.strip()
            if object_label == "":
                continue
            object_id_suffix: str = (
                str(treat_item.treat_id) if treat_item.treat_id is not None else object_label
            )
            relations.append(
                GraphRelationView(
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate="治疗",
                    object_id=f"disease:{object_id_suffix}",
                    object_label=object_label,
                )
            )

        caution_item: RagDrugCaution
        for caution_item in drug.caution:
            object_label = (
                f"{caution_item.crowd}{caution_item.caution_level}"
                if caution_item.caution_level is not None
                else caution_item.crowd
            ).strip()
            if object_label == "":
                continue
            relations.append(
                GraphRelationView(
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate="禁用",
                    object_id=f"group:{caution_item.crowd_id}",
                    object_label=object_label,
                )
            )

        ingredient_item: RagDrugIngredient
        for ingredient_item in drug.ingredients:
            if ingredient_item.ingredient is None:
                continue
            object_label = ingredient_item.ingredient.strip()
            if object_label == "":
                continue
            object_id_suffix = (
                str(ingredient_item.ingredient_id)
                if ingredient_item.ingredient_id is not None
                else object_label
            )
            relations.append(
                GraphRelationView(
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate="成分",
                    object_id=f"drug:{object_id_suffix}",
                    object_label=object_label,
                )
            )

        interaction_item: RagDrugInteraction
        for interaction_item in drug.interaction:
            object_label = interaction_item.name.strip()
            if object_label == "":
                continue
            relations.append(
                GraphRelationView(
                    subject_id=subject_id,
                    subject_label=subject_label,
                    predicate="相互作用",
                    object_id=f"drug:{interaction_item.interaction_id}",
                    object_label=object_label,
                )
            )

        return relations

    def build_evidence_map(
        self,
        request: RagRequest,
        referenced_evidence_ids: set[str],
    ) -> dict[str, EvidenceView]:
        """构造最终被引用证据的展开表。"""

        evidence_map: dict[str, EvidenceView] = {}
        candidate: RagCandidate
        for candidate in select_candidates(request):
            evidence: RagEvidence
            for evidence in select_evidences(candidate, request.max_evidences_per_candidate):
                if evidence.evidence_id not in referenced_evidence_ids:
                    continue
                evidence_map[evidence.evidence_id] = EvidenceView(
                    evidence_id=evidence.evidence_id,
                    drugid=evidence.drugid,
                    drug_name=candidate.name,
                    source=evidence.source,
                    text=evidence.text,
                    retrieval_rank=evidence.retrieval_rank,
                    rerank_rank=evidence.rerank_rank,
                    score=evidence.score,
                    # 目的：把最终被引用证据对应的全部底层图谱关系显式返回给前端展开。
                    graph_relations=self.build_graph_relations(candidate.drug),
                )
        return evidence_map

    def build_risk_summary(
        self,
        selected_drugids: Sequence[str],
        final_candidates: Sequence[RagCandidate],
        on_medicines: Sequence[RetrievalDrugRecMedicine],
    ) -> list[RiskDrugView]:
        """按推荐药物汇总风险信息。"""

        candidate_map: dict[str, RagCandidate] = {
            candidate.drugid: candidate for candidate in final_candidates
        }
        risk_items: list[RiskDrugView] = []
        drugid: str
        for drugid in selected_drugids:
            candidate: RagCandidate | None = candidate_map.get(drugid)
            if candidate is None:
                continue
            cautions: list[str] = [
                f"{item.crowd}{item.caution_level}"
                if item.caution_level is not None
                else item.crowd
                for item in candidate.drug.caution
            ]
            interactions: list[str] = [
                item.name for item in candidate.drug.interaction if item.name.strip() != ""
            ]
            conflicts: list[RiskConflictView] = []
            on_medicine: RetrievalDrugRecMedicine
            for on_medicine in on_medicines:
                interaction_names: list[str] = [
                    item.name for item in on_medicine.interaction if item.name.strip() != ""
                ]
                interaction_name_set: set[str] = set(interaction_names)
                matched_ingredients: list[str] = []
                ingredient_name: str | None
                for ingredient_name in (item.ingredient for item in candidate.drug.ingredients):
                    if ingredient_name is None:
                        continue
                    if ingredient_name in interaction_name_set:
                        matched_ingredients.append(ingredient_name)
                if matched_ingredients:
                    conflicts.append(
                        RiskConflictView(
                            on_medicine_drugid=on_medicine.drugid,
                            on_medicine_name=on_medicine.name,
                            interaction_names=interaction_names,
                            matched_ingredients=matched_ingredients,
                        )
                    )
            risk_items.append(
                RiskDrugView(
                    drugid=candidate.drugid,
                    name=candidate.name,
                    cautions=cautions,
                    interactions=interactions,
                    on_medicine_conflicts=conflicts,
                )
            )
        return risk_items

    def recommend(self, payload: OnlinePatientPayload) -> OnlineRecommendationResponse:
        """执行在线推荐主链路。"""

        timings_ms: dict[str, int] = {}

        patient_start: float = perf_counter()
        patient: RetrievalDrugRecRecord = self.build_online_patient(payload)
        timings_ms["normalize_patient"] = int((perf_counter() - patient_start) * 1000)

        retrieval_start: float = perf_counter()
        retrieved_candidates: list[RetrievedDrugCandidate] = self.retriever.retrieve(
            patient,
            top_k=self.config.retrieval_top_k,
        )
        retrieval_sample: TraceDRSample = self.build_retrieval_sample(patient, retrieved_candidates)
        retrieval_score_map: dict[str, float] = {
            candidate.drugid: candidate.score for candidate in retrieved_candidates
        }
        timings_ms["retrieval"] = int((perf_counter() - retrieval_start) * 1000)

        rerank_start: float = perf_counter()
        rerank_sample: RerankTraceDRSample = self.convert_to_rerank_sample(retrieval_sample)
        ranked_case: RankedCase = self.rank_case(rerank_sample)
        unified_sample = from_retrieval_sample(retrieval_sample)
        rag_case = build_rag_case(unified_sample)
        # 目的：先补药物排序，再补 TraceDR 证据排序，保证 prompt 看到的是完整精排结果。
        reranked_case = apply_ranked_drugs(rag_case, ranked_case.ranked_drugs)
        reranked_case = apply_ranked_evidences(reranked_case, ranked_case.ranked_evidences)
        final_case = freeze_case_candidates(reranked_case, self.config.display_top_k)
        timings_ms["rerank"] = int((perf_counter() - rerank_start) * 1000)

        generation_start: float = perf_counter()
        rag_request: RagRequest = RagRequest(
            case=final_case,
            task=DEFAULT_RECOMMEND_TASK,
            top_k=self.config.display_top_k,
            max_evidences_per_candidate=self.config.max_evidences_per_candidate,
        )
        generation_record: RagGenerationRecord = self.build_generation(
            rag_request,
            self.config.rag_model_name,
            self.config.rag_max_tokens,
            self.config.rag_temperature,
            self.config.rag_timeout_seconds,
        )
        validation_errors: list[str] = validate_generation_record(generation_record)
        timings_ms["generation"] = int((perf_counter() - generation_start) * 1000)

        response_start: float = perf_counter()
        retrieval_candidates: list[CandidateView] = [
            build_candidate_view(candidate, retrieval_score_map)
            for candidate in sorted(final_case.candidates, key=lambda item: item.retrieval_rank)
        ]
        rerank_candidates: list[CandidateView] = [
            build_candidate_view(candidate, retrieval_score_map)
            for candidate in final_case.candidates
        ]
        selected_drugids: list[str] = (
            list(generation_record.parsed_answer.selected_drugids)
            if generation_record.parsed_answer is not None
            else []
        )
        recommendation_items: list[RecommendationItemView] = []
        referenced_evidence_ids: set[str] = set()
        if generation_record.parsed_answer is not None:
            parsed_item: RagGeneratedItem
            for parsed_item in generation_record.parsed_answer.items:
                referenced_evidence_ids.update(parsed_item.evidence_ids)
                recommendation_items.append(
                    RecommendationItemView(
                        drugid=parsed_item.drugid,
                        reason=parsed_item.reason,
                        evidence_ids=list(parsed_item.evidence_ids),
                    )
                )
        evidence_map: dict[str, EvidenceView] = self.build_evidence_map(
            rag_request,
            referenced_evidence_ids,
        )
        risk_summary: list[RiskDrugView] = self.build_risk_summary(
            selected_drugids,
            final_case.candidates,
            patient.on_medicine,
        )
        normalized_patient: NormalizedPatientView = NormalizedPatientView(
            patient_id=patient.id,
            age=patient.age,
            gender=patient.gender,
            group=list(patient.group),
            diagnosis=list(patient.diagnosis),
            symptom=list(patient.symptom),
            antecedents=list(patient.antecedents),
            allergen=list(patient.allergen),
            on_medicines=[
                ResolvedMedicineView(
                    drugid=medicine.drugid,
                    name=medicine.name,
                    CMAN=medicine.CMAN,
                )
                for medicine in patient.on_medicine
            ],
            retrieval_query=build_query_text(patient),
            trace_query=build_patient_query(unified_sample.people),
        )
        response: OnlineRecommendationResponse = OnlineRecommendationResponse(
            normalized_patient=normalized_patient,
            retrieval=RetrievalStageView(
                retrieved_candidate_count=len(retrieved_candidates),
                display_candidate_count=len(retrieval_candidates),
                candidates=retrieval_candidates,
            ),
            rerank=RerankStageView(
                ranked_candidate_count=len(ranked_case.ranked_drugs),
                display_candidate_count=len(rerank_candidates),
                candidates=rerank_candidates,
            ),
            recommendation=RecommendationView(
                success=generation_record.success and len(validation_errors) == 0,
                finish_reason=generation_record.finish_reason,
                error_message=generation_record.error_message,
                trace_id=generation_record.trace_id,
                validation_errors=validation_errors,
                selected_drugids=selected_drugids,
                items=recommendation_items,
            ),
            evidence_map=evidence_map,
            risk_summary=risk_summary,
            pipeline_meta=PipelineMetaView(
                retriever_name=self.config.retriever_name,
                checkpoint_name=resolve_checkpoint_path(self.config.checkpoint_path).name,
                rag_model_name=self.config.rag_model_name,
                retrieval_top_k=self.config.retrieval_top_k,
                display_top_k=self.config.display_top_k,
                max_evidences_per_candidate=self.config.max_evidences_per_candidate,
            ),
            timings_ms=timings_ms,
        )
        timings_ms["assemble_response"] = int((perf_counter() - response_start) * 1000)
        timings_ms["total"] = sum(timings_ms.values())
        return response

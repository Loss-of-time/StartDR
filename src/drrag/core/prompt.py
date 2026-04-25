"""RAG Prompt 构造逻辑。"""

from .adapters import build_patient_query
from .schema import PromptBuildResult, RagCandidate, RagCase, RagEvidence, RagRequest


def _candidate_sort_key(candidate: RagCandidate) -> tuple[int, int]:
    rerank_rank: int = candidate.rerank_rank if candidate.rerank_rank is not None else 10**9
    return rerank_rank, candidate.retrieval_rank


def _evidence_sort_key(evidence: RagEvidence) -> tuple[int, int]:
    rerank_rank: int = evidence.rerank_rank if evidence.rerank_rank is not None else 10**9
    retrieval_rank: int = evidence.retrieval_rank if evidence.retrieval_rank is not None else 10**9
    return rerank_rank, retrieval_rank


def select_candidates(request: RagRequest) -> list[RagCandidate]:
    """按统一规则选择当前请求可见的候选药物。"""

    ranked_candidates: list[RagCandidate] = sorted(request.case.candidates, key=_candidate_sort_key)
    return ranked_candidates[: request.top_k]


def freeze_case_candidates(case: RagCase, top_k: int | None) -> RagCase:
    """按统一选择规则冻结病例候选集。

    Args:
        case: 待处理病例。
        top_k: 保留的候选规模，`None` 表示保留全部。

    Returns:
        候选集顺序与规模均已固定的新病例对象。
    """

    ranked_candidates: list[RagCandidate] = sorted(case.candidates, key=_candidate_sort_key)
    selected_candidates: list[RagCandidate]
    if top_k is None:
        selected_candidates = ranked_candidates
    else:
        selected_candidates = ranked_candidates[:top_k]
    # 目的：把 prompt 侧的排序规则前移到离线产物，保证实验输入文件与实际可见候选完全一致。
    return RagCase(
        patient_id=case.patient_id,
        split=case.split,
        patient=case.patient,
        gold_drugids=list(case.gold_drugids),
        candidates=selected_candidates,
    )


def select_evidences(candidate: RagCandidate, max_count: int) -> list[RagEvidence]:
    """按统一规则选择当前候选药物可见的证据。"""

    ranked_evidences: list[RagEvidence] = sorted(candidate.evidences, key=_evidence_sort_key)
    return ranked_evidences[:max_count]


def build_system_prompt(request: RagRequest) -> str:
    """构造系统提示词。"""

    task_text: str = "推荐药物并给出证据" if request.task == "recommend" else "解释候选药物"
    return (
        "你是一个严格受约束的药物推荐助手。"
        f"你的任务是根据给定患者信息与候选药物证据，{task_text}。"
        "你只能从给定候选 drugid 中选择结果，不能杜撰额外药物。"
        "每条结论必须引用给定 evidence_id。"
        "selected_drugids 与 items 中的 drugid 必须一一对应且顺序一致。"
        "每个 item.reason 必须是简洁中文解释，每个 item.evidence_ids 至少包含一个证据。"
        "你必须输出严格 JSON，不要输出 Markdown，不要输出额外说明。"
        '输出格式为 {"selected_drugids":[...],"items":[{"drugid":"...","reason":"...","evidence_ids":[...]}]}。'
    )


def build_user_prompt(request: RagRequest) -> str:
    """构造用户提示词。"""

    patient = request.case.patient
    patient_lines: list[str] = [
        f"patient_id: {request.case.patient_id}",
        f"年龄: {patient.age}",
        f"性别: {patient.gender}",
        f"人群: {', '.join(patient.group) if patient.group else 'None'}",
        f"诊断: {', '.join(patient.diagnosis) if patient.diagnosis else 'None'}",
        f"症状: {', '.join(patient.symptom) if patient.symptom else 'None'}",
        f"既往史: {', '.join(patient.antecedents) if patient.antecedents else 'None'}",
        f"过敏史: {', '.join(patient.allergen) if patient.allergen else 'None'}",
        f"当前用药: {', '.join(item.name for item in patient.on_medicine) if patient.on_medicine else 'None'}",
        f"检索查询串: {build_patient_query(patient)}",
    ]

    candidate_lines: list[str] = []
    candidate: RagCandidate
    for candidate in select_candidates(request):
        candidate_lines.append(
            f"- drugid={candidate.drugid}; name={candidate.name}; "
            f"retrieval_rank={candidate.retrieval_rank}; "
            f"rerank_rank={candidate.rerank_rank if candidate.rerank_rank is not None else 'None'}; "
            f"rerank_score={candidate.rerank_score if candidate.rerank_score is not None else 'None'}"
        )
        evidence: RagEvidence
        for evidence in select_evidences(candidate, request.max_evidences_per_candidate):
            candidate_lines.append(
                f"  evidence_id={evidence.evidence_id}; source={evidence.source}; text={evidence.text}"
            )

    return (
        "患者信息:\n"
        + "\n".join(patient_lines)
        + "\n\n候选药物与证据:\n"
        + "\n".join(candidate_lines)
        + "\n\n如果没有足够证据支持任何候选药物，请返回空数组。请返回严格 JSON。"
    )


def build_prompt(request: RagRequest) -> PromptBuildResult:
    """构造完整 prompt 与输入 token 估算。"""

    system_prompt: str = build_system_prompt(request)
    user_prompt: str = build_user_prompt(request)
    prompt_text: str = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    selected_candidates: list[RagCandidate] = select_candidates(request)
    evidence_count: int = sum(
        len(select_evidences(candidate, request.max_evidences_per_candidate))
        for candidate in selected_candidates
    )
    return PromptBuildResult(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_text=prompt_text,
        candidate_count=len(selected_candidates),
        evidence_count=evidence_count,
        # 目的：先使用透明且稳定的字符数近似 token，便于离线成本比较。
        input_token_estimate=len(prompt_text),
    )

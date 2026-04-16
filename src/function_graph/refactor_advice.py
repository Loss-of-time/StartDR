"""基于函数图生成重构建议。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .models import FunctionGraphArtifacts, FunctionNode

type SuggestionScope = Literal["function", "module"]
type SuggestionAction = Literal[
    "extract_pure_helper",
    "merge_thin_function",
    "push_side_effect_to_boundary",
    "split_orchestrator",
    "sink_shared_utility",
]
type SuggestionPriority = Literal["high", "medium", "low"]

PRIORITY_WEIGHT: dict[SuggestionPriority, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


@dataclass(slots=True)
class RefactorSuggestion:
    """单条重构建议。

    Attributes:
        target: 建议作用的目标对象。
        scope: 建议作用域，当前支持函数级与模块级。
        action: 建议动作标识。
        priority: 建议优先级。
        summary: 面向人的一句话摘要。
        reason: 触发该建议的原因。
        evidence: 支撑该建议的指标证据。
    """

    target: str
    scope: SuggestionScope
    action: SuggestionAction
    priority: SuggestionPriority
    summary: str
    reason: str
    evidence: list[str]


def build_refactor_suggestions(artifacts: "FunctionGraphArtifacts") -> list[RefactorSuggestion]:
    """根据函数图产出重构建议。

    Args:
        artifacts: 单文件函数图分析结果。

    Returns:
        适合直接执行的重构建议列表。
    """

    suggestions: list[RefactorSuggestion] = []
    for node in artifacts.nodes:
        suggestion = _build_node_suggestion(node)
        if suggestion is not None:
            suggestions.append(suggestion)
    suggestions.extend(_build_module_suggestions(artifacts))
    suggestions.sort(key=lambda item: (PRIORITY_WEIGHT[item.priority], item.target, item.action))
    return suggestions


def _build_base_evidence(node: "FunctionNode") -> list[str]:
    """构造节点级建议的证据列表。

    Args:
        node: 单个函数节点。

    Returns:
        证据字符串列表。
    """

    return [
        f"loc={node.loc}",
        f"complexity={node.complexity}",
        f"fan_in={node.fan_in}",
        f"fan_out={node.fan_out}",
        f"side_effect_score={node.side_effect_score}",
        f"effect_read_score={node.effect_read_score}",
        f"effect_write_score={node.effect_write_score}",
        f"effect_mutation_score={node.effect_mutation_score}",
        f"output_cost={node.output_cost}",
    ]


def _is_shared_side_effect_node(node: "FunctionNode") -> bool:
    """判断节点是否属于高风险共享副作用节点。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中共享副作用规则。
    """

    has_side_effect = node.effect_write_score > 0 or node.effect_mutation_score > 0
    return has_side_effect and node.fan_in >= 2


def _is_orchestrator_node(node: "FunctionNode") -> bool:
    """判断节点是否属于厚编排节点。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中厚编排规则。
    """

    return node.complexity >= 8 and node.fan_out >= 3


def _is_large_pure_node(node: "FunctionNode") -> bool:
    """判断节点是否属于大块纯逻辑节点。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中大块纯逻辑规则。
    """

    # 目的：提高对纯逻辑函数厚度的容忍度，只在规模和复杂度都更高时才建议继续拆分。
    return node.side_effect_score == 0 and node.loc >= 32 and node.complexity >= 8


def _is_thin_passthrough_node(node: "FunctionNode") -> bool:
    """判断节点是否属于适合合并的薄转发函数。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中薄转发规则。
    """

    return (
        node.side_effect_score == 0
        and node.fan_in == 1
        and node.fan_out == 1
        # 目的：提高对薄转发函数的容忍度，只把更极薄的包装层判为可合并。
        and node.loc <= 4
        and node.complexity <= 1
        and node.output_cost <= 1
    )


def _is_thin_leaf_node(node: "FunctionNode") -> bool:
    """判断节点是否属于适合合并的薄叶子函数。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中薄叶子规则。
    """

    return (
        node.side_effect_score == 0
        and node.fan_in == 1
        and node.fan_out == 0
        # 目的：提高对薄叶子函数的容忍度，只合并极小且几乎无输出代价的叶子节点。
        and node.loc <= 3
        and node.complexity <= 1
        and node.output_cost <= 1
    )


def _is_shared_utility_node(node: "FunctionNode") -> bool:
    """判断节点是否属于共享小工具节点。

    Args:
        node: 单个函数节点。

    Returns:
        是否命中共享小工具规则。
    """

    return (
        node.side_effect_score == 0
        and node.loc <= 10
        and node.complexity <= 2
        and node.fan_in >= 2
        and node.fan_out <= 1
    )


def _build_node_suggestion(node: "FunctionNode") -> RefactorSuggestion | None:
    """为单个函数节点生成建议。

    Args:
        node: 单个函数节点。

    Returns:
        若命中规则则返回一条建议，否则返回 `None`。
    """

    base_evidence = _build_base_evidence(node)

    # 目的：优先拦截高扇入的写边界与状态变异节点，这类函数最容易形成难以治理的共享副作用。
    if _is_shared_side_effect_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="push_side_effect_to_boundary",
            priority=_priority_from_score(node, extra_score=4),
            summary="把副作用从共享函数中外移到边界层。",
            reason="该函数被多处依赖且已经存在写边界或共享状态变异，应避免继续作为公共能力扩散。",
            evidence=base_evidence,
        )

    # 目的：优先识别既复杂又负责调度的节点，这类函数最适合先拆成薄编排层。
    if _is_orchestrator_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="split_orchestrator",
            priority=_priority_from_score(node, extra_score=3),
            summary="把厚编排函数拆成更薄的步骤函数。",
            reason="该函数分支较多且依赖下游步骤较多，继续堆逻辑会放大修改波及面。",
            evidence=base_evidence,
        )

    # 目的：单一上游依赖的薄函数继续保留独立层级价值很低，应优先并回调用方减少跳转。
    if _is_thin_passthrough_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="merge_thin_function",
            priority=_priority_from_score(node, extra_score=1),
            summary="把单一上游依赖的薄转发函数合并回调用方。",
            reason="该函数既小又纯，且只承担一层简单转发，单独保留会增加阅读跳转成本。",
            evidence=[*base_evidence, "thin_merge_rule=passthrough"],
        )

    # 目的：只被一处调用的极小纯叶子函数通常不值得维持独立抽象，直接并回更紧凑。
    if _is_thin_leaf_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="merge_thin_function",
            priority=_priority_from_score(node),
            summary="把单一上游依赖的薄叶子函数合并回调用方。",
            reason="该函数没有下游依赖且实现极小，继续拆成独立函数会放大命名与跳转负担。",
            evidence=[*base_evidence, "thin_merge_rule=leaf"],
        )

    # 目的：纯函数一旦变长变复杂，就应优先抽出更稳定的中间计算步骤。
    if _is_large_pure_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="extract_pure_helper",
            priority=_priority_from_score(node, extra_score=2),
            summary="从大块纯逻辑中抽出更小的纯辅助步骤。",
            reason="该函数没有明显副作用，但体积与复杂度都偏高，适合按数据变换步骤继续切分。",
            evidence=base_evidence,
        )

    # 目的：共享且足够小的纯函数应该沉到底层公共工具，减少重复实现。
    if _is_shared_utility_node(node):
        return RefactorSuggestion(
            target=node.qualname,
            scope="function",
            action="sink_shared_utility",
            priority=_priority_from_score(node),
            summary="把稳定小函数沉到底层公共工具层。",
            reason="该函数足够小且被多处复用，单独归到工具层更利于复用与命名稳定。",
            evidence=base_evidence,
        )

    return None


def _build_module_suggestions(artifacts: "FunctionGraphArtifacts") -> list[RefactorSuggestion]:
    """生成模块级建议。

    Args:
        artifacts: 单文件函数图分析结果。

    Returns:
        模块级建议列表。
    """

    boundary_count = _count_boundary_nodes(artifacts)
    orchestration_count = _count_pure_orchestration_nodes(artifacts)
    formatter_count = _count_formatter_nodes(artifacts)
    if boundary_count > 0 and orchestration_count > 0 and formatter_count > 0:
        return [
            RefactorSuggestion(
                target=artifacts.source_path,
                scope="module",
                action="split_orchestrator",
                priority="high",
                summary="把分析流程、格式化输出、边界操作拆开到不同模块。",
                reason="当前文件同时承载主流程编排、结果格式化与边界副作用，职责边界已经开始混杂。",
                evidence=[
                    f"boundary_functions={boundary_count}",
                    f"orchestration_functions={orchestration_count}",
                    f"formatter_functions={formatter_count}",
                ],
            )
        ]
    return []


def _count_boundary_nodes(artifacts: "FunctionGraphArtifacts") -> int:
    """统计边界节点数量。

    Args:
        artifacts: 单文件函数图分析结果。

    Returns:
        命中边界规则的节点数量。
    """

    return sum(
        1
        for node in artifacts.nodes
        if node.effect_read_score > 0
        or node.effect_write_score > 0
        or node.effect_mutation_score > 0
    )


def _count_pure_orchestration_nodes(artifacts: "FunctionGraphArtifacts") -> int:
    """统计纯编排节点数量。

    Args:
        artifacts: 单文件函数图分析结果。

    Returns:
        命中纯编排规则的节点数量。
    """

    return sum(
        1 for node in artifacts.nodes if _is_orchestrator_node(node) and node.side_effect_score == 0
    )


def _count_formatter_nodes(artifacts: "FunctionGraphArtifacts") -> int:
    """统计格式化函数数量。

    Args:
        artifacts: 单文件函数图分析结果。

    Returns:
        命中格式化命名规则的节点数量。
    """

    return sum(
        1
        for node in artifacts.nodes
        if node.simple_name.startswith("build_") or node.simple_name.startswith("render_")
    )


def _priority_from_score(node: "FunctionNode", extra_score: int = 0) -> SuggestionPriority:
    """根据节点代价估计建议优先级。

    Args:
        node: 单个函数节点。
        extra_score: 规则附加分。

    Returns:
        `high`、`medium` 或 `low`。
    """

    score = (
        node.complexity
        + node.fan_in
        + node.fan_out
        + node.effect_read_score
        + node.effect_write_score * 2
        + node.effect_mutation_score * 2
        + max(0, node.loc // 10)
        + extra_score
    )
    if score >= 18:
        return "high"
    if score >= 10:
        return "medium"
    return "low"

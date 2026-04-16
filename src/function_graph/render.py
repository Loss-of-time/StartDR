"""函数图文本与可视化渲染。"""

from .models import FunctionEdge, FunctionGraphArtifacts, FunctionNode
from .refactor_advice import RefactorSuggestion


def _build_suggestion_lines(suggestions: list[RefactorSuggestion]) -> list[str]:
    """生成重构建议 Markdown 行。

    Args:
        suggestions: 重构建议列表。

    Returns:
        Markdown 行列表。
    """

    if not suggestions:
        return ["- 当前未命中启发式重构建议。"]
    lines: list[str] = []
    for suggestion in suggestions:
        evidence_text = ", ".join(suggestion.evidence)
        lines.append(
            "- "
            f"`{suggestion.target}` -> `{suggestion.action}` "
            f"[{suggestion.scope}/{suggestion.priority}]：{suggestion.summary} "
            f"原因：{suggestion.reason} 证据：{evidence_text}"
        )
    return lines


def _build_node_cost_lines(nodes: list[FunctionNode]) -> list[str]:
    """生成节点代价 Markdown 行。

    Args:
        nodes: 函数节点列表。

    Returns:
        Markdown 行列表。
    """

    lines: list[str] = []
    for node in sorted(
        nodes,
        key=lambda item: (-item.fan_in - item.fan_out, -item.complexity, item.qualname),
    ):
        lines.append(
            "- "
            f"`{node.qualname}`: loc={node.loc}, complexity={node.complexity}, "
            f"output_cost={node.output_cost}, side_effect_score={node.side_effect_score}, "
            f"effect_read_score={node.effect_read_score}, "
            f"effect_write_score={node.effect_write_score}, "
            f"effect_mutation_score={node.effect_mutation_score}, "
            f"fan_in={node.fan_in}, fan_out={node.fan_out}"
        )
    return lines


def _build_dot_node_line(node: FunctionNode, roots: set[str]) -> str:
    """生成单个节点的 DOT 行。

    Args:
        node: 函数节点。
        roots: 根节点集合。

    Returns:
        单个节点的 DOT 行文本。
    """

    owner_text = node.owner if node.owner is not None else "模块"
    fillcolor = "#FFE7BA" if node.qualname in roots else "#FFF7E6"
    if node.owner is not None:
        fillcolor = "#FFF1F0"
    label = (
        f"{node.qualname}"
        f"\\nloc={node.loc} cpx={node.complexity}"
        f"\\nout={node.output_cost} io={node.side_effect_score}"
        f" rwm={node.effect_read_score}/{node.effect_write_score}/{node.effect_mutation_score}"
        f"\\nfan={node.fan_in}/{node.fan_out} owner={owner_text}"
    )
    return f'    "{node.qualname}" [label="{label}", fillcolor="{fillcolor}"];'


def _build_dot_edge_line(edge: FunctionEdge) -> str:
    """生成单条边的 DOT 行。

    Args:
        edge: 函数边。

    Returns:
        单条边的 DOT 行文本。
    """

    style = "dashed" if edge.kind == "callback_call" else "solid"
    label = edge.kind
    if edge.via is not None:
        label = f"{edge.kind}\\nvia={edge.via}"
    return f'    "{edge.source}" -> "{edge.target}" [style="{style}", label="{label}"];'


def build_markdown_report(artifacts: FunctionGraphArtifacts) -> str:
    """构造 Markdown 摘要，方便快速阅读。

    Args:
        artifacts: 完整函数图分析结果。

    Returns:
        面向人的 Markdown 报告文本。
    """

    lines = [
        "# 函数图分析报告",
        "",
        f"- 源路径：`{artifacts.source_path}`",
        f"- 节点数：`{artifacts.report.node_count}`",
        f"- 边数：`{artifacts.report.edge_count}`",
        f"- 根节点数：`{artifacts.report.root_count}`",
        f"- merge_excess：`{artifacts.report.merge_excess}`",
        f"- diamond_count：`{artifacts.report.diamond_count}`",
        f"- tree_deviation：`{artifacts.report.tree_deviation}`",
        f"- 重构建议数：`{len(artifacts.suggestions)}`",
        "",
        "## 根节点",
        "",
    ]
    lines.extend(f"- `{root}`" for root in artifacts.report.roots)
    lines.extend(["", "## 枢纽节点", ""])
    lines.extend(f"- `{hub}`" for hub in artifacts.report.top_hubs)
    lines.extend(["", "## 重构建议", ""])
    # 目的：把报告分段文本生成拆细，降低主报告函数的体积与复杂度。
    lines.extend(_build_suggestion_lines(artifacts.suggestions))
    lines.extend(["", "## 节点代价", ""])
    lines.extend(_build_node_cost_lines(artifacts.nodes))
    return "\n".join(lines) + "\n"


def build_dot(artifacts: FunctionGraphArtifacts) -> str:
    """生成 Graphviz DOT。

    Args:
        artifacts: 完整函数图分析结果。

    Returns:
        DOT 文本。
    """

    lines = [
        "// 目的：输出函数级调用与回调关系图，供后续 DAG 压树分析使用。",
        "digraph function_graph {",
        '    graph [rankdir=LR, fontname="Noto Sans CJK SC", fontsize=16, label="函数级依赖图", labelloc="t"];',
        '    node [shape=box, style="rounded,filled", fillcolor="#FFF7E6", color="#C88A2B", fontname="Noto Sans CJK SC"];',
        '    edge [fontname="Noto Sans CJK SC", color="#5B6470"];',
    ]
    roots = set(artifacts.report.roots)
    for node in sorted(artifacts.nodes, key=lambda item: item.qualname):
        lines.append(_build_dot_node_line(node, roots))
    for edge in sorted(
        artifacts.edges,
        key=lambda item: (item.source, item.target, item.kind, item.via or ""),
    ):
        lines.append(_build_dot_edge_line(edge))
    lines.append("}")
    return "\n".join(lines) + "\n"

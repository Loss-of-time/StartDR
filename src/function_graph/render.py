"""函数图文本与可视化渲染。"""

from .models import (
    DataclassNode,
    FunctionEdge,
    FunctionGraphArtifacts,
    FunctionNode,
    PatternCluster,
    SimilarityEdge,
)
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


def _build_pattern_cluster_lines(pattern_clusters: list[PatternCluster]) -> list[str]:
    """生成结构模式聚类 Markdown 行。"""

    if not pattern_clusters:
        return ["- 当前未识别出重复函数簇或相似 dataclass 族。"]
    lines: list[str] = []
    cluster: PatternCluster
    for cluster in pattern_clusters:
        evidence_text: str = ", ".join(cluster.evidence)
        member_text: str = ", ".join(f"`{member}`" for member in cluster.members)
        lines.append(
            "- "
            f"`{cluster.kind}` score={cluster.score:.2f} members=[{member_text}] "
            f"证据：{evidence_text}"
        )
    return lines


def _build_dataclass_lines(dataclass_nodes: list[DataclassNode]) -> list[str]:
    """生成 dataclass 节点 Markdown 行。"""

    if not dataclass_nodes:
        return ["- 当前分析范围内未识别出 dataclass 类型节点。"]
    lines: list[str] = []
    dataclass_node: DataclassNode
    for dataclass_node in dataclass_nodes:
        field_text: str = ", ".join(
            f"{field.name}:{field.annotation or '未知'}" for field in dataclass_node.fields
        )
        method_text: str = ", ".join(dataclass_node.method_names) or "无"
        lines.append(
            "- "
            f"`{dataclass_node.qualname}`: field_count={dataclass_node.field_count}, "
            f"fields=[{field_text}], methods=[{method_text}]"
        )
    return lines


def _build_similarity_edge_lines(similarity_edges: list[SimilarityEdge]) -> list[str]:
    """生成相似性边 Markdown 行。"""

    if not similarity_edges:
        return ["- 当前未识别出结构相似性边。"]
    lines: list[str] = []
    edge: SimilarityEdge
    for edge in similarity_edges:
        evidence_text: str = ", ".join(edge.evidence)
        lines.append(
            "- "
            f"`{edge.source}` -> `{edge.target}` [{edge.kind}] score={edge.score:.2f} "
            f"证据：{evidence_text}"
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


def _build_dot_dataclass_line(node: DataclassNode) -> str:
    """生成单个 dataclass 节点的 DOT 行。"""

    field_names: str = ", ".join(field.name for field in node.fields[:4])
    if node.field_count > 4:
        field_names = f"{field_names}, ..."
    label: str = f"{node.qualname}\\ndataclass fields={node.field_count}\\n[{field_names}]"
    return (
        f'    "{node.qualname}" [label="{label}", shape="note", '
        'fillcolor="#F6FFED", color="#5B8C00"];'
    )


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


def _build_dot_similarity_edge_line(edge: SimilarityEdge) -> str:
    """生成单条结构相似性边的 DOT 行。"""

    style: str = "dashed"
    color: str = "#2F54EB"
    if edge.kind == "near_duplicate_function":
        style = "dotted"
        color = "#13C2C2"
    if edge.kind == "similar_dataclass":
        style = "dashed"
        color = "#389E0D"
    label: str = f"{edge.kind}\\nscore={edge.score:.2f}"
    return (
        f'    "{edge.source}" -> "{edge.target}" '
        f'[style="{style}", color="{color}", label="{label}", constraint="false"];'
    )


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
        f"- dataclass 节点数：`{len(artifacts.dataclass_nodes)}`",
        f"- 相似性边数：`{len(artifacts.similarity_edges)}`",
        f"- 模式簇数：`{len(artifacts.pattern_clusters)}`",
        f"- 重构建议数：`{len(artifacts.suggestions)}`",
        "",
        "## 根节点",
        "",
    ]
    lines.extend(f"- `{root}`" for root in artifacts.report.roots)
    lines.extend(["", "## 枢纽节点", ""])
    lines.extend(f"- `{hub}`" for hub in artifacts.report.top_hubs)
    lines.extend(["", "## 结构模式簇", ""])
    lines.extend(_build_pattern_cluster_lines(artifacts.pattern_clusters))
    lines.extend(["", "## Dataclass 节点", ""])
    lines.extend(_build_dataclass_lines(artifacts.dataclass_nodes))
    lines.extend(["", "## 结构相似性边", ""])
    lines.extend(_build_similarity_edge_lines(artifacts.similarity_edges))
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
    for node in sorted(artifacts.dataclass_nodes, key=lambda item: item.qualname):
        lines.append(_build_dot_dataclass_line(node))
    for edge in sorted(
        artifacts.edges,
        key=lambda item: (item.source, item.target, item.kind, item.via or ""),
    ):
        lines.append(_build_dot_edge_line(edge))
    for edge in sorted(
        artifacts.similarity_edges,
        key=lambda item: (item.kind, item.source, item.target),
    ):
        lines.append(_build_dot_similarity_edge_line(edge))
    lines.append("}")
    return "\n".join(lines) + "\n"

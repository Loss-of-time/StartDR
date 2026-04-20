"""函数图树化建议工具。"""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .graph_metrics import (
    build_child_map,
    build_parent_map,
    compute_diamond_count,
    compute_forest_tree_deviation,
    compute_merge_excess_from_parents,
)
from .models import FunctionEdge, FunctionGraphArtifacts, FunctionGraphReport, FunctionNode


@dataclass(slots=True)
class TreeifyDecision:
    """共享节点的树化决策。"""

    node: str
    action: str
    reason: str
    primary_parent: str | None
    secondary_parents: list[str]


@dataclass(slots=True)
class BackboneMetrics:
    """去交叉后的主干指标。"""

    node_count: int
    edge_count: int
    root_count: int
    merge_excess: int
    diamond_count: int
    tree_deviation: int
    roots: list[str]


@dataclass(slots=True)
class TreeifyArtifacts:
    """树化建议结果。"""

    source_path: str
    decisions: list[TreeifyDecision]
    backbone_metrics: BackboneMetrics
    backbone_roots: list[str]
    utility_nodes: list[str]


def load_graph_artifacts(json_path: Path) -> FunctionGraphArtifacts:
    """从 JSON 恢复函数图 dataclass。"""

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = [FunctionNode(**item) for item in raw["nodes"]]
    edges = [FunctionEdge(**item) for item in raw["edges"]]
    report = FunctionGraphReport(**raw["report"])
    return FunctionGraphArtifacts(
        source_path=raw["source_path"],
        nodes=nodes,
        edges=edges,
        report=report,
    )


def _should_sink_to_utility_layer(node: FunctionNode) -> bool:
    """判断共享节点是否应沉到底层工具层。

    Args:
        node: 单个函数节点。

    Returns:
        是否适合沉底。
    """

    return node.side_effect_score == 0 and node.fan_out <= 1


def _should_duplicate_under_each_parent(node: FunctionNode) -> bool:
    """判断共享节点是否适合复制到各父节点下。

    Args:
        node: 单个函数节点。

    Returns:
        是否适合按父节点复制。
    """

    return (
        node.side_effect_score == 0
        and node.loc <= 8
        and node.complexity <= 1
        and node.output_cost <= 3
    )


def _build_treeify_decision(
    node_name: str,
    parents: list[str],
    node_by_name: dict[str, FunctionNode],
    roots: set[str],
) -> TreeifyDecision:
    """为单个共享节点生成树化决策。

    Args:
        node_name: 节点限定名。
        parents: 父节点列表。
        node_by_name: 节点索引。
        roots: 根节点集合。

    Returns:
        单个共享节点的树化决策。
    """

    node = node_by_name[node_name]
    sorted_parents = sorted(parents)
    if _should_sink_to_utility_layer(node):
        return TreeifyDecision(
            node=node_name,
            action="sink_to_utility_layer",
            reason="共享节点无副作用且扇出很小，更适合沉到底层公共工具层。",
            primary_parent=None,
            secondary_parents=sorted_parents,
        )
    if _should_duplicate_under_each_parent(node):
        return TreeifyDecision(
            node=node_name,
            action="duplicate_under_each_parent",
            reason="节点足够小且纯，复制到各父节点下的维护成本较低。",
            primary_parent=None,
            secondary_parents=sorted_parents,
        )
    primary_parent = choose_primary_parent(parents, node_by_name, roots)
    secondary_parents = sorted(parent for parent in parents if parent != primary_parent)
    return TreeifyDecision(
        node=node_name,
        action="assign_primary_owner",
        reason="节点较重或输出较大，保留一个主归属比复制更稳。",
        primary_parent=primary_parent,
        secondary_parents=secondary_parents,
    )


def choose_primary_parent(
    parents: list[str],
    node_by_name: dict[str, FunctionNode],
    roots: set[str],
) -> str:
    """为需要保留单父归属的共享节点选择主父节点。"""

    def parent_score(parent: str) -> tuple[int, int, int, str]:
        node = node_by_name[parent]
        # 目的：优先保留更靠主干、更像编排层的父节点，让树状主线更稳定。
        root_bonus = 1 if parent in roots else 0
        orchestration_score = node.fan_out
        complexity_penalty = -node.complexity
        return (root_bonus, orchestration_score, complexity_penalty, parent)

    return max(parents, key=parent_score)


def decide_treeify_actions(graph: FunctionGraphArtifacts) -> list[TreeifyDecision]:
    """为所有多父节点生成树化决策。"""

    node_by_name = {node.qualname: node for node in graph.nodes}
    parents_by_node = build_parent_map(graph.edges)
    roots = set(graph.report.roots)
    decisions: list[TreeifyDecision] = []
    for node_name, parents in sorted(parents_by_node.items()):
        if len(parents) <= 1:
            continue
        decisions.append(_build_treeify_decision(node_name, parents, node_by_name, roots))
    return decisions


def build_backbone_edges(
    graph: FunctionGraphArtifacts,
    decisions: list[TreeifyDecision],
) -> list[FunctionEdge]:
    """按树化决策生成主干边。"""

    utility_nodes = {
        decision.node for decision in decisions if decision.action == "sink_to_utility_layer"
    }
    owner_by_node = {
        decision.node: decision.primary_parent
        for decision in decisions
        if decision.action == "assign_primary_owner" and decision.primary_parent is not None
    }
    backbone_edges: list[FunctionEdge] = []
    for edge in graph.edges:
        if edge.source in utility_nodes or edge.target in utility_nodes:
            continue
        primary_parent = owner_by_node.get(edge.target)
        if primary_parent is not None and edge.source != primary_parent:
            continue
        backbone_edges.append(edge)
    return backbone_edges


def compute_backbone_metrics(
    graph: FunctionGraphArtifacts,
    backbone_edges: list[FunctionEdge],
    utility_nodes: set[str],
) -> BackboneMetrics:
    """计算主干图指标。"""

    backbone_node_names = {
        node.qualname for node in graph.nodes if node.qualname not in utility_nodes
    }
    parents_by_node = build_parent_map(backbone_edges)
    children_by_node = {node_name: set[str]() for node_name in backbone_node_names}
    for edge in backbone_edges:
        children_by_node[edge.source].add(edge.target)
    roots = sorted(
        node_name
        for node_name in backbone_node_names
        if len(parents_by_node.get(node_name, [])) == 0
    )
    return BackboneMetrics(
        node_count=len(backbone_node_names),
        edge_count=len(backbone_edges),
        root_count=len(roots),
        merge_excess=compute_merge_excess_from_parents(parents_by_node),
        diamond_count=compute_diamond_count(children_by_node),
        tree_deviation=compute_forest_tree_deviation(
            node_count=len(backbone_node_names),
            edge_count=len(backbone_edges),
            root_count=len(roots),
        ),
        roots=roots,
    )


def collect_backbone_subtree(
    root: str,
    children_by_node: dict[str, list[str]],
) -> list[str]:
    """收集某个根节点下的主干子树。"""

    queue: deque[str] = deque([root])
    visited: set[str] = set()
    ordered_nodes: list[str] = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        ordered_nodes.append(node)
        for child in sorted(children_by_node.get(node, [])):
            queue.append(child)
    return ordered_nodes


def build_markdown_report(
    graph: FunctionGraphArtifacts,
    treeify: TreeifyArtifacts,
    backbone_edges: list[FunctionEdge],
) -> str:
    """生成树化建议 Markdown 报告。"""

    children_by_node = build_child_map(backbone_edges)
    lines = [
        "# 函数图树化建议",
        "",
        f"- 源文件：`{treeify.source_path}`",
        f"- 原始节点数：`{graph.report.node_count}`",
        f"- 原始边数：`{graph.report.edge_count}`",
        f"- 原始 merge_excess：`{graph.report.merge_excess}`",
        f"- 原始 diamond_count：`{graph.report.diamond_count}`",
        f"- 原始 tree_deviation：`{graph.report.tree_deviation}`",
        "",
        "## 主干结果",
        "",
        f"- 主干节点数：`{treeify.backbone_metrics.node_count}`",
        f"- 主干边数：`{treeify.backbone_metrics.edge_count}`",
        f"- 主干根节点：`{', '.join(treeify.backbone_metrics.roots)}`",
        f"- 主干 merge_excess：`{treeify.backbone_metrics.merge_excess}`",
        f"- 主干 diamond_count：`{treeify.backbone_metrics.diamond_count}`",
        f"- 主干 tree_deviation：`{treeify.backbone_metrics.tree_deviation}`",
        "",
        "## 共享节点决策",
        "",
    ]
    # 目的：把不同区块的文本生成拆开，避免树化报告函数继续膨胀。
    lines.extend(_build_decision_lines(treeify.decisions))
    lines.extend(["", "## 建议沉底的公共工具层", ""])
    for node_name in treeify.utility_nodes:
        lines.append(f"- `{node_name}`")
    lines.extend(["", "## 建议主干", ""])
    lines.extend(_build_backbone_lines(treeify.backbone_roots, children_by_node))
    return "\n".join(lines) + "\n"


def _build_decision_lines(decisions: list[TreeifyDecision]) -> list[str]:
    """生成共享节点决策区块文本。

    Args:
        decisions: 树化决策列表。

    Returns:
        Markdown 行列表。
    """

    lines: list[str] = []
    for decision in decisions:
        if decision.action == "sink_to_utility_layer":
            lines.append(
                f"- `{decision.node}` -> `sink_to_utility_layer`："
                f"父节点=`{', '.join(decision.secondary_parents)}`；{decision.reason}"
            )
            continue
        if decision.action == "duplicate_under_each_parent":
            lines.append(
                f"- `{decision.node}` -> `duplicate_under_each_parent`："
                f"父节点=`{', '.join(decision.secondary_parents)}`；{decision.reason}"
            )
            continue
        lines.append(
            f"- `{decision.node}` -> `assign_primary_owner`："
            f"主父节点=`{decision.primary_parent}`，次父节点=`{', '.join(decision.secondary_parents)}`；"
            f"{decision.reason}"
        )
    return lines


def _build_backbone_lines(
    backbone_roots: list[str],
    children_by_node: dict[str, list[str]],
) -> list[str]:
    """生成主干区块文本。

    Args:
        backbone_roots: 主干根节点列表。
        children_by_node: 子节点映射。

    Returns:
        Markdown 行列表。
    """

    lines: list[str] = []
    for root in backbone_roots:
        lines.append(f"- 根节点 `{root}`")
        for node_name in collect_backbone_subtree(root, children_by_node)[1:]:
            lines.append(f"- `{root}` 子树包含 `{node_name}`")
    return lines


def build_dot(
    graph: FunctionGraphArtifacts,
    treeify: TreeifyArtifacts,
    backbone_edges: list[FunctionEdge],
) -> str:
    """生成树化建议 DOT。"""

    utility_nodes = set(treeify.utility_nodes)
    backbone_node_names = {
        node.qualname for node in graph.nodes if node.qualname not in utility_nodes
    }
    lines = [
        "// 目的：展示“树状主干 + 公共工具层”的函数结构建议。",
        "digraph function_graph_treeify {",
        '    graph [rankdir=LR, fontname="Noto Sans CJK SC", fontsize=16, label="函数图树化建议", labelloc="t"];',
        '    node [shape=box, style="rounded,filled", fontname="Noto Sans CJK SC"];',
        '    edge [fontname="Noto Sans CJK SC", color="#5B6470"];',
        "    subgraph cluster_backbone {",
        '        label="树状主干";',
        '        color="#CFE4FF";',
        '        style="rounded";',
    ]
    for node in graph.nodes:
        if node.qualname not in backbone_node_names:
            continue
        fillcolor = "#FFE7BA" if node.qualname in treeify.backbone_roots else "#FFF7E6"
        label = f"{node.qualname}\\nloc={node.loc} cpx={node.complexity}"
        lines.append(
            f'        "{node.qualname}" [label="{label}", fillcolor="{fillcolor}", color="#C88A2B"];'
        )
    lines.append("    }")
    lines.append("    subgraph cluster_utility {")
    lines.append('        label="公共工具层";')
    lines.append('        color="#D5F5E3";')
    lines.append('        style="rounded";')
    for node in graph.nodes:
        if node.qualname not in utility_nodes:
            continue
        label = f"{node.qualname}\\nloc={node.loc} cpx={node.complexity}"
        lines.append(
            f'        "{node.qualname}" [label="{label}", fillcolor="#EAF7EA", color="#4C9A5E"];'
        )
    lines.append("    }")
    for edge in backbone_edges:
        lines.append(f'    "{edge.source}" -> "{edge.target}" [style="solid"];')
    for edge in graph.edges:
        if edge.target in utility_nodes and edge.source not in utility_nodes:
            lines.append(
                f'    "{edge.source}" -> "{edge.target}" [style="dashed", color="#4C9A5E", label="utility"];'
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def analyze_treeify(graph: FunctionGraphArtifacts) -> tuple[TreeifyArtifacts, list[FunctionEdge]]:
    """执行树化分析。"""

    decisions = decide_treeify_actions(graph)
    utility_nodes = sorted(
        decision.node for decision in decisions if decision.action == "sink_to_utility_layer"
    )
    backbone_edges = build_backbone_edges(graph, decisions)
    backbone_metrics = compute_backbone_metrics(graph, backbone_edges, set(utility_nodes))
    treeify_artifacts = TreeifyArtifacts(
        source_path=graph.source_path,
        decisions=decisions,
        backbone_metrics=backbone_metrics,
        backbone_roots=backbone_metrics.roots,
        utility_nodes=utility_nodes,
    )
    return treeify_artifacts, backbone_edges

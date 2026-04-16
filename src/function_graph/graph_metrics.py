"""函数图与树化分析共享图指标函数。"""

from .models import FunctionEdge


def build_parent_map(edges: list[FunctionEdge]) -> dict[str, list[str]]:
    """构造父节点映射。

    Args:
        edges: 函数边列表。

    Returns:
        目标节点到父节点列表的映射。
    """

    parents_by_node: dict[str, list[str]] = {}
    for edge in edges:
        parents_by_node.setdefault(edge.target, []).append(edge.source)
    return parents_by_node


def build_child_map(edges: list[FunctionEdge]) -> dict[str, list[str]]:
    """构造子节点映射。

    Args:
        edges: 函数边列表。

    Returns:
        源节点到子节点列表的映射。
    """

    children_by_node: dict[str, list[str]] = {}
    for edge in edges:
        children_by_node.setdefault(edge.source, []).append(edge.target)
    return children_by_node


def compute_tree_deviation(node_count: int, edge_count: int) -> int:
    """计算无向骨架相对树的偏离度。

    Args:
        node_count: 节点数。
        edge_count: 边数。

    Returns:
        相对单棵树的偏离度。
    """

    if node_count == 0:
        return 0
    return max(0, edge_count - node_count + 1)


def compute_forest_tree_deviation(node_count: int, edge_count: int, root_count: int) -> int:
    """按森林而不是单棵树计算偏离度。

    Args:
        node_count: 节点数。
        edge_count: 边数。
        root_count: 根节点数。

    Returns:
        相对森林结构的偏离度。
    """

    if node_count == 0:
        return 0
    target_edge_count = max(0, node_count - root_count)
    return max(0, edge_count - target_edge_count)


def compute_merge_excess_from_indegree(indegree_by_node: dict[str, int]) -> int:
    """计算多父节点的汇合超额。

    Args:
        indegree_by_node: 节点入度映射。

    Returns:
        汇合超额值。
    """

    return sum(max(0, indegree - 1) for indegree in indegree_by_node.values())


def compute_merge_excess_from_parents(parents_by_node: dict[str, list[str]]) -> int:
    """从父节点映射计算汇合超额。

    Args:
        parents_by_node: 节点父节点映射。

    Returns:
        汇合超额值。
    """

    return sum(max(0, len(parents) - 1) for parents in parents_by_node.values())


def compute_diamond_count(out_neighbors: dict[str, set[str]]) -> int:
    """计算 2x2 菱形结构数量。

    Args:
        out_neighbors: 节点出邻居集合映射。

    Returns:
        菱形结构数量。
    """

    parents = sorted(out_neighbors)
    diamond_count = 0
    for left_index, left_parent in enumerate(parents):
        left_children = out_neighbors[left_parent]
        for right_parent in parents[left_index + 1 :]:
            common_children = left_children & out_neighbors[right_parent]
            if len(common_children) >= 2:
                common_count = len(common_children)
                diamond_count += common_count * (common_count - 1) // 2
    return diamond_count

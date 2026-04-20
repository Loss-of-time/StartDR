"""函数图分析主流程。"""

from pathlib import Path

from .ast_visitors import (
    CallVisitor,
    ComplexityVisitor,
    ReturnVisitor,
    SideEffectVisitor,
    annotation_text,
    collect_function_instance_bindings,
    collect_parameter_names,
)
from .graph_metrics import (
    compute_diamond_count,
    compute_merge_excess_from_indegree,
    compute_tree_deviation,
)
from .models import (
    FunctionEdge,
    FunctionGraphArtifacts,
    FunctionGraphReport,
    FunctionNode,
    ModuleAnalysisContext,
    NodeName,
    RawFunction,
    ScopeName,
)
from .module_resolution import (
    build_module_name,
    collect_imported_symbols,
    collect_module_context,
    collect_module_symbols,
    collect_python_sources,
    merge_class_scope_symbols,
)
from .pattern_analysis import analyze_patterns
from .refactor_advice import build_refactor_suggestions


def _collect_function_artifacts(
    function: RawFunction,
    context: ModuleAnalysisContext,
    class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
    class_attribute_types: dict[ScopeName, dict[str, NodeName]],
    module_symbols: dict[str, dict[str, NodeName]],
) -> tuple[FunctionNode, list[FunctionEdge]]:
    """分析单个函数并生成节点及边。

    Args:
        function: 当前函数定义。
        context: 当前函数所在的模块上下文。
        class_scope_symbols: 全局类成员符号表。
        class_attribute_types: 全局类属性实例类型表。
        module_symbols: 目录内模块顶层符号表。

    Returns:
        单个函数节点与其直接产出的边列表。
    """

    function_node = function.node
    complexity_visitor = ComplexityVisitor()
    return_visitor = ReturnVisitor(annotation_text(function_node.returns))
    side_effect_visitor = SideEffectVisitor(
        parameter_names=collect_parameter_names(function_node.args)
    )
    local_instance_types, _ = collect_function_instance_bindings(
        function=function,
        scope_symbols=context.collector.scope_symbols,
        scope_parents=context.scope_parents,
        class_scope_symbols=class_scope_symbols,
        class_attribute_types=class_attribute_types,
        imported_symbols=context.imported_symbols,
        module_symbols=module_symbols,
    )
    call_visitor = CallVisitor(
        current_scope=function.qualname,
        current_class=function.class_owner,
        scope_symbols=context.collector.scope_symbols,
        scope_parents=context.scope_parents,
        class_scope_symbols=class_scope_symbols,
        class_attribute_types=class_attribute_types,
        imported_symbols=context.imported_symbols,
        module_symbols=module_symbols,
        local_instance_types=local_instance_types,
    )
    for statement in function_node.body:
        complexity_visitor.visit(statement)
        return_visitor.visit(statement)
        side_effect_visitor.visit(statement)
        call_visitor.visit(statement)
    node = FunctionNode(
        qualname=function.qualname,
        simple_name=function.simple_name,
        owner=function.owner,
        lineno=function_node.lineno,
        end_lineno=function_node.end_lineno or function_node.lineno,
        loc=(function_node.end_lineno or function_node.lineno) - function_node.lineno + 1,
        complexity=complexity_visitor.value,
        output_cost=return_visitor.value,
        side_effect_score=side_effect_visitor.total_score,
        effect_read_score=side_effect_visitor.read_score,
        effect_write_score=side_effect_visitor.write_score,
        effect_mutation_score=side_effect_visitor.mutation_score,
    )
    return node, call_visitor.edges


def _collect_nodes_and_edges(
    contexts: list[ModuleAnalysisContext],
    class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
    class_attribute_types: dict[ScopeName, dict[str, NodeName]],
    module_symbols: dict[str, dict[str, NodeName]],
) -> tuple[list[FunctionNode], dict[tuple[str, str, str, str | None], FunctionEdge]]:
    """收集全部函数节点与去重前的边。

    Args:
        contexts: 全部模块上下文。
        class_scope_symbols: 全局类成员符号表。
        class_attribute_types: 全局类属性实例类型表。
        module_symbols: 目录内模块顶层符号表。

    Returns:
        节点列表与按四元组键去重的边映射。
    """

    nodes: list[FunctionNode] = []
    edge_map: dict[tuple[str, str, str, str | None], FunctionEdge] = {}
    for context in contexts:
        for function in context.collector.functions:
            node, edges = _collect_function_artifacts(
                function=function,
                context=context,
                class_scope_symbols=class_scope_symbols,
                class_attribute_types=class_attribute_types,
                module_symbols=module_symbols,
            )
            nodes.append(node)
            for edge in edges:
                edge_map[(edge.source, edge.target, edge.kind, edge.via)] = edge
    return nodes, edge_map


def _collect_class_attribute_types(
    contexts: list[ModuleAnalysisContext],
    class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
    module_symbols: dict[str, dict[str, NodeName]],
) -> dict[ScopeName, dict[str, NodeName]]:
    """收集全部类属性上的实例类型信息。

    Args:
        contexts: 全部模块上下文。
        class_scope_symbols: 全局类成员符号表。
        module_symbols: 目录内模块顶层符号表。

    Returns:
        类限定名到属性实例类型映射。
    """

    class_attribute_types: dict[ScopeName, dict[str, NodeName]] = {}
    context: ModuleAnalysisContext
    function: RawFunction
    attribute_instance_types: dict[str, NodeName]
    for context in contexts:
        for function in context.collector.functions:
            if function.class_owner is None:
                continue
            _, attribute_instance_types = collect_function_instance_bindings(
                function=function,
                scope_symbols=context.collector.scope_symbols,
                scope_parents=context.scope_parents,
                class_scope_symbols=class_scope_symbols,
                class_attribute_types=class_attribute_types,
                imported_symbols=context.imported_symbols,
                module_symbols=module_symbols,
            )
            if not attribute_instance_types:
                continue
            class_attribute_types.setdefault(function.class_owner, {}).update(
                attribute_instance_types
            )
    return class_attribute_types


def _materialize_graph_edges(
    nodes: list[FunctionNode],
    edge_map: dict[tuple[str, str, str, str | None], FunctionEdge],
) -> tuple[list[FunctionEdge], dict[str, int], dict[str, set[str]]]:
    """过滤无效边并回填扇入扇出统计。

    Args:
        nodes: 函数节点列表。
        edge_map: 去重后的边映射。

    Returns:
        有效边列表、入度映射与出邻居映射。
    """

    node_index = {node.qualname: node for node in nodes}
    indegree_by_node = {node.qualname: 0 for node in nodes}
    out_neighbors = {node.qualname: set[str]() for node in nodes}
    filtered_edges: list[FunctionEdge] = []
    for edge in edge_map.values():
        if edge.target not in node_index:
            continue
        filtered_edges.append(edge)
        indegree_by_node[edge.target] += 1
        node_index[edge.source].fan_out += 1
        node_index[edge.target].fan_in += 1
        out_neighbors[edge.source].add(edge.target)
    return filtered_edges, indegree_by_node, out_neighbors


def _build_graph_report(
    nodes: list[FunctionNode],
    filtered_edges: list[FunctionEdge],
    indegree_by_node: dict[str, int],
    out_neighbors: dict[str, set[str]],
) -> FunctionGraphReport:
    """汇总函数图统计指标。

    Args:
        nodes: 函数节点列表。
        filtered_edges: 有效边列表。
        indegree_by_node: 入度映射。
        out_neighbors: 出邻居映射。

    Returns:
        图统计摘要。
    """

    roots = sorted(node_name for node_name, indegree in indegree_by_node.items() if indegree == 0)
    top_hubs = [
        node.qualname
        for node in sorted(
            nodes,
            key=lambda item: (-(item.fan_in + item.fan_out), -item.complexity, item.qualname),
        )[:5]
    ]
    return FunctionGraphReport(
        node_count=len(nodes),
        edge_count=len(filtered_edges),
        root_count=len(roots),
        merge_excess=compute_merge_excess_from_indegree(indegree_by_node),
        diamond_count=compute_diamond_count(out_neighbors),
        tree_deviation=compute_tree_deviation(len(nodes), len(filtered_edges)),
        roots=roots,
        top_hubs=top_hubs,
    )


def _analyze_contexts(
    contexts: list[ModuleAnalysisContext],
    source_path: Path,
) -> FunctionGraphArtifacts:
    """基于一个或多个模块上下文构造完整函数图。

    Args:
        contexts: 单文件或目录整体分析得到的模块上下文列表。
        source_path: 用户传入的源路径。

    Returns:
        完整函数图分析结果。
    """

    module_symbols = collect_module_symbols(contexts)
    for context in contexts:
        context.imported_symbols = collect_imported_symbols(context, module_symbols)
    class_scope_symbols = merge_class_scope_symbols(contexts)
    class_attribute_types = _collect_class_attribute_types(
        contexts=contexts,
        class_scope_symbols=class_scope_symbols,
        module_symbols=module_symbols,
    )
    # 目的：把目录内全部模块的节点和边统一汇总，输出真正的整体函数图。
    nodes, edge_map = _collect_nodes_and_edges(
        contexts,
        class_scope_symbols,
        class_attribute_types,
        module_symbols,
    )
    filtered_edges, indegree_by_node, out_neighbors = _materialize_graph_edges(nodes, edge_map)
    report = _build_graph_report(nodes, filtered_edges, indegree_by_node, out_neighbors)
    artifacts = FunctionGraphArtifacts(
        source_path=str(source_path),
        nodes=sorted(nodes, key=lambda item: item.qualname),
        edges=sorted(
            filtered_edges,
            key=lambda item: (item.source, item.target, item.kind, item.via or ""),
        ),
        report=report,
    )
    (
        artifacts.dataclass_nodes,
        artifacts.similarity_edges,
        artifacts.pattern_clusters,
    ) = analyze_patterns(contexts)
    # 目的：目录整体图也沿用同一套建议生成逻辑，避免只输出结构不输出动作。
    artifacts.suggestions.extend(build_refactor_suggestions(artifacts))
    return artifacts


def analyze_source(source_path: Path) -> FunctionGraphArtifacts:
    """分析单文件函数依赖图。

    Args:
        source_path: 待分析源文件路径。

    Returns:
        包含节点、边、指标与重构建议的完整分析结果。
    """

    context = collect_module_context(source_path=source_path, module_name=None)
    return _analyze_contexts([context], source_path)


def analyze_directory(source_dir: Path) -> FunctionGraphArtifacts:
    """分析目录整体函数依赖图。

    Args:
        source_dir: 待分析目录。

    Returns:
        把目录视为一个整体后的函数图分析结果。
    """

    source_paths = collect_python_sources(source_dir)
    contexts = [
        collect_module_context(
            source_path=source_path,
            module_name=build_module_name(source_dir, source_path),
        )
        for source_path in source_paths
    ]
    return _analyze_contexts(contexts, source_dir)

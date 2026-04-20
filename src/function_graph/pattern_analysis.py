"""函数与 dataclass 结构模式分析。"""

import ast
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

from .ast_visitors import annotation_text, dotted_name
from .models import (
    DataclassField,
    DataclassNode,
    ModuleAnalysisContext,
    PatternCluster,
    RawClass,
    RawFunction,
    SimilarityEdge,
    SimilarityKind,
)

NEAR_DUPLICATE_THRESHOLD = 0.88
NEAR_DUPLICATE_SAME_NAME_THRESHOLD = 0.8
NEAR_DUPLICATE_MIN_LOC = 5
DATACLASS_SIMILARITY_THRESHOLD = 0.6


@dataclass(slots=True)
class _FunctionPatternRecord:
    """函数结构指纹记录。"""

    qualname: str
    simple_name: str
    lineno: int
    loc: int
    normalized_dump: str


class _StructureNormalizer(ast.NodeTransformer):
    """把函数体归一化到结构骨架。"""

    def __init__(self) -> None:
        self._name_tokens: dict[str, str] = {}
        self._counter = 0

    def _tokenize_name(self, name: str) -> str:
        if name not in self._name_tokens:
            self._counter += 1
            self._name_tokens[name] = f"v{self._counter}"
        return self._name_tokens[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(
            ast.Name(id=self._tokenize_name(node.id), ctx=node.ctx),
            node,
        )

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._tokenize_name(node.arg)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        node.attr = "attr"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)
        return node


def analyze_patterns(
    contexts: list[ModuleAnalysisContext],
) -> tuple[list[DataclassNode], list[SimilarityEdge], list[PatternCluster]]:
    """分析结构重复模式与相似 dataclass。

    Args:
        contexts: 全部模块分析上下文。

    Returns:
        dataclass 节点、结构相似性边与模式聚类。
    """

    function_records: list[_FunctionPatternRecord] = _collect_function_pattern_records(contexts)
    dataclass_nodes: list[DataclassNode] = _collect_dataclass_nodes(contexts)

    similarity_edges: list[SimilarityEdge] = []
    pattern_clusters: list[PatternCluster] = []

    exact_edges, exact_clusters = _build_exact_duplicate_function_patterns(function_records)
    near_edges, near_clusters = _build_near_duplicate_function_patterns(
        function_records=function_records,
        existing_edges=exact_edges,
    )
    dataclass_edges, dataclass_clusters = _build_dataclass_similarity_patterns(dataclass_nodes)

    similarity_edges.extend(exact_edges)
    similarity_edges.extend(near_edges)
    similarity_edges.extend(dataclass_edges)
    pattern_clusters.extend(exact_clusters)
    pattern_clusters.extend(near_clusters)
    pattern_clusters.extend(dataclass_clusters)

    similarity_edges.sort(
        key=lambda item: (item.kind, -item.score, item.source, item.target),
    )
    pattern_clusters.sort(
        key=lambda item: (item.kind, -item.score, item.members),
    )
    return dataclass_nodes, similarity_edges, pattern_clusters


def _collect_function_pattern_records(
    contexts: list[ModuleAnalysisContext],
) -> list[_FunctionPatternRecord]:
    """收集函数结构指纹记录。"""

    records: list[_FunctionPatternRecord] = []
    context: ModuleAnalysisContext
    function: RawFunction
    for context in contexts:
        for function in context.collector.functions:
            normalized_dump: str = _build_normalized_function_dump(function.node)
            loc: int = (function.node.end_lineno or function.node.lineno) - function.node.lineno + 1
            records.append(
                _FunctionPatternRecord(
                    qualname=function.qualname,
                    simple_name=function.simple_name,
                    lineno=function.node.lineno,
                    loc=loc,
                    normalized_dump=normalized_dump,
                )
            )
    return records


def _build_normalized_function_dump(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """构造函数体的归一化 AST 文本。"""

    normalized_module: ast.Module = ast.Module(body=node.body, type_ignores=[])
    normalized_module = ast.fix_missing_locations(_StructureNormalizer().visit(normalized_module))
    return ast.dump(
        normalized_module,
        annotate_fields=False,
        include_attributes=False,
    )


def _build_exact_duplicate_function_patterns(
    function_records: list[_FunctionPatternRecord],
) -> tuple[list[SimilarityEdge], list[PatternCluster]]:
    """构造精确重复函数模式。"""

    records_by_dump: dict[str, list[_FunctionPatternRecord]] = defaultdict(list)
    record: _FunctionPatternRecord
    for record in function_records:
        records_by_dump[record.normalized_dump].append(record)

    similarity_edges: list[SimilarityEdge] = []
    pattern_clusters: list[PatternCluster] = []
    group: list[_FunctionPatternRecord]
    for group in records_by_dump.values():
        if len(group) <= 1:
            continue
        sorted_group: list[_FunctionPatternRecord] = sorted(
            group,
            key=lambda item: (item.simple_name, item.qualname),
        )
        evidence: list[str] = [
            f"member_count={len(sorted_group)}",
            f"simple_names={','.join(sorted({item.simple_name for item in sorted_group}))}",
        ]
        pattern_clusters.append(
            PatternCluster(
                kind="exact_duplicate_function",
                score=1.0,
                members=[item.qualname for item in sorted_group],
                evidence=evidence,
            )
        )
        left: _FunctionPatternRecord
        right: _FunctionPatternRecord
        for left, right in combinations(sorted_group, 2):
            similarity_edges.append(
                SimilarityEdge(
                    source=left.qualname,
                    target=right.qualname,
                    kind="exact_duplicate_function",
                    score=1.0,
                    evidence=[
                        f"left_loc={left.loc}",
                        f"right_loc={right.loc}",
                        "normalized_ast=identical",
                    ],
                )
            )
    return similarity_edges, pattern_clusters


def _build_near_duplicate_function_patterns(
    function_records: list[_FunctionPatternRecord],
    existing_edges: list[SimilarityEdge],
) -> tuple[list[SimilarityEdge], list[PatternCluster]]:
    """构造近似重复函数模式。"""

    exact_pairs: set[tuple[str, str]] = {
        _ordered_pair(edge.source, edge.target) for edge in existing_edges
    }
    candidate_edges: list[SimilarityEdge] = []
    left: _FunctionPatternRecord
    right: _FunctionPatternRecord
    for left, right in combinations(function_records, 2):
        pair_key: tuple[str, str] = _ordered_pair(left.qualname, right.qualname)
        if pair_key in exact_pairs:
            continue
        if min(left.loc, right.loc) < NEAR_DUPLICATE_MIN_LOC:
            continue
        score: float = SequenceMatcher(
            None,
            left.normalized_dump,
            right.normalized_dump,
        ).ratio()
        threshold: float = (
            NEAR_DUPLICATE_SAME_NAME_THRESHOLD
            if left.simple_name == right.simple_name
            else NEAR_DUPLICATE_THRESHOLD
        )
        if score < threshold:
            continue
        candidate_edges.append(
            SimilarityEdge(
                source=left.qualname,
                target=right.qualname,
                kind="near_duplicate_function",
                score=round(score, 4),
                evidence=[
                    f"left_loc={left.loc}",
                    f"right_loc={right.loc}",
                    f"same_simple_name={left.simple_name == right.simple_name}",
                ],
            )
        )
    return candidate_edges, _build_similarity_clusters("near_duplicate_function", candidate_edges)


def _collect_dataclass_nodes(
    contexts: list[ModuleAnalysisContext],
) -> list[DataclassNode]:
    """收集 dataclass 类型节点。"""

    dataclass_nodes: list[DataclassNode] = []
    context: ModuleAnalysisContext
    raw_class: RawClass
    for context in contexts:
        for raw_class in context.collector.classes:
            if not _is_dataclass(raw_class.node):
                continue
            dataclass_nodes.append(_build_dataclass_node(raw_class))
    dataclass_nodes.sort(key=lambda item: item.qualname)
    return dataclass_nodes


def _is_dataclass(node: ast.ClassDef) -> bool:
    """判断类定义是否标注了 dataclass。"""

    decorator: ast.expr
    for decorator in node.decorator_list:
        decorator_name: str = dotted_name(decorator)
        if decorator_name.endswith("dataclass"):
            return True
        if isinstance(decorator, ast.Call):
            decorator_name = dotted_name(decorator.func)
            if decorator_name.endswith("dataclass"):
                return True
    return False


def _build_dataclass_node(raw_class: RawClass) -> DataclassNode:
    """把类定义转换成 dataclass 节点。"""

    fields: list[DataclassField] = []
    method_names: list[str] = []
    statement: ast.stmt
    for statement in raw_class.node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            fields.append(
                DataclassField(
                    name=statement.target.id,
                    annotation=annotation_text(statement.annotation),
                    has_default=statement.value is not None,
                    default_kind=_default_kind(statement.value),
                )
            )
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_names.append(statement.name)
    return DataclassNode(
        qualname=raw_class.qualname,
        simple_name=raw_class.simple_name,
        owner=raw_class.owner,
        lineno=raw_class.node.lineno,
        end_lineno=raw_class.node.end_lineno or raw_class.node.lineno,
        field_count=len(fields),
        fields=fields,
        method_names=sorted(method_names),
    )


def _default_kind(value: ast.expr | None) -> str | None:
    """提取字段默认值的大类。"""

    if value is None:
        return None
    if isinstance(value, ast.Constant):
        return type(value.value).__name__
    return type(value).__name__


def _build_dataclass_similarity_patterns(
    dataclass_nodes: list[DataclassNode],
) -> tuple[list[SimilarityEdge], list[PatternCluster]]:
    """构造相似 dataclass 模式。"""

    similarity_edges: list[SimilarityEdge] = []
    left: DataclassNode
    right: DataclassNode
    for left, right in combinations(dataclass_nodes, 2):
        score: float
        evidence: list[str]
        common_field_count: int
        score, evidence, common_field_count = _score_dataclass_similarity(left, right)
        field_name_score: float = _extract_named_score(evidence, "field_jaccard")
        if common_field_count < 3 and field_name_score < 0.75:
            # 目的：避免只有一两个同名字段的小 dataclass 被传递式误聚类。
            continue
        if score < DATACLASS_SIMILARITY_THRESHOLD:
            continue
        similarity_edges.append(
            SimilarityEdge(
                source=left.qualname,
                target=right.qualname,
                kind="similar_dataclass",
                score=round(score, 4),
                evidence=evidence,
            )
        )
    return similarity_edges, _build_similarity_clusters("similar_dataclass", similarity_edges)


def _score_dataclass_similarity(
    left: DataclassNode,
    right: DataclassNode,
) -> tuple[float, list[str], int]:
    """计算两个 dataclass 的相似度。"""

    left_field_names: list[str] = [field.name for field in left.fields]
    right_field_names: list[str] = [field.name for field in right.fields]
    left_field_set: set[str] = set(left_field_names)
    right_field_set: set[str] = set(right_field_names)
    field_name_score: float = _jaccard_score(left_field_set, right_field_set)
    field_order_score: float = SequenceMatcher(
        None,
        left_field_names,
        right_field_names,
    ).ratio()

    left_fields_by_name: dict[str, DataclassField] = {field.name: field for field in left.fields}
    right_fields_by_name: dict[str, DataclassField] = {field.name: field for field in right.fields}
    common_field_names: set[str] = left_field_set & right_field_set
    annotation_match_count: int = 0
    default_match_count: int = 0
    field_name: str
    for field_name in common_field_names:
        if (
            left_fields_by_name[field_name].annotation
            == right_fields_by_name[field_name].annotation
        ):
            annotation_match_count += 1
        if (
            left_fields_by_name[field_name].default_kind
            == right_fields_by_name[field_name].default_kind
        ):
            default_match_count += 1
    if common_field_names:
        annotation_score: float = annotation_match_count / len(common_field_names)
        default_score: float = default_match_count / len(common_field_names)
    else:
        annotation_score = 0.0
        default_score = 0.0

    field_count_ratio: float = min(left.field_count, right.field_count) / max(
        left.field_count,
        right.field_count,
    )
    score: float = (
        field_name_score * 0.45
        + field_order_score * 0.2
        + annotation_score * 0.15
        + default_score * 0.1
        + field_count_ratio * 0.1
    )
    evidence: list[str] = [
        f"field_jaccard={field_name_score:.2f}",
        f"field_order={field_order_score:.2f}",
        f"annotation_overlap={annotation_score:.2f}",
        f"default_overlap={default_score:.2f}",
        f"field_count_ratio={field_count_ratio:.2f}",
        f"common_fields={','.join(sorted(common_field_names))}",
    ]
    return score, evidence, len(common_field_names)


def _jaccard_score(left: set[str], right: set[str]) -> float:
    """计算集合 Jaccard 相似度。"""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _build_similarity_clusters(
    kind: SimilarityKind,
    similarity_edges: list[SimilarityEdge],
) -> list[PatternCluster]:
    """按相似性边聚合模式簇。"""

    neighbors: dict[str, set[str]] = defaultdict(set)
    edge_by_pair: dict[tuple[str, str], SimilarityEdge] = {}
    edge: SimilarityEdge
    for edge in similarity_edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
        edge_by_pair[_ordered_pair(edge.source, edge.target)] = edge

    clusters: list[PatternCluster] = []
    visited: set[str] = set()
    node_name: str
    for node_name in sorted(neighbors):
        if node_name in visited:
            continue
        members: list[str] = []
        pending: list[str] = [node_name]
        current: str
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            pending.extend(sorted(neighbors[current] - visited))
        if len(members) <= 1:
            continue
        member_pairs: list[tuple[str, str]] = list(combinations(sorted(members), 2))
        cluster_edges: list[SimilarityEdge] = [
            edge_by_pair[pair] for pair in member_pairs if pair in edge_by_pair
        ]
        average_score: float = (
            sum(edge.score for edge in cluster_edges) / len(cluster_edges) if cluster_edges else 1.0
        )
        evidence: list[str] = [
            f"member_count={len(members)}",
            f"average_score={average_score:.2f}",
        ]
        clusters.append(
            PatternCluster(
                kind=kind,
                score=round(average_score, 4),
                members=sorted(members),
                evidence=evidence,
            )
        )
    return clusters


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    """把任意无向边端点转成稳定顺序。"""

    if left <= right:
        return left, right
    return right, left


def _extract_named_score(evidence: list[str], prefix: str) -> float:
    """从证据串中提取数值型分数。"""

    item: str
    for item in evidence:
        if not item.startswith(f"{prefix}="):
            continue
        return float(item.split("=", maxsplit=1)[1])
    return 0.0

"""函数图分析共享数据模型。"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .ast_visitors import FunctionDefinitionCollector
    from .refactor_advice import RefactorSuggestion

type ScopeName = str
type NodeName = str
type SimilarityKind = Literal[
    "exact_duplicate_function",
    "near_duplicate_function",
    "similar_dataclass",
]


@dataclass(slots=True)
class FunctionNode:
    """函数节点信息。"""

    qualname: str
    simple_name: str
    owner: str | None
    lineno: int
    end_lineno: int
    loc: int
    complexity: int
    output_cost: int
    side_effect_score: int
    effect_read_score: int = 0
    effect_write_score: int = 0
    effect_mutation_score: int = 0
    fan_in: int = 0
    fan_out: int = 0


@dataclass(slots=True)
class FunctionEdge:
    """函数边信息。"""

    source: str
    target: str
    kind: str
    via: str | None = None


@dataclass(slots=True)
class FunctionGraphReport:
    """函数图统计摘要。"""

    node_count: int
    edge_count: int
    root_count: int
    merge_excess: int
    diamond_count: int
    tree_deviation: int
    roots: list[str]
    top_hubs: list[str]


@dataclass(slots=True)
class FunctionGraphArtifacts:
    """函数图完整分析结果。

    Attributes:
        source_path: 被分析源路径。
        nodes: 函数节点列表。
        edges: 函数边列表。
        report: 图统计摘要。
        suggestions: 基于当前函数图生成的重构建议。
        dataclass_nodes: 识别出的 dataclass 类型节点列表。
        similarity_edges: 重复函数与相似 dataclass 的结构边列表。
        pattern_clusters: 基于结构边聚合出的模式簇。
    """

    source_path: str
    nodes: list[FunctionNode]
    edges: list[FunctionEdge]
    report: FunctionGraphReport
    suggestions: list["RefactorSuggestion"] = field(default_factory=list)
    dataclass_nodes: list["DataclassNode"] = field(default_factory=list)
    similarity_edges: list["SimilarityEdge"] = field(default_factory=list)
    pattern_clusters: list["PatternCluster"] = field(default_factory=list)


@dataclass(slots=True)
class RawFunction:
    """AST 阶段采集到的函数定义。"""

    qualname: str
    simple_name: str
    owner: str | None
    lexical_parent: str | None
    class_owner: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(slots=True)
class RawClass:
    """AST 阶段采集到的类定义。"""

    qualname: str
    simple_name: str
    owner: str | None
    lexical_parent: str | None
    node: ast.ClassDef


@dataclass(slots=True)
class DataclassField:
    """单个 dataclass 字段摘要。"""

    name: str
    annotation: str | None
    has_default: bool
    default_kind: str | None


@dataclass(slots=True)
class DataclassNode:
    """dataclass 类型节点信息。"""

    qualname: str
    simple_name: str
    owner: str | None
    lineno: int
    end_lineno: int
    field_count: int
    fields: list[DataclassField] = field(default_factory=list)
    method_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SimilarityEdge:
    """结构相似性边。"""

    source: str
    target: str
    kind: SimilarityKind
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatternCluster:
    """结构模式聚类结果。"""

    kind: SimilarityKind
    score: float
    members: list[str]
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModuleAnalysisContext:
    """单个模块的分析上下文。"""

    source_path: Path
    module_name: str | None
    module: ast.Module
    collector: "FunctionDefinitionCollector"
    scope_parents: dict[ScopeName, ScopeName | None]
    imported_symbols: dict[str, NodeName] = field(default_factory=dict)

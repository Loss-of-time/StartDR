"""函数图分析共享数据模型。"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .refactor_advice import RefactorSuggestion

type ScopeName = str
type NodeName = str


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
    """

    source_path: str
    nodes: list[FunctionNode]
    edges: list[FunctionEdge]
    report: FunctionGraphReport
    suggestions: list["RefactorSuggestion"] = field(default_factory=list)


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
class ModuleAnalysisContext:
    """单个模块的分析上下文。"""

    source_path: Path
    module_name: str | None
    module: ast.Module
    collector: "FunctionDefinitionCollector"
    scope_parents: dict[ScopeName, ScopeName | None]
    imported_symbols: dict[str, NodeName] = field(default_factory=dict)

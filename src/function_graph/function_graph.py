"""函数级依赖图分析兼容入口。"""

from .analysis_cli import build_output_stem, main
from .analyzer import analyze_directory, analyze_source
from .models import (
    DataclassNode,
    FunctionEdge,
    FunctionGraphArtifacts,
    FunctionGraphReport,
    FunctionNode,
    PatternCluster,
    SimilarityEdge,
)
from .output import DEFAULT_OUTPUT_DIR, ensure_output_dir, render_dot_svg
from .render import build_dot, build_markdown_report

__all__ = [
    "FunctionNode",
    "FunctionEdge",
    "FunctionGraphReport",
    "FunctionGraphArtifacts",
    "DataclassNode",
    "SimilarityEdge",
    "PatternCluster",
    "DEFAULT_OUTPUT_DIR",
    "ensure_output_dir",
    "render_dot_svg",
    "build_output_stem",
    "build_dot",
    "build_markdown_report",
    "analyze_source",
    "analyze_directory",
    "main",
]

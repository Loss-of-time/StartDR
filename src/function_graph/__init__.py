"""函数图分析与重构建议独立项目。"""

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


def main() -> None:
    """延迟导入命令行入口，避免 `python -m` 触发 runpy 警告。"""

    from .analysis_cli import main as entrypoint

    # 目的：把入口函数改成延迟导入，避免包初始化阶段提前加载 CLI 模块。
    entrypoint()


__all__ = [
    "FunctionNode",
    "FunctionEdge",
    "FunctionGraphReport",
    "FunctionGraphArtifacts",
    "DataclassNode",
    "SimilarityEdge",
    "PatternCluster",
    "analyze_source",
    "analyze_directory",
    "main",
]

"""函数级依赖图分析命令行入口。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .analyzer import analyze_directory, analyze_source
from .models import FunctionEdge, FunctionGraphArtifacts, FunctionGraphReport, FunctionNode
from .output import DEFAULT_OUTPUT_DIR, ensure_output_dir, render_dot_svg
from .render import build_dot, build_markdown_report

__all__ = [
    "FunctionNode",
    "FunctionEdge",
    "FunctionGraphReport",
    "FunctionGraphArtifacts",
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


def build_output_stem(source_path: Path) -> str:
    """把源文件路径转换成稳定的输出文件名前缀。

    Args:
        source_path: 用户传入的源路径。

    Returns:
        稳定的输出文件名前缀。
    """

    # 目的：统一独立项目的默认产物命名，避免不同目录下同名文件冲突。
    try:
        normalized_path = source_path.resolve().relative_to(Path.cwd())
    except ValueError:
        normalized_path = source_path
    return str(normalized_path.with_suffix("")).replace("/", "__")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="输出单文件或目录整体的函数级依赖图分析结果。")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    output_stem = build_output_stem(args.source)
    dot_output = output_dir / f"{output_stem}.dot"
    svg_output = output_dir / f"{output_stem}.svg"
    json_output = output_dir / f"{output_stem}.json"
    report_output = output_dir / f"{output_stem}.md"

    artifacts = (
        analyze_directory(args.source) if args.source.is_dir() else analyze_source(args.source)
    )
    dot_output.write_text(build_dot(artifacts), encoding="utf-8")
    json_output.write_text(
        json.dumps(asdict(artifacts), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(build_markdown_report(artifacts), encoding="utf-8")
    render_dot_svg(dot_output, svg_output)


if __name__ == "__main__":
    main()

"""函数图树化建议命令行入口。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .output import DEFAULT_OUTPUT_DIR, ensure_output_dir, render_dot_svg
from .treeify_function_graph import (
    analyze_treeify,
    build_dot,
    build_markdown_report,
    load_graph_artifacts,
)


def main() -> None:
    """命令行入口。"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="基于函数图 JSON 输出树化建议。"
    )
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = ensure_output_dir(args.output_dir)
    output_stem: str = f"{args.graph_json.stem}_treeify"
    report_output: Path = output_dir / f"{output_stem}.md"
    dot_output: Path = output_dir / f"{output_stem}.dot"
    svg_output: Path = output_dir / f"{output_stem}.svg"
    json_output: Path = output_dir / f"{output_stem}.json"

    graph = load_graph_artifacts(args.graph_json)
    treeify_artifacts, backbone_edges = analyze_treeify(graph)
    report_output.write_text(
        build_markdown_report(graph, treeify_artifacts, backbone_edges),
        encoding="utf-8",
    )
    dot_output.write_text(
        build_dot(graph, treeify_artifacts, backbone_edges),
        encoding="utf-8",
    )
    json_output.write_text(
        json.dumps(asdict(treeify_artifacts), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 目的：树化视图和原始函数图保持同一套可视化产物格式。
    render_dot_svg(dot_output, svg_output)


if __name__ == "__main__":
    main()

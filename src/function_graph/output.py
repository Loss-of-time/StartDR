"""函数图产物输出工具。"""

import subprocess
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("output/function_graph")


def ensure_output_dir(output_dir: Path) -> Path:
    """创建当前运行对应的时间戳输出目录。

    Args:
        output_dir: 输出根目录。

    Returns:
        本次运行对应的时间戳子目录路径。
    """

    # 目的：为每次函数图分析生成独立目录，避免不同运行的产物堆在同一层。
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir: Path = output_dir / timestamp
    timestamped_output_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_output_dir


def render_dot_svg(dot_path: Path, svg_path: Path) -> None:
    """把 DOT 文件渲染成 SVG。

    Args:
        dot_path: 已写入磁盘的 DOT 文件路径。
        svg_path: 目标 SVG 文件路径。
    """

    # 目的：同步导出可直接预览的矢量图，避免每次手工执行 dot -Tsvg。
    subprocess.run(
        ["dot", "-Tsvg", str(dot_path), f"-o{svg_path}"],
        check=True,
    )

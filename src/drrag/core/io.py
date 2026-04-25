"""RAG 项目的基础读写工具。"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

type JsonObject = dict[str, object]


def load_jsonl[T](
    path: Path,
    parse_line: Callable[[JsonObject], T],
    limit: int | None = None,
) -> list[T]:
    """读取 `jsonl` 文件。

    Args:
        path: 输入文件路径。
        parse_line: 单行解析函数。
        limit: 最多读取的行数。

    Returns:
        解析后的对象列表。
    """

    with path.open(encoding="utf-8") as file:
        if limit is None:
            return [parse_line(cast(JsonObject, json.loads(line))) for line in file]
        rows: list[T] = []
        for index, line in enumerate(file):
            if index >= limit:
                break
            rows.append(parse_line(cast(JsonObject, json.loads(line))))
        return rows


def write_json(path: Path, row: object) -> None:
    """写出单个 `json` 对象。

    Args:
        path: 输出文件路径。
        row: 待写入对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(row, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl[T](
    path: Path,
    rows: list[T],
    serialize_row: Callable[[T], str],
) -> None:
    """写出 `jsonl` 文件。

    Args:
        path: 输出文件路径。
        rows: 待写入对象列表。
        serialize_row: 单行序列化函数。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(serialize_row(row))
            file.write("\n")

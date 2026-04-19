import json
from collections.abc import Callable, Iterable
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import cast

type JsonObject = dict[str, object]
PICKLE_ROW_STREAM_FORMAT = "pickle_row_stream_v1"


def load_jsonl[T](
    path: Path,
    parse_line: Callable[[JsonObject], T],
    limit: int | None = None,
) -> list[T]:
    # 修复 Pylance 将 json 行对象推断为 object 导致调用处无法按键索引的问题。
    with path.open(encoding="utf-8") as file:
        if limit is None:
            return [parse_line(cast(JsonObject, json.loads(line))) for line in file]
        rows: list[T] = []
        for index, line in enumerate(file):
            if index >= limit:
                break
            rows.append(parse_line(cast(JsonObject, json.loads(line))))
        return rows


def write_jsonl[T](
    path: Path,
    rows: list[T],
    serialize_row: Callable[[T], str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(serialize_row(row))
            file.write("\n")


def load_pickle(path: Path) -> object:
    with path.open("rb") as file:
        return load(file)


def load_pickle_rows[T](
    path: Path,
    limit: int | None = None,
) -> list[T]:
    """读取按行 pickle 的记录文件，兼容旧版整表 `list` 格式。

    Args:
        path: 待读取文件路径。
        limit: 最多读取的记录数。

    Returns:
        记录列表。
    """

    with path.open("rb") as file:
        first_object: object = load(file)
        if isinstance(first_object, dict) and first_object.get("format") == PICKLE_ROW_STREAM_FORMAT:
            rows: list[T] = []
            while limit is None or len(rows) < limit:
                try:
                    rows.append(cast(T, load(file)))
                except EOFError:
                    break
            return rows

    rows = cast(list[T], first_object)
    if limit is None:
        return rows
    return rows[:limit]


def write_pickle_row_stream[T](
    path: Path,
    rows: Iterable[T],
) -> int:
    """把记录流逐条写入 pickle 文件。

    Args:
        path: 输出文件路径。
        rows: 待写入记录迭代器。

    Returns:
        实际写入的记录数。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    row_count: int = 0
    with path.open("wb") as file:
        dump({"format": PICKLE_ROW_STREAM_FORMAT}, file, protocol=HIGHEST_PROTOCOL)
        row: T
        for row in rows:
            dump(row, file, protocol=HIGHEST_PROTOCOL)
            row_count += 1
    return row_count

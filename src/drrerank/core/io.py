import json
from collections.abc import Callable
from pathlib import Path
from pickle import load


def load_jsonl[T](
    path: Path,
    parse_line: Callable[[object], T],
    limit: int | None = None,
) -> list[T]:
    with path.open(encoding="utf-8") as file:
        if limit is None:
            return [parse_line(json.loads(line)) for line in file]
        rows: list[T] = []
        for index, line in enumerate(file):
            if index >= limit:
                break
            rows.append(parse_line(json.loads(line)))
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

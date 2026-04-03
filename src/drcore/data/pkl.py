from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load


def load_pickle(path: Path) -> object:
    """读取 pickle 文件。"""
    with path.open("rb") as file:
        return load(file)


def write_pickle(path: Path, data: object) -> None:
    """写入 pickle 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        dump(data, file, protocol=HIGHEST_PROTOCOL)

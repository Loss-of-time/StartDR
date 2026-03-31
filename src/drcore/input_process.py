import json
import logging
from pathlib import Path
from typing import cast

# 一定要加前面这个点 "."
from .paths import DATA_DIR
from .schema import DrugRecRecord
from .utils.log import setup_logging

# TODO 使用配置文件进行配置
FILE = DATA_DIR / "DrugRec.jsonl"
LOGGER = logging.getLogger(__name__)


def get_patients(number: int) -> list[DrugRecRecord]:
    with open(FILE, encoding="utf-8") as f:
        return [
            cast(DrugRecRecord, json.loads(f.readline())) for _ in range(number)
        ]


def get_all_patients() -> list[DrugRecRecord]:
    return load_jsonl(FILE)


def load_jsonl(path: Path) -> list[DrugRecRecord]:
    with path.open(encoding="utf-8") as f:
        return [cast(DrugRecRecord, json.loads(line)) for line in f]


def load_jsonl_limit(
    path: Path,
    limit: int | None = None,
) -> list[DrugRecRecord]:
    with path.open(encoding="utf-8") as f:
        if limit is None:
            return [cast(DrugRecRecord, json.loads(line)) for line in f]
        records: list[DrugRecRecord] = []
        for index, line in enumerate(f):
            if index >= limit:
                break
            records.append(cast(DrugRecRecord, json.loads(line)))
        return records


def main() -> None:
    log_path = setup_logging()
    patients = get_patients(10)
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("读取前 10 条患者样本成功，共 %s 条。", len(patients))
    LOGGER.info("样本 ID 预览: %s", [patient["id"] for patient in patients[:3]])


if __name__ == "__main__":
    main()

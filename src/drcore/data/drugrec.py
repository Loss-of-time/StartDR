import logging
from pathlib import Path
from typing import cast

from ..schema import DrugRecRecord
from ..utils.log import setup_logging
from ..utils.paths import RESOURCE_DIR
from .jsonl import load_jsonl

DEFAULT_DRUGREC_PATH = RESOURCE_DIR / "DrugRec.jsonl"
LOGGER = logging.getLogger(__name__)


def load_drugrec_records(
    path: Path,
    limit: int | None = None,
) -> list[DrugRecRecord]:
    return load_jsonl(
        path=path,
        parse_line=lambda row: cast(DrugRecRecord, row),
        limit=limit,
    )


def get_patients(number: int) -> list[DrugRecRecord]:
    return load_drugrec_records(DEFAULT_DRUGREC_PATH, limit=number)


def get_all_patients() -> list[DrugRecRecord]:
    return load_drugrec_records(DEFAULT_DRUGREC_PATH)


def main() -> None:
    log_path = setup_logging()
    patients = get_patients(10)
    LOGGER.info("日志文件: %s", log_path.resolve())
    LOGGER.info("读取前 10 条患者样本成功，共 %s 条。", len(patients))
    LOGGER.info("样本 ID 预览: %s", [patient["id"] for patient in patients[:3]])

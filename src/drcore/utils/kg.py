"""负责执行知识图谱查询，并把结果整理成 Python 结构。"""

import logging
from typing import Final, cast

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity

from ..constant import (
    LIST_DRUG_INDEX_QUERY,
    LIST_FULL_DRUG_DETAIL_QUERY,
)
from ..schema import DrugRecMedicine
from .kg_cache_decorator import kg_cache

DRIVER: Final[Driver] = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)

LOGGER = logging.getLogger(__name__)
_DRUG_DETAIL_MAP: dict[str, DrugRecMedicine] | None = None


@kg_cache
def list_drug_ids() -> list[int]:
    with DRIVER.session() as session:
        result = session.run(LIST_DRUG_INDEX_QUERY)
        return [record["drugid"] for record in result]


@kg_cache
def list_full_drug_details() -> list[DrugRecMedicine]:
    with DRIVER.session() as session:
        result = session.run(LIST_FULL_DRUG_DETAIL_QUERY)
        details: list[DrugRecMedicine] = []
        append = details.append
        for record in result:
            row = cast(dict[str, object], record.data())
            drugid = str(row["drugid"])
            append(
                cast(
                    DrugRecMedicine,
                    {
                        "drugid": drugid,
                        "name": row["name"],
                        "CMAN": row["CMAN"],
                        "caution": row["caution"],
                        "ingredients": row["ingredients"],
                        "interaction": row["interaction"],
                        "treat": row["treat"],
                    },
                ),
            )
        return details


def _build_drug_detail_map() -> dict[str, DrugRecMedicine]:
    return {
        detail["drugid"]: detail
        for detail in list_full_drug_details()
    }


def _get_drug_detail_map_singleton() -> dict[str, DrugRecMedicine]:
    global _DRUG_DETAIL_MAP

    if _DRUG_DETAIL_MAP is None:
        _DRUG_DETAIL_MAP = _build_drug_detail_map()

    return _DRUG_DETAIL_MAP


def get_drug_details_by_ids(drug_ids: list[str]) -> list[DrugRecMedicine]:
    drug_detail_map = _get_drug_detail_map_singleton()
    details: list[DrugRecMedicine] = []
    append = details.append

    for drugid in drug_ids:
        detail = drug_detail_map.get(drugid)
        if detail is not None:
            append(detail)

    return details

"""负责执行知识图谱查询，并把结果整理成 Python 结构。"""

import logging
from typing import Final, cast

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity

from .constant import (
    LIST_DRUG_INDEX_QUERY,
    LIST_FULL_DRUG_DETAIL_QUERY,
)
from .schema import DrugRecMedicine
from .utils.kg_cache_decorator import kg_cache
from .utils.log import setup_logging

DRIVER: Final[Driver] = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    # 关闭数据库通知，避免返回 WARNING / INFORMATION 级别提示。
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)

LOGGER = logging.getLogger(__name__)
_DRUG_DETAIL_MAP: dict[str, DrugRecMedicine] | None = None


# 被此装饰器装饰的函数会在首次运行后生成缓存。
# 若 Neo4j 数据或 DrugRec.jsonl 发生变化，需要删除 data\cache 中对应缓存后重新运行。
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


def main() -> None:
    # 由于文本量巨大，禁止使用 logger 打印函数输出
    log_path = setup_logging()
    print("开始获取：")
    drug_detail_map = _get_drug_detail_map_singleton()
    first_drugid = next(iter(drug_detail_map))
    print(first_drugid, drug_detail_map[first_drugid])


if __name__ == "__main__":
    main()

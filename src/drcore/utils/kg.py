"""负责执行知识图谱查询，并把结果整理成 Python 结构。"""

from typing import Final, cast

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity

from ..constant import LIST_FULL_DRUG_DETAIL_QUERY
from ..schema import DrugRecMedicine
from .kg_cache_decorator import kg_cache

DRIVER: Final[Driver] = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)

_DRUG_DETAIL_MAP: dict[str, DrugRecMedicine] | None = None


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
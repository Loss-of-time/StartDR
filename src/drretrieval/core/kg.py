from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import cast

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity

from .schema import DrugRecMedicine, structure, unstructure
from .setting import (
    KG_AUTH,
    KG_BOLT_URL,
    KG_CACHE_PATH,
    LIST_FULL_DRUG_DETAIL_QUERY,
)


def _load_cache(path: Path) -> list[DrugRecMedicine]:
    with path.open("rb") as file:
        return cast(list[DrugRecMedicine], structure(load(file), list[DrugRecMedicine]))


def _write_cache(
    path: Path,
    rows: list[DrugRecMedicine],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        dump(unstructure(rows), file, protocol=HIGHEST_PROTOCOL)


def get_driver() -> Driver:
    return GraphDatabase.driver(
        KG_BOLT_URL,
        auth=KG_AUTH,
        notifications_min_severity=NotificationMinimumSeverity.OFF,
    )


def list_full_drug_details() -> list[DrugRecMedicine]:
    if KG_CACHE_PATH.exists():
        return _load_cache(KG_CACHE_PATH)
    with get_driver().session() as session:
        result = session.run(LIST_FULL_DRUG_DETAIL_QUERY)
        details: list[DrugRecMedicine] = []
        append = details.append
        for record in result:
            row = cast(dict[str, object], record.data())
            row["drugid"] = str(row["drugid"])
            append(cast(DrugRecMedicine, structure(row, DrugRecMedicine)))
    _write_cache(KG_CACHE_PATH, details)
    return details

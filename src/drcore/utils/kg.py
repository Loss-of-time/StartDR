"""负责执行知识图谱查询，并把结果整理成 Python 结构。"""

from typing import Final, cast

from neo4j import Driver, GraphDatabase, NotificationMinimumSeverity

from ..schema import DrugRecMedicine
from .kg_cache_decorator import kg_cache

LIST_FULL_DRUG_DETAIL_QUERY = """
MATCH (drug:`药品`)
OPTIONAL MATCH (drug)-[:用药*0..2]->(fact:`知识组`)-[:用药]->(crowd:`人群`)
OPTIONAL MATCH (fact)-[:用药结果]->(lvl:`用药结果级别`)
WITH drug,
     collect(
         DISTINCT CASE
             WHEN crowd IS NULL OR crowd.name IS NULL THEN NULL
             ELSE {
                 crowd_id: toInteger(last(split(elementId(crowd), ":"))),
                 crowd: crowd.name,
                 caution_levelid: CASE
                     WHEN lvl IS NULL THEN NULL
                     ELSE toInteger(last(split(elementId(lvl), ":")))
                 END,
                 caution_level: lvl.name
             }
         END
     ) AS caution_rows
OPTIONAL MATCH (drug)-[:治疗*0..3]->(treatment:`病症`)
WITH drug, caution_rows,
     collect(
         DISTINCT CASE
             WHEN treatment IS NULL OR treatment.name IS NULL THEN NULL
             ELSE {
                 treat_id: toInteger(last(split(elementId(treatment), ":"))),
                 treat: treatment.name
             }
         END
     ) AS treat_rows
OPTIONAL MATCH (drug)-[:成分*0..3]->(ingredient:`药物`)
WITH drug, caution_rows, treat_rows,
     collect(
         DISTINCT CASE
                WHEN ingredient IS NULL OR ingredient.name IS NULL THEN NULL
                ELSE {
                    ingredient_id: toInteger(last(split(elementId(ingredient), ":"))),
                    ingredient: ingredient.name
                }
            END
     ) AS ingredient_rows
OPTIONAL MATCH (drug)-[:相互作用*0..3]->(interaction:`药物`)
RETURN
    toInteger(last(split(elementId(drug), ":"))) AS drugid,
    drug.name AS name,
    drug.number AS CMAN,
    [x IN caution_rows WHERE x IS NOT NULL] AS caution,
    [x IN collect(
            DISTINCT CASE
                WHEN interaction IS NULL OR interaction.name IS NULL THEN NULL
                ELSE {
                    interaction_id: toInteger(last(split(elementId(interaction), ":"))),
                    name: interaction.name
                }
            END
        ) WHERE x IS NOT NULL] AS interaction,
    [x IN ingredient_rows WHERE x IS NOT NULL] AS ingredients,
    [x IN treat_rows WHERE x IS NOT NULL] AS treat
"""

DRIVER: Final[Driver] = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    notifications_min_severity=NotificationMinimumSeverity.OFF,
)


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

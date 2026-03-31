"""仅定义原始 Cypher 查询语句，不负责连接、执行和结果转换。"""

# 全量导出药品表
LIST_DRUG_INDEX_QUERY = """
MATCH (d:`药品`)
RETURN toInteger(last(split(elementId(d), ":"))) AS drugid,
       d.name AS name,
       d.number AS CMAN
"""

# 全量导出药品详情，字段结构对齐 DrugRecMedicine，供候选补全与图精排使用
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

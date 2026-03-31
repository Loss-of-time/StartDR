"""仅定义原始 Cypher 查询语句，不负责连接、执行和结果转换。"""

# 全量导出药品表
LIST_DRUG_INDEX_QUERY = """
MATCH (d:`药品`)
RETURN toInteger(last(split(elementId(d), ":"))) AS drugid,
       d.name AS name,
       d.number AS CMAN
"""

# 导出药品基础详情，聚合治疗病症、慎用人群及结果、药物成分，供轻量详情展示使用
LIST_SIMPLE_DRUG_DETAIL_QUERY = """
                MATCH (drug:`药品`)
                CALL {
                    WITH drug
                    OPTIONAL MATCH (drug)-[:治疗*0..3]->(treatment:`病症`)
                    RETURN collect(DISTINCT treatment.name) AS treatments
                }
                CALL {
                    WITH drug
                    OPTIONAL MATCH (drug)-[:用药*0..2]->(fact:`知识组`)
                    OPTIONAL MATCH (fact)-[:用药]->(crowd:`人群`)
                    OPTIONAL MATCH (fact)-[:用药结果]->(useResult:`用药结果级别`)
                    RETURN collect(DISTINCT CASE
                        WHEN crowd.name IS NOT NULL AND useResult.name IS NOT NULL THEN crowd.name + useResult.name
                        WHEN crowd.name IS NOT NULL THEN crowd.name
                        ELSE NULL
                    END) AS cautions
                }
                CALL {
                    WITH drug
                    OPTIONAL MATCH (drug)-[:成分*0..3]->(ingre:`药物`)
                    RETURN collect(DISTINCT ingre.name) AS ingredients
                }
RETURN toInteger(last(split(elementId(drug), ":"))) AS drugid,
       drug.name AS name,
       treatments,
       cautions,
       ingredients
                """

# 给每个“药品”节点构建一份可用于文本检索和索引的汇总信息，把药品本身、治疗病症、慎用人群、成分都拉平到一条结果里
LIST_CANDIDATE_TEXT_INDEX_QUERY = """
MATCH (drug:`药品`)
OPTIONAL MATCH (drug)-[:治疗*1..3]->(treatment:`病症`)
WITH toInteger(last(split(elementId(drug), ":"))) AS drugid,
     drug,
     collect(
         DISTINCT CASE
             WHEN treatment IS NULL OR treatment.name IS NULL THEN NULL
             ELSE {
                 treat_id: toInteger(last(split(elementId(treatment), ":"))),
                 treat: treatment.name
             }
         END
     ) AS treat_rows
OPTIONAL MATCH (drug)-[:用药*0..2]->(fact:`知识组`)-[:用药]->(crowd:`人群`)
OPTIONAL MATCH (fact)-[:用药结果]->(lvl:`用药结果级别`)
WITH drugid, drug, treat_rows,
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
OPTIONAL MATCH (drug)-[:成分*0..3]->(ing:`药物`)
RETURN
    drugid,
    drug.name AS name,
    drug.number AS CMAN,
    [x IN treat_rows WHERE x IS NOT NULL] AS treat_rows,
    [x IN caution_rows WHERE x IS NOT NULL] AS caution_rows,
    [x IN collect(
            DISTINCT CASE
                WHEN ing IS NULL OR ing.name IS NULL THEN NULL
                ELSE {
                    ingredient_id: toInteger(last(split(elementId(ing), ":"))),
                    ingredient: ing.name
                }
            END
        ) WHERE x IS NOT NULL] AS ingredient_rows
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

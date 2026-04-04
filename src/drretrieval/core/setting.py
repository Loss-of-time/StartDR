from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resource"
OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_PATIENT_INPUT_DIR = RESOURCE_DIR / "DrugRec0328"
DEFAULT_PATIENT_CANDIDATE_OUTPUT_DIR = RESOURCE_DIR / "patient_candidate"
DEFAULT_RETRIEVER_EVAL_INPUT = RESOURCE_DIR / "DrugRec0328" / "test.jsonl"
DEFAULT_RETRIEVER_EVAL_OUTPUT_DIR = OUTPUT_DIR / "retriever"
DEFAULT_RETRIEVER_NAME = "pyserini_bm25"
DEFAULT_TOP_K = 50

DENSE_MODEL_ID = "DMetaSoul/sbert-chinese-general-v2"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
CACHE_DIR = RESOURCE_DIR / "cache"
PY_SERINI_INDEX_DIR = CACHE_DIR / "pyserini_bm25_zh"
KG_CACHE_PATH = CACHE_DIR / "drretrieval__core__kg__list_full_drug_details.pkl"
KG_BOLT_URL = "bolt://localhost:7687"
KG_AUTH = ("neo4j", "password")

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

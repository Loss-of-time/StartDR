
from typing import TypeAlias, TypedDict

"""
``MINE/kg.py`` 返回的 KG 查询结果行对应的 ``TypedDict`` 定义。

以下结构说明基于 2026-03-27 对本地 Neo4j 上
``LIST_SIMPLE_DRUG_DETAIL_QUERY`` 的一次全量扫描。

- 总行数：113,102
- 每行固定包含 ``drugid``、``name``、``treatments``、``cautions``、
  ``ingredients`` 五个键
- ``drugid`` 始终为 ``int``
- ``name`` 仅有 1 行为 ``None``，其余均为 ``str``
- ``treatments`` / ``cautions`` / ``ingredients`` 始终为 ``list[str]``
- 这三个列表字段允许为空列表，但已观测到的元素中不包含 ``None``
  或空字符串
"""

NullableString: TypeAlias = str | None
NullableInteger: TypeAlias = int | None
TokenizedCorpusWithDrugIds: TypeAlias = tuple[list[list[str]], list[int]]


class SimpleDrugDetailRecord(TypedDict):
    drugid: int
    name: NullableString
    treatments: list[str]
    cautions: list[str]
    ingredients: list[str]


class CandidateTextTreatRow(TypedDict):
    treat_id: int
    treat: str


class CandidateTextCautionRow(TypedDict):
    crowd_id: int
    crowd: str
    caution_levelid: NullableInteger
    caution_level: NullableString


class CandidateTextIngredientRow(TypedDict):
    ingredient_id: int
    ingredient: str


class CandidateTextIndexRecord(TypedDict):
    drugid: int
    name: NullableString
    CMAN: NullableString
    treat_rows: list[CandidateTextTreatRow]
    caution_rows: list[CandidateTextCautionRow]
    ingredient_rows: list[CandidateTextIngredientRow]


__all__ = [
    "CandidateTextCautionRow",
    "CandidateTextIndexRecord",
    "CandidateTextIngredientRow",
    "CandidateTextTreatRow",
    "NullableInteger",
    "NullableString",
    "SimpleDrugDetailRecord",
    "TokenizedCorpusWithDrugIds",
]

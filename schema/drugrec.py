from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

"""
``MINE/data/DrugRec.jsonl`` 规范化结构的 ``TypedDict`` 定义。

以下约定基于 2026-03-27 的一次全量流式扫描与规范化结果。

- 顶层记录总数：21,000
- ``conflict`` 和 ``medicine_num`` 在 2,049 条记录中缺失
- ``medicine_num`` 是源字段，不应直接视为 ``len(medicine)``
- ``ingredients`` 和 ``interaction`` 存在多种对象形态，因此建模为联合类型
- 所有 ``drugid`` 统一为字符串
- 所有 ``NaN`` / ``Infinity`` / ``-Infinity`` 统一为 ``null``
- 除以上两点外，其余字段尽量保持原始结构不变
"""

DatasetSplit: TypeAlias = Literal["train", "dev", "test"]
NullableString: TypeAlias = str | None
NullableInteger: TypeAlias = int | None
NullableCMAN: TypeAlias = str | None


class DrugCaution(TypedDict):
    caution_level: str
    caution_levelid: int
    crowd: str
    crowd_id: int


class DrugIngredientById(TypedDict):
    id: int
    ingredient: NullableString


class DrugIngredientByIngredientId(TypedDict):
    ingredient_id: NullableInteger
    ingredient: NullableString


DrugIngredient: TypeAlias = DrugIngredientById | DrugIngredientByIngredientId


class DrugInteractionById(TypedDict):
    id: int
    name: str


class DrugInteractionByInteractionIdAndName(TypedDict):
    interaction_id: int
    name: str


class DrugInteractionByInteractionIdAndText(TypedDict):
    interaction_id: int
    interaction: str


DrugInteraction: TypeAlias = (
    DrugInteractionById
    | DrugInteractionByInteractionIdAndName
    | DrugInteractionByInteractionIdAndText
)


class DrugTreat(TypedDict):
    treat: str
    treat_id: int


class NullableDrugTreat(TypedDict):
    treat: NullableString
    treat_id: NullableInteger


class DrugRecMedicine(TypedDict):
    CMAN: NullableCMAN
    caution: list[DrugCaution]
    drugid: str
    ingredients: list[DrugIngredient]
    interaction: list[DrugInteraction]
    name: str
    treat: list[DrugTreat]


class DrugRecConflictMedicine(TypedDict):
    CMAN: NullableCMAN
    caution: list[DrugCaution]
    drugid: str
    ingredients: list[DrugIngredient]
    interaction: list[DrugInteraction]
    name: str
    treat: list[DrugTreat]


class DrugRecOnMedicine(TypedDict):
    CMAN: NullableCMAN
    caution: list[DrugCaution]
    drugid: str
    ingredients: list[DrugIngredient]
    interaction: list[DrugInteraction]
    name: str
    treat: list[NullableDrugTreat]


class DrugRecRecord(TypedDict):
    age: int
    allergen: list[str]
    antecedents: list[str]
    diagnosis: list[str]
    gender: str
    group: list[str]
    id: str
    medicine: list[DrugRecMedicine]
    on_medicine: list[DrugRecOnMedicine]
    part: DatasetSplit
    symptom: list[str]
    conflict: NotRequired[list[DrugRecConflictMedicine]]
    medicine_num: NotRequired[int]


__all__ = [
    "DatasetSplit",
    "DrugCaution",
    "DrugIngredient",
    "DrugIngredientById",
    "DrugIngredientByIngredientId",
    "DrugInteraction",
    "DrugInteractionById",
    "DrugInteractionByInteractionIdAndName",
    "DrugInteractionByInteractionIdAndText",
    "DrugRecConflictMedicine",
    "DrugRecMedicine",
    "DrugRecOnMedicine",
    "DrugRecRecord",
    "DrugTreat",
    "NullableCMAN",
    "NullableDrugTreat",
    "NullableInteger",
    "NullableString",
]

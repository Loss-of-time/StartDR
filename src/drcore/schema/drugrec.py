from typing import Literal, NotRequired, TypedDict

"""
``resource/DrugRec.jsonl`` 规范化结构的 ``TypedDict`` 定义。

以下约定基于 2026-03-31 的一次全量流式扫描与字段统一结果。

- 顶层记录总数：21,000
- ``conflict`` 和 ``medicine_num`` 在 2,049 条记录中缺失
- ``medicine_num`` 是源字段，不应直接视为 ``len(medicine)``
- ``ingredients`` 固定使用 ``ingredient_id``
- ``interaction`` 固定使用 ``interaction_id`` 与 ``name``
- 所有 ``drugid`` 统一为字符串
- 所有 ``NaN`` / ``Infinity`` / ``-Infinity`` 统一为 ``null``
- ``medicine`` / ``on_medicine`` / ``conflict`` 统一使用同一套药品结构
- ``treat`` 使用可空字段覆盖非空字段，由所属字段名表达业务语义
- 除以上两点外，其余字段尽量保持原始结构不变
"""

type DatasetSplit = Literal["train", "dev", "test"]
type NullableString = str | None
type NullableInteger = int | None
type NullableCMAN = str | None


class DrugCaution(TypedDict):
    caution_level: NullableString
    caution_levelid: NullableInteger
    crowd: str
    crowd_id: int


class DrugIngredient(TypedDict):
    ingredient_id: NullableInteger
    ingredient: NullableString


class DrugInteraction(TypedDict):
    interaction_id: int
    name: str


class DrugTreat(TypedDict):
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


class DrugRecRecord(TypedDict):
    age: int
    allergen: list[str]
    antecedents: list[str]
    diagnosis: list[str]
    gender: str
    group: list[str]
    id: str
    medicine: list[DrugRecMedicine]
    on_medicine: list[DrugRecMedicine]
    part: DatasetSplit
    symptom: list[str]
    conflict: NotRequired[list[DrugRecMedicine]]
    medicine_num: NotRequired[int]


__all__ = [
    "DatasetSplit",
    "DrugCaution",
    "DrugIngredient",
    "DrugInteraction",
    "DrugRecMedicine",
    "DrugRecRecord",
    "DrugTreat",
    "NullableCMAN",
    "NullableInteger",
    "NullableString",
]

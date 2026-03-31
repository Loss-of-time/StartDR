from typing import TypeAlias

"""
检索阶段共享的非业务实体类型定义。

药品实体本身统一复用 ``MINE.schema.drugrec.DrugRecMedicine``，
不再在此处按检索阶段重复建模。
"""

TokenizedCorpusWithDrugIds: TypeAlias = tuple[list[list[str]], list[str]]

__all__ = ["TokenizedCorpusWithDrugIds"]

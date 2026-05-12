"""在线链路共享的 caution 过滤工具。"""

from collections.abc import Sequence
from typing import Protocol


class CautionLike(Protocol):
    """可参与在线 caution 展示的最小字段协议。"""

    caution_level: str | None
    crowd: str


def build_visible_caution_text(caution: CautionLike) -> str | None:
    """构造在线链路可见的 caution 文本。

    Args:
        caution: 任意包含 `crowd` 与 `caution_level` 的 caution 对象。

    Returns:
        可展示的 caution 文本；若当前项不应按禁用语义消费则返回 `None`。
    """

    if caution.caution_level is None:
        return None
    display_value: str = f"{caution.crowd}{caution.caution_level}".strip()
    if display_value == "":
        return None
    return display_value


def build_visible_caution_texts(cautions: Sequence[CautionLike]) -> list[str]:
    """批量构造在线链路可见的 caution 文本列表。

    Args:
        cautions: 原始 caution 列表。

    Returns:
        过滤后的展示文本列表。
    """

    visible_values: list[str] = []
    caution: CautionLike
    for caution in cautions:
        display_value: str | None = build_visible_caution_text(caution)
        if display_value is None:
            continue
        visible_values.append(display_value)
    return visible_values

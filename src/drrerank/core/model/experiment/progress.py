"""统一训练进度输出。"""

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, cast

from tqdm import tqdm


class ProgressHandle[ItemT](Protocol):
    """统一的进度显示协议。"""

    def __enter__(self) -> "ProgressHandle[ItemT]":
        """进入进度上下文并返回自身。"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """退出进度上下文。"""
        ...

    def __iter__(self) -> Iterator[ItemT]:
        """迭代底层数据并推动进度前进。"""
        ...

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """设置附加状态文本。"""
        ...

    def close(self) -> None:
        """关闭进度显示并输出最终状态。"""
        ...


@dataclass(slots=True)
class SilentProgress[ItemT]:
    """非 TTY 环境下的静默进度显示器。"""

    items: Iterable[ItemT]
    desc: str
    leave: bool = False
    total: int | None = None
    _closed: bool = False

    def __enter__(self) -> "SilentProgress[ItemT]":
        """进入上下文并返回自身。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """退出上下文并关闭进度显示器。"""

        self.close()
        return None

    def __iter__(self) -> Iterator[ItemT]:
        """迭代底层数据且不额外输出文本。"""

        yield from self.items
        self.close()

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """兼容 tqdm 接口但不输出附加状态。"""

    def close(self) -> None:
        """关闭进度显示器。"""

        self._closed = True


def build_progress[ItemT](
    items: Iterable[ItemT],
    *,
    desc: str,
    leave: bool = False,
    total: int | None = None,
) -> ProgressHandle[ItemT]:
    """按终端能力构造统一进度显示器。

    Args:
        items: 待迭代的数据。
        desc: 进度描述文本。
        leave: 是否保留最终进度条。
        total: 显式总步数，默认按可求长对象自动推断。

    Returns:
        统一进度显示对象。
    """

    if sys.stdout.isatty():
        return cast(
            ProgressHandle[ItemT],
            tqdm(
                items,
                desc=desc,
                leave=leave,
                total=total,
                file=sys.stdout,
                dynamic_ncols=True,
            ),
        )
    # 目的：在 uv、IDE 日志面板等非 TTY 环境下避免周期性刷屏，仅保留 epoch 摘要输出。
    return SilentProgress(
        items=items,
        desc=desc,
        leave=leave,
        total=total,
    )

"""统一训练进度输出。"""

import sys
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass, field
from time import monotonic
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
class PlainProgress[ItemT]:
    """非 TTY 环境下的文本进度显示器。"""

    items: Iterable[ItemT]
    desc: str
    leave: bool = False
    total: int | None = None
    min_interval_seconds: float = 1.0
    current: int = 0
    postfix: str = ""
    _started: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _last_render_at: float = field(default=0.0, init=False)
    _last_message: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """补全缺失的总步数。"""

        if self.total is not None:
            return
        if isinstance(self.items, Sized):
            self.total = len(self.items)

    def __enter__(self) -> "PlainProgress[ItemT]":
        """进入上下文并输出初始状态。"""

        self._ensure_started()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """退出上下文并关闭进度显示。"""

        self.close()
        return None

    def __iter__(self) -> Iterator[ItemT]:
        """迭代底层数据并按固定频率打印当前进度。"""

        self._ensure_started()
        item: ItemT
        for item in self.items:
            yield item
            self.current += 1
            self._maybe_render(force=self.total is not None and self.current >= self.total)
        self.close()

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        """设置附加状态文本。"""

        self.postfix = s
        if refresh:
            self._maybe_render(force=False)

    def close(self) -> None:
        """关闭进度显示并输出最终状态。"""

        if self._closed:
            return
        self._maybe_render(force=True)
        self._closed = True

    def _ensure_started(self) -> None:
        """确保至少输出一次初始进度。"""

        if self._started:
            return
        self._started = True
        self._maybe_render(force=True)

    def _maybe_render(self, force: bool) -> None:
        """按频率限制输出文本进度。"""

        if self._closed:
            return
        current_time: float = monotonic()
        if not force and current_time - self._last_render_at < self.min_interval_seconds:
            return
        message: str = self._build_message()
        if message == self._last_message:
            return
        print(message, file=sys.stdout, flush=True)
        self._last_render_at = current_time
        self._last_message = message

    def _build_message(self) -> str:
        """构造单行文本进度消息。"""

        progress_text: str
        if self.total is None:
            progress_text = f"{self.current}"
        else:
            progress_text = f"{self.current}/{self.total}"
        message: str = f"{self.desc}: {progress_text}"
        if self.postfix != "":
            message = f"{message} {self.postfix}"
        return message


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
    # 目的：在 uv、IDE 日志面板等非 TTY 环境下退化成稳定文本进度，避免训练过程完全无反馈。
    return PlainProgress(
        items=items,
        desc=desc,
        leave=leave,
        total=total,
    )

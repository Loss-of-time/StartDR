from collections.abc import Callable
from functools import wraps
from inspect import signature
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load

from ..constant import CACHE_DIR
from .paths import PROJECT_DIR


def _build_cache_prefix[R](func: Callable[[], R]) -> str:
    relative_file = Path(func.__code__.co_filename).resolve().relative_to(PROJECT_DIR)
    relative_stem = "__".join(relative_file.with_suffix("").parts)
    return f"{relative_stem}__{func.__name__}"


def kg_cache[R](func: Callable[[], R]) -> Callable[[], R]:
    if signature(func).parameters:
        raise ValueError("kg_cache 只允许装饰无参数函数")

    # 缓存名只绑定函数定义位置与函数名，不受运行入口影响。
    cache_prefix = _build_cache_prefix(func)
    cache_path = CACHE_DIR / f"{cache_prefix}.pkl"

    @wraps(func)  # 保留原函数元信息
    def wrapper() -> R:
        # 确保目录存在
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 若已存在对应缓存则组读
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                return load(cache_file)

        # 否则运行该函数
        result = func()
        with cache_path.open("wb") as cache_file:
            dump(result, cache_file, protocol=HIGHEST_PROTOCOL)
        return result

    return wrapper

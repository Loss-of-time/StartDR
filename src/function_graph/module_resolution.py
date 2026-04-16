"""目录级模块解析与上下文收集。"""

import ast
from pathlib import Path

from .ast_visitors import FunctionDefinitionCollector
from .models import ModuleAnalysisContext, NodeName, ScopeName


def build_module_name(source_root: Path, source_path: Path) -> str:
    """把目录内源文件路径转换成稳定模块名。

    Args:
        source_root: 目录整体分析的根目录。
        source_path: 单个 Python 源文件路径。

    Returns:
        以传入目录名为包前缀的模块名。
    """

    relative_path = source_path.relative_to(source_root)
    module_parts = [source_root.name]
    relative_parts = list(relative_path.parts)
    if source_path.name == "__init__.py":
        module_parts.extend(relative_parts[:-1])
    else:
        module_parts.extend(relative_parts[:-1])
        module_parts.append(source_path.stem)
    return ".".join(module_parts)


def collect_python_sources(source_path: Path) -> list[Path]:
    """收集待分析的 Python 源文件列表。

    Args:
        source_path: 用户传入的文件或目录路径。

    Returns:
        单文件列表或目录下全部 Python 文件列表。
    """

    if source_path.is_file():
        return [source_path]
    # 目的：目录整体分析时把所有 Python 模块并入同一张图，而不是拆成多个单文件结果。
    return sorted(path for path in source_path.rglob("*.py") if path.is_file())


def collect_module_context(
    source_path: Path,
    module_name: str | None,
) -> ModuleAnalysisContext:
    """为单个源文件构造模块分析上下文。

    Args:
        source_path: 单个 Python 源文件路径。
        module_name: 目录整体分析时的模块名；单文件分析时为 `None`。

    Returns:
        单个模块的分析上下文。
    """

    module = ast.parse(source_path.read_text(encoding="utf-8"))
    collector = FunctionDefinitionCollector(module_name=module_name)
    collector.visit(module)
    scope_parents = {function.qualname: function.lexical_parent for function in collector.functions}
    return ModuleAnalysisContext(
        source_path=source_path,
        module_name=module_name,
        module=module,
        collector=collector,
        scope_parents=scope_parents,
    )


def collect_module_symbols(
    contexts: list[ModuleAnalysisContext],
) -> dict[str, dict[str, NodeName]]:
    """收集目录内模块的顶层符号表。

    Args:
        contexts: 全部模块上下文。

    Returns:
        模块名到顶层可见符号的映射。
    """

    module_symbols: dict[str, dict[str, NodeName]] = {}
    for context in contexts:
        if context.module_name is None:
            continue
        module_symbols[context.module_name] = dict(context.collector.scope_symbols.get(None, {}))
    return module_symbols


def resolve_import_from_module(
    current_module_name: str,
    source_path: Path,
    imported_module: str | None,
    level: int,
) -> str | None:
    """解析 `from ... import ...` 的基准模块名。

    Args:
        current_module_name: 当前模块名。
        source_path: 当前模块文件路径。
        imported_module: `from` 后的模块文本。
        level: 相对导入层级。

    Returns:
        归一化后的基准模块名；无法解析时返回 `None`。
    """

    if level == 0:
        return imported_module
    current_package = current_module_name
    if source_path.name != "__init__.py":
        current_package = current_module_name.rsplit(".", maxsplit=1)[0]
    package_parts = current_package.split(".")
    base_length = len(package_parts) - (level - 1)
    if base_length <= 0:
        return None
    module_parts = package_parts[:base_length]
    if imported_module is not None:
        module_parts.extend(imported_module.split("."))
    return ".".join(module_parts)


def collect_imported_symbols(
    context: ModuleAnalysisContext,
    module_symbols: dict[str, dict[str, NodeName]],
) -> dict[str, NodeName]:
    """收集目录内模块的导入绑定。

    Args:
        context: 当前模块上下文。
        module_symbols: 全部内部模块的顶层符号表。

    Returns:
        当前模块中可解析到的导入名映射。
    """

    imported_symbols: dict[str, NodeName] = {}
    available_modules = set(module_symbols)
    if context.module_name is None:
        return imported_symbols
    for statement in context.module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname is not None:
                    if alias.name in available_modules:
                        imported_symbols[alias.asname] = alias.name
                    continue
                binding_name = alias.name.split(".", maxsplit=1)[0]
                if binding_name in available_modules:
                    imported_symbols[binding_name] = binding_name
            continue
        if not isinstance(statement, ast.ImportFrom):
            continue
        base_module = resolve_import_from_module(
            current_module_name=context.module_name,
            source_path=context.source_path,
            imported_module=statement.module,
            level=statement.level,
        )
        if base_module is None:
            continue
        for alias in statement.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if base_module in module_symbols and alias.name in module_symbols[base_module]:
                imported_symbols[local_name] = module_symbols[base_module][alias.name]
                continue
            nested_module_name = f"{base_module}.{alias.name}"
            if nested_module_name in available_modules:
                imported_symbols[local_name] = nested_module_name
    return imported_symbols


def merge_class_scope_symbols(
    contexts: list[ModuleAnalysisContext],
) -> dict[ScopeName, dict[str, NodeName]]:
    """合并全部模块的类成员符号表。

    Args:
        contexts: 全部模块上下文。

    Returns:
        全局类成员符号表。
    """

    class_scope_symbols: dict[ScopeName, dict[str, NodeName]] = {}
    for context in contexts:
        class_scope_symbols.update(context.collector.class_scope_symbols)
    return class_scope_symbols

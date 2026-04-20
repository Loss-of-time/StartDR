"""函数图分析使用的 AST 访问器。"""

import ast

from .models import FunctionEdge, NodeName, RawClass, RawFunction, ScopeName

HEAVY_RETURN_NAMES = {
    "csr_matrix",
    "Data",
    "tensor",
    "empty",
    "full",
    "arange",
    "zeros",
    "ones",
}
LISTLIKE_RETURN_NODES = (
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
)
READ_EFFECT_NAMES = {
    "load_jsonl",
    "load_pickle",
    "dill.load",
    "json.load",
    "pickle.load",
}
WRITE_EFFECT_NAMES = {
    "print",
    "mkdir",
    "pickle_dump",
    "dump",
    "json.dump",
    "pickle.dump",
    "session.run",
    "subprocess.run",
}
MUTATION_METHOD_NAMES = {
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
    "add",
}
POSITIONAL_CALLBACK_ARGUMENT_INDEXES: dict[str, tuple[int, ...]] = {
    "filter": (0,),
    "map": (0,),
    "reduce": (0,),
    # 目的：补齐项目内高阶 IO 工具的位置回调识别，避免 parse_line/serialize_row 形成假根节点。
    "load_jsonl": (1,),
    "write_jsonl": (2,),
}
CALLBACK_KEYWORD_NAMES = {
    "callback",
    "default_factory",
    "factory",
    "func",
    "handler",
    "hook",
    "key",
    "predicate",
    "visitor",
}


class FunctionDefinitionCollector(ast.NodeVisitor):
    """收集模块中的顶层函数与局部函数定义。"""

    def __init__(self, module_name: str | None = None) -> None:
        self.module_name = module_name
        self._scope_stack: list[str] = []
        self._function_stack: list[str] = []
        self._class_stack: list[str] = []
        self.functions: list[RawFunction] = []
        self.classes: list[RawClass] = []
        self.scope_symbols: dict[ScopeName | None, dict[str, NodeName]] = {None: {}}
        self.class_scope_symbols: dict[ScopeName, dict[str, NodeName]] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._collect_class(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._collect_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._collect_function(node)

    def _collect_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        owner = self._scope_stack[-1] if self._scope_stack else None
        lexical_parent = self._function_stack[-1] if self._function_stack else None
        class_owner = self._class_stack[-1] if self._class_stack else None
        # 目的：限定名只基于最近一层归属作用域扩展，避免重复拼接祖先前缀。
        qualname = node.name if owner is None else f"{owner}.{node.name}"
        if owner is None and self.module_name is not None:
            # 目的：目录整体分析时为顶层定义补模块前缀，避免跨文件同名节点冲突。
            qualname = f"{self.module_name}.{node.name}"
        raw_function = RawFunction(
            qualname=qualname,
            simple_name=node.name,
            owner=owner,
            lexical_parent=lexical_parent,
            class_owner=class_owner,
            node=node,
        )
        self.functions.append(raw_function)
        # 目的：只把裸名可见的符号记入词法作用域，避免把类方法误当成模块函数。
        if lexical_parent is not None:
            self.scope_symbols.setdefault(lexical_parent, {})[node.name] = qualname
        elif class_owner is None:
            self.scope_symbols.setdefault(None, {})[node.name] = qualname
        # 目的：单独记录类成员，供 self.xxx()/cls.xxx() 这类方法调用解析。
        if class_owner is not None and owner == class_owner:
            self.class_scope_symbols.setdefault(class_owner, {})[node.name] = qualname
        self.scope_symbols.setdefault(qualname, {})
        self._scope_stack.append(qualname)
        self._function_stack.append(qualname)
        self.generic_visit(node)
        self._function_stack.pop()
        self._scope_stack.pop()

    def _collect_class(self, node: ast.ClassDef) -> None:
        owner = self._scope_stack[-1] if self._scope_stack else None
        lexical_parent = self._function_stack[-1] if self._function_stack else None
        # 目的：类也参与限定名拼接，避免不同类中的同名方法互相覆盖。
        qualname = node.name if owner is None else f"{owner}.{node.name}"
        if owner is None and self.module_name is not None:
            # 目的：目录整体分析时为顶层类补模块前缀，让跨文件方法解析有稳定归属。
            qualname = f"{self.module_name}.{node.name}"
        self.classes.append(
            RawClass(
                qualname=qualname,
                simple_name=node.name,
                owner=owner,
                lexical_parent=lexical_parent,
                node=node,
            )
        )
        # 目的：记录类名本身，让 ClassName.method() 也能回溯到类成员表。
        if lexical_parent is not None:
            self.scope_symbols.setdefault(lexical_parent, {})[node.name] = qualname
        elif not self._class_stack:
            self.scope_symbols.setdefault(None, {})[node.name] = qualname
        self.class_scope_symbols.setdefault(qualname, {})
        self._scope_stack.append(qualname)
        self._class_stack.append(qualname)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()


class FunctionBodyVisitor(ast.NodeVisitor):
    """遍历函数体时跳过内部定义，避免把子函数内容错误算进外层函数。"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None


class ComplexityVisitor(FunctionBodyVisitor):
    """粗粒度圈复杂度估计。"""

    def __init__(self) -> None:
        self.value = 1

    def visit_If(self, node: ast.If) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.value += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.value += max(1, len(node.handlers))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.value += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.value += max(1, len(node.ifs))
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.value += max(1, len(node.cases))
        self.generic_visit(node)


class ReturnVisitor(FunctionBodyVisitor):
    """估计函数输出代价。"""

    def __init__(self, annotation_text_value: str | None) -> None:
        self.annotation_text = annotation_text_value
        self.value = self._annotation_cost(annotation_text_value)

    def visit_Return(self, node: ast.Return) -> None:
        self.value = max(self.value, self._expr_cost(node.value))
        self.generic_visit(node)

    def _annotation_cost(self, annotation_text_value: str | None) -> int:
        if annotation_text_value is None:
            return 1
        if (
            "csr_matrix" in annotation_text_value
            or "torch.Tensor" in annotation_text_value
            or "Data" in annotation_text_value
        ):
            return 5
        if (
            "list[" in annotation_text_value
            or "dict[" in annotation_text_value
            or "tuple[" in annotation_text_value
        ):
            return 3
        if "|" in annotation_text_value:
            return 2
        return 1

    def _expr_cost(self, expr: ast.expr | None) -> int:
        if expr is None:
            return 1
        if isinstance(expr, ast.Call):
            called_name = dotted_name(expr.func)
            if called_name in HEAVY_RETURN_NAMES:
                return 5
            if called_name.endswith("tensor") or called_name.endswith("csr_matrix"):
                return 5
            return 2
        if isinstance(expr, LISTLIKE_RETURN_NODES):
            return 3
        if isinstance(expr, ast.BinOp):
            return max(self._expr_cost(expr.left), self._expr_cost(expr.right))
        if isinstance(expr, ast.Name):
            return 2
        return 1


class SideEffectVisitor(FunctionBodyVisitor):
    """估计函数副作用强度。"""

    def __init__(self, parameter_names: set[str]) -> None:
        self.parameter_names = parameter_names
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.read_score = 0
        self.write_score = 0
        self.mutation_score = 0

    def visit_Call(self, node: ast.Call) -> None:
        called_name = dotted_name(node.func)
        # 目的：把 IO 规则判断拆细，降低单个访问器方法的分支密度。
        self._score_call_effect(called_name, node)
        if self._is_mutation_call(node):
            self.mutation_score += 2
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._score_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._score_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._score_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._score_targets(node.targets)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    @property
    def total_score(self) -> int:
        """返回聚合后的副作用分数。"""

        return self.read_score + self.write_score * 2 + self.mutation_score * 2

    def _score_call_effect(self, called_name: str, node: ast.Call) -> None:
        """为单个调用累计读写副作用分数。

        Args:
            called_name: 点分调用名。
            node: 调用表达式节点。
        """

        if self._is_open_call(called_name):
            self._score_open_call(node)
            return
        if self._is_known_read_call(called_name):
            self.read_score += 1
            return
        if self._is_known_write_call(called_name):
            self.write_score += 2
            return
        if self._is_suffix_write_call(called_name):
            self.write_score += 2
            return
        if self._is_suffix_read_call(called_name):
            self.read_score += 1

    def _score_open_call(self, node: ast.Call) -> None:
        """为 `open` 调用累计读写分数。

        Args:
            node: 调用表达式节点。
        """

        if self._is_write_mode_open(node):
            self.write_score += 2
            return
        self.read_score += 1

    def _is_known_read_call(self, called_name: str) -> bool:
        """判断调用名是否命中显式读边界规则。"""

        return called_name in READ_EFFECT_NAMES

    def _is_known_write_call(self, called_name: str) -> bool:
        """判断调用名是否命中显式写边界规则。"""

        return called_name in WRITE_EFFECT_NAMES

    def _is_suffix_write_call(self, called_name: str) -> bool:
        """判断调用名是否命中写后缀规则。"""

        return called_name.endswith(
            (".mkdir", ".write", ".write_text", ".write_bytes", ".dump", ".save", ".run")
        )

    def _is_suffix_read_call(self, called_name: str) -> bool:
        """判断调用名是否命中读后缀规则。"""

        return called_name.endswith((".load", ".read", ".read_text", ".read_bytes"))

    def _score_targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            self._score_target(target)

    def _score_target(self, target: ast.expr) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._score_target(element)
            return
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            # 目的：属性写入和容器位写入通常都会改动外部可观察状态。
            self.mutation_score += 2
            return
        if isinstance(target, ast.Name) and target.id in self.global_names | self.nonlocal_names:
            self.mutation_score += 2

    def _is_open_call(self, called_name: str) -> bool:
        return called_name == "open" or called_name.endswith(".open")

    def _is_write_mode_open(self, node: ast.Call) -> bool:
        mode_node: ast.expr | None = None
        if len(node.args) >= 2:
            mode_node = node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_node = keyword.value
                break
        if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
            return False
        return any(flag in mode_node.value for flag in {"w", "a", "x", "+"})

    def _is_mutation_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        attr_name = node.func.attr
        if attr_name in MUTATION_METHOD_NAMES:
            return self._is_shared_mutation_base(node.func.value)
        if attr_name.endswith("_") and not attr_name.startswith("__"):
            return self._is_shared_mutation_base(node.func.value)
        return False

    def _is_shared_mutation_base(self, base: ast.expr) -> bool:
        if isinstance(base, ast.Name):
            return (
                base.id in self.parameter_names
                or base.id in self.global_names
                or base.id in self.nonlocal_names
            )
        if isinstance(base, ast.Attribute):
            return isinstance(base.value, ast.Name) and base.value.id in {"self", "cls"}
        if isinstance(base, ast.Subscript):
            return True
        return False


class InstanceBindingVisitor(FunctionBodyVisitor):
    """收集函数体内的实例类型绑定。"""

    def __init__(
        self,
        current_scope: str,
        current_class: str | None,
        scope_symbols: dict[ScopeName | None, dict[str, NodeName]],
        scope_parents: dict[ScopeName, ScopeName | None],
        class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
        class_attribute_types: dict[ScopeName, dict[str, NodeName]],
        imported_symbols: dict[str, NodeName],
        module_symbols: dict[str, dict[str, NodeName]],
    ) -> None:
        self.current_scope = current_scope
        self.current_class = current_class
        self.scope_symbols = scope_symbols
        self.scope_parents = scope_parents
        self.class_scope_symbols = class_scope_symbols
        self.class_attribute_types = class_attribute_types
        self.imported_symbols = imported_symbols
        self.module_symbols = module_symbols
        self.local_instance_types: dict[str, NodeName] = {}
        if current_class is None:
            self.attribute_instance_types: dict[str, NodeName] = {}
        else:
            self.attribute_instance_types = dict(class_attribute_types.get(current_class, {}))

    def collect_parameter_types(self, arguments: ast.arguments) -> None:
        """收集带类注解的形参实例类型。"""

        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameter: ast.arg
        for parameter in parameters:
            instance_type = self._resolve_annotation_class(parameter.annotation)
            if instance_type is None:
                continue
            # 目的：把带类注解的形参提前记成实例，补齐 model.predict_logits 这类调用边。
            self.local_instance_types[parameter.arg] = instance_type
        if arguments.vararg is not None:
            instance_type = self._resolve_annotation_class(arguments.vararg.annotation)
            if instance_type is not None:
                self.local_instance_types[arguments.vararg.arg] = instance_type
        if arguments.kwarg is not None:
            instance_type = self._resolve_annotation_class(arguments.kwarg.annotation)
            if instance_type is not None:
                self.local_instance_types[arguments.kwarg.arg] = instance_type

    def visit_Assign(self, node: ast.Assign) -> None:
        instance_type = self._resolve_instance_type(node.value)
        if instance_type is not None:
            target: ast.expr
            for target in node.targets:
                self._bind_target(target, instance_type)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        instance_type = self._resolve_annotation_class(node.annotation)
        if instance_type is None and node.value is not None:
            instance_type = self._resolve_instance_type(node.value)
        if instance_type is not None:
            self._bind_target(node.target, instance_type)
        self.generic_visit(node)

    def _bind_target(self, target: ast.expr, instance_type: NodeName) -> None:
        """把解析出的实例类型绑定到赋值目标。"""

        if isinstance(target, ast.Name):
            self.local_instance_types[target.id] = instance_type
            return
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and self.current_class is not None
        ):
            # 目的：记录 self.xxx / cls.xxx 的实例类型，供其他方法内解析 self.xxx(...)。
            self.attribute_instance_types[target.attr] = instance_type

    def _resolve_annotation_class(self, annotation: ast.expr | None) -> NodeName | None:
        """解析类型注解中的类引用。"""

        if annotation is None:
            return None
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            left = self._resolve_annotation_class(annotation.left)
            if left is not None:
                return left
            return self._resolve_annotation_class(annotation.right)
        if isinstance(annotation, ast.Subscript):
            inner = self._resolve_annotation_class(annotation.slice)
            if inner is not None:
                return inner
            return self._resolve_annotation_class(annotation.value)
        return self._resolve_class_reference(annotation)

    def _resolve_instance_type(self, expr: ast.expr) -> NodeName | None:
        """解析表达式对应的实例类型。"""

        if isinstance(expr, ast.Name):
            return self.local_instance_types.get(expr.id)
        if isinstance(expr, ast.Attribute):
            target = self._resolve_value_owner(expr)
            if target in self.class_scope_symbols:
                return target
            return None
        if isinstance(expr, ast.Call):
            class_reference = self._resolve_class_reference(expr.func)
            if class_reference is not None:
                return class_reference
            if isinstance(expr.func, ast.Attribute):
                # 目的：保留 Model(...).to(device) 这类链式构造后的实例类型。
                return self._resolve_instance_type(expr.func.value)
        return None

    def _resolve_class_reference(self, expr: ast.expr) -> NodeName | None:
        """解析表达式是否直接引用了当前分析范围内的类。"""

        target = self._resolve_value_owner(expr)
        if target in self.class_scope_symbols:
            return target
        return None

    def _resolve_value_owner(self, expr: ast.expr) -> NodeName | None:
        """解析表达式指向的模块、类、方法或实例类型。"""

        if isinstance(expr, ast.Name):
            target = self._resolve_symbol_name(expr.id)
            if target is not None:
                return target
            return self.local_instance_types.get(expr.id)
        if not isinstance(expr, ast.Attribute):
            return None
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id in {"self", "cls"}
            and self.current_class is not None
        ):
            attribute_type = self.attribute_instance_types.get(expr.attr)
            if attribute_type is not None:
                return attribute_type
            return self.class_scope_symbols.get(self.current_class, {}).get(expr.attr)
        owner = self._resolve_value_owner(expr.value)
        if owner is None:
            return None
        if owner in self.module_symbols:
            module_symbol = self.module_symbols[owner].get(expr.attr)
            if module_symbol is not None:
                return module_symbol
            nested_module_name = f"{owner}.{expr.attr}"
            if nested_module_name in self.module_symbols:
                return nested_module_name
        class_member = self.class_scope_symbols.get(owner, {}).get(expr.attr)
        if class_member is not None:
            return class_member
        return self.class_attribute_types.get(owner, {}).get(expr.attr)

    def _resolve_symbol_name(self, name: str) -> NodeName | None:
        """解析当前作用域可见的裸名符号。"""

        target = resolve_scoped_name(
            current_scope=self.current_scope,
            scope_symbols=self.scope_symbols,
            scope_parents=self.scope_parents,
            name=name,
        )
        if target is not None:
            return target
        return self.imported_symbols.get(name)


class CallVisitor(FunctionBodyVisitor):
    """提取函数体中的直接调用与回调引用。"""

    def __init__(
        self,
        current_scope: str,
        current_class: str | None,
        scope_symbols: dict[ScopeName | None, dict[str, NodeName]],
        scope_parents: dict[ScopeName, ScopeName | None],
        class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
        class_attribute_types: dict[ScopeName, dict[str, NodeName]],
        imported_symbols: dict[str, NodeName],
        module_symbols: dict[str, dict[str, NodeName]],
        local_instance_types: dict[str, NodeName],
    ) -> None:
        self.current_scope = current_scope
        self.current_class = current_class
        self.scope_symbols = scope_symbols
        self.scope_parents = scope_parents
        self.class_scope_symbols = class_scope_symbols
        self.class_attribute_types = class_attribute_types
        self.imported_symbols = imported_symbols
        self.module_symbols = module_symbols
        self.local_instance_types = local_instance_types
        self.edges: list[FunctionEdge] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = self._resolve_direct_target(node.func)
        if target is not None:
            self.edges.append(
                FunctionEdge(
                    source=self.current_scope,
                    target=target,
                    kind="direct_call",
                )
            )
        else:
            callee_name = dotted_name(node.func)
            for argument in self._iter_callback_arguments(node):
                callback_target = self._resolve_callback(argument)
                if callback_target is None:
                    continue
                self.edges.append(
                    FunctionEdge(
                        source=self.current_scope,
                        target=callback_target,
                        kind="callback_call",
                        via=callee_name or None,
                    )
                )
        self.generic_visit(node)

    def _iter_callback_arguments(self, node: ast.Call) -> list[ast.expr]:
        callback_arguments: list[ast.expr] = []
        callee_leaf_name = dotted_name(node.func).rsplit(".", maxsplit=1)[-1]
        # 目的：只在明确声明的位置参数槽位上识别回调，避免把普通参数误判成函数边。
        for callback_index in POSITIONAL_CALLBACK_ARGUMENT_INDEXES.get(callee_leaf_name, ()):
            if callback_index < len(node.args):
                callback_arguments.append(node.args[callback_index])
        for keyword in node.keywords:
            if keyword.arg in CALLBACK_KEYWORD_NAMES:
                callback_arguments.append(keyword.value)
        return callback_arguments

    def _resolve_callback(self, expr: ast.expr) -> str | None:
        return self._resolve_direct_target(expr)

    def _resolve_direct_target(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            direct_target = self._resolve_name(expr.id)
            if direct_target is not None:
                return self._materialize_direct_target(direct_target)
            instance_type = self.local_instance_types.get(expr.id)
            if instance_type is not None:
                return self.class_scope_symbols.get(instance_type, {}).get("forward")
            return None
        if isinstance(expr, ast.Attribute):
            return self._resolve_attribute(expr)
        return None

    def _materialize_direct_target(self, target: str) -> str | None:
        """把类名或函数名转换成最终可落边目标。"""

        if target in self.class_scope_symbols:
            # 目的：把类构造调用落到 __init__，避免类名本身在成图阶段被过滤掉。
            return self.class_scope_symbols[target].get("__init__")
        return target

    def _resolve_name(self, name: str) -> str | None:
        target = resolve_scoped_name(
            current_scope=self.current_scope,
            scope_symbols=self.scope_symbols,
            scope_parents=self.scope_parents,
            name=name,
        )
        if target is not None:
            return target
        return self.imported_symbols.get(name)

    def _resolve_attribute(self, expr: ast.Attribute) -> str | None:
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id in {"self", "cls"}
            and self.current_class is not None
        ):
            class_method = self.class_scope_symbols.get(self.current_class, {}).get(expr.attr)
            if class_method is not None:
                return class_method
            attribute_type = self.class_attribute_types.get(self.current_class, {}).get(expr.attr)
            if attribute_type is not None:
                return self.class_scope_symbols.get(attribute_type, {}).get("forward")
            return None
        owner = self._resolve_attribute_owner(expr.value)
        if owner is None:
            return None
        if owner in self.module_symbols:
            module_symbol = self.module_symbols[owner].get(expr.attr)
            if module_symbol is not None:
                return self._materialize_direct_target(module_symbol)
            nested_module_name = f"{owner}.{expr.attr}"
            if nested_module_name in self.module_symbols:
                return nested_module_name
        class_method = self.class_scope_symbols.get(owner, {}).get(expr.attr)
        if class_method is not None:
            return class_method
        attribute_type = self.class_attribute_types.get(owner, {}).get(expr.attr)
        if attribute_type is not None:
            return self.class_scope_symbols.get(attribute_type, {}).get("forward")
        return None

    def _resolve_attribute_owner(self, expr: ast.expr) -> str | None:
        """解析属性基对象归属到的模块、类或实例类型。"""

        if isinstance(expr, ast.Name):
            if expr.id in {"self", "cls"} and self.current_class is not None:
                return self.current_class
            owner = self._resolve_name(expr.id)
            if owner is not None:
                return owner
            return self.local_instance_types.get(expr.id)
        if not isinstance(expr, ast.Attribute):
            return None
        if (
            isinstance(expr.value, ast.Name)
            and expr.value.id in {"self", "cls"}
            and self.current_class is not None
        ):
            attribute_type = self.class_attribute_types.get(self.current_class, {}).get(expr.attr)
            if attribute_type is not None:
                return attribute_type
            return self.class_scope_symbols.get(self.current_class, {}).get(expr.attr)
        owner = self._resolve_attribute_owner(expr.value)
        if owner is None:
            return None
        if owner in self.module_symbols:
            module_symbol = self.module_symbols[owner].get(expr.attr)
            if module_symbol is not None:
                return module_symbol
            nested_module_name = f"{owner}.{expr.attr}"
            if nested_module_name in self.module_symbols:
                return nested_module_name
        class_method = self.class_scope_symbols.get(owner, {}).get(expr.attr)
        if class_method is not None:
            return class_method
        return self.class_attribute_types.get(owner, {}).get(expr.attr)


def dotted_name(node: ast.AST) -> str:
    """把简单的 Name/Attribute 还原成点分名称。"""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    return ""


def annotation_text(annotation: ast.expr | None) -> str | None:
    """提取返回注解的源文本。"""

    if annotation is None:
        return None
    return ast.unparse(annotation)


def collect_parameter_names(arguments: ast.arguments) -> set[str]:
    """提取函数参数名集合。

    Args:
        arguments: 函数参数 AST 节点。

    Returns:
        当前函数签名中的全部参数名。
    """

    parameter_names = {arg.arg for arg in arguments.posonlyargs}
    parameter_names.update(arg.arg for arg in arguments.args)
    parameter_names.update(arg.arg for arg in arguments.kwonlyargs)
    if arguments.vararg is not None:
        parameter_names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        parameter_names.add(arguments.kwarg.arg)
    return parameter_names


def collect_function_instance_bindings(
    function: RawFunction,
    scope_symbols: dict[ScopeName | None, dict[str, NodeName]],
    scope_parents: dict[ScopeName, ScopeName | None],
    class_scope_symbols: dict[ScopeName, dict[str, NodeName]],
    class_attribute_types: dict[ScopeName, dict[str, NodeName]],
    imported_symbols: dict[str, NodeName],
    module_symbols: dict[str, dict[str, NodeName]],
) -> tuple[dict[str, NodeName], dict[str, NodeName]]:
    """收集单个函数中的局部实例绑定和类属性实例绑定。"""

    visitor = InstanceBindingVisitor(
        current_scope=function.qualname,
        current_class=function.class_owner,
        scope_symbols=scope_symbols,
        scope_parents=scope_parents,
        class_scope_symbols=class_scope_symbols,
        class_attribute_types=class_attribute_types,
        imported_symbols=imported_symbols,
        module_symbols=module_symbols,
    )
    visitor.collect_parameter_types(function.node.args)
    statement: ast.stmt
    for statement in function.node.body:
        visitor.visit(statement)
    return visitor.local_instance_types, visitor.attribute_instance_types


def resolve_scoped_name(
    current_scope: str,
    scope_symbols: dict[ScopeName | None, dict[str, NodeName]],
    scope_parents: dict[ScopeName, ScopeName | None],
    name: str,
) -> str | None:
    """按词法作用域解析裸名。

    Args:
        current_scope: 当前函数限定名。
        scope_symbols: 各作用域可见符号表。
        scope_parents: 各函数作用域的父作用域映射。
        name: 待解析符号名。

    Returns:
        解析出的限定名；若不存在则返回 `None`。
    """

    scope: str | None = current_scope
    while scope is not None:
        scoped_names = scope_symbols.get(scope, {})
        if name in scoped_names:
            return scoped_names[name]
        scope = scope_parents[scope]
    module_names = scope_symbols.get(None, {})
    return module_names.get(name)

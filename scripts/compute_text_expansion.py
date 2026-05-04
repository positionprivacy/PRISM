#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

from repo_config import PROJECT_ROOT, get_analysis_dir


BASE_DIR = PROJECT_ROOT
ANALYSIS_DIR = get_analysis_dir("analysis")

UNKNOWN = object()

TEXT_CONSTRUCTORS = {
    "Text": {"args": "first"},
    "MarkupText": {"args": "first"},
    "Title": {"args": "first"},
    "Paragraph": {"args": "all"},
    "BulletedList": {"args": "all"},
    "Tex": {"args": "all"},
    "MathTex": {"args": "all"},
    "TexText": {"args": "all"},
    "SingleStringMathTex": {"args": "all"},
    "Code": {"args": "first", "keywords": ("code",)},
    "TextMobject": {"args": "all"},
    "TexMobject": {"args": "all"},
    "BraceLabel": {"args": "index:1", "keywords": ("text", "label_constructor", "label")},
    "LabeledDot": {"args": "first", "keywords": ("label",)},
    "LabeledLine": {"args": "index:2", "keywords": ("label",)},
    "Table": {"args": "index:0", "keywords": ("row_labels", "col_labels", "top_left_entry")},
    "MathTable": {"args": "index:0", "keywords": ("row_labels", "col_labels", "top_left_entry")},
    "MobjectTable": {"args": "index:0", "keywords": ("row_labels", "col_labels", "top_left_entry")},
    "DecimalTable": {"args": "index:0", "keywords": ("row_labels", "col_labels", "top_left_entry")},
    "IntegerTable": {"args": "index:0", "keywords": ("row_labels", "col_labels", "top_left_entry")},
    "Matrix": {"args": "index:0"},
    "MobjectMatrix": {"args": "index:0"},
    "DecimalMatrix": {"args": "index:0"},
    "IntegerMatrix": {"args": "index:0"},
    "BarChart": {"keywords": ("bar_names", "y_axis_label", "x_axis_label")},
}

TEXT_METHODS = {
    "add_labels": {"args": "first"},
    "get_axis_labels": {"args": "all", "keywords": ("x_label", "y_label")},
    "get_x_axis_label": {"args": "all", "keywords": ("label",)},
    "get_y_axis_label": {"args": "all", "keywords": ("label",)},
    "get_graph_label": {"args": "all", "keywords": ("label",)},
    "add_coordinates": {"args": "first"},
    "set_title": {"args": "first"},
}

PROMPT_BODY_MARKERS = [
    "Markdown 讲义如下：",
    "Markdown 讲义如下:",
    "The Markdown lecture notes are as follows:",
    "The Markdown lecture notes are as follows：",
]

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
LATEX_TEXT_CMD_PATTERN = re.compile(
    r"\\(?:text|mathrm|operatorname|mathbf|mathit|textbf|textit|mathsf|mathtt)\{([^{}]*)\}"
)
CONTEXT_BLOCK_PATTERN = re.compile(
    r"<!--\s*CONTEXT:BEGIN\s*-->.*?<!--\s*CONTEXT:END\s*-->",
    flags=re.S,
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.S)
LATEX_COMMAND_PATTERN = re.compile(r"\\[A-Za-z]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
IDENT_ONLY_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
CALL_NAMES = sorted(set(TEXT_CONSTRUCTORS) | set(TEXT_METHODS), key=len, reverse=True)
CALL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?P<prefix>\.??)(?P<name>" + "|".join(map(re.escape, CALL_NAMES)) + r")\s*\(")
SIMPLE_ASSIGN_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*(?:#.*)?$")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pct(value):
    return f"{100.0 * float(value):.1f}%"


def num(value):
    return f"{float(value):.4f}"


def normalize_prompt_body(prompt_text: str) -> str:
    body = prompt_text
    for marker in PROMPT_BODY_MARKERS:
        if marker in body:
            body = body.split(marker, 1)[1]
            break
    body = CONTEXT_BLOCK_PATTERN.sub(" ", body)
    body = HTML_COMMENT_PATTERN.sub(" ", body)
    return body.strip()


def latex_to_plain(text: str) -> str:
    result = text
    previous = None
    while previous != result:
        previous = result
        result = LATEX_TEXT_CMD_PATTERN.sub(r" \1 ", result)
    result = result.replace("\\\\", " ")
    result = LATEX_COMMAND_PATTERN.sub(" ", result)
    result = result.replace("{", " ").replace("}", " ")
    result = result.replace("^", " ").replace("_", " ")
    result = result.replace("&", " ")
    result = result.replace("$", " ")
    return WHITESPACE_PATTERN.sub(" ", result).strip()


def normalize_segment(text: str) -> str:
    text = latex_to_plain(text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if not text:
        return ""
    lowered = []
    for char in text:
        if "A" <= char <= "Z":
            lowered.append(char.lower())
        else:
            lowered.append(char)
    return "".join(lowered)


def tokenize_text(text: str):
    cleaned = normalize_segment(text)
    return TOKEN_PATTERN.findall(cleaned)


def flatten_to_strings(value):
    if value is UNKNOWN or value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        output = []
        for item in value:
            output.extend(flatten_to_strings(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(flatten_to_strings(item))
        return output
    return []


def bind_target(target, value, env):
    if isinstance(target, ast.Name):
        env[target.id] = value
        return
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)):
        for child, child_value in zip(target.elts, value):
            bind_target(child, child_value, env)


class CodeTextExtractor:
    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.global_functions = {}
        self.class_methods = {}
        self.global_statements = []
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef):
                self.global_functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods[child.name] = child
                self.class_methods[node.name] = methods
            else:
                self.global_statements.append(node)
        self.global_env = {}

    def eval_literal(self, node, env):
        if node is None:
            return UNKNOWN
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, self.global_env.get(node.id, UNKNOWN))
        if isinstance(node, ast.List):
            values = []
            for item in node.elts:
                value = self.eval_literal(item, env)
                if value is UNKNOWN:
                    return UNKNOWN
                values.append(value)
            return values
        if isinstance(node, ast.Tuple):
            values = []
            for item in node.elts:
                value = self.eval_literal(item, env)
                if value is UNKNOWN:
                    return UNKNOWN
                values.append(value)
            return tuple(values)
        if isinstance(node, ast.Set):
            values = []
            for item in node.elts:
                value = self.eval_literal(item, env)
                if value is UNKNOWN:
                    return UNKNOWN
                values.append(value)
            return values
        if isinstance(node, ast.Dict):
            values = {}
            for key, value_node in zip(node.keys, node.values):
                key_value = self.eval_literal(key, env)
                value_value = self.eval_literal(value_node, env)
                if key_value is UNKNOWN or value_value is UNKNOWN:
                    return UNKNOWN
                values[key_value] = value_value
            return values
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    formatted = self.eval_literal(value.value, env)
                    if formatted is UNKNOWN:
                        return UNKNOWN
                    parts.extend(flatten_to_strings(formatted))
                else:
                    return UNKNOWN
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.eval_literal(node.left, env)
            right = self.eval_literal(node.right, env)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            if isinstance(left, tuple) and isinstance(right, tuple):
                return left + right
            return UNKNOWN
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self.eval_literal(node.operand, env)
            if isinstance(operand, (int, float)):
                return -operand
            return UNKNOWN
        if isinstance(node, ast.Subscript):
            value = self.eval_literal(node.value, env)
            index = self.eval_literal(node.slice, env)
            if value is UNKNOWN or index is UNKNOWN:
                return UNKNOWN
            try:
                return value[index]
            except Exception:
                return UNKNOWN
        if isinstance(node, ast.Call):
            callee_name = self.get_callee_name(node.func)
            if callee_name == "str" and node.args:
                value = self.eval_literal(node.args[0], env)
                if value is UNKNOWN:
                    return UNKNOWN
                if isinstance(value, (list, tuple, set, dict)):
                    return UNKNOWN
                return str(value)
            if callee_name == "range":
                values = [self.eval_literal(arg, env) for arg in node.args]
                if any(value is UNKNOWN for value in values):
                    return UNKNOWN
                try:
                    return list(range(*values))
                except Exception:
                    return UNKNOWN
            if callee_name == "enumerate" and node.args:
                iterable = self.eval_literal(node.args[0], env)
                if isinstance(iterable, (list, tuple)):
                    start = 0
                    if len(node.args) > 1:
                        start = self.eval_literal(node.args[1], env)
                        if start is UNKNOWN:
                            return UNKNOWN
                    return list(enumerate(iterable, start))
                return UNKNOWN
            if callee_name == "zip":
                iterables = [self.eval_literal(arg, env) for arg in node.args]
                if any(not isinstance(item, (list, tuple)) for item in iterables):
                    return UNKNOWN
                return list(zip(*iterables))
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                template = self.eval_literal(node.func.value, env)
                if not isinstance(template, str):
                    return UNKNOWN
                args = [self.eval_literal(arg, env) for arg in node.args]
                kwargs = {
                    keyword.arg: self.eval_literal(keyword.value, env)
                    for keyword in node.keywords
                    if keyword.arg
                }
                if any(value is UNKNOWN for value in args) or any(
                    value is UNKNOWN for value in kwargs.values()
                ):
                    return UNKNOWN
                try:
                    return template.format(*args, **kwargs)
                except Exception:
                    return UNKNOWN
            return UNKNOWN
        if isinstance(node, ast.ListComp):
            return self.eval_comprehension(node, env)
        if isinstance(node, ast.Tuple):
            return tuple(self.eval_literal(item, env) for item in node.elts)
        return UNKNOWN

    def eval_condition(self, node, env):
        value = self.eval_literal(node, env)
        if isinstance(value, bool):
            return value
        return UNKNOWN

    def eval_comprehension(self, node, env):
        if len(node.generators) != 1:
            return UNKNOWN
        generator = node.generators[0]
        iterable = self.eval_literal(generator.iter, env)
        if not isinstance(iterable, (list, tuple)):
            return UNKNOWN
        values = []
        for item in iterable:
            child_env = dict(env)
            bind_target(generator.target, item, child_env)
            include = True
            for condition in generator.ifs:
                result = self.eval_condition(condition, child_env)
                if result is UNKNOWN or not result:
                    include = False
                    break
            if not include:
                continue
            value = self.eval_literal(node.elt, child_env)
            if value is UNKNOWN:
                return UNKNOWN
            values.append(value)
        return values

    def get_callee_name(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None

    def collect_text_from_call(self, node, env):
        callee_name = self.get_callee_name(node.func)
        spec = TEXT_CONSTRUCTORS.get(callee_name)
        method_spec = None
        if spec is None and isinstance(node.func, ast.Attribute):
            method_spec = TEXT_METHODS.get(node.func.attr)
        config = spec or method_spec
        if config is None:
            return []
        collected = []
        arg_mode = config.get("args")
        if arg_mode == "first":
            if node.args:
                collected.extend(flatten_to_strings(self.eval_literal(node.args[0], env)))
        elif arg_mode == "all":
            for arg in node.args:
                collected.extend(flatten_to_strings(self.eval_literal(arg, env)))
        elif isinstance(arg_mode, str) and arg_mode.startswith("index:"):
            index = int(arg_mode.split(":", 1)[1])
            if len(node.args) > index:
                collected.extend(flatten_to_strings(self.eval_literal(node.args[index], env)))
        for keyword in node.keywords:
            if keyword.arg and keyword.arg in config.get("keywords", ()):
                collected.extend(flatten_to_strings(self.eval_literal(keyword.value, env)))
        return [item for item in collected if isinstance(item, str) and item.strip()]

    def simulate_function_call(self, func_def, arg_values, current_class_name, caller_env, call_stack):
        func_key = (current_class_name or "", func_def.name, len(call_stack))
        if func_key in call_stack:
            return []
        local_env = dict(self.global_env)
        local_env.update(caller_env)
        positional = list(getattr(func_def.args, "posonlyargs", [])) + list(func_def.args.args)
        defaults = list(func_def.args.defaults)
        default_offset = len(positional) - len(defaults)
        for index, param in enumerate(positional):
            if index < len(arg_values):
                local_env[param.arg] = arg_values[index]
            elif index >= default_offset:
                default_value = self.eval_literal(defaults[index - default_offset], local_env)
                if default_value is not UNKNOWN:
                    local_env[param.arg] = default_value
        child_stack = set(call_stack)
        child_stack.add(func_key)
        return self.process_statements(
            func_def.body,
            local_env,
            current_class_name=current_class_name,
            call_stack=child_stack,
        )

    def inspect_expr(self, node, env, current_class_name, call_stack):
        texts = []
        if node is None:
            return texts
        if isinstance(node, ast.Call):
            texts.extend(self.collect_text_from_call(node, env))
            callee_name = self.get_callee_name(node.func)
            arg_values = [self.eval_literal(arg, env) for arg in node.args]
            if isinstance(node.func, ast.Name) and callee_name in self.global_functions:
                texts.extend(
                    self.simulate_function_call(
                        self.global_functions[callee_name],
                        arg_values,
                        current_class_name=current_class_name,
                        caller_env=env,
                        call_stack=call_stack,
                    )
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and current_class_name in self.class_methods
                and callee_name in self.class_methods[current_class_name]
            ):
                texts.extend(
                    self.simulate_function_call(
                        self.class_methods[current_class_name][callee_name],
                        arg_values,
                        current_class_name=current_class_name,
                        caller_env=env,
                        call_stack=call_stack,
                    )
                )
        for child in ast.iter_child_nodes(node):
            texts.extend(self.inspect_expr(child, env, current_class_name, call_stack))
        return texts

    def process_statements(self, statements, env, current_class_name, call_stack):
        collected = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                collected.extend(self.inspect_expr(statement.value, env, current_class_name, call_stack))
                value = self.eval_literal(statement.value, env)
                if value is not UNKNOWN:
                    for target in statement.targets:
                        bind_target(target, value, env)
            elif isinstance(statement, ast.AnnAssign):
                if statement.value is not None:
                    collected.extend(self.inspect_expr(statement.value, env, current_class_name, call_stack))
                    value = self.eval_literal(statement.value, env)
                    if value is not UNKNOWN:
                        bind_target(statement.target, value, env)
            elif isinstance(statement, ast.Expr):
                collected.extend(self.inspect_expr(statement.value, env, current_class_name, call_stack))
            elif isinstance(statement, ast.For):
                collected.extend(self.inspect_expr(statement.iter, env, current_class_name, call_stack))
                iterable = self.eval_literal(statement.iter, env)
                if isinstance(iterable, (list, tuple)):
                    for item in iterable:
                        child_env = dict(env)
                        bind_target(statement.target, item, child_env)
                        collected.extend(
                            self.process_statements(
                                statement.body,
                                child_env,
                                current_class_name=current_class_name,
                                call_stack=call_stack,
                            )
                        )
                        if statement.orelse:
                            collected.extend(
                                self.process_statements(
                                    statement.orelse,
                                    child_env,
                                    current_class_name=current_class_name,
                                    call_stack=call_stack,
                                )
                            )
                else:
                    collected.extend(
                        self.process_statements(
                            statement.body,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
            elif isinstance(statement, ast.If):
                collected.extend(self.inspect_expr(statement.test, env, current_class_name, call_stack))
                verdict = self.eval_condition(statement.test, env)
                if verdict is True:
                    collected.extend(
                        self.process_statements(
                            statement.body,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
                elif verdict is False:
                    collected.extend(
                        self.process_statements(
                            statement.orelse,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
                else:
                    collected.extend(
                        self.process_statements(
                            statement.body,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
                    collected.extend(
                        self.process_statements(
                            statement.orelse,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
            elif isinstance(statement, ast.While):
                collected.extend(self.inspect_expr(statement.test, env, current_class_name, call_stack))
                collected.extend(
                    self.process_statements(
                        statement.body,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
                collected.extend(
                    self.process_statements(
                        statement.orelse,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
            elif isinstance(statement, ast.With):
                for item in statement.items:
                    collected.extend(self.inspect_expr(item.context_expr, env, current_class_name, call_stack))
                collected.extend(
                    self.process_statements(
                        statement.body,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
            elif isinstance(statement, ast.Try):
                collected.extend(
                    self.process_statements(
                        statement.body,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
                for handler in statement.handlers:
                    collected.extend(
                        self.process_statements(
                            handler.body,
                            dict(env),
                            current_class_name=current_class_name,
                            call_stack=call_stack,
                        )
                    )
                collected.extend(
                    self.process_statements(
                        statement.orelse,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
                collected.extend(
                    self.process_statements(
                        statement.finalbody,
                        dict(env),
                        current_class_name=current_class_name,
                        call_stack=call_stack,
                    )
                )
            elif isinstance(statement, ast.Return):
                if statement.value is not None:
                    collected.extend(self.inspect_expr(statement.value, env, current_class_name, call_stack))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        collected.extend(self.inspect_expr(child, env, current_class_name, call_stack))
        return collected

    def extract(self):
        self.global_env = {}
        self.process_statements(self.global_statements, self.global_env, current_class_name=None, call_stack=set())
        collected = []
        for class_name, methods in self.class_methods.items():
            if "construct" in methods:
                env = dict(self.global_env)
                collected.extend(
                    self.process_statements(
                        methods["construct"].body,
                        env,
                        current_class_name=class_name,
                        call_stack=set(),
                    )
                )
        return collected


def eval_expr_literal(expr_text: str, env=None):
    try:
        expr = ast.parse(expr_text, mode="eval")
    except Exception:
        return UNKNOWN
    evaluator = CodeTextExtractor(ast.parse(""))
    evaluator.global_env = dict(env or {})
    return evaluator.eval_literal(expr.body, dict(env or {}))


def extract_parenthesized_loose(text: str, open_paren_index: int):
    depth = 1
    cursor = open_paren_index + 1
    start = cursor
    quote = None
    triple = False
    escaped = False
    while cursor < len(text):
        chunk3 = text[cursor : cursor + 3]
        char = text[cursor]
        if quote is not None:
            if triple:
                if chunk3 == quote * 3:
                    quote = None
                    triple = False
                    cursor += 3
                    continue
                cursor += 1
                continue
            if escaped:
                escaped = False
                cursor += 1
                continue
            if char == "\\":
                escaped = True
                cursor += 1
                continue
            if char == quote:
                quote = None
                cursor += 1
                continue
            if char == "\n":
                quote = None
                cursor += 1
                continue
            cursor += 1
            continue
        if chunk3 in ('"""', "'''"):
            quote = char
            triple = True
            cursor += 3
            continue
        if char in ("'", '"'):
            quote = char
            triple = False
            cursor += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:cursor]
        cursor += 1
    return text[start:]


def split_top_level_args_loose(text: str):
    parts = []
    start = 0
    depth_paren = depth_brack = depth_brace = 0
    quote = None
    triple = False
    escaped = False
    cursor = 0
    while cursor < len(text):
        chunk3 = text[cursor : cursor + 3]
        char = text[cursor]
        if quote is not None:
            if triple:
                if chunk3 == quote * 3:
                    quote = None
                    triple = False
                    cursor += 3
                    continue
                cursor += 1
                continue
            if escaped:
                escaped = False
                cursor += 1
                continue
            if char == "\\":
                escaped = True
                cursor += 1
                continue
            if char == quote:
                quote = None
                cursor += 1
                continue
            if char == "\n":
                quote = None
                cursor += 1
                continue
            cursor += 1
            continue
        if chunk3 in ('"""', "'''"):
            quote = char
            triple = True
            cursor += 3
            continue
        if char in ("'", '"'):
            quote = char
            triple = False
            cursor += 1
            continue
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            parts.append(text[start:cursor].strip())
            start = cursor + 1
        cursor += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_keyword_arg_loose(arg_text: str):
    depth_paren = depth_brack = depth_brace = 0
    quote = None
    triple = False
    escaped = False
    for index, char in enumerate(arg_text):
        chunk3 = arg_text[index : index + 3]
        if quote is not None:
            if triple:
                if chunk3 == quote * 3:
                    quote = None
                    triple = False
                continue
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote or char == "\n":
                quote = None
                continue
            continue
        if chunk3 in ('"""', "'''"):
            quote = char
            triple = True
            continue
        if char in ("'", '"'):
            quote = char
            triple = False
            continue
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "=" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            left = arg_text[:index].strip()
            right = arg_text[index + 1 :].strip()
            if IDENT_ONLY_PATTERN.fullmatch(left):
                return left, right
    return None, arg_text.strip()


def heuristic_text_from_expr(expr_text: str):
    cleaned = expr_text
    cleaned = re.sub(r"(?<![A-Za-z0-9_])[rRuUbBfF]{1,3}(?=(\"\"\"|'''|\"|'))", "", cleaned)
    cleaned = cleaned.replace('"""', " ").replace("'''", " ")
    cleaned = cleaned.replace('"', " ").replace("'", " ")
    cleaned = re.sub(r"\b(True|False|None)\b", " ", cleaned)
    cleaned = re.sub(r"[+\-*/%=,:;\[\]{}()]", " ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned if tokenize_text(cleaned) else ""


def extract_fallback_from_expr(expr_text: str, env):
    expr_text = expr_text.strip()
    if not expr_text:
        return []
    if IDENT_ONLY_PATTERN.fullmatch(expr_text):
        value = env.get(expr_text, UNKNOWN)
        if value is not UNKNOWN:
            return [item for item in flatten_to_strings(value) if isinstance(item, str) and item.strip()]
        return []
    value = eval_expr_literal(expr_text, env)
    if value is not UNKNOWN:
        return [item for item in flatten_to_strings(value) if isinstance(item, str) and item.strip()]
    heuristic = heuristic_text_from_expr(expr_text)
    return [heuristic] if heuristic else []


def build_tolerant_env(code_text: str):
    env = {}
    for line in code_text.splitlines():
        match = SIMPLE_ASSIGN_PATTERN.match(line)
        if not match:
            continue
        name, expr_text = match.groups()
        value = eval_expr_literal(expr_text, env)
        if value is not UNKNOWN:
            env[name] = value
            continue
        heuristic = heuristic_text_from_expr(expr_text)
        if heuristic:
            env[name] = heuristic
    return env


def analyze_one_script_fallback(code_text: str):
    env = build_tolerant_env(code_text)
    collected = []
    for match in CALL_PATTERN.finditer(code_text):
        name = match.group("name")
        prefix = match.group("prefix")
        if prefix == "." and name not in TEXT_METHODS:
            continue
        config = TEXT_METHODS.get(name) if prefix == "." else TEXT_CONSTRUCTORS.get(name)
        if config is None:
            config = TEXT_METHODS.get(name) or TEXT_CONSTRUCTORS.get(name)
        if config is None:
            continue
        arg_blob = extract_parenthesized_loose(code_text, match.end() - 1)
        args = split_top_level_args_loose(arg_blob)
        selected_exprs = []
        arg_mode = config.get("args")
        positional = []
        keywords = {}
        for arg in args:
            key, value = split_keyword_arg_loose(arg)
            if key is None:
                positional.append(value)
            else:
                keywords[key] = value
        if arg_mode == "first":
            if positional:
                selected_exprs.append(positional[0])
        elif arg_mode == "all":
            selected_exprs.extend(positional)
        elif isinstance(arg_mode, str) and arg_mode.startswith("index:"):
            index = int(arg_mode.split(":", 1)[1])
            if len(positional) > index:
                selected_exprs.append(positional[index])
        for key in config.get("keywords", ()):
            if key in keywords:
                selected_exprs.append(keywords[key])
        for expr_text in selected_exprs:
            collected.extend(extract_fallback_from_expr(expr_text, env))
    return collected


def analyze_one_script(code_text: str):
    tree = ast.parse(code_text)
    extractor = CodeTextExtractor(tree)
    return extractor.extract()


def analyze_one_script_tolerant(code_text: str):
    try:
        return analyze_one_script(code_text), "ast"
    except SyntaxError:
        return analyze_one_script_fallback(code_text), "fallback_scan"


def compute_text_stats(prompt_text: str, extracted_segments):
    prompt_body = normalize_prompt_body(prompt_text)
    prompt_tokens = tokenize_text(prompt_body)
    prompt_token_count = len(prompt_tokens)
    prompt_counter = Counter(prompt_tokens)

    raw_segments = [segment for segment in extracted_segments if normalize_segment(segment)]
    unique_segments = []
    seen = set()
    for segment in raw_segments:
        normalized = normalize_segment(segment)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_segments.append(normalized)

    raw_token_total = sum(len(tokenize_text(segment)) for segment in raw_segments)
    unique_token_total = 0
    novel_token_total = 0
    overlap_token_total = 0
    for segment in unique_segments:
        tokens = tokenize_text(segment)
        if not tokens:
            continue
        unique_token_total += len(tokens)
        overlap = sum((Counter(tokens) & prompt_counter).values())
        overlap_token_total += overlap
        novel_token_total += max(len(tokens) - overlap, 0)

    denominator = max(prompt_token_count, 1)
    return {
        "prompt_body_text": prompt_body,
        "prompt_token_count": prompt_token_count,
        "display_segment_count_raw": len(raw_segments),
        "display_segment_count_unique": len(unique_segments),
        "display_token_count_raw": raw_token_total,
        "display_token_count_unique": unique_token_total,
        "display_overlap_token_count": overlap_token_total,
        "display_novel_token_count": novel_token_total,
        "text_expand_ratio_raw": raw_token_total / denominator,
        "text_expand_ratio_unique": unique_token_total / denominator,
        "info_expand_ratio": novel_token_total / denominator,
        "novelty_ratio": (novel_token_total / unique_token_total) if unique_token_total else 0.0,
        "display_text_unique_preview": unique_segments[:20],
    }


def load_manifest_difficulty_map(manifest_path: Path):
    mapping = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if sample_id:
                mapping[sample_id] = row.get("difficulty")
    return mapping


def aggregate_detail_rows(detail_rows):
    ok_rows = [row for row in detail_rows if row.get("status") == "ok"]
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [row for row in ok_rows if row.get("difficulty") == difficulty]
        by_difficulty[difficulty] = {
            "count": len(subset),
            "avg_text_expand_ratio_raw": mean([row.get("text_expand_ratio_raw", 0.0) for row in subset]),
            "avg_text_expand_ratio_unique": mean([row.get("text_expand_ratio_unique", 0.0) for row in subset]),
            "avg_info_expand_ratio": mean([row.get("info_expand_ratio", 0.0) for row in subset]),
            "avg_novelty_ratio": mean([row.get("novelty_ratio", 0.0) for row in subset]),
            "avg_prompt_token_count": mean([row.get("prompt_token_count", 0) for row in subset]),
            "avg_display_token_count_unique": mean(
                [row.get("display_token_count_unique", 0) for row in subset]
            ),
        }
    return {
        "formula": {
            "text_expand_ratio_unique": "unique_display_text_tokens / prompt_body_tokens",
            "info_expand_ratio": "novel_display_text_tokens / prompt_body_tokens",
            "novelty_ratio": "novel_display_text_tokens / unique_display_text_tokens",
        },
        "total_records": len(detail_rows),
        "scored_count": len(ok_rows),
        "avg_text_expand_ratio_raw": mean([row.get("text_expand_ratio_raw", 0.0) for row in ok_rows]),
        "avg_text_expand_ratio_unique": mean([row.get("text_expand_ratio_unique", 0.0) for row in ok_rows]),
        "avg_info_expand_ratio": mean([row.get("info_expand_ratio", 0.0) for row in ok_rows]),
        "avg_novelty_ratio": mean([row.get("novelty_ratio", 0.0) for row in ok_rows]),
        "avg_prompt_token_count": mean([row.get("prompt_token_count", 0) for row in ok_rows]),
        "avg_display_token_count_unique": mean(
            [row.get("display_token_count_unique", 0) for row in ok_rows]
        ),
        "fallback_count": sum(1 for row in ok_rows if row.get("extract_mode") == "fallback_scan"),
        "parse_fail_count": sum(1 for row in detail_rows if row.get("status") == "parse_fail"),
        "missing_input_count": sum(1 for row in detail_rows if row.get("status") == "missing_input"),
        "by_difficulty": by_difficulty,
    }


def score_one_generation_run(gen_dir: Path):
    results = load_json(gen_dir / "results.json")
    details = results.get("details", [])
    meta_dir = gen_dir / "meta"
    prompt_dir = gen_dir / "prompt_snapshots"
    code_dir = gen_dir / "cleaned_scripts"
    output_dir = gen_dir / "info_expand_final"
    manifest_path = None
    for row in details:
        path = row.get("manifest_path")
        if path:
            manifest_path = Path(path)
            break
    difficulty_map = load_manifest_difficulty_map(manifest_path) if manifest_path and manifest_path.exists() else {}

    scored_rows = []
    for row in details:
        sample_id = row["id"]
        meta_path = meta_dir / f"{sample_id}.json"
        prompt_path = prompt_dir / f"{sample_id}.txt"
        code_path = code_dir / f"{sample_id}.py"
        difficulty = difficulty_map.get(row.get("source_id")) or difficulty_map.get(sample_id)
        base = {
            "id": sample_id,
            "source_id": row.get("source_id"),
            "difficulty": difficulty,
            "prompt_path": str(prompt_path),
            "code_path": str(code_path),
        }
        if not prompt_path.exists() or not code_path.exists():
            base["status"] = "missing_input"
            scored_rows.append(base)
            continue
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")
        code_text = code_path.read_text(encoding="utf-8", errors="ignore")
        try:
            extracted_segments, extract_mode = analyze_one_script_tolerant(code_text)
        except Exception as exc:
            base["status"] = "parse_fail"
            base["parse_error"] = f"{type(exc).__name__}: {exc}"
            scored_rows.append(base)
            continue
        stats = compute_text_stats(prompt_text, extracted_segments)
        base.update(stats)
        base["status"] = "ok"
        base["extract_mode"] = extract_mode
        base["meta_path"] = str(meta_path) if meta_path.exists() else None
        scored_rows.append(base)

    summary = aggregate_detail_rows(scored_rows)
    summary["gen_dir"] = str(gen_dir)
    summary["manifest_path"] = str(manifest_path) if manifest_path else None
    write_jsonl(output_dir / "info_expand_scores.jsonl", scored_rows)
    write_json(output_dir / "info_expand_scores.json", scored_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def refresh_analysis_tables():
    model_summary_path = ANALYSIS_DIR / "model_summary.json"
    difficulty_summary_path = ANALYSIS_DIR / "difficulty_summary.json"
    model_rows = load_json(model_summary_path)
    difficulty_rows = load_json(difficulty_summary_path)

    for row in model_rows:
        gen_dir = BASE_DIR / row["gen_dir"]
        info_summary_path = gen_dir / "info_expand_final" / "summary.json"
        if not info_summary_path.exists():
            continue
        info_summary = load_json(info_summary_path)
        row["info_expand_source"] = str(Path(row["gen_dir"]) / "info_expand_final" / "info_expand_scores.jsonl")
        row["text_expand_ratio_raw_mean"] = info_summary["avg_text_expand_ratio_raw"]
        row["text_expand_ratio_unique_mean"] = info_summary["avg_text_expand_ratio_unique"]
        row["info_expand_ratio_mean"] = info_summary["avg_info_expand_ratio"]
        row["novelty_ratio_mean"] = info_summary["avg_novelty_ratio"]

    difficulty_index = {(row["model_key"], row["lang"], row["difficulty"]): row for row in difficulty_rows}
    for row in model_rows:
        gen_dir = BASE_DIR / row["gen_dir"]
        info_summary_path = gen_dir / "info_expand_final" / "summary.json"
        if not info_summary_path.exists():
            continue
        info_summary = load_json(info_summary_path)
        for difficulty, values in info_summary.get("by_difficulty", {}).items():
            key = (row["model_key"], row["lang"], difficulty)
            if key not in difficulty_index:
                continue
            target = difficulty_index[key]
            target["text_expand_ratio_raw_mean"] = values["avg_text_expand_ratio_raw"]
            target["text_expand_ratio_unique_mean"] = values["avg_text_expand_ratio_unique"]
            target["info_expand_ratio_mean"] = values["avg_info_expand_ratio"]
            target["novelty_ratio_mean"] = values["avg_novelty_ratio"]

    write_json(model_summary_path, model_rows)
    write_csv(ANALYSIS_DIR / "model_summary.csv", model_rows)
    ordered_difficulty_rows = [difficulty_index[(row["model_key"], row["lang"], row["difficulty"])] for row in difficulty_rows]
    write_json(difficulty_summary_path, ordered_difficulty_rows)
    write_csv(ANALYSIS_DIR / "difficulty_summary.csv", ordered_difficulty_rows)

    table_lines = [
        "# Model Results Table",
        "",
        "| Model | Lang | Gen | Audit | Visual | Render | PADVC_raw | PADVC_center | uPADVC | TD_raw | TD_center | uTD | InfoExpand | Novelty |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    pass_lines = [
        "# Model Results Table",
        "",
        "",
        "| Model | Lang | Pass@1 | Audit | Visual | PADVC_raw | PADVC_center | uPADVC | TD_raw | TD_center | uTD | InfoExpand | Novelty |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    quick_lines = [
        "# Model Result Analysis",
        "",
        "## Overall Summary",
        "",
        "| model | lang | gen | audit | visual | render | PADVC_raw | PADVC_center | uPADVC | TD_raw | TD_center | uTD | InfoExpand | Novelty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        info_expand = num(row.get("info_expand_ratio_mean", 0.0))
        novelty = num(row.get("novelty_ratio_mean", 0.0))
        table_lines.append(
            "| {model} | {lang} | {gen} | {audit} | {visual} | {render} | {padvc_raw} | {padvc_center} | {u_padvc} | {td_raw} | {td_center} | {u_td} | {info_expand} | {novelty} |".format(
                model=row["model"],
                lang=row["lang"],
                gen=pct(row["generation_success_rate"]),
                audit=pct(row["audit_success_rate"]),
                visual=pct(row["visual_pass_rate"]),
                render=pct(row["render_success_rate"]),
                padvc_raw=num(row["padvc_raw_mean_rendered"]),
                padvc_center=num(row["padvc_center_mean_rendered"]),
                u_padvc=num(row["u_padvc_mean_all"]),
                td_raw=num(row["td_raw_mean_rendered"]),
                td_center=num(row["td_center_mean_rendered"]),
                u_td=num(row["u_td_mean_all"]),
                info_expand=info_expand,
                novelty=novelty,
            )
        )
        pass_lines.append(
            "| {model} | {lang} | {pass1} | {audit} | {visual} | {padvc_raw} | {padvc_center} | {u_padvc} | {td_raw} | {td_center} | {u_td} | {info_expand} | {novelty} |".format(
                model=row["model"],
                lang=row["lang"],
                pass1=pct(row["render_success_rate"]),
                audit=pct(row["audit_success_rate"]),
                visual=pct(row["visual_pass_rate"]),
                padvc_raw=num(row["padvc_raw_mean_rendered"]),
                padvc_center=num(row["padvc_center_mean_rendered"]),
                u_padvc=num(row["u_padvc_mean_all"]),
                td_raw=num(row["td_raw_mean_rendered"]),
                td_center=num(row["td_center_mean_rendered"]),
                u_td=num(row["u_td_mean_all"]),
                info_expand=info_expand,
                novelty=novelty,
            )
        )
        quick_lines.append(
            "| {model} | {lang} | {gen} | {audit} | {visual} | {render} | {padvc_raw} | {padvc_center} | {u_padvc} | {td_raw} | {td_center} | {u_td} | {info_expand} | {novelty} |".format(
                model=row["model"],
                lang=row["lang"],
                gen=pct(row["generation_success_rate"]),
                audit=pct(row["audit_success_rate"]),
                visual=pct(row["visual_pass_rate"]),
                render=pct(row["render_success_rate"]),
                padvc_raw=num(row["padvc_raw_mean_rendered"]),
                padvc_center=num(row["padvc_center_mean_rendered"]),
                u_padvc=num(row["u_padvc_mean_all"]),
                td_raw=num(row["td_raw_mean_rendered"]),
                td_center=num(row["td_center_mean_rendered"]),
                u_td=num(row["u_td_mean_all"]),
                info_expand=info_expand,
                novelty=novelty,
            )
        )

    info_sorted = sorted(model_rows, key=lambda row: row.get("info_expand_ratio_mean", 0.0), reverse=True)
    quick_lines.extend(["", "## InfoExpand Ranking", ""])
    for index, row in enumerate(info_sorted, start=1):
        quick_lines.append(
            f"{index}. {row['model']} {row['lang']}: InfoExpand={num(row.get('info_expand_ratio_mean', 0.0))}, Novelty={num(row.get('novelty_ratio_mean', 0.0))}, Gen={pct(row['generation_success_rate'])}"
        )

    (ANALYSIS_DIR / "model_results_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    (ANALYSIS_DIR / "model_results_table_pass1.md").write_text(
        "\n".join(pass_lines) + "\n",
        encoding="utf-8",
    )
    (ANALYSIS_DIR / "quick_report.md").write_text("\n".join(quick_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-dir",
        nargs="+",
        help="One or more generation run directories containing results.json, prompt_snapshots/, and cleaned_scripts/.",
    )
    parser.add_argument(
        "--analysis-model-summary",
        default=str(ANALYSIS_DIR / "model_summary.json"),
        help="Model summary file used for aggregate local report refresh.",
    )
    parser.add_argument(
        "--skip-analysis-refresh",
        action="store_true",
        help="Only score --generation-dir runs; do not refresh aggregate analysis tables.",
    )
    args = parser.parse_args()

    if args.generation_dir:
        run_reports = []
        for raw_dir in args.generation_dir:
            gen_dir = Path(raw_dir)
            summary = score_one_generation_run(gen_dir)
            report = {
                "gen_dir": str(gen_dir),
                "summary_path": str(gen_dir / "info_expand_final" / "summary.json"),
                "scores_path": str(gen_dir / "info_expand_final" / "info_expand_scores.jsonl"),
                "avg_text_expand_ratio_unique": summary["avg_text_expand_ratio_unique"],
                "avg_info_expand_ratio": summary["avg_info_expand_ratio"],
                "avg_novelty_ratio": summary["avg_novelty_ratio"],
                "scored_count": summary["scored_count"],
            }
            run_reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        if args.skip_analysis_refresh:
            return

    model_rows = load_json(Path(args.analysis_model_summary))
    run_reports = []
    for row in model_rows:
        gen_dir = BASE_DIR / row["gen_dir"]
        summary = score_one_generation_run(gen_dir)
        run_reports.append(
            {
                "model": row["model"],
                "lang": row["lang"],
                "gen_dir": row["gen_dir"],
                "avg_text_expand_ratio_unique": summary["avg_text_expand_ratio_unique"],
                "avg_info_expand_ratio": summary["avg_info_expand_ratio"],
                "avg_novelty_ratio": summary["avg_novelty_ratio"],
                "scored_count": summary["scored_count"],
            }
        )
        print(
            json.dumps(
                {
                    "model": row["model"],
                    "lang": row["lang"],
                    "avg_info_expand_ratio": summary["avg_info_expand_ratio"],
                    "avg_novelty_ratio": summary["avg_novelty_ratio"],
                    "scored_count": summary["scored_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_json(ANALYSIS_DIR / "info_expand_run_reports.json", run_reports)
    refresh_analysis_tables()
    print(ANALYSIS_DIR / "info_expand_run_reports.json")
    print(ANALYSIS_DIR / "model_results_table.md")


if __name__ == "__main__":
    main()

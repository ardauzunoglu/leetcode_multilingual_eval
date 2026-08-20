"""Convert LeetCodeDataset records to a safe language-neutral representation."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .languages import get_adapter
from .schema import ArgumentSpec, FunctionSpec, SCHEMA_VERSION, TypeSpec, write_jsonl


class UnsupportedTask(ValueError):
    """A dataset feature cannot be translated faithfully by this evaluator."""


PRIMITIVES = {"int", "float", "str", "bool"}
LIST_NAMES = {"List", "list"}
OPTIONAL_NAMES = {"Optional"}


def _subscript_items(node: ast.Subscript) -> list[ast.AST]:
    value = node.slice
    return list(value.elts) if isinstance(value, ast.Tuple) else [value]


def parse_type(node: ast.AST | None) -> TypeSpec:
    if node is None:
        raise UnsupportedTask("missing type annotation")
    if isinstance(node, ast.Constant) and node.value is None:
        return TypeSpec("none")
    if isinstance(node, ast.Name):
        if node.id in PRIMITIVES:
            return TypeSpec(node.id)
        if node.id in {"None", "NoneType"}:
            return TypeSpec("none")
        raise UnsupportedTask(f"unsupported type {node.id}")
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        name = node.value.id
        items = _subscript_items(node)
        if name in LIST_NAMES and len(items) == 1:
            return TypeSpec("list", (parse_type(items[0]),))
        if name in OPTIONAL_NAMES and len(items) == 1:
            return TypeSpec("optional", (parse_type(items[0]),))
        if name in {"Tuple", "tuple"} and items:
            if len(items) == 2 and isinstance(items[1], ast.Constant) and items[1].value is Ellipsis:
                raise UnsupportedTask("variable-length tuples are unsupported")
            return TypeSpec("tuple", tuple(parse_type(item) for item in items))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = parse_type(node.left), parse_type(node.right)
        if left.kind == "none":
            return TypeSpec("optional", (right,))
        if right.kind == "none":
            return TypeSpec("optional", (left,))
    try:
        rendered = ast.unparse(node)
    except Exception:
        rendered = node.__class__.__name__
    raise UnsupportedTask(f"unsupported type annotation {rendered}")


def parse_function(starter_code: str) -> FunctionSpec:
    source = starter_code.rstrip()
    candidates = [source, source + "\n        pass", source + "\n    pass"]
    tree: ast.Module | None = None
    last_error: SyntaxError | None = None
    for candidate in candidates:
        try:
            tree = ast.parse(candidate + "\n")
            break
        except SyntaxError as exc:
            last_error = exc
    if tree is None:
        raise UnsupportedTask(f"cannot parse starter code: {last_error}")

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        raise UnsupportedTask("starter code contains no function")
    function = functions[0]
    if isinstance(function, ast.AsyncFunctionDef):
        raise UnsupportedTask("async functions are unsupported")

    arguments: list[ArgumentSpec] = []
    positional = list(function.args.posonlyargs) + list(function.args.args)
    for argument in positional:
        if argument.arg in {"self", "cls"}:
            continue
        arguments.append(ArgumentSpec(argument.arg, parse_type(argument.annotation)))
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise UnsupportedTask("variadic and keyword-only signatures are unsupported")
    return FunctionSpec(function.name, tuple(arguments), parse_type(function.returns))


def _literal(node: ast.AST, label: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise UnsupportedTask(f"{label} is not a literal") from exc


def parse_input(value: Any, function: FunctionSpec) -> list[Any]:
    if isinstance(value, Mapping):
        try:
            return [value[arg.name] for arg in function.arguments]
        except KeyError as exc:
            raise UnsupportedTask(f"input is missing argument {exc.args[0]}") from exc
    if not isinstance(value, str):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(function.arguments) == 1:
                return [value]
            return list(value)
        raise UnsupportedTask("input must be a string, object, or array")
    try:
        call = ast.parse(f"__candidate({value})", mode="eval").body
    except SyntaxError as exc:
        raise UnsupportedTask(f"cannot parse test input {value!r}") from exc
    if not isinstance(call, ast.Call) or any(keyword.arg is None for keyword in call.keywords):
        raise UnsupportedTask("test input must be positional or named literals")
    positional = [_literal(item, "test input") for item in call.args]
    named = {str(item.arg): _literal(item.value, "test input") for item in call.keywords}
    if positional and named:
        values = positional[:]
        for argument in function.arguments[len(positional):]:
            if argument.name not in named:
                raise UnsupportedTask(f"input is missing argument {argument.name}")
            values.append(named.pop(argument.name))
        if named:
            raise UnsupportedTask(f"input has unknown arguments: {', '.join(sorted(named))}")
        return values
    if named:
        unknown = set(named) - {arg.name for arg in function.arguments}
        if unknown:
            raise UnsupportedTask(f"input has unknown arguments: {', '.join(sorted(unknown))}")
        try:
            return [named[arg.name] for arg in function.arguments]
        except KeyError as exc:
            raise UnsupportedTask(f"input is missing argument {exc.args[0]}") from exc
    return positional


def parse_output(value: Any, return_type: TypeSpec | None = None) -> Any:
    if not isinstance(value, str):
        return value
    if return_type is not None and return_type.kind == "str":
        # LeetCodeDataset stores string outputs as bare text rather than as
        # Python string literals. Still accept quoted strings in custom files.
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        return parsed if isinstance(parsed, str) else value
    if return_type is not None and return_type.kind == "float":
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise UnsupportedTask(f"cannot parse expected output {value!r}") from exc


def value_matches_type(value: Any, type_spec: TypeSpec) -> bool:
    kind = type_spec.kind
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if kind == "str":
        return isinstance(value, str)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "none":
        return value is None
    if kind == "optional":
        return value is None or value_matches_type(value, type_spec.args[0])
    if kind == "list":
        return isinstance(value, list) and all(value_matches_type(item, type_spec.args[0]) for item in value)
    if kind == "tuple":
        return (
            isinstance(value, (list, tuple))
            and len(value) == len(type_spec.args)
            and all(value_matches_type(item, item_type) for item, item_type in zip(value, type_spec.args))
        )
    return False


def normalize_tests(row: Mapping[str, Any], function: FunctionSpec, max_tests: int | None) -> list[dict[str, Any]]:
    raw_tests = row.get("input_output")
    if not isinstance(raw_tests, list) or not raw_tests:
        raise UnsupportedTask("input_output is missing or empty")
    tests: list[dict[str, Any]] = []
    for index, raw_test in enumerate(raw_tests[:max_tests] if max_tests else raw_tests):
        if not isinstance(raw_test, Mapping) or "input" not in raw_test or "output" not in raw_test:
            raise UnsupportedTask(f"test {index} does not contain input and output")
        args = parse_input(raw_test["input"], function)
        expected = parse_output(raw_test["output"], function.returns)
        if len(args) != len(function.arguments):
            raise UnsupportedTask(
                f"test {index} has {len(args)} arguments; expected {len(function.arguments)}"
            )
        for argument, value in zip(function.arguments, args):
            if not value_matches_type(value, argument.type):
                raise UnsupportedTask(
                    f"test {index} value for {argument.name} does not match {argument.type.display()}"
                )
        if not value_matches_type(expected, function.returns):
            raise UnsupportedTask(
                f"test {index} output does not match {function.returns.display()}"
            )
        tests.append({"args": args, "expected": expected})
    return tests


def iter_source_rows(input_path: Path | None, dataset_name: str, split: str) -> Iterator[dict[str, Any]]:
    if input_path is not None:
        from .schema import read_jsonl

        yield from read_jsonl(input_path)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    yield from load_dataset(dataset_name, split=split)


def prepare_dataset(
    *,
    input_path: Path | None,
    dataset_name: str,
    split: str,
    languages: Sequence[str],
    output_path: Path,
    manifest_path: Path,
    max_tasks: int | None,
    max_tests: int | None,
    common_only: bool,
) -> dict[str, Any]:
    source_rows = []
    reasons: Counter[str] = Counter()
    for source_index, row in enumerate(iter_source_rows(input_path, dataset_name, split)):
        if max_tasks is not None and len(source_rows) >= max_tasks:
            break
        task_id = str(row.get("task_id") or "")
        try:
            if not task_id:
                raise UnsupportedTask("missing task_id")
            function = parse_function(str(row.get("starter_code") or ""))
            tests = normalize_tests(row, function, max_tests)
            source_rows.append((source_index, row, function, tests))
        except UnsupportedTask as exc:
            reasons[str(exc)] += 1

    prepared: list[dict[str, Any]] = []
    eligible_by_task: dict[str, set[str]] = {}
    candidates_by_task: dict[str, list[dict[str, Any]]] = {}
    language_reasons: Counter[str] = Counter()
    for source_index, row, function, tests in source_rows:
        task_id = str(row["task_id"])
        task_candidates = []
        eligible = set()
        for language in languages:
            adapter = get_adapter(language)
            reason = adapter.unsupported_reason(function)
            if reason:
                language_reasons[f"{language}: {reason}"] += 1
                continue
            prompt = adapter.make_prompt(str(row.get("problem_description") or ""), function)
            task_candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task_id,
                    "question_id": row.get("question_id"),
                    "difficulty": row.get("difficulty"),
                    "tags": row.get("tags") or [],
                    "source_index": source_index,
                    "language": language,
                    "function": function.to_json(),
                    "prompt": prompt,
                    "tests": tests,
                }
            )
            eligible.add(language)
        eligible_by_task[task_id] = eligible
        candidates_by_task[task_id] = task_candidates

    required = set(languages)
    for task_id, task_candidates in candidates_by_task.items():
        if common_only and eligible_by_task[task_id] != required:
            continue
        prepared.extend(task_candidates)
    prepared.sort(key=lambda row: (row["source_index"], languages.index(row["language"])))
    write_jsonl(output_path, prepared)

    counts = Counter(row["language"] for row in prepared)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(input_path) if input_path else dataset_name,
        "split": split,
        "languages": list(languages),
        "common_only": common_only,
        "max_tests": max_tests,
        "source_tasks_parsed": len(source_rows),
        "prepared_records": len(prepared),
        "prepared_tasks_by_language": dict(sorted(counts.items())),
        "dataset_rejections": dict(reasons.most_common()),
        "language_rejections": dict(language_reasons.most_common()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

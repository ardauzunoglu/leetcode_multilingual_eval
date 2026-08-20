"""Language prompt, harness, and compiler adapters.

The built-in normalized contract deliberately uses a free function in every
language. This removes LeetCode's Python ``Solution`` wrapper from the metric
and lets every language execute the same literal input/output cases.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schema import FunctionSpec, TypeSpec


PASS_PLACEHOLDER = "__LC_EVAL_PASS_TOKEN__"


LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "mojo🔥": "mojo",
}


def canonical_language(value: str) -> str:
    result = LANGUAGE_ALIASES.get(value.lower(), value.lower())
    if result not in ADAPTERS:
        raise ValueError(f"unsupported language {value!r}; choose from {', '.join(sorted(ADAPTERS))}")
    return result


def extract_code(text: str) -> str:
    """Extract the largest fenced code block, or return the plain response."""

    blocks = re.findall(r"```[^\n`]*\n(.*?)```", text, flags=re.DOTALL)
    code = max(blocks, key=len) if blocks else text
    return code.strip() + "\n"


def _walk_types(type_spec: TypeSpec) -> Iterable[TypeSpec]:
    for argument in type_spec.args:
        yield from _walk_types(argument)
    yield type_spec


def _all_types(function: FunctionSpec) -> set[TypeSpec]:
    result = set(_walk_types(function.returns))
    for argument in function.arguments:
        result.update(_walk_types(argument.type))
    return result


def _contains_kind(type_spec: TypeSpec, kinds: set[str]) -> bool:
    return type_spec.kind in kinds or any(_contains_kind(item, kinds) for item in type_spec.args)


def _common_unsupported(function: FunctionSpec, supported: set[str]) -> str | None:
    if function.returns.kind == "none":
        return "void/in-place outputs are not represented by input_output"
    for type_spec in _all_types(function):
        if type_spec.kind not in supported:
            return f"type {type_spec.display()} is unsupported"
    return None


def _problem_prompt(description: str, language: str, signature: str, notes: str = "") -> str:
    extra = f"\n{notes.strip()}\n" if notes.strip() else "\n"
    return (
        f"Solve the following programming problem in {language}.\n\n"
        f"Use only {language} and do not use syntax or imports from another language.\n\n"
        f"{description.strip()}\n\n"
        "Implement exactly this free-function interface:\n\n"
        f"```{language}\n{signature}\n```\n"
        f"{extra}"
        "Return source code only. Include any imports you need, but do not define "
        "a main function, tests, or the interface more than once."
    )


@dataclass(frozen=True)
class Program:
    filename: str
    source: str
    compile_command: tuple[str, ...]
    run_command: tuple[str, ...]


class LanguageAdapter(ABC):
    name: str

    @abstractmethod
    def unsupported_reason(self, function: FunctionSpec) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def make_prompt(self, description: str, function: FunctionSpec) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_program(
        self, function: FunctionSpec, tests: list[Mapping[str, Any]], candidate: str
    ) -> Program:
        raise NotImplementedError


class PythonAdapter(LanguageAdapter):
    name = "python"
    supported = {"int", "float", "str", "bool", "list", "tuple", "optional"}

    def unsupported_reason(self, function: FunctionSpec) -> str | None:
        return _common_unsupported(function, self.supported)

    def type_name(self, value: TypeSpec) -> str:
        if value.kind in {"int", "float", "str", "bool"}:
            return value.kind
        if value.kind == "list":
            return f"list[{self.type_name(value.args[0])}]"
        if value.kind == "tuple":
            return f"tuple[{', '.join(self.type_name(item) for item in value.args)}]"
        if value.kind == "optional":
            return f"{self.type_name(value.args[0])} | None"
        raise AssertionError(value)

    def signature(self, function: FunctionSpec) -> str:
        args = ", ".join(f"{arg.name}: {self.type_name(arg.type)}" for arg in function.arguments)
        return f"def {function.name}({args}) -> {self.type_name(function.returns)}:\n    ..."

    def make_prompt(self, description: str, function: FunctionSpec) -> str:
        return _problem_prompt(description, "python", self.signature(function))

    def make_program(
        self, function: FunctionSpec, tests: list[Mapping[str, Any]], candidate: str
    ) -> Program:
        checks = []
        for index, test in enumerate(tests):
            args = ", ".join(repr(value) for value in test["args"])
            checks.append(f"    _actual = {function.name}({args})")
            checks.append(f"    _expected = {test['expected']!r}")
            checks.append(
                f"    assert _lc_equal(_actual, _expected), "
                f"f'test {index} failed: {{_actual!r}} != {{_expected!r}}'"
            )
        prelude = """\
from __future__ import annotations
import bisect
import collections
import functools
import heapq
import itertools
import math
import operator
import random
from collections import *
from functools import *
from itertools import *
from typing import *

"""
        equality = """\

def _lc_equal(left, right):
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_lc_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_lc_equal(left[k], right[k]) for k in left)
    return left == right

def _lc_main():
"""
        checks.append(f"    print({PASS_PLACEHOLDER!r})")
        source = prelude + extract_code(candidate) + equality + "\n".join(checks) + "\n\n_lc_main()\n"
        return Program(
            filename="candidate.py",
            source=source,
            compile_command=("python", "-I", "-m", "py_compile", "candidate.py"),
            run_command=("python", "-I", "candidate.py"),
        )


class CppAdapter(LanguageAdapter):
    name = "cpp"
    supported = {"int", "float", "str", "bool", "list"}

    def unsupported_reason(self, function: FunctionSpec) -> str | None:
        return _common_unsupported(function, self.supported)

    def type_name(self, value: TypeSpec) -> str:
        names = {"int": "long long", "float": "double", "str": "std::string", "bool": "bool"}
        if value.kind in names:
            return names[value.kind]
        if value.kind == "list":
            return f"std::vector<{self.type_name(value.args[0])}>"
        raise AssertionError(value)

    def signature(self, function: FunctionSpec, declaration: bool = False) -> str:
        args = ", ".join(f"{self.type_name(arg.type)} {arg.name}" for arg in function.arguments)
        suffix = ";" if declaration else " { /* implementation */ }"
        return f"{self.type_name(function.returns)} {function.name}({args}){suffix}"

    def make_prompt(self, description: str, function: FunctionSpec) -> str:
        return _problem_prompt(
            description,
            "cpp",
            self.signature(function),
            "Use C++20. Arguments are passed by value, so they may be mutated locally.",
        )

    def literal(self, value: Any, type_spec: TypeSpec) -> str:
        if type_spec.kind == "int":
            return str(value)
        if type_spec.kind == "float":
            if not math.isfinite(float(value)):
                raise ValueError("non-finite test floats are unsupported")
            return repr(float(value))
        if type_spec.kind == "bool":
            return "true" if value else "false"
        if type_spec.kind == "str":
            return f"std::string({json.dumps(value, ensure_ascii=False)})"
        if type_spec.kind == "list":
            items = ", ".join(self.literal(item, type_spec.args[0]) for item in value)
            return f"{self.type_name(type_spec)}{{{items}}}"
        raise AssertionError(type_spec)

    def make_program(
        self, function: FunctionSpec, tests: list[Mapping[str, Any]], candidate: str
    ) -> Program:
        checks = []
        for index, test in enumerate(tests):
            args = ", ".join(
                self.literal(value, argument.type)
                for value, argument in zip(test["args"], function.arguments)
            )
            expected = self.literal(test["expected"], function.returns)
            checks.extend(
                [
                    f"    auto actual_{index} = {function.name}({args});",
                    f"    auto expected_{index} = {expected};",
                    f"    if (!lc_equal(actual_{index}, expected_{index})) {{",
                    f'        std::cerr << "test {index} failed\\n";',
                    "        return 1;",
                    "    }",
                ]
            )
        prelude = """\
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

"""
        equality = """\

template <typename T>
bool lc_equal(const T& left, const T& right) { return left == right; }

inline bool lc_equal(double left, double right) {
    return std::abs(left - right) <= 1e-9 * (1.0 + std::abs(right));
}

template <typename T>
bool lc_equal(const std::vector<T>& left, const std::vector<T>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!lc_equal(left[i], right[i])) return false;
    }
    return true;
}

int main() {
"""
        checks.append(f'    std::cout << "{PASS_PLACEHOLDER}\\n";')
        source = prelude + extract_code(candidate) + equality + "\n".join(checks) + "\n    return 0;\n}\n"
        return Program(
            filename="candidate.cpp",
            source=source,
            compile_command=(
                "g++", "-O2", "-std=c++20", "-pipe", "candidate.cpp", "-o", "candidate"
            ),
            run_command=("./candidate",),
        )


class CAdapter(LanguageAdapter):
    name = "c"
    supported = {"int", "float", "str", "bool", "list"}

    def unsupported_reason(self, function: FunctionSpec) -> str | None:
        return _common_unsupported(function, self.supported)

    def _type_token(self, value: TypeSpec) -> str:
        names = {"int": "Int", "float": "Float", "str": "String", "bool": "Bool"}
        if value.kind in names:
            return names[value.kind]
        if value.kind == "list":
            return "List" + self._type_token(value.args[0])
        raise AssertionError(value)

    def type_name(self, value: TypeSpec) -> str:
        names = {"int": "long long", "float": "double", "str": "const char *", "bool": "bool"}
        if value.kind in names:
            return names[value.kind]
        if value.kind == "list":
            return "LC" + self._type_token(value)
        raise AssertionError(value)

    def list_definitions(self, function: FunctionSpec) -> str:
        lists = sorted(
            (item for item in _all_types(function) if item.kind == "list"),
            key=lambda item: item.display().count("list"),
        )
        lines = []
        for item in lists:
            lines.append(
                f"typedef struct {{ {self.type_name(item.args[0])} *data; size_t len; }} "
                f"{self.type_name(item)};"
            )
        return "\n".join(lines)

    def signature(self, function: FunctionSpec, declaration: bool = False) -> str:
        args = ", ".join(f"{self.type_name(arg.type)} {arg.name}" for arg in function.arguments)
        if not args:
            args = "void"
        suffix = ";" if declaration else " { /* implementation */ }"
        return f"{self.type_name(function.returns)} {function.name}({args}){suffix}"

    def make_prompt(self, description: str, function: FunctionSpec) -> str:
        definitions = self.list_definitions(function)
        signature = definitions + ("\n\n" if definitions else "") + self.signature(function)
        notes = (
            "Use C17. LCList... values are {data, len} structs. Returned arrays may use heap or "
            "static storage; the harness does not free them. Do not repeat the typedefs in your answer."
        )
        return _problem_prompt(description, "c", signature, notes)

    def literal(self, value: Any, type_spec: TypeSpec) -> str:
        if type_spec.kind == "int":
            return str(value)
        if type_spec.kind == "float":
            if not math.isfinite(float(value)):
                raise ValueError("non-finite test floats are unsupported")
            return repr(float(value))
        if type_spec.kind == "bool":
            return "true" if value else "false"
        if type_spec.kind == "str":
            return json.dumps(value, ensure_ascii=False)
        if type_spec.kind == "list":
            if not value:
                return f"({self.type_name(type_spec)}){{.data = NULL, .len = 0}}"
            items = ", ".join(self.literal(item, type_spec.args[0]) for item in value)
            return (
                f"({self.type_name(type_spec)}){{.data = ({self.type_name(type_spec.args[0])}[])"
                f"{{{items}}}, .len = {len(value)}}}"
            )
        raise AssertionError(type_spec)

    def equality_helpers(self, function: FunctionSpec) -> str:
        lines = [
            "static bool lc_eq_Int(long long a, long long b) { return a == b; }",
            "static bool lc_eq_Float(double a, double b) { return fabs(a-b) <= 1e-9*(1.0+fabs(b)); }",
            "static bool lc_eq_String(const char *a, const char *b) { return a && b ? strcmp(a,b)==0 : a==b; }",
            "static bool lc_eq_Bool(bool a, bool b) { return a == b; }",
        ]
        lists = sorted(
            (item for item in _all_types(function) if item.kind == "list"),
            key=lambda item: item.display().count("list"),
        )
        for item in lists:
            token, child = self._type_token(item), self._type_token(item.args[0])
            ctype = self.type_name(item)
            lines.append(
                f"static bool lc_eq_{token}({ctype} a, {ctype} b) {{ "
                "if (a.len != b.len) return false; "
                f"for (size_t i=0; i<a.len; ++i) if (!lc_eq_{child}(a.data[i], b.data[i])) return false; "
                "return true; }"
            )
        return "\n".join(lines)

    def make_program(
        self, function: FunctionSpec, tests: list[Mapping[str, Any]], candidate: str
    ) -> Program:
        checks = []
        return_token = self._type_token(function.returns)
        return_type = self.type_name(function.returns)
        for index, test in enumerate(tests):
            args = ", ".join(
                self.literal(value, argument.type)
                for value, argument in zip(test["args"], function.arguments)
            )
            expected = self.literal(test["expected"], function.returns)
            checks.extend(
                [
                    f"    {return_type} actual_{index} = {function.name}({args});",
                    f"    {return_type} expected_{index} = {expected};",
                    f"    if (!lc_eq_{return_token}(actual_{index}, expected_{index})) {{",
                    f'        fprintf(stderr, "test {index} failed\\n");',
                    "        return 1;",
                    "    }",
                ]
            )
        prelude = """\
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

"""
        definitions = self.list_definitions(function)
        checks.append(f'    printf("{PASS_PLACEHOLDER}\\n");')
        source = (
            prelude
            + definitions
            + "\n\n"
            + extract_code(candidate)
            + "\n"
            + self.equality_helpers(function)
            + "\n\nint main(void) {\n"
            + "\n".join(checks)
            + "\n    return 0;\n}\n"
        )
        return Program(
            filename="candidate.c",
            source=source,
            compile_command=(
                "gcc", "-O2", "-std=c17", "-pipe", "candidate.c", "-lm", "-o", "candidate"
            ),
            run_command=("./candidate",),
        )


class MojoAdapter(LanguageAdapter):
    name = "mojo"
    supported = {"int", "float", "str", "bool", "list"}

    def unsupported_reason(self, function: FunctionSpec) -> str | None:
        return _common_unsupported(function, self.supported)

    def type_name(self, value: TypeSpec) -> str:
        names = {"int": "Int", "float": "Float64", "str": "String", "bool": "Bool"}
        if value.kind in names:
            return names[value.kind]
        if value.kind == "list":
            return f"List[{self.type_name(value.args[0])}]"
        raise AssertionError(value)

    def signature(self, function: FunctionSpec) -> str:
        rendered = []
        for argument in function.arguments:
            convention = "var " if argument.type.kind in {"list", "str"} else ""
            rendered.append(f"{convention}{argument.name}: {self.type_name(argument.type)}")
        return (
            f"def {function.name}({', '.join(rendered)}) -> {self.type_name(function.returns)}:\n"
            "    ..."
        )

    def make_prompt(self, description: str, function: FunctionSpec) -> str:
        return _problem_prompt(
            description,
            "mojo",
            self.signature(function),
            "Target Mojo 1.0. Collection arguments are owned (`var`) so they may be mutated.",
        )

    def literal(self, value: Any, type_spec: TypeSpec) -> str:
        if type_spec.kind == "int":
            return str(value)
        if type_spec.kind == "float":
            if not math.isfinite(float(value)):
                raise ValueError("non-finite test floats are unsupported")
            return repr(float(value))
        if type_spec.kind == "bool":
            return "True" if value else "False"
        if type_spec.kind == "str":
            return f"String({json.dumps(value, ensure_ascii=False)})"
        if type_spec.kind == "list":
            if not value:
                return f"{self.type_name(type_spec)}()"
            items = ", ".join(self.literal(item, type_spec.args[0]) for item in value)
            return f"[{items}]"
        raise AssertionError(type_spec)

    def make_program(
        self, function: FunctionSpec, tests: list[Mapping[str, Any]], candidate: str
    ) -> Program:
        checks = []
        for index, test in enumerate(tests):
            args = ", ".join(
                self.literal(value, argument.type)
                for value, argument in zip(test["args"], function.arguments)
            )
            expected = self.literal(test["expected"], function.returns)
            checks.extend(
                [
                    f"    var actual_{index} = {function.name}({args})",
                    f"    var expected_{index}: {self.type_name(function.returns)} = {expected}",
                    f"    if actual_{index} != expected_{index}:",
                    f'        raise Error("test {index} failed")',
                ]
            )
        checks.append(f'    print("{PASS_PLACEHOLDER}")')
        source = extract_code(candidate) + "\ndef main() raises:\n" + "\n".join(checks) + "\n"
        return Program(
            filename="candidate.mojo",
            source=source,
            compile_command=(
                "mojo", "build", "--num-threads", "1", "-o", "candidate", "candidate.mojo"
            ),
            run_command=("./candidate",),
        )


ADAPTERS: dict[str, LanguageAdapter] = {
    adapter.name: adapter
    for adapter in (PythonAdapter(), CppAdapter(), CAdapter(), MojoAdapter())
}


def get_adapter(language: str) -> LanguageAdapter:
    return ADAPTERS[canonical_language(language)]

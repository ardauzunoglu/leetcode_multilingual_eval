"""Internal schema and safe JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TypeSpec:
    """Small language-neutral subset of Python type annotations."""

    kind: str
    args: tuple["TypeSpec", ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "args": [arg.to_json() for arg in self.args]}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TypeSpec":
        return cls(
            kind=str(value["kind"]),
            args=tuple(cls.from_json(arg) for arg in value.get("args", [])),
        )

    def display(self) -> str:
        if not self.args:
            return self.kind
        return f"{self.kind}[{', '.join(arg.display() for arg in self.args)}]"


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    type: TypeSpec

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type.to_json()}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ArgumentSpec":
        return cls(str(value["name"]), TypeSpec.from_json(value["type"]))


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    arguments: tuple[ArgumentSpec, ...]
    returns: TypeSpec

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": [arg.to_json() for arg in self.arguments],
            "returns": self.returns.to_json(),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "FunctionSpec":
        return cls(
            name=str(value["name"]),
            arguments=tuple(ArgumentSpec.from_json(arg) for arg in value["arguments"]),
            returns=TypeSpec.from_json(value["returns"]),
        )


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


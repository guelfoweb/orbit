from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .schema import ParityMode

SCHEMA_VERSION = 1
CAPABILITY_REQUIREMENTS = {
    "chat": ("chat",),
    "tools": ("tools",),
    "artifacts": ("write_artifact", "verify_artifact"),
}


class FixtureError(ValueError):
    pass


@dataclass(frozen=True)
class RequestSpec:
    prompt: str
    tools: bool


@dataclass(frozen=True)
class ToolExpectation:
    name: str
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ArtifactExpectation:
    path: str
    json_equals: Any
    publication: bool
    verification: bool

    @property
    def canonical_json_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.json_equals)).hexdigest()


@dataclass(frozen=True)
class ExpectSpec:
    finish_reason: str
    max_model_calls: int
    route: str | None = None
    tool_calls: tuple[ToolExpectation, ...] = ()
    exact_output: str | None = None
    final_reports_workdir: bool = False
    artifact: ArtifactExpectation | None = None


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    capability: str
    profiles: tuple[str, ...]
    request: RequestSpec
    expect: ExpectSpec
    parity_mode: ParityMode
    fixture_hash: str


@dataclass(frozen=True)
class FixtureSet:
    schema_version: int
    fixtures: tuple[FixtureSpec, ...]
    content_hash: str


def load_fixture_set(path: Path | str) -> FixtureSet:
    return load_fixture_text(Path(path).read_text(encoding="utf-8"))


def load_fixture_text(text: str) -> FixtureSet:
    try:
        raw = json.loads(text, object_pairs_hook=_unique, parse_constant=lambda item: _fail("invalid_constant", item))
    except FixtureError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise FixtureError(f"invalid_json: {error}") from error
    root = _object(raw, "document", {"schema_version", "fixtures"})
    version = _integer(root["schema_version"], "schema_version")
    if version != SCHEMA_VERSION:
        _fail("unsupported_schema_version", str(version))
    if not isinstance(root["fixtures"], list):
        _fail("invalid_type", "fixtures must be an array")
    fixtures = tuple(_fixture(item, version) for item in root["fixtures"])
    names = [item.name for item in fixtures]
    if len(names) != len(set(names)):
        _fail("duplicate_fixture", "fixture names must be unique")
    return FixtureSet(version, fixtures, hashlib.sha256(_canonical(root)).hexdigest())


def _fixture(value: Any, version: int) -> FixtureSpec:
    required = {"name", "capability", "profiles", "request", "expect", "parity"}
    row = _object(value, "fixture", required)
    name = _identifier(row["name"], "name")
    capability = _identifier(row["capability"], "capability")
    if capability not in CAPABILITY_REQUIREMENTS:
        _fail("unsupported_capability", capability)
    if not isinstance(row["profiles"], list) or not row["profiles"]:
        _fail("invalid_type", f"{name}.profiles must be a non-empty array")
    profiles = tuple(_string(item, f"{name}.profiles") for item in row["profiles"])
    if len(profiles) != len(set(profiles)):
        _fail("duplicate_profile", name)
    parity = _object(row["parity"], f"{name}.parity", {"mode"})
    try:
        parity_mode = ParityMode(_string(parity["mode"], f"{name}.parity.mode"))
    except ValueError:
        _fail("invalid_parity_mode", name)
    request = _request(row["request"], name)
    expect = _expect(row["expect"], name)
    _validate_contract(name, capability, request, expect)
    digest = hashlib.sha256(_canonical({"schema_version": version, "fixture": row})).hexdigest()
    return FixtureSpec(name, capability, profiles, request, expect, parity_mode, digest)


def _request(value: Any, name: str) -> RequestSpec:
    row = _object(value, f"{name}.request", {"prompt", "tools"})
    if not isinstance(row["tools"], bool):
        _fail("invalid_type", f"{name}.request.tools")
    return RequestSpec(_string(row["prompt"], f"{name}.request.prompt"), row["tools"])


def _expect(value: Any, name: str) -> ExpectSpec:
    required = {"finish_reason", "max_model_calls"}
    optional = {"route", "tool_calls", "exact_output", "final_reports_workdir", "artifact"}
    row = _object(value, f"{name}.expect", required, optional)
    raw_calls = row.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        _fail("invalid_type", f"{name}.expect.tool_calls")
    report_workdir = row.get("final_reports_workdir", False)
    if not isinstance(report_workdir, bool):
        _fail("invalid_type", f"{name}.expect.final_reports_workdir")
    return ExpectSpec(
        finish_reason=_string(row["finish_reason"], f"{name}.expect.finish_reason"),
        max_model_calls=_positive(row["max_model_calls"], f"{name}.expect.max_model_calls"),
        route=_optional_string(row.get("route"), f"{name}.expect.route"),
        tool_calls=tuple(_tool(item, name) for item in raw_calls),
        exact_output=_optional_string(row.get("exact_output"), f"{name}.expect.exact_output"),
        final_reports_workdir=report_workdir,
        artifact=_artifact(row["artifact"], name) if "artifact" in row else None,
    )


def _tool(value: Any, name: str) -> ToolExpectation:
    row = _object(value, f"{name}.tool_call", {"name"}, {"arguments"})
    arguments = row.get("arguments")
    if arguments is not None:
        arguments = _object(arguments, f"{name}.tool_call.arguments", set(), set(arguments) if isinstance(arguments, dict) else set())
    return ToolExpectation(_string(row["name"], f"{name}.tool_call.name"), arguments)


def _artifact(value: Any, name: str) -> ArtifactExpectation:
    keys = {"path", "json_equals", "publication", "verification"}
    row = _object(value, f"{name}.artifact", keys)
    if not isinstance(row["publication"], bool) or not isinstance(row["verification"], bool):
        _fail("invalid_type", f"{name}.artifact publication/verification")
    return ArtifactExpectation(_relative_path(row["path"], f"{name}.artifact.path"), row["json_equals"], row["publication"], row["verification"])


def _object(value: Any, path: str, required: set[str], optional: set[str] = set()) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid_type", f"{path} must be an object")
    unknown = sorted(set(value) - required - optional)
    missing = sorted(required - set(value))
    if unknown:
        _fail("unknown_key", f"{path}.{unknown[0]}")
    if missing:
        _fail("missing_key", f"{path}.{missing[0]}")
    return value


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_key", key)
        result[key] = value
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid_type", f"{path} must be a non-empty string")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _identifier(value: Any, path: str) -> str:
    result = _string(value, path)
    if re.fullmatch(r"[a-z][a-z0-9_]*", result) is None:
        _fail("invalid_value", f"{path} is not an identifier")
    return result


def _relative_path(value: Any, path: str) -> str:
    result = _string(value, path)
    parsed = PurePosixPath(result)
    if "\x00" in result or result == "." or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != result:
        _fail("invalid_value", f"{path} is not a confined relative path")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("invalid_type", f"{path} must be an integer")
    return value


def _positive(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result < 1:
        _fail("invalid_value", f"{path} must be positive")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_contract(name: str, capability: str, request: RequestSpec, expect: ExpectSpec) -> None:
    has_tool_contract = bool(expect.route or expect.tool_calls or expect.final_reports_workdir)
    if capability == "chat" and (request.tools or has_tool_contract or expect.artifact is not None):
        _fail("invalid_fixture_contract", f"{name} chat fixture exposes tool behavior")
    if capability == "tools" and (not request.tools or expect.artifact is not None):
        _fail("invalid_fixture_contract", f"{name} tools fixture is inconsistent")
    if capability == "artifacts" and (not request.tools or expect.artifact is None or expect.route is not None):
        _fail("invalid_fixture_contract", f"{name} artifact fixture is inconsistent")


def _fail(code: str, detail: str):
    raise FixtureError(f"{code}: {detail}")

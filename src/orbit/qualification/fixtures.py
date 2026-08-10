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
    "state_reuse": ("route_prefix_reuse",),
    "document.full_analysis": ("full_document_analysis",),
}

LIFECYCLE_OPERATIONS = frozenset({
    "reset_invalidation", "cancellation", "restore_failure_fallback",
    "repeated_restore_rss", "teardown_cleanup",
})


class FixtureError(ValueError):
    pass


@dataclass(frozen=True)
class RequestSpec:
    prompt: str
    tools: bool
    full_request: bool = False


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
class FileSpec:
    path: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkspaceSpec:
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class WorkflowExpectation:
    files: tuple[FileSpec, ...]
    unchanged_files: tuple[str, ...]
    max_tool_calls: int
    timeout_seconds: int
    test_runner: str | None = None
    require_recovery: bool = False


@dataclass(frozen=True)
class LifecycleExpectation:
    operation: str
    min_restores: int = 0


@dataclass(frozen=True)
class DocumentExpectation:
    coverage: str | None
    answer_contains: str | None = None


@dataclass(frozen=True)
class ExpectSpec:
    finish_reason: str
    max_model_calls: int
    route: str | None = None
    tool_calls: tuple[ToolExpectation, ...] = ()
    exact_output: str | None = None
    final_reports_workdir: bool = False
    artifact: ArtifactExpectation | None = None
    workflow: WorkflowExpectation | None = None
    lifecycle: LifecycleExpectation | None = None
    document: DocumentExpectation | None = None


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    capability: str
    profiles: tuple[str, ...]
    request: RequestSpec
    expect: ExpectSpec
    parity_mode: ParityMode
    fixture_hash: str
    workspace: WorkspaceSpec | None = None


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
    row = _object(value, "fixture", required, {"workspace"})
    name = _identifier(row["name"], "name")
    capability = _string(row["capability"], "capability")
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
    workspace = _workspace(row["workspace"], name) if "workspace" in row else None
    _validate_contract(name, capability, request, expect, workspace)
    digest = hashlib.sha256(_canonical({"schema_version": version, "fixture": row})).hexdigest()
    return FixtureSpec(name, capability, profiles, request, expect, parity_mode, digest, workspace)


def _request(value: Any, name: str) -> RequestSpec:
    row = _object(value, f"{name}.request", {"prompt", "tools"}, {"full_request"})
    full_request = row.get("full_request", False)
    if not isinstance(row["tools"], bool) or not isinstance(full_request, bool):
        _fail("invalid_type", f"{name}.request.tools")
    return RequestSpec(
        _string(row["prompt"], f"{name}.request.prompt"),
        row["tools"],
        full_request,
    )


def _expect(value: Any, name: str) -> ExpectSpec:
    required = {"finish_reason", "max_model_calls"}
    optional = {
        "route", "tool_calls", "exact_output", "final_reports_workdir", "artifact",
        "workflow", "lifecycle", "document",
    }
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
        workflow=_workflow(row["workflow"], name) if "workflow" in row else None,
        lifecycle=_lifecycle(row["lifecycle"], name) if "lifecycle" in row else None,
        document=_document(row["document"], name) if "document" in row else None,
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


def _workspace(value: Any, name: str) -> WorkspaceSpec:
    row = _object(value, f"{name}.workspace", {"files"})
    files = _files(row["files"], f"{name}.workspace.files")
    _unique_paths(files, "duplicate_workspace_path", name)
    return WorkspaceSpec(files)


def _workflow(value: Any, name: str) -> WorkflowExpectation:
    required = {"files", "unchanged_files", "max_tool_calls", "timeout_seconds"}
    row = _object(value, f"{name}.workflow", required, {"test_runner", "require_recovery"})
    files = _files(row["files"], f"{name}.workflow.files")
    _unique_paths(files, "duplicate_expected_path", name)
    raw_unchanged = row["unchanged_files"]
    if not isinstance(raw_unchanged, list):
        _fail("invalid_type", f"{name}.workflow.unchanged_files must be an array")
    unchanged = tuple(_relative_path(item, f"{name}.workflow.unchanged_files") for item in raw_unchanged)
    if len(unchanged) != len(set(unchanged)):
        _fail("duplicate_unchanged_path", name)
    if set(unchanged) & {item.path for item in files}:
        _fail("invalid_fixture_contract", f"{name} expected and unchanged files overlap")
    test_runner = row.get("test_runner")
    if test_runner is not None and test_runner != "python_unittest":
        _fail("invalid_test_runner", name)
    require_recovery = row.get("require_recovery", False)
    if not isinstance(require_recovery, bool):
        _fail("invalid_type", f"{name}.workflow.require_recovery")
    return WorkflowExpectation(
        files=files,
        unchanged_files=unchanged,
        max_tool_calls=_bounded_positive(row["max_tool_calls"], f"{name}.workflow.max_tool_calls", 16),
        timeout_seconds=_bounded_positive(row["timeout_seconds"], f"{name}.workflow.timeout_seconds", 1800),
        test_runner=test_runner,
        require_recovery=require_recovery,
    )


def _lifecycle(value: Any, name: str) -> LifecycleExpectation:
    row = _object(value, f"{name}.lifecycle", {"operation"}, {"min_restores"})
    operation = _string(row["operation"], f"{name}.lifecycle.operation")
    if operation not in LIFECYCLE_OPERATIONS:
        _fail("invalid_lifecycle_operation", operation)
    min_restores = row.get("min_restores", 0)
    min_restores = _bounded_positive(min_restores, f"{name}.lifecycle.min_restores", 50) if min_restores else 0
    if operation == "repeated_restore_rss" and min_restores < 2:
        _fail("invalid_fixture_contract", f"{name} requires at least two restores")
    if operation != "repeated_restore_rss" and min_restores:
        _fail("invalid_fixture_contract", f"{name} min_restores is only valid for RSS qualification")
    return LifecycleExpectation(operation, min_restores)


def _document(value: Any, name: str) -> DocumentExpectation:
    required = {"coverage"}
    row = _object(value, f"{name}.document", required, {"answer_contains"})
    coverage = row["coverage"]
    if coverage is not None and coverage not in {"none", "complete"}:
        _fail("invalid_value", f"{name}.document.coverage")
    return DocumentExpectation(
        coverage=coverage,
        answer_contains=_optional_string(row.get("answer_contains"), f"{name}.document.answer_contains"),
    )


def _files(value: Any, path: str) -> tuple[FileSpec, ...]:
    if not isinstance(value, list) or not value:
        _fail("invalid_type", f"{path} must be a non-empty array")
    files = []
    for index, item in enumerate(value):
        row = _object(item, f"{path}[{index}]", {"path", "content"})
        files.append(FileSpec(
            _relative_path(row["path"], f"{path}[{index}].path"),
            _text(row["content"], f"{path}[{index}].content"),
        ))
    return tuple(files)


def _unique_paths(files: tuple[FileSpec, ...], code: str, detail: str) -> None:
    paths = tuple(item.path for item in files)
    if len(paths) != len(set(paths)):
        _fail(code, detail)


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


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail("invalid_type", f"{path} must be a string")
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


def _bounded_positive(value: Any, path: str, maximum: int) -> int:
    result = _positive(value, path)
    if result > maximum:
        _fail("invalid_value", f"{path} must be at most {maximum}")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_contract(
    name: str,
    capability: str,
    request: RequestSpec,
    expect: ExpectSpec,
    workspace: WorkspaceSpec | None,
) -> None:
    if capability == "document.full_analysis":
        document = expect.document
        if (
            document is None or not request.tools or workspace is None
            or len(workspace.files) != 1 or request.full_request
            or expect.artifact is not None or expect.workflow is not None or expect.lifecycle is not None
            or expect.tool_calls or expect.final_reports_workdir
        ):
            _fail("invalid_fixture_contract", f"{name} document fixture is inconsistent")
        if document.coverage is not None:
            if expect.route != "FILESYSTEM":
                _fail("invalid_fixture_contract", f"{name} document route is inconsistent")
        elif expect.route != "CHAT":
            _fail("invalid_fixture_contract", f"{name} inert document expectation is inconsistent")
        return
    if expect.document is not None:
        _fail("invalid_fixture_contract", f"{name} document expectation requires document.full_analysis")
    if capability == "state_reuse":
        if expect.lifecycle is None or expect.workflow is not None or expect.artifact is not None:
            _fail("invalid_fixture_contract", f"{name} state-reuse fixture is inconsistent")
        if workspace is not None or expect.route is not None or expect.tool_calls or expect.exact_output is not None:
            _fail("invalid_fixture_contract", f"{name} state-reuse fixture exposes unrelated behavior")
        return
    if expect.lifecycle is not None:
        _fail("invalid_fixture_contract", f"{name} lifecycle expectation requires state_reuse")
    has_tool_contract = bool(expect.route or expect.tool_calls or expect.final_reports_workdir)
    if capability == "chat" and (request.tools or has_tool_contract or expect.artifact is not None):
        _fail("invalid_fixture_contract", f"{name} chat fixture exposes tool behavior")
    if capability == "tools" and (not request.tools or expect.artifact is not None):
        _fail("invalid_fixture_contract", f"{name} tools fixture is inconsistent")
    if capability == "artifacts" and (not request.tools or expect.artifact is None or expect.route is not None):
        _fail("invalid_fixture_contract", f"{name} artifact fixture is inconsistent")
    if request.full_request and not (
        capability == "tools" and expect.route == "CHAT"
        and not expect.tool_calls and expect.max_model_calls >= 2
    ):
        _fail("invalid_fixture_contract", f"{name} full request fixture is inconsistent")
    if expect.workflow is None:
        if workspace is not None:
            _fail("invalid_fixture_contract", f"{name} workspace requires a workflow")
        return
    if (
        capability != "tools"
        or not request.tools
        or workspace is None
        or expect.route is not None
        or expect.tool_calls
        or expect.exact_output is not None
        or expect.final_reports_workdir
        or expect.artifact is not None
    ):
        _fail("invalid_fixture_contract", f"{name} workflow fixture is inconsistent")
    workspace_paths = {item.path for item in workspace.files}
    if set(expect.workflow.unchanged_files) - workspace_paths:
        _fail("invalid_fixture_contract", f"{name} unchanged file is absent from workspace")


def _fail(code: str, detail: str):
    raise FixtureError(f"{code}: {detail}")

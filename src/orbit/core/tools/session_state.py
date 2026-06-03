from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PURE_TOOL_NAMES = {"read_file", "list_files", "stat_path", "search_web", "fetch_url"}
WRITE_TOOL_NAMES = {"write_file", "append_file", "replace_in_file", "make_directory", "delete_path"}
CONTENT_WRITE_TOOL_NAMES = {"write_file", "append_file", "replace_in_file"}
READ_GUARD_WARN_THRESHOLD = 1
TRUST_WARN_THRESHOLD = 3
TRUST_DROP_THRESHOLD = 5
DEDUP_WINDOW = 5


@dataclass
class ReadBeforeWriteGuard:
    workdir: Path
    read_paths: set[str] = field(default_factory=set)
    written_paths: set[str] = field(default_factory=set)
    warned_paths: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.read_paths.clear()
        self.written_paths.clear()
        self.warned_paths.clear()

    def record_read(self, raw_path: str) -> None:
        canonical = self._canonical_path(raw_path)
        if canonical is None:
            return
        self.read_paths.add(canonical)
        self.warned_paths.discard(canonical)

    def record_write(self, raw_path: str) -> None:
        canonical = self._canonical_path(raw_path)
        if canonical is None:
            return
        self.written_paths.add(canonical)
        self.read_paths.add(canonical)

    def check_write(self, raw_path: str) -> tuple[bool, str | None]:
        canonical = self._canonical_path(raw_path)
        if canonical is None:
            return True, None
        path = self.workdir / canonical
        if not path.exists():
            return True, None
        if canonical in self.read_paths or canonical in self.written_paths:
            return True, None
        if canonical in self.warned_paths:
            self.record_write(raw_path)
            return True, "overwriting unread file after one prior warning"
        self.warned_paths.add(canonical)
        return (
            False,
            (
                f"refused: {raw_path} exists but has not been read in this session. "
                "Call read_file first to inspect the current content, or retry once if you intend a full replacement."
            ),
        )

    def _canonical_path(self, raw_path: str) -> str | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        path = Path(raw_path)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(self.workdir.resolve())
            except ValueError:
                return None
            return relative.as_posix()
        return Path(raw_path).as_posix().lstrip("./")


@dataclass
class ToolDedupCache:
    recent: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    window_size: int = DEDUP_WINDOW

    def reset(self) -> None:
        self.recent.clear()

    def lookup(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if name not in PURE_TOOL_NAMES:
            return None
        signature = self._signature(arguments)
        for cached_name, cached_signature, result in reversed(self.recent):
            if cached_name == name and cached_signature == signature:
                copy = dict(result)
                copy["_dedup_cached"] = True
                return copy
        return None

    def record(self, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        if name not in PURE_TOOL_NAMES:
            return
        if not isinstance(result, dict) or result.get("ok") is not True:
            return
        signature = self._signature(arguments)
        self.recent = [item for item in self.recent if not (item[0] == name and item[1] == signature)]
        self.recent.append((name, signature, dict(result)))
        while len(self.recent) > self.window_size:
            self.recent.pop(0)

    @staticmethod
    def _signature(arguments: dict[str, Any]) -> str:
        return json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)


@dataclass
class ToolTrustDecay:
    scores: dict[str, int] = field(default_factory=dict)
    warn_threshold: int = TRUST_WARN_THRESHOLD
    drop_threshold: int = TRUST_DROP_THRESHOLD

    def reset(self) -> None:
        self.scores.clear()

    def record(self, tool_name: str, success: bool, *, guard_failure: bool = False) -> None:
        if not tool_name or guard_failure:
            return
        if success:
            self.scores[tool_name] = 0
            return
        self.scores[tool_name] = self.scores.get(tool_name, 0) + 1

    def filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ok: list[dict[str, Any]] = []
        warned: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", {}) or {}
            name = fn.get("name")
            if not isinstance(name, str):
                ok.append(tool)
                continue
            level = self.level(name)
            if level == "drop":
                continue
            if level == "warn":
                warned.append(tool)
            else:
                ok.append(tool)
        return ok + warned

    def level(self, tool_name: str) -> str:
        count = self.scores.get(tool_name, 0)
        if count >= self.drop_threshold:
            return "drop"
        if count >= self.warn_threshold:
            return "warn"
        return "ok"


@dataclass
class ToolSessionState:
    workdir: Path
    read_guard: ReadBeforeWriteGuard = field(init=False)
    dedup: ToolDedupCache = field(init=False)
    trust: ToolTrustDecay = field(init=False)

    def __post_init__(self) -> None:
        self.read_guard = ReadBeforeWriteGuard(self.workdir)
        self.dedup = ToolDedupCache()
        self.trust = ToolTrustDecay()

    def reset(self) -> None:
        self.read_guard.reset()
        self.dedup.reset()
        self.trust.reset()

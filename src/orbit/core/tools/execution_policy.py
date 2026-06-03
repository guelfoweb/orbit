from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .session_state import CONTENT_WRITE_TOOL_NAMES, WRITE_TOOL_NAMES, ToolSessionState


class ToolExecutionPolicy:
    def __init__(self, workdir: Path) -> None:
        self._state = ToolSessionState(workdir)

    def reset(self) -> None:
        self._state.reset()

    def filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._state.trust.filter_tools(tools)

    def call_tool(
        self,
        *,
        registry: Any,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        guarded = self._check_write_guard(name=name, arguments=arguments)
        if guarded is not None:
            return guarded, 0
        cached = self._state.dedup.lookup(name, arguments)
        if cached is not None:
            return cached, 0
        started_at = time.monotonic_ns()
        result = registry.call(name, arguments)
        elapsed_ns = time.monotonic_ns() - started_at
        self._record_effects(name=name, arguments=arguments, result=result)
        return result, elapsed_ns

    def rehydrate_from_messages(self, messages: list[dict[str, Any]]) -> None:
        self.reset()
        for message in messages:
            if message.get("role") != "tool":
                continue
            name = message.get("tool_name")
            content = message.get("content")
            if not isinstance(name, str) or not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                continue
            path = payload.get("path")
            if name == "read_file" and isinstance(path, str):
                self._state.read_guard.record_read(path)
            if name in WRITE_TOOL_NAMES and isinstance(path, str):
                self._state.read_guard.record_write(path)

    def _check_write_guard(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if name not in CONTENT_WRITE_TOOL_NAMES:
            return None
        path = arguments.get("path")
        if not isinstance(path, str):
            return None
        allowed, reason = self._state.read_guard.check_write(path)
        if allowed:
            return None
        return {"ok": False, "error": reason, "_guarded": True, "path": path}

    def _record_effects(self, *, name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        success = bool(result.get("ok"))
        guard_failure = bool(result.get("_guarded"))
        self._state.trust.record(name, success, guard_failure=guard_failure)
        if success and name == "read_file":
            path = result.get("path") or arguments.get("path")
            if isinstance(path, str):
                self._state.read_guard.record_read(path)
        if success and name in WRITE_TOOL_NAMES:
            path = result.get("path") or arguments.get("path")
            if isinstance(path, str):
                self._state.read_guard.record_write(path)
        self._state.dedup.record(name, arguments, result)

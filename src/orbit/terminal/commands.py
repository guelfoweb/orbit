from __future__ import annotations

from dataclasses import replace
import json

from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.config import AppConfig
from orbit.terminal.command_registry import COMMANDS
from orbit.terminal.runtime_status import collect_runtime_status, format_status_panel
from orbit.terminal.think_mode import think_text
from orbit.terminal.tool_mode import ToolSpec


MIN_MAX_TOKENS = 32
MAX_MAX_TOKENS = 4096


def help_text() -> str:
    width = max(len(command.usage) for command in COMMANDS) + 2
    lines: list[str] = []
    category = None
    for command in COMMANDS:
        if command.category != category:
            if lines:
                lines.append("")
            category = command.category
            lines.append(category)
        lines.append(f"{command.usage:<{width}}{command.description}")
    return "\n".join(lines)


def health_text(backend: LlamaServerBackend, config: AppConfig) -> str:
    healthy = backend.health()
    lines = [
        "Health",
        "------",
        f"base_url: {config.base_url}",
        f"server: {'ok' if healthy else 'unavailable'}",
    ]
    if not healthy:
        lines.append("hint: start a local backend before launching Orbit")
        return "\n".join(lines)
    info = backend.model_info()
    display_model = (info.id if info and info.id else None) or backend.display_model_name() or "unknown"
    server_tools = _server_tool_names(backend)
    lines.extend(
        [
            f"model: {display_model}",
            f"context: {info.context_length if info and info.context_length is not None else 'unknown'}",
            f"server_tools: {len(server_tools)} available",
        ]
    )
    return "\n".join(lines)


def tools_text(current: ToolSpec | None = None) -> str:
    lines = []
    if current is not None:
        lines.extend([f"tools: {current}", ""])
    lines.extend(
        [
            "Use:",
            "  /tools off = chat only",
            "  /tools on  = unrestricted local shell for files, web, edits, system, and automation",
            "  /tools status = show detected local document capabilities",
            "  /tools refresh = refresh detected local document capabilities",
        ]
    )
    return "\n".join(lines)


def think_mode_text(current: bool | None = None) -> str:
    return think_text(current)


def runtime_status(
    runtime: ChatRuntime,
    config: AppConfig,
    backend: LlamaServerBackend,
    *,
    tools_mode: ToolSpec | None = None,
) -> str:
    status = collect_runtime_status(runtime, config, backend, tools_mode=tools_mode)
    return format_status_panel(status)


def props_text(backend: LlamaServerBackend) -> str:
    try:
        props = backend.backend_props()
    except (LlamaServerError, OSError, ValueError) as exc:
        return f"error: backend properties unavailable: {exc}"
    return json.dumps(props if isinstance(props, dict) else {}, ensure_ascii=False, indent=2, sort_keys=True)


def set_max_tokens(config: AppConfig, value: str) -> tuple[AppConfig, str]:
    value = value.strip()
    if not value:
        return config, f"max_tokens: {config.max_tokens}"
    try:
        parsed = int(value)
    except ValueError:
        return config, f"error: max_tokens must be an integer between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
    if parsed < MIN_MAX_TOKENS or parsed > MAX_MAX_TOKENS:
        return config, f"error: max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
    return replace(config, max_tokens=parsed), f"max_tokens: {parsed}"


def reset_session(runtime: ChatRuntime, session: SessionStore | None) -> str:
    runtime.reset()
    if session:
        session.clear()
    return "session reset"


def _server_tool_names(backend: LlamaServerBackend) -> list[str]:
    names = []
    for item in backend.server_tools():
        name = item.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(names)

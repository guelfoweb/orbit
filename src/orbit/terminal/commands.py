from __future__ import annotations

from dataclasses import replace

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.sessions import SessionStore
from orbit.terminal.config import AppConfig
from orbit.terminal.runtime_status import collect_runtime_status, format_status_panel
from orbit.terminal.think_mode import think_text
from orbit.terminal.tool_mode import ToolSpec


MIN_MAX_TOKENS = 32
MAX_MAX_TOKENS = 4096


def help_text() -> str:
    commands = [
        ("/compact [tools]", "Compact memory; use tools for old tool results."),
        ("/continue", "Continue the last answer if it reached max_tokens."),
        ("/health", "Check backend health."),
        ("/help", "Show this help."),
        ("/max-tokens [n]", "Show or set output token limit for following turns."),
        ("/think [off|on]", "Show or set thinking visibility."),
        ("/reset", "Clear current conversation and saved session."),
        ("/sessions clear", "Delete all saved sessions for this workdir."),
        ("/status [ctx]", "Show runtime status or estimated context usage."),
        ("/tools [off|on|status|refresh]", "Show tool access or local capabilities."),
        ("/exit", "Exit interactive mode."),
    ]
    width = max(len(command) for command, _ in commands) + 2
    return "\n".join(f"{command:<{width}}{description}" for command, description in commands)


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

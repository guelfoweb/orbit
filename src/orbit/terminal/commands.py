from __future__ import annotations

from dataclasses import replace

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.session_memory import DEFAULT_CONTEXT_TOKENS, SOFT_MEMORY_RATIO, estimate_message_tokens
from orbit.runtime.sessions import SessionStore
from orbit.runtime.tools import tool_names
from orbit.terminal.config import AppConfig
from orbit.terminal.tool_mode import ToolSpec


MIN_MAX_TOKENS = 32
MAX_MAX_TOKENS = 4096


def help_text() -> str:
    return "\n".join(
        [
            "/continue        Continue the last answer if it reached max_tokens.",
            "/health           Check llama-server health.",
            "/help             Show this help.",
            "/max-tokens [n]   Show or set output token limit for following turns.",
            "/reset            Clear current conversation and saved session.",
            "/sessions clear   Delete all saved sessions for this workdir.",
            "/status           Show runtime, session, and backend capabilities.",
            "/tools [spec]     Show or set tools: off, on, files, edit, web, shell.",
            "/exit             Exit interactive mode.",
        ]
    )


def tools_text(current: ToolSpec | None = None) -> str:
    lines = []
    if current is not None:
        lines.extend([f"tools: {current}", ""])
    lines.extend(
        [
            "Use:",
            "  /tools files = read/inspect local files",
            "  /tools edit  = create/modify/delete files or directories",
            "  /tools web   = search/fetch URLs",
            "  /tools shell = read-only local/system commands",
        ]
    )
    return "\n".join(lines)


def runtime_status(
    runtime: ChatRuntime,
    config: AppConfig,
    backend: LlamaServerBackend,
    *,
    tools_mode: ToolSpec | None = None,
) -> str:
    info = backend.model_info()
    display_model = (info.id if info and info.id else None) or backend.display_model_name() or "unknown"
    lines = [
        "Backend",
        "-------",
        f"base_url: {config.base_url}",
        f"server: {'ok' if backend.health() else 'unavailable'}",
        f"model: {display_model}",
        "",
        "Runtime",
        "-------",
        f"temperature: {config.temperature}",
        f"max_tokens: {config.max_tokens}",
        f"context_tokens_override: {config.context_tokens if config.context_tokens is not None else 'off'}",
        f"messages: {len(runtime.messages)}",
        f"estimated_context_tokens: {estimate_message_tokens(runtime.messages)}",
        f"system: {'off' if config.no_system else 'on'}",
        "",
        "Workdir",
        "-------",
        f"workdir: {config.workdir}",
        "",
        "Tools",
        "-------",
        f"tools_mode: {_format_tools_mode(tools_mode)}",
        f"tools_llama_server: {_format_server_tools(backend)}",
        f"tools_orbit_only: {', '.join(_orbit_only_tool_names(_server_tool_names(backend))) or 'none'}",
        "",
        "Memory",
        "-------",
        *_memory_status_lines(runtime),
    ]
    if info:
        lines.extend(
            [
                "",
                "Model",
                "-------",
                f"capabilities: {', '.join(info.capabilities) if info.capabilities else 'unknown'}",
                f"context: {info.context_length if info.context_length is not None else 'unknown'}",
                f"parameters: {_format_count(info.parameter_count)}",
                f"model_size: {_format_bytes(info.size_bytes)}",
            ]
        )
    return "\n".join(lines)


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


def _memory_status_lines(runtime: ChatRuntime) -> list[str]:
    window = runtime.context_tokens or DEFAULT_CONTEXT_TOKENS
    threshold = int(window * SOFT_MEMORY_RATIO)
    last_success = runtime.last_memory_refresh
    last_attempt = runtime.last_memory_refresh_attempt
    cooldown_remaining = _memory_cooldown_remaining(runtime)
    lines = [
        f"memory_refresh_threshold: {threshold}/{window}",
        f"memory_refreshes: {runtime.memory_refreshes}",
        f"total_tokens_saved: {runtime.total_memory_tokens_saved}",
        f"memory_cooldown: {'active' if cooldown_remaining > 0 else 'inactive'} ({cooldown_remaining} message(s) remaining)",
    ]
    if last_success:
        saved = max(0, last_success.estimated_tokens_before - last_success.estimated_tokens_after)
        lines.extend(
            [
                f"last_refresh_before: {last_success.estimated_tokens_before}",
                f"last_refresh_after: {last_success.estimated_tokens_after}",
                f"last_refresh_saved: {saved}",
            ]
        )
    else:
        lines.extend(["last_refresh_before: none", "last_refresh_after: none", "last_refresh_saved: 0"])
    if last_attempt:
        outcome = "success" if last_attempt.changed else "discarded"
        lines.append(f"last_refresh_outcome: {outcome} ({last_attempt.reason})")
    else:
        lines.append("last_refresh_outcome: none")
    return lines


def _memory_cooldown_remaining(runtime: ChatRuntime) -> int:
    if runtime.last_memory_refresh_message_count is None:
        return 0
    used = len(runtime.messages) - runtime.last_memory_refresh_message_count
    return max(0, runtime.memory_refresh_cooldown_messages - used)


def _format_tools_mode(mode: ToolSpec | None) -> str:
    if mode is None:
        return "n/a"
    return mode


def _format_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return str(value)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value} B"


def _format_server_tools(backend: LlamaServerBackend) -> str:
    names = _server_tool_names(backend)
    return ", ".join(names) if names else "unavailable"


def _server_tool_names(backend: LlamaServerBackend) -> list[str]:
    names = []
    for item in backend.server_tools():
        name = item.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(names)


def _orbit_only_tool_names(server_tools: list[str]) -> list[str]:
    hidden = set(server_tools)
    if "exec_shell_command" in server_tools:
        hidden.update({"stat_path"})
    if "edit_file" in server_tools or "apply_diff" in server_tools:
        hidden.update({"append_file", "replace_in_file"})
    return [name for name in tool_names() if name not in hidden]

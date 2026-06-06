from __future__ import annotations

from dataclasses import replace

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.session_memory import estimate_message_tokens
from orbit.runtime.sessions import SessionStore
from orbit.runtime.tools import tool_names
from orbit.terminal.config import AppConfig


MIN_MAX_TOKENS = 32
MAX_MAX_TOKENS = 4096


def help_text() -> str:
    return "\n".join(
        [
            "/health           Check llama-server health.",
            "/help             Show this help.",
            "/max-tokens       Show current output token limit.",
            "/max-tokens <n>   Set output token limit for following turns.",
            "/reset            Clear current conversation and saved session.",
            "/status           Show runtime, session, and backend capabilities.",
            "/tools            Show available local tools.",
            "/exit             Exit interactive mode.",
        ]
    )


def runtime_status(runtime: ChatRuntime, config: AppConfig, backend: LlamaServerBackend) -> str:
    info = backend.model_info()
    display_model = (info.id if info and info.id else None) or backend.display_model_name() or config.model
    lines = [
        f"base_url: {config.base_url}",
        f"server: {'ok' if backend.health() else 'unavailable'}",
        f"model: {display_model}",
        f"temperature: {config.temperature}",
        f"max_tokens: {config.max_tokens}",
        f"context_tokens_override: {config.context_tokens if config.context_tokens is not None else 'off'}",
        f"messages: {len(runtime.messages)}",
        f"estimated_context_tokens: {estimate_message_tokens(runtime.messages)}",
        f"system: {'off' if config.no_system else 'on'}",
        f"workdir: {config.workdir}",
        f"tools: {', '.join(tool_names())}",
    ]
    if info:
        lines.extend(
            [
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

from __future__ import annotations

from dataclasses import replace

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime import ChatRuntime
from orbit.runtime.session_memory import DEFAULT_CONTEXT_TOKENS, SOFT_MEMORY_RATIO, estimate_message_tokens
from orbit.runtime.sessions import SessionStore
from orbit.runtime.tools import tool_names
from orbit.terminal.config import AppConfig
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
    info = backend.model_info()
    props = backend.backend_props()
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
        f"thinking_mode: {'on' if config.think else 'off'}",
        "",
        "Workdir",
        "-------",
        f"workdir: {config.workdir}",
        "",
        "Tools",
        "-------",
        f"tools_mode: {_format_tools_mode(tools_mode)}",
        f"model_tools: {', '.join(tool_names())}",
        "",
        "Memory",
        "-------",
        *_memory_status_lines(runtime),
        "",
        "Mutation verification",
        "-------",
        f"mutation_verifications: {runtime.mutation_verifications}",
        f"mutation_verification_repairs: {runtime.mutation_verification_repairs}",
        f"mutation_verification_failures: {runtime.mutation_verification_failures}",
        f"mutation_semantic_repairs: {runtime.mutation_semantic_repairs}",
        f"mutation_semantic_repair_commands: {runtime.mutation_semantic_repair_commands}",
        f"mutation_semantic_repair_failures: {runtime.mutation_semantic_repair_failures}",
        "",
        "Content evidence guard",
        "-------",
        f"content_evidence_guard_nudges: {runtime.content_evidence_guard_nudges}",
        f"content_evidence_guard_commands: {runtime.content_evidence_guard_commands}",
        f"content_evidence_guard_successes: {runtime.content_evidence_guard_successes}",
        f"content_evidence_guard_failures: {runtime.content_evidence_guard_failures}",
        "",
        "Completion guard",
        "-------",
        f"completion_guard_nudges: {runtime.completion_guard_nudges}",
        f"completion_guard_commands: {runtime.completion_guard_commands}",
        f"completion_guard_successes: {runtime.completion_guard_successes}",
        f"completion_guard_failures: {runtime.completion_guard_failures}",
        "",
        "Minimal patch guard",
        "-------",
        f"minimal_patch_guard_nudges: {runtime.minimal_patch_guard_nudges}",
        f"minimal_patch_guard_commands: {runtime.minimal_patch_guard_commands}",
        f"minimal_patch_guard_successes: {runtime.minimal_patch_guard_successes}",
        f"minimal_patch_guard_failures: {runtime.minimal_patch_guard_failures}",
    ]
    if props:
        lines.extend(
            [
                "",
                "Backend runtime",
                "---------------",
                f"backend: {_str_value(props.get('backend'))}",
                f"backend_mode: {_str_value(props.get('backend_mode'))}",
                f"session_id: {_str_value(props.get('session_id'))}",
                f"cached_tokens: {_int_value(props.get('cached_tokens'))}",
                f"in_flight: {_boolish_value(props.get('in_flight'))}",
                f"threads: {_int_value(props.get('threads'))}",
                f"threads_batch: {_int_value(props.get('threads_batch'))}",
                f"ctx_size: {_int_value(props.get('ctx_size'))}",
                f"batch_size: {_int_value(props.get('batch_size'))}",
                f"ubatch_size: {_int_value(props.get('ubatch_size'))}",
                f"parallel_slots: {_int_value(props.get('parallel_slots'))}",
                f"mtp_available: {_boolish_value(props.get('mtp_available'))}",
                f"mtp_enabled: {_boolish_value(props.get('mtp_enabled'))}",
                f"multimodal_available: {_boolish_value(props.get('multimodal_available'))}",
            ]
        )
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


def _str_value(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _int_value(value: object) -> str:
    return str(value) if isinstance(value, int) else "unknown"


def _boolish_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "unknown"


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


def _server_tool_names(backend: LlamaServerBackend) -> list[str]:
    names = []
    for item in backend.server_tools():
        name = item.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(names)

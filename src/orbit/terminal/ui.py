from __future__ import annotations

import sys
import threading

from ..core.agent import AgentLoop, TurnStatus
from ..core.events import (
    DebugTimingEvent,
    EmptyReplyRetryEvent,
    ModelRequestEvent,
    RepeatedToolRetryEvent,
    SessionAutoCompactEvent,
    ToolRouteEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingUnavailableEvent,
    ToolCallEvent,
    ToolResultCompactEvent,
    ToolResultEvent,
)
from ..tooling.registry import ToolRegistry


STATUS_COLOR = "\x1b[90m"
PROMPT_COLOR = "\x1b[96m"
RESET_COLOR = "\x1b[0m"
LIVE_OUTPUT_LOCK = threading.Lock()


def print_help() -> None:
    print("Commands:")
    print("  /compact")
    print("  /debug")
    print("  /exit")
    print("  /help")
    print("  /reset")
    print("  /sessions clear")
    print("  /skill clear | list | show | use <ref>")
    print("  /status")
    print("  /think on|off|auto")
    print("  /thinking on|off")
    print("  /tools")


def print_tools(registry: ToolRegistry) -> None:
    for tool in registry.definitions():
        fn = tool["function"]
        print(f"- {fn['name']}: {fn['description']}")


def format_user_prompt(text: str) -> str:
    line = f"> {text}"
    if sys.stdout.isatty():
        return f"{PROMPT_COLOR}{line}{RESET_COLOR}"
    return line


def format_input_prompt() -> str:
    if sys.stdout.isatty():
        return f"\001{PROMPT_COLOR}\002> \001{RESET_COLOR}\002"
    return "> "


def format_status(status: TurnStatus) -> str:
    if status.context_window:
        if status.usage_ratio is not None:
            context_text = f"{status.context_window} ({status.usage_ratio * 100:.1f}%)"
        else:
            context_text = str(status.context_window)
    else:
        context_text = "unknown"
    input_text = str(status.prompt_tokens) if status.prompt_tokens is not None else f"~{status.estimated_prompt_tokens}"
    output_text = str(status.output_tokens) if status.output_tokens is not None else "-"
    token_text = _format_token_flow(
        input_text=input_text,
        output_text=output_text,
        prefill_tps=status.prefill_tps,
        decode_tps=status.decode_tps,
    )
    show_thinking_state = getattr(status, "show_thinking_state", "off")
    line = (
        f"{status.active_model} | "
        f"ctx: {context_text} | "
        f"tk: {token_text}"
    )
    if not (status.think_state == "off" and show_thinking_state == "off"):
        line = f"{line} | think: {status.think_state} | show-thinking: {show_thinking_state}"
    warning = _format_status_warning(status.warning)
    if warning:
        line = f"{line} | {warning}"
    if sys.stdout.isatty():
        return f"{STATUS_COLOR}{line}{RESET_COLOR}"
    return line


def format_startup_line(text: str) -> str:
    if sys.stdout.isatty():
        return f"{STATUS_COLOR}{text}{RESET_COLOR}"
    return text


def _format_token_flow(
    *,
    input_text: str,
    output_text: str,
    prefill_tps: float | None,
    decode_tps: float | None,
) -> str:
    left = f"{input_text} ({prefill_tps:.1f}/s)" if prefill_tps is not None else input_text
    right = f"{output_text} ({decode_tps:.1f}/s)" if decode_tps is not None else output_text
    return f"{left} -> {right}"


def format_skill(agent: AgentLoop) -> str:
    if agent.skill is None:
        return "skill: -"
    return f"skill: {agent.skill.name} | path: {agent.skill.path}"


def format_runtime_status(runtime) -> str:
    status = runtime.agent.current_status()
    metadata = getattr(runtime, "model_metadata", None)
    capabilities = getattr(metadata, "capabilities", ()) or ()
    capability_text = ", ".join(capabilities) if capabilities else "-"
    tools_state = "enabled" if getattr(runtime, "tools_enabled", False) else "disabled"
    skill = runtime.agent.skill.name if runtime.agent.skill is not None else "-"
    lines = [
        f"model: {status.active_model}",
        f"capabilities: {capability_text}",
        f"ctx: {status.context_window or 'unknown'} | used: ~{status.estimated_prompt_tokens}"
        + (f" ({status.usage_ratio * 100:.1f}%)" if status.usage_ratio is not None else ""),
        f"session: {runtime.session_name} | msg: {status.session_turns}",
        f"workdir: {runtime.config.workdir}",
        f"skill: {skill}",
        f"tools: {tools_state}",
        f"think: {status.think_state} | show-thinking: {getattr(status, 'show_thinking_state', 'off')}",
    ]
    return "\n".join(lines)


def print_live_event(event) -> None:
    if isinstance(event, DebugTimingEvent):
        detail = f" {event.detail}" if event.detail else ""
        _print_live_line(f"└ timing {event.phase}: {event.elapsed_ms:.1f}ms{detail}")
        return
    if isinstance(event, ModelRequestEvent):
        return
    if isinstance(event, SessionAutoCompactEvent):
        detail = event.reason or "context pressure"
        if event.level and not detail.startswith(f"{event.level} "):
            detail = f"{event.level} {detail}"
        _print_live_line(f"└ auto-compact ({detail})")
        return
    if isinstance(event, ToolResultCompactEvent):
        detail = event.reason or "tool result pressure"
        if event.tool_name:
            detail = f"{detail}, tool={event.tool_name}"
        _print_live_line(f"└ auto-compact ({detail})")
        return
    if isinstance(event, EmptyReplyRetryEvent):
        _print_live_line("└ [retry] empty reply")
        return
    if isinstance(event, ToolRouteEvent):
        if tuple(event.categories) != ("filesystem", "write", "shell", "web"):
            _print_live_line(f"└ route: {event.intent} -> {', '.join(event.categories)}")
        return
    if isinstance(event, ThinkingStartEvent):
        _print_thinking_start()
        return
    if isinstance(event, ThinkingChunkEvent):
        _print_thinking_chunk(event.text)
        return
    if isinstance(event, ThinkingEndEvent):
        _print_thinking_end()
        return
    if isinstance(event, ThinkingUnavailableEvent):
        _print_live_line("└ thinking unavailable for this model; continuing without thinking")
        return
    if isinstance(event, RepeatedToolRetryEvent):
        detail = _extract_repeated_tool_detail(event.detail)
        if detail:
            _print_live_line(f"└ [retry] repeated tool blocked ({detail})")
        else:
            _print_live_line("└ [retry] repeated tool blocked")
        return
    if isinstance(event, ToolCallEvent):
        detail = _render_tool_detail(event.name, event.arguments or {})
        _print_live_line(f"└ {detail or event.name}")
        return
    if isinstance(event, ToolResultEvent):
        if event.ok:
            return
        for line in _format_tool_error_lines(event):
            _print_live_line(line)


def make_live_event_printer(*, debug_timing: bool = False):
    def printer(event) -> None:
        if debug_timing and isinstance(event, ToolResultEvent) and event.elapsed_ms is not None:
            state = "ok" if event.ok else "error"
            _print_live_line(f"└ {event.name} {state} · {event.elapsed_ms:.1f}ms")
        if isinstance(event, ToolRouteEvent) and not debug_timing:
            return
        print_live_event(event)

    return printer


def _format_status_warning(warning: str | None) -> str | None:
    if not warning:
        return None
    if warning == "critical: context window nearly exhausted":
        return "context high: consider /compact"
    if warning == "warning: context window getting tight":
        return "context rising: consider /compact soon"
    return warning


def _render_tool_detail(name: str, arguments: dict) -> str:
    if not isinstance(arguments, dict):
        return ""
    if name == "bash":
        command = arguments.get("command")
        if isinstance(command, str) and command.strip():
            return _trim_preview(command)
    for key in ("path", "url", "query", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return f"{name}: {_trim_preview(value)}"
    return ""


def _trim_preview(value: object, limit: int = 120) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_tool_error_lines(event: ToolResultEvent) -> list[str]:
    message = str(event.error or "tool error")
    returncode = event.returncode
    stderr = _trim_preview(event.stderr)
    stdout = _trim_preview(event.stdout)
    detail_parts: list[str] = []
    if isinstance(returncode, int):
        detail_parts.append(f"rc={returncode}")
    if stderr:
        detail_parts.append(f"stderr={stderr}")
    elif stdout:
        detail_parts.append(f"stdout={stdout}")
    label = f"{event.name} error" if event.name else "tool error"
    if detail_parts:
        lines = [f"  {label} ({', '.join(detail_parts)})"]
    else:
        lines = [f"  {label}: {message}"]
    hint = _tool_error_hint(event)
    if hint:
        lines.append(f"  hint: {hint}")
    return lines


def _tool_error_hint(event: ToolResultEvent) -> str | None:
    text = " ".join(
        part
        for part in (
            event.error or "",
            event.stderr or "",
            event.stdout or "",
        )
        if part
    ).lower()
    if "timed out" in text or "timeout" in text:
        if event.name == "bash":
            return "target a smaller path or use a bounded inspection command"
        if event.name == "fetch_url":
            return "retry with a smaller page chunk or a more specific query"
        return "retry with a narrower request"
    if "outside" in text and "workdir" in text:
        return "use a path inside the configured workdir"
    if "read_file" in text and event.name in {"write_file", "append_file", "replace_in_file"}:
        return "read the existing file first, then retry the edit"
    return None


def _extract_repeated_tool_detail(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    marker = "Repeated call:"
    if marker not in value:
        return ""
    tail = value.split(marker, 1)[1].strip()
    if "." in tail:
        tail = tail.split(".", 1)[0].strip()
    return _trim_preview(tail)


def _print_live_line(text: str) -> None:
    with LIVE_OUTPUT_LOCK:
        if sys.stderr.isatty():
            print(f"{STATUS_COLOR}{text}{RESET_COLOR}", file=sys.stderr)
            return
        print(text, file=sys.stderr)


def _print_thinking_start() -> None:
    _print_live_line("└ thinking")


def _print_thinking_chunk(text: str) -> None:
    if not isinstance(text, str) or not text:
        return
    lines = text.splitlines() or [text]
    with LIVE_OUTPUT_LOCK:
        for line in lines:
            if sys.stderr.isatty():
                print(f"{STATUS_COLOR}  {line}{RESET_COLOR}", file=sys.stderr)
            else:
                print(f"  {line}", file=sys.stderr)


def _print_thinking_end() -> None:
    with LIVE_OUTPUT_LOCK:
        print("", file=sys.stderr)

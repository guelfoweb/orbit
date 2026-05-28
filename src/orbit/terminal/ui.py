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
    print("  /exit")
    print("  /help")
    print("  /reset")
    print("  /sessions clear")
    print("  /skill clear (restore default)")
    print("  /skill list")
    print("  /skill show")
    print("  /skill use <ref>")
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
    rate_parts: list[str] = []
    if status.prefill_tps is not None:
        rate_parts.append(f"pf: {status.prefill_tps:.1f}/s")
    if status.decode_tps is not None:
        rate_parts.append(f"gen: {status.decode_tps:.1f}/s")
    show_thinking_state = getattr(status, "show_thinking_state", "off")
    line = (
        f"{status.active_model} | "
        f"ctx: {context_text} | "
        f"msg: {status.session_turns} | "
        f"tk_in: {input_text} | "
        f"tk_out: {output_text}"
    )
    if rate_parts:
        line = f"{line} | {' | '.join(part.replace('pf:', 'tk_pf:').replace('gen:', 'tk_gen') for part in rate_parts)}"
    if not (status.think_state == "off" and show_thinking_state == "off"):
        line = f"{line} | think: {status.think_state} | show-thinking: {show_thinking_state}"
    if status.warning:
        line = f"{line} | {status.warning}"
    if sys.stdout.isatty():
        return f"{STATUS_COLOR}{line}{RESET_COLOR}"
    return line


def format_startup_line(text: str) -> str:
    if sys.stdout.isatty():
        return f"{STATUS_COLOR}{text}{RESET_COLOR}"
    return text


def format_skill(agent: AgentLoop) -> str:
    if agent.skill is None:
        return "skill: -"
    return f"skill: {agent.skill.name} | path: {agent.skill.path}"


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
            _print_live_line(f"└ timing tool:{event.name}: {event.elapsed_ms:.1f}ms")
        print_live_event(event)

    return printer


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
    if detail_parts:
        lines = [f"  error ({', '.join(detail_parts)})"]
    else:
        lines = [f"  error: {message}"]
    return lines


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

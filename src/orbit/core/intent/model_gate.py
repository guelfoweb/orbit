from __future__ import annotations

import json
import time
from typing import Any, Callable

from .gate import intent_gate_messages, parse_intent_gate_reply
from .tool_gate import parse_tool_gate_reply, tool_gate_decision, tool_gate_messages
from ..events import EventSink
from ..tools.router import ToolRoute


UpdateMetrics = Callable[[Any, dict[str, Any]], None]
EmitTiming = Callable[[EventSink | None, str, int, str | None], None]


def model_confirms_tool_route(
    *,
    client: Any,
    user_input: str,
    route: ToolRoute,
    metrics: Any,
    update_metrics: UpdateMetrics,
    emit_timing: EmitTiming,
    on_event: EventSink | None,
) -> bool:
    messages = intent_gate_messages(user_input=user_input, route=route)
    started_at = time.monotonic_ns()
    try:
        response = client.chat(
            messages=messages,
            tools=[],
            options={"temperature": 0.0},
            think=False,
        )
        update_metrics(metrics, response)
    except Exception:
        emit_timing(on_event, "intent-check", started_at, f"{route.intent} -> fail-open")
        return True
    parsed = parse_intent_gate_reply((response.get("message") or {}).get("content"))
    outcome = "unclear, fail-open" if parsed is None else ("YES" if parsed else "NO")
    emit_timing(on_event, "intent-check", started_at, f"{route.intent} -> {outcome}")
    return True if parsed is None else parsed


def model_confirms_tool_call(
    *,
    client: Any,
    user_input: str,
    route: ToolRoute | None,
    tool_name: str,
    arguments: dict[str, Any],
    messages: list[dict[str, Any]],
    metrics: Any,
    update_metrics: UpdateMetrics,
    emit_timing: EmitTiming,
    on_event: EventSink | None,
) -> bool:
    if tool_name == "fetch_url" and fetch_url_matches_recent_search_result(arguments, messages):
        return True
    decision = tool_gate_decision(
        user_input=user_input,
        route=route,
        tool_name=tool_name,
        arguments=arguments,
    )
    if not decision.confirm:
        return True
    if route is None:
        return True
    started_at = time.monotonic_ns()
    gate_messages = tool_gate_messages(
        user_input=user_input,
        route=route,
        tool_name=tool_name,
        arguments=arguments,
        reason=decision.reason,
    )
    try:
        response = client.chat(
            messages=gate_messages,
            tools=[],
            options={"temperature": 0.0},
            think=False,
        )
        update_metrics(metrics, response)
    except Exception:
        emit_timing(on_event, "tool-check", started_at, f"{tool_name} -> fail-open")
        return True
    parsed = parse_tool_gate_reply((response.get("message") or {}).get("content"))
    outcome = "unclear, fail-open" if parsed is None else ("YES" if parsed else "NO")
    emit_timing(on_event, "tool-check", started_at, f"{tool_name} -> {outcome}")
    return True if parsed is None else parsed


def fetch_url_matches_recent_search_result(arguments: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return False
    target = url.strip()
    for message in reversed(messages):
        if message.get("role") == "user":
            return False
        if message.get("role") != "tool" or message.get("tool_name") != "search_web":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for item in results:
            if isinstance(item, dict) and item.get("url") == target:
                return True
    return False

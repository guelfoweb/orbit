from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable


DEFAULT_SESSION_ID = "default"


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    session_id: str
    thinking: bool | None
    stop: tuple[str, ...]
    stream: bool
    tools: list[dict[str, Any]]
    route_prefix_anchor: bool
    allow_mtp_experimental: bool | None
    final_prefix_experiment: bool


@dataclass(frozen=True)
class ContinueRequest:
    max_tokens: int
    thinking: bool | None
    stop: tuple[str, ...]
    stream: bool


def parse_chat_request(payload: dict[str, Any]) -> ChatRequest:
    return ChatRequest(
        messages=_messages_from_payload(payload),
        max_tokens=_int_value(payload.get("max_tokens"), 256),
        temperature=_float_value(payload.get("temperature"), 0.0),
        session_id=_session_id(payload.get("session_id")),
        thinking=_thinking_mode(payload.get("thinking")),
        stop=_stop_sequences(payload.get("stop")),
        stream=payload.get("stream") is True,
        tools=_tools_from_payload(payload.get("tools")),
        route_prefix_anchor=payload.get("route_prefix_anchor") is True,
        allow_mtp_experimental=_optional_bool(payload.get("allow_mtp_experimental")),
        final_prefix_experiment=payload.get("final_prefix_experiment") is True,
    )


def parse_continue_request(payload: dict[str, Any]) -> ContinueRequest:
    return ContinueRequest(
        max_tokens=_int_value(payload.get("max_tokens"), 256),
        thinking=_thinking_mode(payload.get("thinking")),
        stop=_stop_sequences(payload.get("stop")),
        stream=payload.get("stream") is True,
    )


def native_chat_response(
    *,
    content: str,
    model: str,
    finish_reason: str,
    session_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    reused_prompt_tokens: int,
    evaluated_prompt_tokens: int,
    prefill_ms: float,
    generation_ms: float,
    cancelled: bool,
) -> dict[str, Any]:
    return {
        "content": content,
        "model": model,
        "session_id": session_id,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": reused_prompt_tokens,
                "reused_tokens": reused_prompt_tokens,
                "evaluated_tokens": evaluated_prompt_tokens,
            },
        },
        "timings": {
            "prompt_ms": prefill_ms,
            "predicted_ms": generation_ms,
            "prompt_per_second": _rate(evaluated_prompt_tokens, prefill_ms),
            "predicted_per_second": _rate(completion_tokens, generation_ms),
        },
        "native": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "reused_prompt_tokens": reused_prompt_tokens,
            "evaluated_prompt_tokens": evaluated_prompt_tokens,
            "prefill_ms": prefill_ms,
            "generation_ms": generation_ms,
            "cancelled": cancelled,
        },
    }


def openai_chat_response(result: dict[str, Any], *, content: str | None = None) -> dict[str, Any]:
    return {
        "model": result["model"],
        "choices": [
            {
                "message": {"role": "assistant", "content": result["content"] if content is None else content},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
        "timings": result["timings"],
    }


def sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_data(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode("utf-8")
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def validate_session_id(session_id: str) -> None:
    if session_id != DEFAULT_SESSION_ID:
        raise ValueError("only the default native session is supported in this experiment")


def _messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("messages")
    if not isinstance(raw, list):
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        raise ValueError("messages must be a list")
    messages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("messages must contain objects")
        role = item.get("role")
        if not isinstance(role, str):
            raise ValueError("message role must be a string")
        message = dict(item)
        message["role"] = role
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
    if not messages:
        raise ValueError("messages must not be empty")
    return messages


def _tools_from_payload(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _thinking_mode(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _stop_sequences(value: object) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _session_id(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_SESSION_ID


def _int_value(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError("max_tokens must be a positive integer")


def _float_value(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _rate(tokens: int, ms: float) -> float | None:
    if tokens <= 0 or ms <= 0:
        return None
    return tokens / (ms / 1000.0)


def trim_at_stop(content: str, stops: Iterable[str]) -> tuple[str, bool]:
    first: int | None = None
    for stop in stops:
        idx = content.find(stop)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is None:
        return content, False
    return content[:first], True

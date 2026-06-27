from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from itertools import count
from typing import Any, Callable, Iterator

from orbit.backend.base import ChatResult, Message, StreamProgress
from orbit.runtime.session_memory import estimate_message_tokens, estimate_text_tokens


_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar("orbit_kv_diag_phase", default=None)
_TOOLS_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("orbit_kv_diag_tools_mode", default=None)
_REQUEST: contextvars.ContextVar["_RequestState | None"] = contextvars.ContextVar("orbit_kv_diag_request", default=None)
_CALL_IDS = count(1)
_REQUEST_IDS = count(1)
_LAST_BY_SCENARIO: dict[str, dict[str, str | None]] = {}
_LAST_LAYOUT_BY_SCENARIO: dict[str, list[dict[str, Any]]] = {}
_LAST_MODEL_CALL: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptFingerprint:
    prompt_chars: int
    prompt_tokens_estimate: int
    stable_prefix_chars: int
    stable_prefix_hash: str
    full_prompt_hash: str
    first_user_message_hash: str | None
    tool_schema_hash: str
    capability_summary_hash: str | None
    runtime_policy_hash: str | None
    conversation_prefix_hash: str
    prompt_layout_hash: str
    prompt_layout: list[dict[str, Any]]


@dataclass
class _RequestState:
    session_id_hash: str
    request_id: str
    started: float
    pass_index: int = 0
    model_calls: list[dict[str, Any]] = field(default_factory=list)


def enabled() -> bool:
    value = os.environ.get("ORBIT_KV_DIAG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def reset_diagnostics_for_tests() -> None:
    global _CALL_IDS, _REQUEST_IDS, _LAST_MODEL_CALL
    _CALL_IDS = count(1)
    _REQUEST_IDS = count(1)
    _LAST_BY_SCENARIO.clear()
    _LAST_LAYOUT_BY_SCENARIO.clear()
    _LAST_MODEL_CALL = None


def current_tools_mode() -> str | None:
    return _TOOLS_MODE.get()


@contextlib.contextmanager
def request_context(*, session_id: str | None = None) -> Iterator[None]:
    if not enabled() or _REQUEST.get() is not None:
        yield
        return
    state = _RequestState(
        session_id_hash=_hash(session_id or "default"),
        request_id=f"req_{next(_REQUEST_IDS):06d}",
        started=time.monotonic(),
    )
    token = _REQUEST.set(state)
    try:
        yield
    finally:
        _emit_request_summary(state)
        _REQUEST.reset(token)


def user_request(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with request_context(session_id=getattr(self, "diagnostic_session_id", "default")):
            return method(self, *args, **kwargs)

    return wrapped


@contextlib.contextmanager
def model_call_context(*, phase: str, tools_mode: str | None = None) -> Iterator[None]:
    phase_token = _PHASE.set(phase)
    tools_token = _TOOLS_MODE.set(tools_mode) if tools_mode is not None else None
    try:
        yield
    finally:
        _PHASE.reset(phase_token)
        if tools_token is not None:
            _TOOLS_MODE.reset(tools_token)


def instrument_backend(backend: Any) -> Any:
    if not enabled():
        return backend
    if getattr(backend, "_orbit_kv_diag_wrapped", False):
        return backend
    return _DiagnosticBackend(backend)


def fingerprint_prompt(messages: list[Message], tools: list[dict[str, Any]] | None = None) -> PromptFingerprint:
    rendered_messages = _canonical_json(messages)
    tool_schema = _canonical_json(tools or [])
    prompt_layout = _prompt_layout(messages, tools or [])
    runtime_policy = _runtime_policy(messages)
    capability_summary = _capability_summary(messages)
    stable_components = {
        "runtime_policy": runtime_policy,
        "tool_schema": tool_schema,
        "capability_summary": capability_summary,
    }
    first_user = _first_user_message(messages)
    conversation_prefix = messages[:-1] if messages else []
    return PromptFingerprint(
        prompt_chars=len(rendered_messages) + len(tool_schema),
        prompt_tokens_estimate=estimate_message_tokens(messages),
        stable_prefix_chars=sum(len(value or "") for value in stable_components.values()),
        stable_prefix_hash=_hash(stable_components),
        full_prompt_hash=_hash({"messages": messages, "tools": tools or []}),
        first_user_message_hash=_hash(first_user) if first_user is not None else None,
        tool_schema_hash=_hash(tool_schema),
        capability_summary_hash=_hash(capability_summary) if capability_summary is not None else None,
        runtime_policy_hash=_hash(runtime_policy) if runtime_policy is not None else None,
        conversation_prefix_hash=_hash(conversation_prefix),
        prompt_layout_hash=_hash(_layout_identity(prompt_layout)),
        prompt_layout=prompt_layout,
    )


class _DiagnosticBackend:
    _orbit_kv_diag_wrapped = True

    def __init__(self, backend: Any) -> None:
        object.__setattr__(self, "_backend", backend)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_backend":
            object.__setattr__(self, name, value)
            return
        setattr(self._backend, name, value)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        return self._record_call(
            messages,
            tools=tools,
            streamed=False,
            invoke=lambda: self._backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools),
        )

    def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        timings = _ProgressTimings()

        def wrapped_progress(progress: StreamProgress) -> None:
            timings.observe(progress)
            if on_progress is not None:
                on_progress(progress)

        return self._record_call(
            messages,
            tools=tools,
            streamed=True,
            progress_timings=timings,
            invoke=lambda: self._backend.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                on_delta=on_delta,
                on_progress=wrapped_progress if on_progress is not None else None,
            ),
        )

    def continue_current(self, *args: Any, **kwargs: Any) -> ChatResult:
        call_context = _next_call_context(_PHASE.get() or "continue", _TOOLS_MODE.get())
        started = time.monotonic()
        result = self._backend.continue_current(*args, **kwargs)
        event = {
            "event": "kv_diag_model_call",
            **call_context,
            "phase": call_context["phase"],
            "tools_mode": call_context["tools_mode"],
            "streamed": kwargs.get("on_delta") is not None,
            **_empty_prompt_fingerprint_fields(),
            "wall_ms": _elapsed_ms(started),
            **_result_metrics(result),
        }
        _record_model_call(event)
        _emit(event)
        return result

    def _record_call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        streamed: bool,
        invoke: Callable[[], ChatResult],
        progress_timings: "_ProgressTimings | None" = None,
    ) -> ChatResult:
        phase = _PHASE.get() or "unknown"
        tools_mode = _TOOLS_MODE.get() or ("on" if tools else "off")
        call_context = _next_call_context(phase, tools_mode)
        fingerprint = fingerprint_prompt(messages, tools)
        started = time.monotonic()
        result = invoke()
        components = _component_changes(call_context["request_id"], phase, tools_mode, fingerprint)
        layout_common_prefix = _layout_common_prefix(call_context["request_id"], phase, tools_mode, fingerprint)
        event = {
            "event": "kv_diag_model_call",
            **call_context,
            "streamed": streamed,
            "prompt_chars": fingerprint.prompt_chars,
            "prompt_tokens_estimate": fingerprint.prompt_tokens_estimate,
            "stable_prefix_chars": fingerprint.stable_prefix_chars,
            "stable_prefix_hash": fingerprint.stable_prefix_hash,
            "full_prompt_hash": fingerprint.full_prompt_hash,
            "first_user_message_hash": fingerprint.first_user_message_hash,
            "tool_schema_hash": fingerprint.tool_schema_hash,
            "capability_summary_hash": fingerprint.capability_summary_hash,
            "runtime_policy_hash": fingerprint.runtime_policy_hash,
            "conversation_prefix_hash": fingerprint.conversation_prefix_hash,
            "prompt_layout_hash": fingerprint.prompt_layout_hash,
            "prompt_layout_order": [block["component"] for block in fingerprint.prompt_layout],
            "prompt_layout": fingerprint.prompt_layout,
            "prompt_layout_common_prefix": layout_common_prefix,
            "changed_components": components,
            "wall_ms": _elapsed_ms(started),
            "prefill_ms": progress_timings.prefill_ms(started) if progress_timings is not None else None,
            "generation_ms": progress_timings.generation_ms() if progress_timings is not None else None,
            **_result_metrics(result),
        }
        _record_model_call(event)
        _emit(event)
        return result


class _ProgressTimings:
    def __init__(self) -> None:
        self.first_prefill_at: float | None = None
        self.first_generation_at: float | None = None
        self.last_generation_at: float | None = None

    def observe(self, progress: StreamProgress) -> None:
        now = time.monotonic()
        if progress.phase == "prefill" and self.first_prefill_at is None:
            self.first_prefill_at = now
        if progress.phase == "generation":
            if self.first_generation_at is None:
                self.first_generation_at = now
            self.last_generation_at = now

    def prefill_ms(self, started: float) -> int | None:
        if self.first_generation_at is None:
            return None
        return int((self.first_generation_at - started) * 1000)

    def generation_ms(self) -> int | None:
        if self.first_generation_at is None or self.last_generation_at is None:
            return None
        return int((self.last_generation_at - self.first_generation_at) * 1000)


def _result_metrics(result: ChatResult) -> dict[str, Any]:
    prompt_tokens = result.prompt_tokens
    cached_tokens = result.cached_tokens
    evaluated_tokens = None
    if prompt_tokens is not None and cached_tokens is not None:
        evaluated_tokens = max(0, prompt_tokens - cached_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "evaluated_tokens": evaluated_tokens,
        "reused_tokens": cached_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": result.completion_tokens,
        "prefill_tps": result.prompt_tokens_per_second,
        "generation_tps": result.generation_tokens_per_second,
        "finish_reason": result.finish_reason,
    }


def emit_footer_metrics(
    result: ChatResult,
    *,
    elapsed_seconds: float | None = None,
    estimated_context_tokens: int | None = None,
    context_tokens: int | None = None,
) -> None:
    if not enabled() or _LAST_MODEL_CALL is None:
        return
    prompt_tokens = result.prompt_tokens
    cache_percent = None
    if prompt_tokens and result.cached_tokens is not None:
        cache_percent = (result.cached_tokens / prompt_tokens) * 100
    _emit(
        {
            "event": "kv_diag_footer_metrics",
            "session_id_hash": _LAST_MODEL_CALL.get("session_id_hash"),
            "request_id": _LAST_MODEL_CALL.get("request_id"),
            "model_call_id": _LAST_MODEL_CALL.get("model_call_id"),
            "call_id": _LAST_MODEL_CALL.get("call_id"),
            "pass_index": _LAST_MODEL_CALL.get("pass_index"),
            "phase": _LAST_MODEL_CALL.get("phase"),
            "footer": {
                "model": result.model,
                "ctx_used": estimated_context_tokens,
                "ctx_total": context_tokens,
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
                "cached_tokens": result.cached_tokens,
                "cache_percent": cache_percent,
                "prefill_tok_s": result.prompt_tokens_per_second,
                "generation_tok_s": result.generation_tokens_per_second,
                "finish_reason": result.finish_reason,
                "wall_ms": int(elapsed_seconds * 1000) if elapsed_seconds is not None else None,
            },
        }
    )


def emit_route_outcome(
    *,
    outcome: str,
    finish_reason: str | None,
    decision_type: str | None,
    output_chars: int | None,
    output_tokens: int | None,
    retry_reason: str | None,
) -> None:
    if not enabled() or _LAST_MODEL_CALL is None:
        return
    _emit(
        {
            "event": "kv_diag_route_outcome",
            "session_id_hash": _LAST_MODEL_CALL.get("session_id_hash"),
            "request_id": _LAST_MODEL_CALL.get("request_id"),
            "model_call_id": _LAST_MODEL_CALL.get("model_call_id"),
            "call_id": _LAST_MODEL_CALL.get("call_id"),
            "pass_index": _LAST_MODEL_CALL.get("pass_index"),
            "phase": _LAST_MODEL_CALL.get("phase"),
            "tools_mode": _LAST_MODEL_CALL.get("tools_mode"),
            "finish_reason": finish_reason,
            "decision_type": decision_type,
            "output_chars": output_chars,
            "output_tokens": output_tokens,
            "retry_reason": retry_reason,
            "outcome": outcome,
        }
    )


def _next_call_context(phase: str, tools_mode: str | None) -> dict[str, Any]:
    request = _REQUEST.get()
    call_id = next(_CALL_IDS)
    model_call_id = f"mc_{call_id:06d}"
    if request is None:
        return {
            "session_id_hash": None,
            "request_id": None,
            "model_call_id": model_call_id,
            "call_id": call_id,
            "pass_index": None,
            "phase": phase,
            "tools_mode": tools_mode,
        }
    request.pass_index += 1
    return {
        "session_id_hash": request.session_id_hash,
        "request_id": request.request_id,
        "model_call_id": model_call_id,
        "call_id": call_id,
        "pass_index": request.pass_index,
        "phase": phase,
        "tools_mode": tools_mode,
    }


def _record_model_call(event: dict[str, Any]) -> None:
    global _LAST_MODEL_CALL
    _LAST_MODEL_CALL = {
        key: event.get(key)
        for key in ("session_id_hash", "request_id", "model_call_id", "call_id", "pass_index", "phase", "tools_mode")
    }
    request = _REQUEST.get()
    if request is not None:
        request.model_calls.append(event)


def _empty_prompt_fingerprint_fields() -> dict[str, Any]:
    return {
        "prompt_chars": None,
        "prompt_tokens_estimate": None,
        "stable_prefix_chars": None,
        "stable_prefix_hash": None,
        "full_prompt_hash": None,
        "first_user_message_hash": None,
        "tool_schema_hash": None,
        "capability_summary_hash": None,
        "runtime_policy_hash": None,
        "conversation_prefix_hash": None,
        "prompt_layout_hash": None,
        "prompt_layout_order": [],
        "prompt_layout": [],
        "prompt_layout_common_prefix": None,
        "changed_components": [],
    }


def _emit_request_summary(state: _RequestState) -> None:
    calls = state.model_calls
    _emit(
        {
            "event": "kv_diag_request_summary",
            "session_id_hash": state.session_id_hash,
            "request_id": state.request_id,
            "model_calls": len(calls),
            "phases": [call.get("phase") for call in calls],
            "total_prompt_tokens": _sum_optional(call.get("prompt_tokens") for call in calls),
            "total_cached_tokens": _sum_optional(call.get("cached_tokens") for call in calls),
            "total_evaluated_tokens": _sum_optional(call.get("evaluated_tokens") for call in calls),
            "total_prefill_ms": _sum_optional(call.get("prefill_ms") for call in calls),
            "total_generation_ms": _sum_optional(call.get("generation_ms") for call in calls),
            "wall_ms": _elapsed_ms(state.started),
            "finish_reasons": [call.get("finish_reason") for call in calls],
        }
    )


def _sum_optional(values: Iterator[Any]) -> int | None:
    total = 0
    seen = False
    for value in values:
        if isinstance(value, (int, float)):
            total += int(value)
            seen = True
    return total if seen else None


def _component_changes(request_id: str | None, phase: str, tools_mode: str | None, fingerprint: PromptFingerprint) -> list[str]:
    scenario = f"{phase}:{tools_mode or 'unknown'}:{fingerprint.first_user_message_hash or 'no_user'}"
    current = {
        "stable_prefix": fingerprint.stable_prefix_hash,
        "tool_schema": fingerprint.tool_schema_hash,
        "capability_summary": fingerprint.capability_summary_hash,
        "runtime_policy": fingerprint.runtime_policy_hash,
        "conversation_prefix": fingerprint.conversation_prefix_hash,
        "full_prompt": fingerprint.full_prompt_hash,
    }
    previous = _LAST_BY_SCENARIO.get(scenario)
    _LAST_BY_SCENARIO[scenario] = current
    if previous is None:
        return []
    changed = [name for name, value in current.items() if previous.get(name) != value]
    for name in changed:
        _emit(
            {
                "event": "kv_diag_prefix_mismatch",
                "request_id": request_id,
                "component": name,
                "previous_hash": previous.get(name),
                "current_hash": current.get(name),
            }
        )
    return changed


def _layout_common_prefix(
    request_id: str | None,
    phase: str,
    tools_mode: str | None,
    fingerprint: PromptFingerprint,
) -> dict[str, Any]:
    scenario = f"{phase}:{tools_mode or 'unknown'}:{fingerprint.first_user_message_hash or 'no_user'}"
    current = fingerprint.prompt_layout
    previous = _LAST_LAYOUT_BY_SCENARIO.get(scenario)
    _LAST_LAYOUT_BY_SCENARIO[scenario] = current
    if previous is None:
        return {
            "previous_seen": False,
            "common_blocks": 0,
            "common_chars_estimate": 0,
            "common_tokens_estimate": 0,
            "common_prefix_hash": None,
            "first_divergence_component": None,
            "previous_first_divergence_component": None,
        }
    common = 0
    for previous_block, current_block in zip(previous, current):
        if previous_block.get("hash") != current_block.get("hash") or previous_block.get("component") != current_block.get("component"):
            break
        common += 1
    common_blocks = current[:common]
    current_divergence = current[common].get("component") if common < len(current) else None
    previous_divergence = previous[common].get("component") if common < len(previous) else None
    event = {
        "previous_seen": True,
        "common_blocks": common,
        "common_chars_estimate": sum(int(block.get("chars", 0)) for block in common_blocks),
        "common_tokens_estimate": sum(int(block.get("tokens_estimate", 0)) for block in common_blocks),
        "common_prefix_hash": _hash(_layout_identity(common_blocks)) if common_blocks else None,
        "first_divergence_component": current_divergence,
        "previous_first_divergence_component": previous_divergence,
    }
    if current_divergence or previous_divergence:
        _emit(
            {
                "event": "kv_diag_prompt_layout_mismatch",
                "request_id": request_id,
                "phase": phase,
                "tools_mode": tools_mode,
                **event,
            }
        )
    return event


def _prompt_layout(messages: list[Message], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    char_offset = 0
    token_offset = 0
    for index, message in enumerate(messages):
        component = _message_component(message, index=index)
        serialized = _canonical_json(message)
        tokens = estimate_message_tokens([message])
        block = _layout_block(
            index=len(blocks),
            component=component,
            source="messages",
            role=str(message.get("role") or "unknown"),
            serialized=serialized,
            token_estimate=tokens,
            char_offset=char_offset,
            token_offset=token_offset,
        )
        blocks.append(block)
        char_offset = block["end_char_estimate"]
        token_offset = block["end_token_estimate"]
    if tools:
        serialized = _canonical_json(tools)
        tokens = estimate_text_tokens(serialized)
        block = _layout_block(
            index=len(blocks),
            component="tool_schema_parameter",
            source="tools_parameter",
            role=None,
            serialized=serialized,
            token_estimate=tokens,
            char_offset=char_offset,
            token_offset=token_offset,
        )
        block["tool_count"] = len(tools)
        blocks.append(block)
    return blocks


def _layout_block(
    *,
    index: int,
    component: str,
    source: str,
    role: str | None,
    serialized: str,
    token_estimate: int,
    char_offset: int,
    token_offset: int,
) -> dict[str, Any]:
    chars = len(serialized)
    return {
        "index": index,
        "component": component,
        "source": source,
        "role": role,
        "hash": _hash(serialized),
        "chars": chars,
        "tokens_estimate": token_estimate,
        "start_char_estimate": char_offset,
        "end_char_estimate": char_offset + chars,
        "start_token_estimate": token_offset,
        "end_token_estimate": token_offset + token_estimate,
    }


def _message_component(message: Message, *, index: int) -> str:
    role = str(message.get("role") or "unknown")
    content = message.get("content")
    if role == "system" and isinstance(content, str):
        if content.startswith("Local tools available:"):
            return "capability_summary"
        return "runtime_policy" if index == 0 else "system_instruction"
    if role == "user":
        return "user_message"
    if role == "assistant":
        return "assistant_history"
    if role == "tool":
        return "tool_result"
    return f"{role}_message"


def _layout_identity(layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "component": block.get("component"),
            "source": block.get("source"),
            "role": block.get("role"),
            "hash": block.get("hash"),
        }
        for block in layout
    ]


def _emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path = os.environ.get("ORBIT_KV_DIAG_FILE")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return
    print(line, file=sys.stderr, flush=True)


def _runtime_policy(messages: list[Message]) -> str | None:
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and not content.startswith("Local tools available:"):
                return content
    return None


def _capability_summary(messages: list[Message]) -> str | None:
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.startswith("Local tools available:"):
            return content
    return None


def _first_user_message(messages: list[Message]) -> object | None:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content")
    return None


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _hash(value: object) -> str:
    text = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)

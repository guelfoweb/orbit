from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Iterator

from orbit.backend.base import ChatResult, Message, StreamProgress
from orbit.runtime.session_memory import estimate_message_tokens


_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar("orbit_kv_diag_phase", default=None)
_TOOLS_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("orbit_kv_diag_tools_mode", default=None)
_CALL_IDS = count(1)
_LAST_BY_SCENARIO: dict[str, dict[str, str | None]] = {}


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


def enabled() -> bool:
    value = os.environ.get("ORBIT_KV_DIAG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def reset_diagnostics_for_tests() -> None:
    global _CALL_IDS
    _CALL_IDS = count(1)
    _LAST_BY_SCENARIO.clear()


def current_tools_mode() -> str | None:
    return _TOOLS_MODE.get()


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
        started = time.monotonic()
        result = self._backend.continue_current(*args, **kwargs)
        _emit(
            {
                "event": "kv_diag_model_call",
                "call_id": next(_CALL_IDS),
                "phase": _PHASE.get() or "continue",
                "tools_mode": _TOOLS_MODE.get(),
                "streamed": kwargs.get("on_delta") is not None,
                "wall_ms": _elapsed_ms(started),
                **_result_metrics(result),
            }
        )
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
        call_id = next(_CALL_IDS)
        phase = _PHASE.get() or "unknown"
        tools_mode = _TOOLS_MODE.get() or ("on" if tools else "off")
        fingerprint = fingerprint_prompt(messages, tools)
        started = time.monotonic()
        result = invoke()
        components = _component_changes(phase, tools_mode, fingerprint)
        _emit(
            {
                "event": "kv_diag_model_call",
                "call_id": call_id,
                "phase": phase,
                "tools_mode": tools_mode,
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
                "changed_components": components,
                "wall_ms": _elapsed_ms(started),
                "prefill_ms": progress_timings.prefill_ms(started) if progress_timings is not None else None,
                "generation_ms": progress_timings.generation_ms() if progress_timings is not None else None,
                **_result_metrics(result),
            }
        )
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


def _component_changes(phase: str, tools_mode: str | None, fingerprint: PromptFingerprint) -> list[str]:
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
    return [name for name, value in current.items() if previous.get(name) != value]


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

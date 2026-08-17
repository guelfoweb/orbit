from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from orbit.backend.base import ChatResult, Message, StreamProgress, TokenCount


EVIDENCE_REF_MARKER = "tool_evidence_ref: true"
DEFAULT_NEXT_ACTION_RESERVE = 256
DEFAULT_SAFETY_MARGIN = 256


class ContextAdmissionError(ValueError):
    """The exact model input cannot be admitted without losing required context."""


@dataclass(frozen=True)
class ContextBudget:
    context_tokens: int
    output_reserve: int
    next_action_reserve: int = DEFAULT_NEXT_ACTION_RESERVE
    safety_margin: int = DEFAULT_SAFETY_MARGIN

    @property
    def input_limit(self) -> int:
        return max(
            0,
            self.context_tokens
            - self.output_reserve
            - self.next_action_reserve
            - self.safety_margin,
        )


@dataclass(frozen=True)
class ContextPlan:
    status: str
    messages: tuple[Message, ...]
    input_limit: int
    tokens_before: int | None
    tokens_after: int | None
    compacted_turns: int = 0
    externalized_evidence_ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def admitted(self) -> bool:
        return self.status in {"unchanged", "compacted"}


@dataclass(frozen=True)
class _Turn:
    start: int
    end: int
    final_index: int | None
    evidence_ids: tuple[str, ...]
    contains_embedded_system: bool = False


def plan_context(
    messages: list[Message],
    *,
    budget: ContextBudget,
    available_evidence_ids: Iterable[str] = (),
    covered_evidence_ids: Iterable[str] = (),
    count_tokens: Callable[[list[Message]], int],
) -> ContextPlan:
    """Build a deterministic prompt view without semantic summarization.

    Only completed tool-bearing turns whose exact evidence is both externally
    available and covered by a completed visible answer may have tool-result
    content replaced by a deterministic exact-evidence reference. The complete
    user/assistant/tool role structure, ordinary conversation, and the active
    turn remain intact. If those reductions are insufficient, admission fails
    closed.
    """

    source = [dict(message) for message in messages]
    if not _budget_is_valid(budget):
        return _blocked(source, budget, None, "invalid-context-reserve")
    try:
        before = _count(count_tokens, source)
    except (TypeError, ValueError):
        return _blocked(source, budget, None, "exact-token-count-unavailable")
    if budget.input_limit <= 0:
        return _blocked(source, budget, before, "invalid-or-exhausted-reserve")
    try:
        turns = _parse_turns(source)
    except ValueError as exc:
        return _blocked(source, budget, before, f"invalid-message-structure:{exc}")
    if before <= budget.input_limit:
        return ContextPlan(
            status="unchanged",
            messages=tuple(source),
            input_limit=budget.input_limit,
            tokens_before=before,
            tokens_after=before,
        )

    available = frozenset(available_evidence_ids)
    covered = frozenset(covered_evidence_ids)
    selected: set[int] = set()
    omitted_ids: list[str] = []
    projected = source
    after = before
    for turn_index, turn in enumerate(turns):
        if not _eligible_tool_turn(turn, available=available, covered=covered):
            continue
        selected.add(turn_index)
        omitted_ids.extend(turn.evidence_ids)
        projected = _project(source, turns, selected)
        try:
            after = _count(count_tokens, projected)
        except (TypeError, ValueError):
            return ContextPlan(
                status="blocked",
                messages=tuple(projected),
                input_limit=budget.input_limit,
                tokens_before=before,
                tokens_after=None,
                compacted_turns=len(selected),
                externalized_evidence_ids=tuple(omitted_ids),
                reason="exact-token-count-unavailable",
            )
        if after <= budget.input_limit:
            return ContextPlan(
                status="compacted",
                messages=tuple(projected),
                input_limit=budget.input_limit,
                tokens_before=before,
                tokens_after=after,
                compacted_turns=len(selected),
                externalized_evidence_ids=tuple(omitted_ids),
            )

    return ContextPlan(
        status="blocked",
        messages=tuple(projected),
        input_limit=budget.input_limit,
        tokens_before=before,
        tokens_after=after,
        compacted_turns=len(selected),
        externalized_evidence_ids=tuple(omitted_ids),
        reason="required-context-does-not-fit",
    )


def plan_exact_context(
    messages: list[Message],
    *,
    backend: object,
    output_reserve: int,
    next_action_reserve: int,
    configured_context_tokens: int | None,
    tools: list[dict[str, Any]] | None,
    thinking: bool,
    available_evidence_ids: Iterable[str] = (),
    covered_evidence_ids: Iterable[str] = (),
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    count_chat_override: Callable[..., object] | None = None,
) -> ContextPlan:
    """Plan admission from two attested renders of the actual backend input."""

    source = [dict(message) for message in messages]
    count_chat = count_chat_override if count_chat_override is not None else getattr(backend, "count_chat_tokens", None)
    if not callable(count_chat):
        return _blocked_without_budget(source, "exact-token-count-unavailable")
    first = _safe_count_chat(count_chat, source, tools=tools, thinking=thinking)
    second = _safe_count_chat(count_chat, source, tools=tools, thinking=thinking)
    if not _same_exact_count(first, second):
        return _blocked_without_budget(source, "tokenizer-template-or-context-changed")
    assert first is not None and first.context_tokens is not None
    active_context = first.context_tokens
    if isinstance(configured_context_tokens, int) and configured_context_tokens > 0:
        active_context = min(active_context, configured_context_tokens)
    budget = ContextBudget(
        context_tokens=active_context,
        output_reserve=output_reserve,
        next_action_reserve=next_action_reserve,
        safety_margin=safety_margin,
    )
    source_key = _message_identity(source)
    exact_cache: dict[str, int] = {source_key: first.tokens}

    def exact_counter(candidate: list[Message]) -> int:
        key = _message_identity(candidate)
        cached = exact_cache.get(key)
        if cached is not None:
            return cached
        candidate_first = _safe_count_chat(count_chat, candidate, tools=tools, thinking=thinking)
        candidate_second = _safe_count_chat(count_chat, candidate, tools=tools, thinking=thinking)
        if not _same_exact_count(candidate_first, candidate_second):
            raise ValueError("exact token identity unavailable")
        assert candidate_first is not None
        if candidate_first.context_tokens != first.context_tokens:
            raise ValueError("active context changed")
        exact_cache[key] = candidate_first.tokens
        return candidate_first.tokens

    return plan_context(
        source,
        budget=budget,
        available_evidence_ids=available_evidence_ids,
        covered_evidence_ids=covered_evidence_ids,
        count_tokens=exact_counter,
    )


class ContextManagedBackend:
    """Runtime-side admission wrapper; the wrapped backend still owns inference."""

    def __init__(
        self,
        backend: object,
        prepare: Callable[[list[Message], int, list[dict[str, Any]] | None, bool, bool], list[Message]],
    ) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_prepare", prepare)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_backend", "_prepare"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._backend, name, value)

    def _messages(
        self,
        messages: list[Message],
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        *,
        artifact_content: bool = False,
    ) -> list[Message]:
        thinking = False if artifact_content else bool(getattr(self._backend, "thinking", False))
        return self._prepare(messages, max_tokens, tools, thinking, artifact_content)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        prepared = self._messages(messages, max_tokens, tools)
        return self._backend.chat(prepared, temperature=temperature, max_tokens=max_tokens, tools=tools)

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
        prepared = self._messages(messages, max_tokens, tools)
        return self._backend.chat_stream(
            prepared,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            on_delta=on_delta,
            on_progress=on_progress,
        )

    def artifact_content_stream(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        on_delta: Callable[[str], None],
        on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> ChatResult:
        generate = getattr(self._backend, "artifact_content_stream", None)
        if not callable(generate):
            raise RuntimeError("artifact content generation requires the native Orbit backend")
        prepared = self._messages(messages, max_tokens, None, artifact_content=True)
        return generate(
            prepared,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_delta,
            on_progress=on_progress,
        )


def _parse_turns(messages: list[Message]) -> list[_Turn]:
    prefix_done = False
    starts: list[int] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in {"system", "developer"} and not prefix_done:
            continue
        prefix_done = True
        if role == "user":
            starts.append(index)
        elif role in {"system", "developer"}:
            continue
        elif role not in {"assistant", "tool"}:
            raise ValueError(f"unsupported-role-{role}")
        elif not starts:
            raise ValueError("history-does-not-start-with-user")
    if not starts:
        return []

    turns: list[_Turn] = []
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(messages)
        pending: list[str] = []
        final_index: int | None = None
        terminal_seen = False
        evidence_ids: list[str] = []
        for index in range(start + 1, end):
            message = messages[index]
            role = message.get("role")
            if role in {"system", "developer"}:
                continue
            if role == "assistant":
                if terminal_seen:
                    raise ValueError("message-after-terminal-assistant")
                if pending:
                    raise ValueError("assistant-before-tool-results")
                calls = message.get("tool_calls")
                if calls is None:
                    terminal_seen = True
                    content = message.get("content")
                    final_index = index if isinstance(content, str) and content.strip() else None
                    continue
                if not isinstance(calls, list) or not calls:
                    raise ValueError("invalid-tool-call-list")
                pending = [_tool_call_id(call) for call in calls]
                final_index = None
                continue
            if role != "tool":
                raise ValueError(f"unsupported-role-{role}")
            if terminal_seen:
                raise ValueError("message-after-terminal-assistant")
            if not pending:
                raise ValueError("orphan-tool-result")
            tool_call_id = message.get("tool_call_id")
            if tool_call_id != pending[0]:
                raise ValueError("tool-result-order-or-id-mismatch")
            pending.pop(0)
            evidence_id = message.get("evidence_id")
            content = message.get("content")
            if not isinstance(evidence_id, str) or not evidence_id:
                evidence_ids.clear()
            elif _is_externalized_evidence_ref(content, evidence_id):
                evidence_ids.append(evidence_id)
            else:
                evidence_ids.clear()
        if pending:
            raise ValueError("missing-tool-result")
        tool_count = sum(1 for message in messages[start:end] if message.get("role") == "tool")
        turns.append(
            _Turn(
                start=start,
                end=end,
                final_index=final_index,
                evidence_ids=tuple(evidence_ids) if len(evidence_ids) == tool_count else (),
                contains_embedded_system=any(
                    message.get("role") in {"system", "developer"}
                    for message in messages[start + 1 : end]
                ),
            )
        )
    return turns


def _tool_call_id(call: object) -> str:
    if not isinstance(call, dict):
        raise ValueError("invalid-tool-call")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("missing-tool-call-id")
    return call_id


def _count(counter: Callable[[list[Message]], int], messages: list[Message]) -> int:
    value = counter(messages)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid exact token count")
    return value


def _budget_is_valid(budget: ContextBudget) -> bool:
    values = (
        budget.context_tokens,
        budget.output_reserve,
        budget.next_action_reserve,
        budget.safety_margin,
    )
    return (
        all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        and budget.context_tokens > 0
        and budget.output_reserve >= 0
        and budget.next_action_reserve >= 0
        and budget.safety_margin >= 0
    )


def _is_externalized_evidence_ref(content: object, evidence_id: str) -> bool:
    if not isinstance(content, str) or not content.startswith(f"{EVIDENCE_REF_MARKER}\n"):
        return False
    return f"evidence_id: {evidence_id}" in content.splitlines()[1:]


def _eligible_tool_turn(
    turn: _Turn,
    *,
    available: frozenset[str],
    covered: frozenset[str],
) -> bool:
    if turn.final_index is None or not turn.evidence_ids or turn.contains_embedded_system:
        return False
    evidence = frozenset(turn.evidence_ids)
    return evidence.issubset(available) and evidence.issubset(covered)


def _project(messages: list[Message], turns: list[_Turn], selected: set[int]) -> list[Message]:
    first_turn = turns[0].start if turns else len(messages)
    projected = [dict(message) for message in messages[:first_turn]]
    for index, turn in enumerate(turns):
        if index not in selected:
            projected.extend(dict(message) for message in messages[turn.start : turn.end])
            continue
        for message in messages[turn.start : turn.end]:
            copied = dict(message)
            if copied.get("role") == "tool":
                evidence_id = copied.get("evidence_id")
                assert isinstance(evidence_id, str)
                copied["content"] = _archive_tool_reference(evidence_id)
            projected.append(copied)
    return projected


def _blocked(messages: list[Message], budget: ContextBudget, tokens: int | None, reason: str) -> ContextPlan:
    return ContextPlan(
        status="blocked",
        messages=tuple(messages),
        input_limit=budget.input_limit,
        tokens_before=tokens,
        tokens_after=tokens,
        reason=reason,
    )


def _blocked_without_budget(messages: list[Message], reason: str) -> ContextPlan:
    return ContextPlan(
        status="blocked",
        messages=tuple(messages),
        input_limit=0,
        tokens_before=None,
        tokens_after=None,
        reason=reason,
    )


def _archive_tool_reference(evidence_id: str) -> str:
    return "\n".join(
        (
            "tool_evidence_ref: true",
            "archived: true",
            f"evidence_id: {evidence_id}",
            f"exact_content_ref: evidence:{evidence_id}",
        )
    )


def _safe_count_chat(
    counter: Callable[..., object],
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None,
    thinking: bool,
) -> TokenCount | None:
    try:
        value = counter(messages, tools=tools, thinking=thinking)
    except Exception:
        return None
    return value if isinstance(value, TokenCount) else None


def _exact_count(value: TokenCount | None) -> bool:
    return (
        value is not None
        and isinstance(value.tokens, int)
        and value.tokens >= 0
        and isinstance(value.context_tokens, int)
        and value.context_tokens > 0
        and isinstance(value.rendered_hash, str)
        and len(value.rendered_hash) == 64
        and isinstance(value.token_hash, str)
        and len(value.token_hash) == 64
    )


def _same_exact_count(first: TokenCount | None, second: TokenCount | None) -> bool:
    return _exact_count(first) and _exact_count(second) and first == second


def _message_identity(messages: list[Message]) -> str:
    import hashlib
    import json

    rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

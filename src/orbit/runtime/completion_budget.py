from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MAX_TOKENS = 256

ROUTE_MAX_TOKENS = 64
TOOL_CALL_MAX_TOKENS = 96
FILE_RECOVERY_TOOL_CALL_MAX_TOKENS = 64
CHAT_DEFAULT_MAX_TOKENS = 192
CHAT_MAX_TOKENS = 256

FINAL_SMALL_MAX_TOKENS = 96
FINAL_SHELL_ERROR_MAX_TOKENS = 128
FINAL_SYSTEM_INFO_MAX_TOKENS = 160
FINAL_MEDIUM_MAX_TOKENS = 192
FINAL_WEB_SEARCH_MAX_TOKENS = 192
FINAL_READ_MAX_TOKENS = 256

CHAT_FINAL_RETRY_MAX_TOKENS = 128
CHAT_FINAL_RETRY_AFTER_LENGTH_MAX_TOKENS = 192
FINAL_REPAIR_MAX_TOKENS = 128
FINAL_REPAIR_AFTER_LENGTH_MAX_TOKENS = 192
FINAL_REPAIR_CAP_MAX_TOKENS = 160


@dataclass(frozen=True)
class CompletionBudget:
    requested_max_tokens: int

    def internal(self, cap: int) -> int:
        return max(1, min(self.requested_max_tokens, cap))

    def user_visible(self) -> int:
        return max(1, self.requested_max_tokens)


def resolve_max_tokens(
    completion_kind: str,
    requested_max_tokens: int | None = None,
    evidence_kind: str | None = None,
    evidence_chars: int | None = None,
    previous_finish_reason: str | None = None,
) -> int:
    """Resolve bounded output budgets from structural runtime state only."""

    requested = _positive_or_none(requested_max_tokens)
    kind = completion_kind.strip().lower()
    evidence = (evidence_kind or "").strip().lower()
    previous = (previous_finish_reason or "").strip().lower()

    if kind == "route":
        return _internal(requested, ROUTE_MAX_TOKENS)
    if kind == "tool_call":
        return _internal(requested, TOOL_CALL_MAX_TOKENS)
    if kind == "tool_call_file_recovery":
        return _internal(requested, FILE_RECOVERY_TOOL_CALL_MAX_TOKENS)
    if kind == "chat":
        if requested is None:
            return CHAT_DEFAULT_MAX_TOKENS
        return _floor_and_cap(requested, 64, CHAT_MAX_TOKENS)
    if kind == "final_from_tool":
        return _final_from_tool_tokens(requested, evidence, evidence_chars)
    if kind == "chat_final_retry":
        target = CHAT_FINAL_RETRY_AFTER_LENGTH_MAX_TOKENS if previous == "length" else CHAT_FINAL_RETRY_MAX_TOKENS
        return _floor_and_cap(requested, target, target)
    if kind == "final_from_tool_retry":
        target = CHAT_FINAL_RETRY_AFTER_LENGTH_MAX_TOKENS if previous == "length" else CHAT_FINAL_RETRY_MAX_TOKENS
        return _floor_and_cap(requested, target, target)
    if kind == "repair":
        target = FINAL_REPAIR_AFTER_LENGTH_MAX_TOKENS if previous == "length" else FINAL_REPAIR_MAX_TOKENS
        cap = FINAL_REPAIR_AFTER_LENGTH_MAX_TOKENS if previous == "length" else FINAL_REPAIR_CAP_MAX_TOKENS
        return _floor_and_cap(requested, target, cap)
    return _floor_and_cap(requested, DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS)


def _final_from_tool_tokens(requested: int | None, evidence_kind: str, evidence_chars: int | None) -> int:
    if evidence_kind == "web_search":
        return _floor_and_cap(requested, FINAL_WEB_SEARCH_MAX_TOKENS, FINAL_WEB_SEARCH_MAX_TOKENS)
    if evidence_kind in {"read", "fetch"}:
        return _floor_and_cap(requested, FINAL_READ_MAX_TOKENS, FINAL_READ_MAX_TOKENS)
    if evidence_kind == "shell_error":
        return _floor_and_cap(requested, FINAL_SHELL_ERROR_MAX_TOKENS, FINAL_SHELL_ERROR_MAX_TOKENS)
    if evidence_kind == "system_info":
        return _floor_and_cap(requested, FINAL_SYSTEM_INFO_MAX_TOKENS, FINAL_SYSTEM_INFO_MAX_TOKENS)
    if evidence_kind in {"shell", "unknown", "grep_search"} and evidence_chars is not None and evidence_chars <= 500:
        return _floor_and_cap(requested, FINAL_SMALL_MAX_TOKENS, FINAL_SMALL_MAX_TOKENS)
    if evidence_kind in {"shell", "unknown", "grep_search"}:
        return _floor_and_cap(requested, FINAL_MEDIUM_MAX_TOKENS, FINAL_MEDIUM_MAX_TOKENS)
    return _floor_and_cap(requested, FINAL_MEDIUM_MAX_TOKENS, DEFAULT_MAX_TOKENS)


def _internal(requested: int | None, cap: int) -> int:
    if requested is None:
        return cap
    return max(1, min(requested, cap))


def _floor_and_cap(requested: int | None, floor: int, cap: int) -> int:
    base = floor if requested is None else max(requested, floor)
    return max(1, min(base, cap))


def _positive_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return max(1, int(value))

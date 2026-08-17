#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, Message
from orbit.runtime.context_manager import ContextBudget, plan_context
from orbit.runtime.full_document import required_full_document_context
from orbit.runtime.session_memory import estimate_message_tokens, maybe_refresh_memory


CONTEXT_TOKENS = 8192
OUTPUT_RESERVE = 256


class _SummaryBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        self.prompt_tokens += estimate_message_tokens(messages)
        return ChatResult(
            content="Deterministic benchmark placeholder for a model-generated memory summary.",
            model="benchmark-double",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


def _tool_turn(number: int) -> list[Message]:
    call_id = f"call-{number}"
    evidence_id = f"ev-{number}"
    return [
        {"role": "user", "content": f"Inspect artifact {number} and report the exact result."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": f"item-{number}.txt"})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "read_file",
            "evidence_id": evidence_id,
            "content": (
                "tool_evidence_ref: true\n"
                f"evidence_id: {evidence_id}\n"
                "raw_ref: evidence:stored\n"
                "size: 1200 chars, 20 lines\n"
                "compat_excerpt:\n"
                + (f"result-{number} " * 90)
            ),
        },
        {"role": "assistant", "content": f"Verified visible result {number}."},
    ]


def _plain_turn(number: int, *, chars: int = 500) -> list[Message]:
    return [
        {"role": "user", "content": f"Question {number}: " + ("q" * chars)},
        {"role": "assistant", "content": f"Answer {number}: " + ("a" * chars)},
    ]


def _workloads() -> dict[str, tuple[list[Message], set[str]]]:
    tools = [{"role": "system", "content": "stable system"}]
    for number in range(30):
        tools.extend(_tool_turn(number))
    tools.append({"role": "user", "content": "Give a concise follow-up using the verified results."})

    mixed = [{"role": "system", "content": "stable system"}]
    for number in range(8):
        mixed.extend(_plain_turn(number, chars=180))
    for number in range(16):
        mixed.extend(_tool_turn(number))
    mixed.append({"role": "user", "content": "Continue from the established results."})

    long_chat = [{"role": "system", "content": "stable system"}]
    for number in range(34):
        long_chat.extend(_plain_turn(number, chars=420))
    long_chat.append({"role": "user", "content": "What constraint did I state first?"})

    near_limit = [{"role": "system", "content": "stable system"}]
    for number in range(19):
        near_limit.extend(_plain_turn(number, chars=760))
    near_limit.append({"role": "user", "content": "Continue without losing any prior requirement."})

    return {
        "long_normal_conversation": (long_chat, set()),
        "repeated_large_tool_observations": (tools, {f"ev-{number}" for number in range(30)}),
        "mixed_assistant_tool_history": (mixed, {f"ev-{number}" for number in range(16)}),
        "near_context_limit": (near_limit, set()),
    }


def _sha(messages: list[Message] | tuple[Message, ...]) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _visible_dialogue(messages: list[Message] | tuple[Message, ...]) -> list[tuple[str, object]]:
    return [
        (str(message.get("role")), message.get("content"))
        for message in messages
        if message.get("role") == "user"
        or (message.get("role") == "assistant" and not message.get("tool_calls"))
    ]


def _measure(name: str, messages: list[Message], evidence_ids: set[str]) -> dict[str, object]:
    original_tokens = estimate_message_tokens(messages)
    baseline_messages = [dict(message) for message in messages]
    baseline_backend = _SummaryBackend()
    baseline_refresh = maybe_refresh_memory(
        baseline_messages,
        backend=baseline_backend,
        context_tokens=CONTEXT_TOKENS,
        temperature=0,
    )
    plan = plan_context(
        messages,
        budget=ContextBudget(
            context_tokens=CONTEXT_TOKENS,
            output_reserve=OUTPUT_RESERVE,
        ),
        available_evidence_ids=evidence_ids,
        covered_evidence_ids=evidence_ids,
        count_tokens=estimate_message_tokens,
    )
    repeated = plan_context(
        messages,
        budget=ContextBudget(
            context_tokens=CONTEXT_TOKENS,
            output_reserve=OUTPUT_RESERVE,
        ),
        available_evidence_ids=evidence_ids,
        covered_evidence_ids=evidence_ids,
        count_tokens=estimate_message_tokens,
    )
    return {
        "workload": name,
        "baseline": {
            "prompt_tokens_before": original_tokens,
            "prompt_tokens_after": estimate_message_tokens(baseline_messages),
            "peak_context_occupancy": round(original_tokens / CONTEXT_TOKENS, 4),
            "context_management_model_calls": baseline_backend.calls,
            "context_management_evaluated_tokens": baseline_backend.prompt_tokens,
            "cached_tokens": None,
            "ttft_ms": None,
            "status": baseline_refresh.reason,
            "semantic_output_parity": "not provable: history is replaced by a model-generated summary",
        },
        "candidate": {
            "prompt_tokens_before": plan.tokens_before,
            "prompt_tokens_after": plan.tokens_after,
            "peak_context_occupancy": round(plan.tokens_after / CONTEXT_TOKENS, 4),
            "context_management_model_calls": 0,
            "context_management_evaluated_tokens": 0,
            "cached_tokens": None,
            "ttft_ms": None,
            "status": plan.status,
            "reason": plan.reason,
            "compacted_turns": plan.compacted_turns,
            "externalized_evidence": len(plan.externalized_evidence_ids),
            "visible_dialogue_preserved": _visible_dialogue(messages) == _visible_dialogue(plan.messages),
            "stable_projection": _sha(plan.messages) == _sha(repeated.messages),
        },
    }


def main() -> int:
    measurements = [_measure(name, messages, evidence) for name, (messages, evidence) in _workloads().items()]
    full_document_prompt_tokens = 6900
    measurements.append(
        {
            "workload": "full_document_request",
            "baseline": {
                "prompt_tokens": full_document_prompt_tokens,
                "required_context": required_full_document_context(full_document_prompt_tokens, 1024),
                "context_management_model_calls": 0,
                "status": "existing exact full-document admission",
            },
            "candidate": {
                "prompt_tokens": full_document_prompt_tokens,
                "required_context": required_full_document_context(full_document_prompt_tokens, 1024),
                "context_management_model_calls": 0,
                "status": "delegated unchanged to exact full-document admission",
            },
        }
    )
    cache_messages = [{"role": "system", "content": "stable system"}, *_tool_turn(1)]
    cache_budget = ContextBudget(8192, 256)
    cache_plan = plan_context(
        cache_messages,
        budget=cache_budget,
        available_evidence_ids={"ev-1"},
        covered_evidence_ids={"ev-1"},
        count_tokens=estimate_message_tokens,
    )
    measurements.append(
        {
            "workload": "qwen3_coder_cache_sensitive",
            "baseline": {"prompt_sha256": _sha(cache_messages), "ordinary_arbitrary_lcp": "disabled by profile"},
            "candidate": {
                "prompt_sha256": _sha(cache_plan.messages),
                "byte_identical_under_budget": list(cache_plan.messages) == cache_messages,
                "ordinary_arbitrary_lcp": "unchanged/disabled by profile",
                "dedicated_checkpoint_policy": "not touched",
            },
        }
    )
    print(json.dumps({"policy": "deterministic-first-v0", "measurements": measurements}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

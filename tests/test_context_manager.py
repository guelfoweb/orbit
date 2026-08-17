from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.context_manager import ContextAdmissionError, ContextBudget, plan_context, plan_exact_context
from orbit.runtime.evidence import EvidenceStore, tool_evidence_ref
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.session_memory import estimate_message_tokens


def _tool_turn(number: int, *, payload_chars: int = 1200) -> list[dict[str, object]]:
    call_id = f"call-{number}"
    evidence_id = f"ev-{number}"
    return [
        {"role": "user", "content": f"request {number}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "read_file",
            "evidence_id": evidence_id,
            "content": f"tool_evidence_ref: true\nevidence_id: {evidence_id}\n" + ("x" * payload_chars),
        },
        {"role": "assistant", "content": f"grounded final {number}"},
    ]


class _ExactBackend:
    def __init__(self, *, context_tokens: int = 4096) -> None:
        self.context_tokens = context_tokens
        self.thinking = False
        self.calls = 0
        self.count_calls = 0
        self.artifact_count_calls = 0
        self.messages_by_call: list[list[dict[str, object]]] = []
        self.tools_by_count: list[object] = []
        self.thinking_by_count: list[bool] = []

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.count_calls += 1
        self.tools_by_count.append(tools)
        self.thinking_by_count.append(thinking)
        rendered = json.dumps(
            {"messages": messages, "tools": tools or [], "thinking": thinking},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return TokenCount(
            tokens=max(1, len(rendered.encode("utf-8")) // 4),
            context_tokens=self.context_tokens,
            rendered_hash=digest,
            token_hash=digest,
        )

    def count_artifact_content_tokens(self, messages, *, tools=None, thinking=False):
        self.artifact_count_calls += 1
        rendered = json.dumps(
            {"artifact_messages": messages, "tools": tools or [], "thinking": thinking},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        return TokenCount(
            tokens=max(1, len(rendered.encode("utf-8")) // 4),
            context_tokens=self.context_tokens,
            rendered_hash=digest,
            token_hash=digest,
        )

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls += 1
        self.messages_by_call.append([dict(message) for message in messages])
        exact = "EXACT_VALUE_123" if any(
            "deterministic_evidence_rehydration" in str(message.get("content", ""))
            for message in messages
        ) else ""
        return ChatResult(
            content=exact or "done",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )


def _stored_tool_turn(store: EvidenceStore, number: int, raw: str) -> tuple[list[dict[str, object]], str]:
    call_id = f"call-{number}"
    turn_id = f"turn-{number}"
    record = store.add(
        "read_file",
        raw,
        metadata={
            "tool_call_id": call_id,
            "user_turn_id": turn_id,
            "produced_by_phase": "tool_call",
        },
    )
    return (
        [
            {"role": "user", "content": f"request {number}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "read_file",
                "evidence_id": record.evidence_id,
                "user_turn_id": turn_id,
                "content": tool_evidence_ref(record),
            },
            {"role": "assistant", "content": f"visible final {number}"},
        ],
        record.evidence_id,
    )


class ContextManagerTests(unittest.TestCase):
    def test_under_budget_is_byte_for_byte_unchanged(self) -> None:
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]

        plan = plan_context(messages, budget=ContextBudget(4096, 256), count_tokens=estimate_message_tokens)

        self.assertEqual(plan.status, "unchanged")
        self.assertEqual(list(plan.messages), messages)

    def test_compacts_oldest_covered_externalized_tool_turn_only_as_needed(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            *_tool_turn(1),
            *_tool_turn(2),
            {"role": "user", "content": "current question"},
        ]
        budget = ContextBudget(context_tokens=700, output_reserve=128, next_action_reserve=64, safety_margin=32)

        plan = plan_context(
            messages,
            budget=budget,
            available_evidence_ids={"ev-1", "ev-2"},
            covered_evidence_ids={"ev-1", "ev-2"},
            count_tokens=estimate_message_tokens,
        )

        self.assertTrue(plan.admitted)
        self.assertEqual(plan.status, "compacted")
        self.assertGreaterEqual(plan.compacted_turns, 1)
        self.assertLess(plan.tokens_after, plan.tokens_before)
        self.assertEqual(plan.messages[-1], {"role": "user", "content": "current question"})
        self.assertEqual(
            [message.get("role") for message in plan.messages],
            [message.get("role") for message in messages],
        )
        self.assertTrue(
            any(
                message.get("role") == "tool" and "evidence:ev-1" in str(message.get("content"))
                for message in plan.messages
            )
        )
        rendered = "\n".join(str(message.get("content", "")) for message in plan.messages)
        self.assertIn("grounded final 1", rendered)

    def test_multiple_terminal_assistant_messages_fail_closed(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            *_tool_turn(1, payload_chars=3000),
            {"role": "assistant", "content": ""},
        ]

        plan = plan_context(
            messages,
            budget=ContextBudget(600, 128, 64, 32),
            available_evidence_ids={"ev-1"},
            covered_evidence_ids={"ev-1"},
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertIn("message-after-terminal-assistant", str(plan.reason))

    def test_never_compacts_uncovered_or_unavailable_evidence(self) -> None:
        messages = [{"role": "system", "content": "system"}, *_tool_turn(1, payload_chars=3000)]
        budget = ContextBudget(600, 128, 64, 32)

        unavailable = plan_context(
            messages,
            budget=budget,
            available_evidence_ids=set(),
            covered_evidence_ids={"ev-1"},
            count_tokens=estimate_message_tokens,
        )
        uncovered = plan_context(
            messages,
            budget=budget,
            available_evidence_ids={"ev-1"},
            covered_evidence_ids=set(),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(unavailable.status, "blocked")
        self.assertEqual(uncovered.status, "blocked")
        self.assertEqual(list(unavailable.messages), messages)
        self.assertEqual(list(uncovered.messages), messages)

    def test_long_plain_chat_fails_closed_instead_of_semantic_summary(self) -> None:
        messages: list[dict[str, object]] = [{"role": "system", "content": "system"}]
        for number in range(12):
            messages.extend(
                [
                    {"role": "user", "content": f"question {number} " + ("q" * 400)},
                    {"role": "assistant", "content": f"answer {number} " + ("a" * 400)},
                ]
            )

        plan = plan_context(
            messages,
            budget=ContextBudget(1200, 256, 128, 64),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.reason, "required-context-does-not-fit")
        self.assertEqual(plan.compacted_turns, 0)
        self.assertEqual(list(plan.messages), messages)

    def test_active_tool_group_is_preserved(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            *_tool_turn(1, payload_chars=2000),
            {"role": "user", "content": "active"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "active-call", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "active-call",
                "name": "read_file",
                "evidence_id": "active-ev",
                "content": "tool_evidence_ref: true\nevidence_id: active-ev\nactive result",
            },
        ]

        plan = plan_context(
            messages,
            budget=ContextBudget(800, 128, 64, 32),
            available_evidence_ids={"ev-1", "active-ev"},
            covered_evidence_ids={"ev-1"},
            count_tokens=estimate_message_tokens,
        )

        self.assertTrue(plan.admitted)
        self.assertEqual(plan.messages[-1]["tool_call_id"], "active-call")
        self.assertIn("active result", str(plan.messages[-1]["content"]))

    def test_malformed_tool_sequence_fails_closed(self) -> None:
        messages = [
            {"role": "user", "content": "request"},
            {"role": "tool", "tool_call_id": "orphan", "name": "read_file", "content": "result"},
        ]

        plan = plan_context(
            messages,
            budget=ContextBudget(256, 64, 32, 16),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertTrue(plan.reason.startswith("invalid-message-structure:"))

    def test_parallel_tool_results_keep_declared_order(self) -> None:
        messages = [
            {"role": "user", "content": "request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "name": "two", "content": "bad order"},
        ]

        plan = plan_context(
            messages,
            budget=ContextBudget(256, 64, 32, 16),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertIn("tool-result-order-or-id-mismatch", str(plan.reason))

    def test_reserves_are_subtracted_from_admission(self) -> None:
        plan = plan_context(
            [{"role": "user", "content": "x" * 600}],
            budget=ContextBudget(context_tokens=256, output_reserve=64, next_action_reserve=32, safety_margin=16),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.input_limit, 144)
        self.assertEqual(plan.status, "blocked")

    def test_projection_is_stable_for_cache_sensitive_callers(self) -> None:
        messages = [{"role": "system", "content": "system"}, *_tool_turn(1, payload_chars=3000)]
        kwargs = {
            "budget": ContextBudget(600, 128, 64, 32),
            "available_evidence_ids": {"ev-1"},
            "covered_evidence_ids": {"ev-1"},
            "count_tokens": estimate_message_tokens,
        }

        first = plan_context(messages, **kwargs)
        second = plan_context(messages, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first.status, "compacted")

    def test_missing_exact_token_identity_fails_closed(self) -> None:
        plan = plan_context(
            [{"role": "user", "content": "hello"}],
            budget=ContextBudget(4096, 256),
            count_tokens=lambda _messages: None,  # type: ignore[return-value]
        )

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.reason, "exact-token-count-unavailable")
        self.assertIsNone(plan.tokens_before)

    def test_invalid_reserve_fails_closed(self) -> None:
        plan = plan_context(
            [{"role": "user", "content": "hello"}],
            budget=ContextBudget(4096, -1),
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.reason, "invalid-context-reserve")

    def test_changed_render_or_token_identity_fails_closed(self) -> None:
        class ChangingBackend(_ExactBackend):
            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                value = super().count_chat_tokens(messages, tools=tools, thinking=thinking)
                if self.count_calls == 2:
                    return TokenCount(
                        tokens=value.tokens,
                        context_tokens=value.context_tokens,
                        rendered_hash=value.rendered_hash,
                        token_hash="f" * 64,
                    )
                return value

        plan = plan_exact_context(
            [{"role": "user", "content": "hello"}],
            backend=ChangingBackend(),
            output_reserve=64,
            next_action_reserve=0,
            configured_context_tokens=4096,
            tools=None,
            thinking=False,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.reason, "tokenizer-template-or-context-changed")

    def test_marker_text_without_matching_evidence_identity_is_not_compacted(self) -> None:
        messages = [{"role": "system", "content": "system"}, *_tool_turn(1, payload_chars=3000)]
        messages[3]["content"] = "prefix " + ("x" * 3000) + "\ntool_evidence_ref: true\nevidence_id: ev-1"

        plan = plan_context(
            messages,
            budget=ContextBudget(600, 128, 64, 32),
            available_evidence_ids={"ev-1"},
            covered_evidence_ids={"ev-1"},
            count_tokens=estimate_message_tokens,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.compacted_turns, 0)

    def test_exact_admission_counts_actual_tools_and_thinking_twice(self) -> None:
        backend = _ExactBackend()
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]

        plan = plan_exact_context(
            messages,
            backend=backend,
            output_reserve=128,
            next_action_reserve=256,
            configured_context_tokens=4096,
            tools=tools,
            thinking=True,
        )

        self.assertEqual(plan.status, "unchanged")
        self.assertEqual(list(plan.messages), messages)
        self.assertEqual(backend.count_calls, 2)
        self.assertEqual(backend.tools_by_count, [tools, tools])
        self.assertEqual(backend.thinking_by_count, [True, True])

    def test_runtime_compacts_only_reattested_covered_tool_turns_and_rehydrates_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            turns: list[dict[str, object]] = []
            evidence_ids: list[str] = []
            for number in range(1, 5):
                prefix = "EXACT_VALUE_123\n" if number == 1 else f"value {number}\n"
                turn, evidence_id = _stored_tool_turn(store, number, prefix + (chr(96 + number) * 700))
                turns.extend(turn)
                evidence_ids.append(evidence_id)
            first_id = evidence_ids[0]
            backend = _ExactBackend(context_tokens=1300)
            runtime = ChatRuntime(
                backend=backend,
                system_prompt="system",
                messages=[{"role": "system", "content": "system"}, *turns],
                context_tokens=1300,
                evidence_store=store,
            )
            runtime.completed_evidence_ids.update(evidence_ids)

            result = runtime.ask_chat("current question", temperature=0, max_tokens=64)

            self.assertEqual(result.content, "done")
            self.assertIsNotNone(runtime.last_context_plan)
            assert runtime.last_context_plan is not None
            self.assertEqual(runtime.last_context_plan.status, "compacted")
            self.assertGreater(runtime.last_context_plan.tokens_before, runtime.last_context_plan.tokens_after)
            sent = backend.messages_by_call[-1]
            self.assertNotIn(tool_evidence_ref(store.records[first_id]), [message.get("content") for message in sent])
            self.assertTrue(any(f"evidence:{first_id}" in str(message.get("content")) for message in sent))

            followup = runtime.ask_chat(
                f"Return the exact value from evidence:{first_id}",
                temperature=0,
                max_tokens=64,
            )

            self.assertEqual(followup.content, "EXACT_VALUE_123")
            self.assertEqual(runtime.last_context_rehydrated_ids, (first_id,))
            rehydrated = "\n".join(str(message.get("content", "")) for message in backend.messages_by_call[-1])
            self.assertIn("EXACT_VALUE_123", rehydrated)
            self.assertIn(store.records[first_id].raw_sha256, rehydrated)

    def test_runtime_rejects_protocol_markers_before_exact_rehydration(self) -> None:
        markers = (
            "<|im_start|>",
            "<|im_end|>",
            "</s>",
            "<|endoftext|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<tool_call>",
            "</tool_call>",
            "<|tool_response>",
            "<tool_response|>",
            "<start_of_turn>",
            "<end_of_turn>",
        )
        for marker in markers:
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                store = EvidenceStore(Path(tmp) / "evidence")
                turn, evidence_id = _stored_tool_turn(store, 1, f"before {marker} after")
                backend = _ExactBackend(context_tokens=4096)
                runtime = ChatRuntime(
                    backend=backend,
                    messages=[*turn],
                    context_tokens=4096,
                    evidence_store=store,
                )
                runtime.completed_evidence_ids.add(evidence_id)

                with self.assertRaises(ContextAdmissionError):
                    runtime.ask_chat(f"Return evidence:{evidence_id}", temperature=0, max_tokens=64)

                self.assertEqual(backend.calls, 0)

    def test_runtime_binds_evidence_to_tool_call_name_and_user_turn(self) -> None:
        mutations = {
            "call": lambda turn: (
                turn[1]["tool_calls"][0].__setitem__("id", "different-call"),
                turn[2].__setitem__("tool_call_id", "different-call"),
            ),
            "name": lambda turn: turn[2].__setitem__("name", "different_tool"),
            "turn": lambda turn: turn[2].__setitem__("user_turn_id", "different-turn"),
        }
        for label, mutate in mutations.items():
            with self.subTest(field=label), tempfile.TemporaryDirectory() as tmp:
                store = EvidenceStore(Path(tmp) / "evidence")
                turn, evidence_id = _stored_tool_turn(store, 1, "x" * 1800)
                mutate(turn)
                backend = _ExactBackend(context_tokens=470)
                runtime = ChatRuntime(
                    backend=backend,
                    messages=[*turn],
                    context_tokens=470,
                    evidence_store=store,
                )
                runtime.completed_evidence_ids.add(evidence_id)

                with self.assertRaises(ContextAdmissionError):
                    runtime.ask_chat("follow up", temperature=0, max_tokens=64)

                self.assertEqual(backend.calls, 0)
                assert runtime.last_context_plan is not None
                self.assertEqual(runtime.last_context_plan.compacted_turns, 0)

    def test_declared_non_native_backend_keeps_normal_path_unchanged(self) -> None:
        class NonNativeBackend(_ExactBackend):
            def supports_exact_context_admission(self):
                return False

        backend = NonNativeBackend(context_tokens=128)
        runtime = ChatRuntime(backend=backend, context_tokens=128)
        messages = [{"role": "user", "content": "x" * 4000}]

        with model_call_context(phase="chat_final", tools_mode="off"):
            runtime.backend.chat(messages, temperature=0, max_tokens=64)

        self.assertEqual(backend.count_calls, 0)
        self.assertEqual(backend.messages_by_call[-1], messages)

    def test_unknown_exact_count_capability_fails_closed(self) -> None:
        class UnknownBackend(_ExactBackend):
            def supports_exact_context_admission(self):
                return None

        backend = UnknownBackend(context_tokens=4096)
        runtime = ChatRuntime(backend=backend, context_tokens=4096)

        with model_call_context(phase="chat_final", tools_mode="off"):
            with self.assertRaises(ContextAdmissionError):
                runtime.backend.chat([{"role": "user", "content": "hello"}], temperature=0, max_tokens=64)

        self.assertEqual(backend.calls, 0)

    def test_chat_thinking_phase_reserves_next_action(self) -> None:
        backend = _ExactBackend(context_tokens=4096)
        runtime = ChatRuntime(backend=backend, context_tokens=4096)
        messages = [{"role": "user", "content": "think"}]

        with model_call_context(phase="chat_thinking", tools_mode="off"):
            runtime.backend.chat(messages, temperature=0, max_tokens=64)

        assert runtime.last_context_plan is not None
        self.assertEqual(runtime.last_context_plan.input_limit, 4096 - 64 - 256 - 256)

    def test_stale_evidence_prevents_compaction_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence")
            turn, evidence_id = _stored_tool_turn(store, 1, "x" * 1100)
            (store.root / f"{evidence_id}.txt").unlink()
            backend = _ExactBackend(context_tokens=600)
            runtime = ChatRuntime(
                backend=backend,
                messages=[*turn],
                context_tokens=600,
                evidence_store=store,
            )
            runtime.completed_evidence_ids.add(evidence_id)

            with self.assertRaises(ContextAdmissionError):
                runtime.ask_chat("follow up", temperature=0, max_tokens=64)

            self.assertEqual(backend.calls, 0)
            self.assertIsNotNone(runtime.last_context_plan)
            assert runtime.last_context_plan is not None
            self.assertEqual(runtime.last_context_plan.compacted_turns, 0)

    def test_full_document_phase_bypasses_generic_manager(self) -> None:
        backend = _ExactBackend(context_tokens=128)
        runtime = ChatRuntime(backend=backend, context_tokens=128)
        messages = [{"role": "user", "content": "x" * 2000}]

        with model_call_context(phase="full_document", tools_mode="on"):
            runtime.backend.chat(messages, temperature=0, max_tokens=64)

        self.assertEqual(backend.count_calls, 0)
        self.assertEqual(backend.messages_by_call[-1], messages)

    def test_artifact_content_uses_phase_specific_exact_counter(self) -> None:
        backend = _ExactBackend(context_tokens=4096)
        runtime = ChatRuntime(backend=backend, context_tokens=4096)
        messages = [{"role": "user", "content": "artifact content"}]

        prepared = runtime._prepare_model_context(messages, 512, None, False, True)

        self.assertEqual(prepared, messages)
        self.assertEqual(backend.artifact_count_calls, 2)
        self.assertEqual(backend.count_calls, 0)


if __name__ == "__main__":
    unittest.main()

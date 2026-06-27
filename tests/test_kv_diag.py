from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.kv_diag import (
    emit_route_outcome,
    fingerprint_prompt,
    instrument_backend,
    model_call_context,
    request_context,
    reset_diagnostics_for_tests,
)
from orbit.terminal.status import format_turn_status


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[Message] = []
        self.tools = None

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        self.messages = messages
        self.tools = tools
        return ChatResult(
            content="ok",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=4,
            prompt_tokens_per_second=12.5,
            generation_tokens_per_second=3.5,
        )


class SequenceBackend:
    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.calls = 0

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
        self.calls += 1
        if not self.results:
            raise AssertionError("unexpected backend call")
        return self.results.pop(0)


def _result(content: str, *, finish_reason: str, prompt_tokens: int = 10, completion_tokens: int = 2, cached_tokens: int = 0) -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=[],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        prompt_tokens_per_second=10.0,
        generation_tokens_per_second=3.0,
    )


class KVDiagTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_diagnostics_for_tests()

    def test_diag_default_off_does_not_write_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "0", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("secret prompt text", temperature=0, max_tokens=32)

            self.assertFalse(log_path.exists())

    def test_diag_on_writes_hashes_not_raw_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("secret prompt text", temperature=0, max_tokens=32)

            payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["event"], "kv_diag_model_call")
        self.assertIn("request_id", payload)
        self.assertIn("model_call_id", payload)
        self.assertEqual(payload["pass_index"], 1)
        self.assertEqual(payload["phase"], "chat_final")
        self.assertIn("stable_prefix_hash", payload)
        self.assertIn("full_prompt_hash", payload)
        self.assertEqual(payload["prompt_tokens"], 10)
        self.assertEqual(payload["cached_tokens"], 4)
        self.assertEqual(payload["reused_tokens"], 4)
        self.assertEqual(payload["evaluated_tokens"], 6)
        self.assertNotIn("secret prompt text", json.dumps(payload))

    def test_stable_prefix_hash_is_stable_for_identical_inputs(self) -> None:
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
        ]
        first = fingerprint_prompt(messages, tools=[])
        second = fingerprint_prompt(messages, tools=[])

        self.assertEqual(first.stable_prefix_hash, second.stable_prefix_hash)
        self.assertEqual(first.tool_schema_hash, second.tool_schema_hash)
        self.assertEqual(first.full_prompt_hash, second.full_prompt_hash)

    def test_capability_summary_hash_changes_when_summary_changes(self) -> None:
        base = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}]
        with_python = [
            *base,
            {"role": "system", "content": "Local tools available: python3.\nUnavailable: pandoc."},
        ]
        with_pandoc = [
            *base,
            {"role": "system", "content": "Local tools available: python3, pandoc.\nUnavailable: none."},
        ]

        first = fingerprint_prompt(with_python, tools=[])
        second = fingerprint_prompt(with_pandoc, tools=[])

        self.assertNotEqual(first.capability_summary_hash, second.capability_summary_hash)
        self.assertNotEqual(first.stable_prefix_hash, second.stable_prefix_hash)

    def test_tool_schema_hash_changes_with_tools_on_off(self) -> None:
        messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}]
        off = fingerprint_prompt(messages, tools=[])
        on = fingerprint_prompt(
            messages,
            tools=[{"type": "function", "function": {"name": "system_info", "parameters": {}}}],
        )

        self.assertNotEqual(off.tool_schema_hash, on.tool_schema_hash)
        self.assertNotEqual(off.stable_prefix_hash, on.stable_prefix_hash)

    def test_consecutive_same_prompt_reports_no_component_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("same prompt", temperature=0, max_tokens=32)
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("same prompt", temperature=0, max_tokens=32)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            calls = [line for line in lines if line["event"] == "kv_diag_model_call"]

        self.assertEqual(calls[0]["changed_components"], [])
        self.assertEqual(calls[1]["changed_components"], [])

    def test_pass_index_increments_inside_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                messages = [{"role": "user", "content": "prompt"}]
                with request_context(session_id="session"):
                    with model_call_context(phase="first", tools_mode="off"):
                        backend.chat(messages, temperature=0, max_tokens=32)
                    with model_call_context(phase="second", tools_mode="off"):
                        backend.chat(messages, temperature=0, max_tokens=32)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            calls = [line for line in lines if line["event"] == "kv_diag_model_call"]
            summaries = [line for line in lines if line["event"] == "kv_diag_request_summary"]

        self.assertEqual([call["pass_index"] for call in calls], [1, 2])
        self.assertEqual(calls[0]["request_id"], calls[1]["request_id"])
        self.assertNotEqual(calls[0]["model_call_id"], calls[1]["model_call_id"])
        self.assertEqual(summaries[0]["model_calls"], 2)
        self.assertEqual(summaries[0]["phases"], ["first", "second"])
        self.assertEqual(summaries[0]["total_prompt_tokens"], 20)
        self.assertEqual(summaries[0]["total_cached_tokens"], 8)
        self.assertEqual(summaries[0]["total_evaluated_tokens"], 12)

    def test_each_user_turn_gets_distinct_request_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("first", temperature=0, max_tokens=32)
                runtime.ask_chat("second", temperature=0, max_tokens=32)

            calls = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "kv_diag_model_call"
            ]

        self.assertNotEqual(calls[0]["request_id"], calls[1]["request_id"])
        self.assertEqual(calls[0]["pass_index"], 1)
        self.assertEqual(calls[1]["pass_index"], 1)

    def test_footer_metrics_correlate_to_last_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                result = runtime.ask_chat("footer prompt", temperature=0, max_tokens=32)
                format_turn_status(result, elapsed_seconds=1.5, estimated_context_tokens=20, context_tokens=100)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            call = next(line for line in lines if line["event"] == "kv_diag_model_call")
            footer = next(line for line in lines if line["event"] == "kv_diag_footer_metrics")

        self.assertEqual(footer["request_id"], call["request_id"])
        self.assertEqual(footer["model_call_id"], call["model_call_id"])
        self.assertEqual(footer["pass_index"], call["pass_index"])
        self.assertEqual(footer["footer"]["input_tokens"], 10)
        self.assertEqual(footer["footer"]["wall_ms"], 1500)

    def test_prefix_mismatch_event_contains_only_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                first = [
                    {"role": "system", "content": "Local tools available: python3."},
                    {"role": "user", "content": "same user prompt"},
                ]
                second = [
                    {"role": "system", "content": "Local tools available: python3, file."},
                    {"role": "user", "content": "same user prompt"},
                ]
                with request_context(session_id="session"):
                    with model_call_context(phase="tool_call", tools_mode="on"):
                        backend.chat(first, temperature=0, max_tokens=32)
                    with model_call_context(phase="tool_call", tools_mode="on"):
                        backend.chat(second, temperature=0, max_tokens=32)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            mismatches = [line for line in lines if line["event"] == "kv_diag_prefix_mismatch"]
            raw_log = json.dumps(lines)

        self.assertTrue(any(event["component"] == "capability_summary" for event in mismatches))
        self.assertNotIn("same user prompt", raw_log)
        self.assertNotIn("python3, file", raw_log)

    def test_route_direct_final_stop_is_observed_without_behavior_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=SequenceBackend([_result("direct answer", finish_reason="stop")]), system_prompt=None)
                result = runtime.ask_auto(
                    "hi",
                    temperature=0,
                    max_tokens=32,
                    workdir=Path(tmp),
                    allowed_tool_names=("exec_shell_full_command", "fetch_url", "list_directory", "system_info"),
                )

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            outcomes = [line for line in lines if line["event"] == "kv_diag_route_outcome"]

        self.assertEqual(result.content, "direct answer")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome"], "route_direct_final_stop")
        self.assertEqual(outcomes[0]["phase"], "route")
        self.assertEqual(outcomes[0]["finish_reason"], "stop")
        self.assertIsNone(outcomes[0]["decision_type"])
        self.assertNotIn("direct answer", json.dumps(outcomes))

    def test_route_no_decision_length_retry_is_observed_without_behavior_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            backend = SequenceBackend(
                [
                    _result("truncated route prose", finish_reason="length", prompt_tokens=12, completion_tokens=128, cached_tokens=8),
                    _result("final answer", finish_reason="stop", prompt_tokens=20, completion_tokens=3, cached_tokens=19),
                ]
            )
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                result = runtime.ask_auto(
                    "hi, tell me something about yourself",
                    temperature=0,
                    max_tokens=32,
                    workdir=Path(tmp),
                    allowed_tool_names=("exec_shell_full_command", "fetch_url", "list_directory", "system_info"),
                )

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            outcomes = [line for line in lines if line["event"] == "kv_diag_route_outcome"]
            calls = [line for line in lines if line["event"] == "kv_diag_model_call"]

        self.assertEqual(result.content, "final answer")
        self.assertEqual(backend.calls, 2)
        self.assertEqual([call["phase"] for call in calls], ["route", "chat_final_retry"])
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome"], "route_no_decision_length_retry")
        self.assertEqual(outcomes[0]["retry_reason"], "length_without_decision")
        self.assertEqual(outcomes[0]["output_tokens"], 128)
        self.assertNotIn("truncated route prose", json.dumps(outcomes))
        self.assertNotIn("final answer", json.dumps(outcomes))

    def test_route_outcome_event_is_metadata_only_for_required_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                with request_context(session_id="session"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat([{"role": "user", "content": "secret prompt"}], temperature=0, max_tokens=32)
                    for outcome, decision_type, retry_reason in (
                        ("route_parsed_tool", "FILESYSTEM", None),
                        ("route_parsed_chat", "CHAT", None),
                        ("route_invalid_output", None, "empty_response"),
                        ("route_other_retry", None, "explicit_web_search"),
                    ):
                        emit_route_outcome(
                            outcome=outcome,
                            finish_reason="stop",
                            decision_type=decision_type,
                            output_chars=123,
                            output_tokens=7,
                            retry_reason=retry_reason,
                        )

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            outcomes = [line for line in lines if line["event"] == "kv_diag_route_outcome"]
            raw_log = json.dumps(lines)

        self.assertEqual(
            [event["outcome"] for event in outcomes],
            ["route_parsed_tool", "route_parsed_chat", "route_invalid_output", "route_other_retry"],
        )
        self.assertTrue(all(event["request_id"] for event in outcomes))
        self.assertTrue(all(event["model_call_id"] for event in outcomes))
        self.assertNotIn("secret prompt", raw_log)


if __name__ == "__main__":
    unittest.main()

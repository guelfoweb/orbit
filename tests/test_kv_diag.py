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
    emit_evidence_lineage,
    emit_route_outcome,
    fingerprint_prompt,
    instrument_backend,
    model_call_context,
    request_context,
    reset_diagnostics_for_tests,
)
from orbit.native_llama.kv_diag import (
    build_prompt_component_tokens,
    emit_prompt_cache_event as emit_native_prompt_cache_event,
    emit_route_prefix_anchor_event as emit_native_route_prefix_anchor_event,
    request_context as native_request_context,
    reset_diagnostics_for_tests as reset_native_diagnostics_for_tests,
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


class LlamaServerBackend(FakeBackend):
    def __init__(self, *, native: bool = True) -> None:
        super().__init__()
        self._props_cache = {"backend": "orbit-native"} if native else {}


class ContinueBackend:
    def continue_current(self, *, max_tokens: int, on_delta=None, on_progress=None) -> ChatResult:
        return ChatResult(
            content="ok",
            model="fake",
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=3,
            completion_tokens=1,
            cached_tokens=1,
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=3.0,
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
        reset_native_diagnostics_for_tests()

    def test_diag_default_off_does_not_write_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "0", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("placeholder payload alpha", temperature=0, max_tokens=32)

            self.assertFalse(log_path.exists())

    def test_diag_on_writes_hashes_not_raw_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=FakeBackend(), system_prompt=None)
                runtime.ask_chat("placeholder payload alpha", temperature=0, max_tokens=32)

            payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["event"], "kv_diag_model_call")
        self.assertIn("request_id", payload)
        self.assertIn("model_call_id", payload)
        self.assertEqual(payload["pass_index"], 1)
        self.assertEqual(payload["phase"], "chat_final")
        self.assertIn("stable_prefix_hash", payload)
        self.assertIn("full_prompt_hash", payload)
        self.assertIn("prompt_layout_hash", payload)
        self.assertIn("prompt_layout", payload)
        self.assertIn("prompt_layout_common_prefix", payload)
        self.assertIn("request_envelope", payload)
        self.assertEqual(payload["request_envelope"]["message_count"], 2)
        self.assertEqual(payload["request_envelope"]["role_sequence"], ["system", "user"])
        self.assertFalse(payload["request_envelope"]["tools_parameter_present"])
        self.assertEqual(payload["prompt_tokens"], 10)
        self.assertEqual(payload["cached_tokens"], 4)
        self.assertEqual(payload["reused_tokens"], 4)
        self.assertEqual(payload["evaluated_tokens"], 6)
        self.assertNotIn("placeholder payload alpha", json.dumps(payload))

    def test_request_envelope_diagnostics_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(LlamaServerBackend())
                messages = [
                    {"role": "system", "content": "runtime policy placeholder"},
                    {"role": "user", "content": "placeholder payload delta"},
                ]
                tools = [{"type": "function", "function": {"name": "list_directory", "parameters": {"marker": "schema payload"}}}]
                with request_context(session_id="session-key-placeholder"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat(messages, temperature=0, max_tokens=32, tools=tools)

            payload = next(
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "kv_diag_model_call"
            )
            envelope = payload["request_envelope"]
            raw_log = json.dumps(payload)

        self.assertEqual(envelope["backend_class"], "LlamaServerBackend")
        self.assertEqual(envelope["endpoint"], "/chat/stream")
        self.assertFalse(envelope["stream"])
        self.assertTrue(envelope["cache_prompt"])
        self.assertFalse(envelope["continue_current"])
        self.assertFalse(envelope["session_identity_present"])
        self.assertIsNone(envelope["session_identity_hash"])
        self.assertEqual(envelope["message_count"], 2)
        self.assertEqual(envelope["role_sequence"], ["system", "user"])
        self.assertTrue(envelope["tools_parameter_present"])
        self.assertEqual(envelope["tool_count"], 1)
        self.assertIn("runtime_session_key_hash", envelope)
        self.assertIn("prompt_layout_common_tokens_estimate", envelope)
        self.assertNotIn("placeholder payload delta", raw_log)
        self.assertNotIn("runtime policy placeholder", raw_log)
        self.assertNotIn("schema payload", raw_log)
        self.assertNotIn("session-key-placeholder", raw_log)

    def test_continue_current_envelope_is_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(ContinueBackend())
                with request_context(session_id="session-key-placeholder"):
                    with model_call_context(phase="continue", tools_mode="off"):
                        backend.continue_current(max_tokens=8)

            payload = next(
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "kv_diag_model_call"
            )
            envelope = payload["request_envelope"]
            raw_log = json.dumps(payload)

        self.assertEqual(envelope["backend_class"], "ContinueBackend")
        self.assertIsNone(envelope["endpoint"])
        self.assertFalse(envelope["stream"])
        self.assertIsNone(envelope["cache_prompt"])
        self.assertTrue(envelope["continue_current"])
        self.assertEqual(envelope["message_count"], 0)
        self.assertEqual(envelope["role_sequence"], [])
        self.assertFalse(envelope["tools_parameter_present"])
        self.assertEqual(envelope["tool_count"], 0)
        self.assertNotIn("session-key-placeholder", raw_log)

    def test_prompt_layout_diagnostics_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                messages = [
                    {"role": "system", "content": "policy text"},
                    {"role": "user", "content": "placeholder payload beta"},
                ]
                tools = [{"type": "function", "function": {"name": "system_info", "parameters": {"placeholder": "schema metadata"}}}]
                with request_context(session_id="session"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat(messages, temperature=0, max_tokens=32, tools=tools)

            payload = next(
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "kv_diag_model_call"
            )
            raw_log = json.dumps(payload)

        self.assertEqual(payload["prompt_layout_order"], ["runtime_policy", "user_message", "tool_schema_parameter"])
        self.assertEqual(payload["prompt_layout"][0]["source"], "messages")
        self.assertEqual(payload["prompt_layout"][-1]["source"], "tools_parameter")
        self.assertEqual(payload["prompt_layout"][-1]["tool_count"], 1)
        self.assertIn("start_token_estimate", payload["prompt_layout"][0])
        self.assertIn("end_token_estimate", payload["prompt_layout"][0])
        self.assertFalse(payload["prompt_layout_common_prefix"]["previous_seen"])
        self.assertNotIn("placeholder payload beta", raw_log)
        self.assertNotIn("schema metadata", raw_log)
        self.assertNotIn("policy text", raw_log)

    def test_route_prefix_boundary_diagnostics_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                messages = [
                    {"role": "system", "content": "route policy placeholder"},
                    {"role": "user", "content": "placeholder route payload"},
                ]
                with request_context(session_id="session"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat(messages, temperature=0, max_tokens=32)

            payload = next(
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "kv_diag_model_call"
            )
            boundary = payload["route_prefix_boundary"]
            raw_log = json.dumps(payload)

        self.assertTrue(boundary["route_prefix_boundary_available"])
        self.assertIsNone(boundary["failure_reason"])
        self.assertIsInstance(boundary["stable_prefix_hash"], str)
        self.assertGreater(boundary["stable_prefix_char_len"], 0)
        self.assertGreater(boundary["stable_prefix_token_count_estimate"], 0)
        self.assertGreater(boundary["dynamic_suffix_char_len"], 0)
        self.assertEqual(boundary["first_dynamic_component"], "user_message")
        self.assertNotIn("placeholder route payload", raw_log)
        self.assertNotIn("route policy placeholder", raw_log)

    def test_route_prefix_boundary_hash_changes_with_schema_and_capabilities(self) -> None:
        messages = [
            {"role": "system", "content": "route policy placeholder"},
            {"role": "system", "content": "Local tools available: python3."},
            {"role": "user", "content": "placeholder route payload"},
        ]
        first = fingerprint_prompt(
            messages,
            tools=[{"type": "function", "function": {"name": "tool_alpha", "parameters": {}}}],
        )
        second = fingerprint_prompt(
            [
                {"role": "system", "content": "route policy placeholder"},
                {"role": "system", "content": "Local tools available: python3, file."},
                {"role": "user", "content": "placeholder route payload"},
            ],
            tools=[{"type": "function", "function": {"name": "tool_alpha", "parameters": {}}}],
        )
        third = fingerprint_prompt(
            messages,
            tools=[{"type": "function", "function": {"name": "tool_beta", "parameters": {}}}],
        )

        from orbit.runtime.kv_diag import _route_prefix_boundary_metadata

        first_boundary = _route_prefix_boundary_metadata("route", "on", first)
        second_boundary = _route_prefix_boundary_metadata("route", "on", second)
        third_boundary = _route_prefix_boundary_metadata("route", "on", third)

        self.assertTrue(first_boundary["route_prefix_boundary_available"])
        self.assertNotEqual(first_boundary["stable_prefix_hash"], second_boundary["stable_prefix_hash"])
        self.assertNotEqual(first_boundary["stable_prefix_hash"], third_boundary["stable_prefix_hash"])

    def test_prompt_layout_mismatch_event_contains_only_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                first = [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "placeholder repeated payload"},
                ]
                second = [
                    {"role": "system", "content": "policy"},
                    {"role": "assistant", "content": "placeholder assistant payload"},
                    {"role": "user", "content": "placeholder repeated payload"},
                ]
                with request_context(session_id="session"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat(first, temperature=0, max_tokens=32)
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat(second, temperature=0, max_tokens=32)

            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            mismatch = next(line for line in lines if line["event"] == "kv_diag_prompt_layout_mismatch")
            second_call = [line for line in lines if line["event"] == "kv_diag_model_call"][-1]
            raw_log = json.dumps(lines)

        self.assertEqual(mismatch["common_blocks"], 1)
        self.assertEqual(mismatch["first_divergence_component"], "assistant_history")
        self.assertEqual(mismatch["previous_first_divergence_component"], "user_message")
        self.assertTrue(second_call["prompt_layout_common_prefix"]["previous_seen"])
        self.assertEqual(second_call["prompt_layout_common_prefix"]["common_blocks"], 1)
        self.assertNotIn("placeholder repeated payload", raw_log)
        self.assertNotIn("placeholder assistant payload", raw_log)

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
                    {"role": "user", "content": "placeholder repeated payload"},
                ]
                second = [
                    {"role": "system", "content": "Local tools available: python3, file."},
                    {"role": "user", "content": "placeholder repeated payload"},
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
        self.assertNotIn("placeholder repeated payload", raw_log)
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
        self.assertEqual(outcomes[0]["route_output_class"], "direct_prose")
        self.assertFalse(outcomes[0]["route_output_canonical"])
        self.assertFalse(outcomes[0]["route_parser_accepted"])
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
        self.assertEqual(outcomes[0]["route_output_class"], "malformed")
        self.assertEqual(outcomes[0]["route_output_tokens"], 128)
        self.assertEqual(outcomes[0]["route_finish_reason"], "length")
        self.assertNotIn("truncated route prose", json.dumps(outcomes))
        self.assertNotIn("final answer", json.dumps(outcomes))

    def test_route_control_only_length_is_classified_without_behavior_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            backend = SequenceBackend(
                [
                    _result("", finish_reason="length", prompt_tokens=12, completion_tokens=64, cached_tokens=8),
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
            outcome = next(line for line in lines if line["event"] == "kv_diag_route_outcome")

        self.assertEqual(result.content, "final answer")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(outcome["route_output_class"], "control_loop")
        self.assertEqual(outcome["route_output_reason"], "empty_visible_control_output")
        self.assertNotIn("final answer", json.dumps(outcome))

    def test_invalid_route_retry_is_classified_without_changing_tool_fallback(self) -> None:
        class RouteRetryBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return _result("route prose", finish_reason="stop")
                if self.calls == 2:
                    return _result("retry prose", finish_reason="length", completion_tokens=64)
                if self.calls == 3:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "exec_shell_full_command",
                                    "arguments": '{"command":"printf route-diagnostic-result"}',
                                },
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=10.0,
                        generation_tokens_per_second=3.0,
                    )
                return _result("route diagnostic result", finish_reason="stop")

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            backend = RouteRetryBackend()
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=backend, system_prompt="route system")
                result = runtime.ask_auto(
                    "search online for route diagnostic information",
                    temperature=0,
                    max_tokens=32,
                    workdir=Path(tmp),
                    allowed_tool_names=("exec_shell_full_command",),
                )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            outcomes = [event for event in events if event["event"] == "kv_diag_route_outcome"]

        self.assertEqual(result.content, "route diagnostic result")
        self.assertEqual(backend.calls, 5)
        self.assertEqual(
            [event["outcome"] for event in outcomes],
            ["route_other_retry", "route_retry_invalid_output"],
        )
        self.assertEqual(outcomes[1]["route_output_class"], "malformed")
        self.assertFalse(outcomes[1]["route_parser_accepted"])
        self.assertNotIn("route prose", json.dumps(outcomes))
        self.assertNotIn("retry prose", json.dumps(outcomes))

    def test_valid_route_retry_is_classified_independently_and_keeps_tool_selection(self) -> None:
        backend = SequenceBackend(
            [
                _result("route prose", finish_reason="stop"),
                _result('{"command":"printf route-valid-retry"}', finish_reason="stop"),
                _result("route valid retry", finish_reason="stop"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                runtime = ChatRuntime(backend=backend, system_prompt="route system")
                result = runtime.ask_auto(
                    "search online for route validation information",
                    temperature=0,
                    max_tokens=32,
                    workdir=Path(tmp),
                    allowed_tool_names=("exec_shell_full_command",),
                )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            outcomes = [event for event in events if event["event"] == "kv_diag_route_outcome"]

        self.assertEqual(result.content, "route valid retry")
        self.assertEqual(backend.calls, 3)
        self.assertEqual(
            [(event["outcome"], event["route_output_class"]) for event in outcomes],
            [("route_other_retry", "malformed"), ("route_parsed_tool", "canonical")],
        )
        tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
        self.assertEqual(tool_messages[-1]["name"], "exec_shell_full_command")
        self.assertIn("route-valid-retry", tool_messages[-1]["content"])
        self.assertNotIn("route prose", json.dumps(outcomes))
        self.assertNotIn("route-valid-retry", json.dumps(outcomes))

    def test_route_outcome_event_is_metadata_only_for_required_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                backend = instrument_backend(FakeBackend())
                with request_context(session_id="session"):
                    with model_call_context(phase="route", tools_mode="on"):
                        backend.chat([{"role": "user", "content": "placeholder payload gamma"}], temperature=0, max_tokens=32)
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
        self.assertNotIn("placeholder payload gamma", raw_log)

    def test_native_cache_diagnostics_default_off_does_not_write_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "0", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload={"cache_prompt": True}):
                    emit_native_prompt_cache_event(
                        prompt_tokens=[1, 2, 3],
                        previous_prompt_tokens=[1, 2],
                        reused_prompt_tokens=2,
                        output_tokens=1,
                        cancelled=False,
                        slot_id="default",
                    )

            self.assertFalse(log_path.exists())

    def test_native_cache_diagnostics_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            payload = {
                "cache_prompt": True,
                "stream": True,
                "_orbit_kv_phase": "final_from_tool",
                "_orbit_kv_tools_mode": "on",
                "session_id": "session-key-placeholder",
                "messages": [
                    {"role": "system", "content": "runtime policy placeholder"},
                    {"role": "user", "content": "placeholder payload epsilon"},
                ],
                "tools": [{"type": "function", "function": {"name": "fetch_url", "parameters": {"marker": "schema payload"}}}],
            }
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload=payload):
                    emit_native_prompt_cache_event(
                        prompt_tokens=[11, 22, 33, 44],
                        previous_prompt_tokens=[11, 22, 99],
                        reused_prompt_tokens=2,
                        output_tokens=3,
                        cancelled=False,
                        slot_id="default",
                    )

            event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            raw_log = json.dumps(event)

        self.assertEqual(event["event"], "kv_diag_native_cache")
        self.assertEqual(event["backend_request_id"], "native_req_000001")
        self.assertEqual(event["endpoint"], "/chat/stream")
        self.assertTrue(event["stream"])
        self.assertTrue(event["cache_prompt"])
        self.assertEqual(event["phase"], "final_from_tool")
        self.assertEqual(event["tools_mode"], "on")
        self.assertEqual(event["slot_id"], "default")
        self.assertEqual(event["prompt_tokens"], 4)
        self.assertEqual(event["previous_prompt_tokens"], 3)
        self.assertIsNotNone(event["previous_tokenized_prompt_hash"])
        self.assertNotEqual(event["previous_tokenized_prompt_hash"], event["current_tokenized_prompt_hash"])
        self.assertEqual(event["current_tokenized_prompt_hash"], event["tokenized_prompt_hash"])
        self.assertEqual(event["tokenized_prefix_length"], 2)
        self.assertEqual(event["longest_common_prefix_tokens"], 2)
        self.assertEqual(event["first_mismatch_index"], 2)
        self.assertEqual(event["first_mismatch_token"], 2)
        self.assertEqual(event["previous_token_at_mismatch"], 99)
        self.assertEqual(event["current_token_at_mismatch"], 33)
        self.assertEqual(event["cached_tokens"], 2)
        self.assertEqual(event["evaluated_tokens"], 2)
        self.assertEqual(event["output_tokens"], 3)
        self.assertIsNone(event["cache_miss_reason"])
        self.assertEqual(event["message_count"], 2)
        self.assertEqual(event["role_sequence"], ["system", "user"])
        self.assertTrue(event["tools_parameter_present"])
        self.assertEqual(event["tool_count"], 1)
        self.assertNotIn("placeholder payload epsilon", raw_log)
        self.assertNotIn("runtime policy placeholder", raw_log)
        self.assertNotIn("schema payload", raw_log)
        self.assertNotIn("session-key-placeholder", raw_log)
        self.assertNotIn("11, 22", raw_log)

    def test_native_cache_miss_reason_reports_prefix_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload={"cache_prompt": True, "messages": []}):
                    emit_native_prompt_cache_event(
                        prompt_tokens=[5, 6, 7],
                        previous_prompt_tokens=[1, 2, 3],
                        reused_prompt_tokens=0,
                        output_tokens=1,
                        cancelled=False,
                        slot_id="default",
                    )

            event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(event["cache_miss_reason"], "prefix_mismatch_at_token_0")
        self.assertEqual(event["longest_common_prefix_tokens"], 0)
        self.assertEqual(event["first_mismatch_index"], 0)
        self.assertEqual(event["first_mismatch_token"], 0)
        self.assertEqual(event["previous_token_at_mismatch"], 1)
        self.assertEqual(event["current_token_at_mismatch"], 5)

    def test_native_prompt_component_tokens_are_metadata_only(self) -> None:
        def token_count(text: str) -> int:
            return len([part for part in text.replace("\n", " ").split(" ") if part])

        messages = [
            {"role": "system", "content": "system policy"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "placeholder component payload"},
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "evidence_context:",
                        "- evidence 1:",
                        "tool_evidence_card: true",
                        "kind: shell",
                        "stdout_excerpt: concise summary",
                        "bounded_raw_excerpt:",
                        "raw detail line",
                    ]
                ),
            },
        ]
        component_tokens = build_prompt_component_tokens(
            messages=messages,
            prompt_tokens_total=64,
            token_count=token_count,
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            payload = {
                "cache_prompt": True,
                "stream": True,
                "_orbit_kv_phase": "chat_final",
                "_orbit_kv_tools_mode": "on",
                "messages": messages,
            }
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload=payload):
                    emit_native_prompt_cache_event(
                        prompt_tokens=list(range(64)),
                        previous_prompt_tokens=[0, 1, 99],
                        reused_prompt_tokens=2,
                        output_tokens=1,
                        cancelled=False,
                        slot_id="default",
                        component_tokens=component_tokens,
                    )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            event = next(item for item in events if item["event"] == "kv_diag_prompt_component_tokens")
            raw_log = json.dumps(event)

        components = event["components"]
        self.assertEqual(event["phase"], "chat_final")
        self.assertEqual(event["prompt_tokens_total"], 64)
        self.assertEqual(event["cached_tokens"], 2)
        self.assertEqual(event["evaluated_tokens"], 62)
        self.assertEqual(event["role_sequence"], ["system", "assistant", "user", "system"])
        self.assertEqual(event["evidence_card_count"], 1)
        self.assertEqual(event["evidence_kinds"], ["shell"])
        self.assertEqual(components["system_prompt_tokens"], 2)
        self.assertEqual(components["assistant_history_tokens"], 2)
        self.assertEqual(components["user_message_tokens"], 3)
        self.assertEqual(components["conversation_window_tokens"], 5)
        self.assertGreater(components["evidence_total_tokens"], 0)
        self.assertGreater(components["evidence_raw_excerpt_tokens"], 0)
        self.assertGreater(components["evidence_summary_tokens"], 0)
        self.assertGreaterEqual(components["template_overhead_tokens"], 0)
        self.assertEqual(components["unknown_tokens"], 0)
        self.assertNotIn("placeholder component payload", raw_log)
        self.assertNotIn("raw detail line", raw_log)
        self.assertNotIn("system policy", raw_log)

    def test_native_evidence_card_tokens_are_metadata_only(self) -> None:
        def token_count(text: str) -> int:
            return len([part for part in text.replace("\n", " ").split(" ") if part])

        messages = [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "placeholder question"},
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "evidence_context:",
                        "- evidence 1:",
                        "tool_evidence_card: true",
                        "kind: shell",
                        "status: error",
                        "raw_ref: evidence:secret-raw-ref",
                        "command: secret-command --with path.txt",
                        "path: /secret/path.txt",
                        "stdout_excerpt: concise summary",
                        "bounded_raw_excerpt:",
                        "raw detail line",
                    ]
                ),
            },
        ]
        component_tokens = build_prompt_component_tokens(
            messages=messages,
            prompt_tokens_total=48,
            token_count=token_count,
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            payload = {
                "cache_prompt": True,
                "_orbit_kv_phase": "chat_final",
                "_orbit_kv_tools_mode": "on",
                "messages": messages,
            }
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload=payload):
                    emit_native_prompt_cache_event(
                        prompt_tokens=list(range(48)),
                        previous_prompt_tokens=[0, 1],
                        reused_prompt_tokens=1,
                        output_tokens=1,
                        cancelled=False,
                        slot_id="default",
                        component_tokens=component_tokens,
                    )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            event = next(item for item in events if item["event"] == "kv_diag_evidence_card_tokens")
            raw_log = json.dumps(event)

        self.assertEqual(event["phase"], "chat_final")
        self.assertEqual(event["tools_mode"], "on")
        self.assertEqual(event["card_index"], 1)
        self.assertEqual(event["kind"], "shell")
        self.assertEqual(event["status"], "error")
        self.assertIsNotNone(event["evidence_id_hash"])
        self.assertIsNotNone(event["command_hash"])
        self.assertIsNotNone(event["path_hash"])
        self.assertGreater(event["metadata_tokens"], 0)
        self.assertGreater(event["raw_excerpt_tokens"], 0)
        self.assertGreater(event["summary_tokens"], 0)
        self.assertGreater(event["wrapper_tokens"], 0)
        self.assertEqual(event["unknown_tokens"], 0)
        self.assertEqual(
            event["total_tokens"],
            event["metadata_tokens"] + event["raw_excerpt_tokens"] + event["summary_tokens"] + event["wrapper_tokens"],
        )
        self.assertTrue(event["has_raw_excerpt"])
        self.assertTrue(event["has_summary"])
        self.assertTrue(event["is_error_status"])
        self.assertNotIn("secret-command", raw_log)
        self.assertNotIn("/secret/path.txt", raw_log)
        self.assertNotIn("raw detail line", raw_log)
        self.assertNotIn("evidence:secret-raw-ref", raw_log)

    def test_runtime_evidence_lineage_is_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with request_context(session_id="session-a"):
                    with model_call_context(phase="chat_final", tools_mode="on"):
                        emit_evidence_lineage(
                            {
                                "context_kind": "final",
                                "card_index": 1,
                                "evidence_id": "ev_secret_raw",
                                "evidence_sequence": 0,
                                "tool_call_id": "call_secret",
                                "user_turn_id": None,
                                "producer_model_call_id": None,
                                "produced_by_phase": None,
                                "kind": "shell",
                                "status": "error",
                            }
                        )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            event = next(item for item in events if item["event"] == "kv_diag_evidence_lineage")
            raw_log = json.dumps(event)

        self.assertEqual(event["phase"], "chat_final")
        self.assertEqual(event["tools_mode"], "on")
        self.assertEqual(event["context_kind"], "final")
        self.assertEqual(event["card_index"], 1)
        self.assertEqual(event["evidence_sequence"], 0)
        self.assertIsNotNone(event["evidence_id_hash"])
        self.assertIsNotNone(event["tool_call_id_hash"])
        self.assertIsNone(event["user_turn_id"])
        self.assertIsNone(event["producer_model_call_id"])
        self.assertIsNone(event["produced_by_phase"])
        self.assertEqual(event["kind"], "shell")
        self.assertEqual(event["status"], "error")
        self.assertNotIn("ev_secret_raw", raw_log)
        self.assertNotIn("call_secret", raw_log)

    def test_prompt_component_tokens_preserve_zero_values(self) -> None:
        component_tokens = build_prompt_component_tokens(
            messages=[],
            prompt_tokens_total=0,
            token_count=lambda text: len(text.split()),
        )

        self.assertEqual(component_tokens["evidence_card_count"], 0)
        self.assertEqual(component_tokens["evidence_kinds"], [])
        self.assertEqual(component_tokens["evidence_cards"], [])
        for value in component_tokens["components"].values():
            self.assertEqual(value, 0)

    def test_native_route_prefix_anchor_diagnostics_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "diag.jsonl"
            payload = {
                "cache_prompt": True,
                "stream": True,
                "route_prefix_anchor": True,
                "messages": [
                    {"role": "system", "content": "runtime route policy placeholder"},
                    {"role": "user", "content": "placeholder route request"},
                ],
            }
            metadata = {
                "route_anchor_enabled": True,
                "route_anchor_attempted": True,
                "route_anchor_hit": True,
                "route_anchor_miss": False,
                "capture_attempted": False,
                "restore_attempted": True,
                "restore_used": True,
                "fallback_reason": None,
                "prefix_hash": "prefix-hash-placeholder",
                "prefix_token_count": 693,
                "checkpoint_size": 2048,
                "checkpoint_size_bytes": 2048,
                "checkpoint_age_ms": 42,
                "anchor_invalidated": False,
                "invalidation_reason": None,
                "cached_tokens": 693,
                "evaluated_tokens": 21,
                "lcp_tokens": 693,
                "phase": "route",
            }
            with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": str(log_path)}, clear=False):
                with native_request_context(endpoint="/chat/stream", payload=payload):
                    emit_native_route_prefix_anchor_event(metadata)

            event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            raw_log = json.dumps(event)

        self.assertEqual(event["event"], "kv_diag_route_prefix_anchor")
        self.assertEqual(event["phase"], "route")
        self.assertTrue(event["route_anchor_enabled"])
        self.assertTrue(event["route_anchor_attempted"])
        self.assertTrue(event["route_anchor_hit"])
        self.assertTrue(event["restore_attempted"])
        self.assertTrue(event["restore_used"])
        self.assertEqual(event["prefix_hash"], "prefix-hash-placeholder")
        self.assertEqual(event["prefix_token_count"], 693)
        self.assertEqual(event["cached_tokens"], 693)
        self.assertEqual(event["evaluated_tokens"], 21)
        self.assertEqual(event["lcp_tokens"], 693)
        self.assertEqual(event["checkpoint_size_bytes"], 2048)
        self.assertEqual(event["checkpoint_age_ms"], 42)
        self.assertFalse(event["anchor_invalidated"])
        self.assertIsNone(event["invalidation_reason"])
        self.assertEqual(event["role_sequence"], ["system", "user"])
        self.assertNotIn("runtime route policy placeholder", raw_log)
        self.assertNotIn("placeholder route request", raw_log)


if __name__ == "__main__":
    unittest.main()

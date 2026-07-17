from __future__ import annotations

import json
import os
from pathlib import Path
import random
import string
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult, Message
from orbit.runtime import ChatRuntime
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_healing import (
    analyze_tool_attempt,
    build_tool_call_repair,
    correlate_tool_attempt_shadow,
    observe_tool_attempt_shadow,
    reset_tool_call_healing_diagnostics,
    tool_call_healing_status,
    validate_active_tool_calls,
)
from orbit.runtime.tools import ToolResult, default_tool_names, tool_definitions
from orbit.tool_healing_config import resolve_tool_call_healing, resolve_tool_call_healing_shadow


class ToolHealingShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._post_tool_final_reuse_env = mock.patch.dict(
            os.environ,
            {"ORBIT_POST_TOOL_FINAL_REUSE": "0"},
        )
        self._post_tool_final_reuse_env.start()
        self.addCleanup(self._post_tool_final_reuse_env.stop)

    def _analyze(
        self,
        text: str,
        *,
        allowed: tuple[str, ...] | None = None,
        user_prompt: str = "perform the requested operation",
        tool_calls: list[dict[str, object]] | None = None,
        workdir: Path | None = None,
        finish_reason: str | None = None,
    ):
        return analyze_tool_attempt(
            text=text,
            tool_calls=tool_calls or [],
            tool_definitions=tool_definitions(),
            allowed_tool_names=allowed or default_tool_names(),
            workdir=workdir or Path.cwd(),
            user_prompt=user_prompt,
            finish_reason=finish_reason,
        )

    def _repair(
        self,
        text: str,
        *,
        allowed: tuple[str, ...] | None = None,
        user_prompt: str = "perform the requested operation",
        finish_reason: str | None = "stop",
        workdir: Path | None = None,
    ):
        return build_tool_call_repair(
            text=text,
            tool_calls=[],
            tool_definitions=tool_definitions(),
            allowed_tool_names=allowed or default_tool_names(),
            workdir=workdir or Path.cwd(),
            user_prompt=user_prompt,
            finish_reason=finish_reason,
        )

    def test_healing_config_is_on_by_default_and_invalid_fails_closed(self) -> None:
        default = resolve_tool_call_healing({})
        self.assertTrue(default.enabled)
        self.assertEqual(default.source, "default")
        self.assertFalse(resolve_tool_call_healing({"ORBIT_TOOL_CALL_HEALING": "0"}).enabled)
        self.assertTrue(resolve_tool_call_healing({"ORBIT_TOOL_CALL_HEALING": "1"}).enabled)
        invalid = resolve_tool_call_healing({"ORBIT_TOOL_CALL_HEALING": "true"})
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.validation_error, "invalid_healing_value")

    def test_healing_status_counts_repairs_and_structural_rejections_without_payloads(self) -> None:
        reset_tool_call_healing_diagnostics()
        try:
            no_attempt = build_tool_call_repair(
                text="ordinary prose", tool_calls=[], tool_definitions=tool_definitions(),
                allowed_tool_names=default_tool_names(), workdir=Path.cwd(),
                user_prompt="fixture", finish_reason="stop",
            )
            rejected = build_tool_call_repair(
                text='<|tool_call>{"name":"system_info","arguments":{"extra":"SECRET"},}<tool_call|>',
                tool_calls=[], tool_definitions=tool_definitions(),
                allowed_tool_names=default_tool_names(), workdir=Path.cwd(),
                user_prompt="fixture", finish_reason="stop",
            )
            repaired = build_tool_call_repair(
                text='<|tool_call>{"name":"system_info","arguments":{},}<tool_call|>',
                tool_calls=[], tool_definitions=tool_definitions(),
                allowed_tool_names=default_tool_names(), workdir=Path.cwd(),
                user_prompt="fixture", finish_reason="stop",
            )
            status = tool_call_healing_status()
        finally:
            reset_tool_call_healing_diagnostics()

        self.assertEqual(no_attempt.reason, "no_attempt")
        self.assertEqual(rejected.reason, "canonical_validation_failed")
        self.assertIsNotNone(repaired.tool_call)
        self.assertTrue(status["tool_call_healing_enabled"])
        self.assertEqual(status["tool_call_healing_source"], "default")
        self.assertEqual(status["tool_call_healing_repair_count"], 1)
        self.assertEqual(status["tool_call_healing_rejection_count"], 1)
        self.assertEqual(
            status["tool_call_healing_last_rules"],
            ["remove_known_envelope", "remove_trailing_comma"],
        )
        self.assertNotIn("SECRET", json.dumps(status))
        self.assertNotIn("arguments", json.dumps(status))

    def test_healing_effective_status_respects_canonical_gate_kill_switch(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"},
            clear=False,
        ):
            status = tool_call_healing_status()

        self.assertFalse(status["tool_call_healing_enabled"])
        self.assertEqual(status["tool_call_healing_source"], "default")
        self.assertEqual(status["tool_call_healing_blocked_reason"], "canonical_gate_disabled")

    def test_opt_in_repair_whitelist_produces_complete_certificates(self) -> None:
        cases = {
            "envelope": '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
            "trailing": '<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
            "arguments_string": '<tool_call>{"name":"system_info","arguments":"{}"}</tool_call>',
            "registered_wrapper": '[TOOL_CALLS]system_info[ARGS]{}',
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                repaired = self._repair(text)
                self.assertEqual(repaired.reason, "formal_repair_authorized")
                self.assertIsNotNone(repaired.tool_call)
                certificate = repaired.certificate
                self.assertIsNotNone(certificate)
                assert certificate is not None
                self.assertTrue(all((
                    certificate.candidate_count_one,
                    certificate.tool_name_unchanged,
                    certificate.argument_count_unchanged,
                    certificate.argument_keys_unchanged,
                    certificate.argument_values_unchanged,
                    certificate.idempotent,
                    certificate.schema_valid,
                    certificate.policy_valid,
                    certificate.permission_valid,
                    certificate.operational_limits_valid,
                    certificate.finish_complete,
                )))
                self.assertIn("remove_known_envelope", certificate.repair_categories)

    def test_opt_in_repair_rejects_ambiguous_and_unsafe_inputs(self) -> None:
        cases = (
            ("plain_json", '{"name":"system_info","arguments":{}}', "insufficient_template_evidence"),
            ("quoted_example", 'Example: <tool_call>{"name":"system_info","arguments":{}}</tool_call>', "insufficient_template_evidence"),
            ("markdown", '```json\n<tool_call>{"name":"system_info","arguments":{}}</tool_call>\n```', "insufficient_template_evidence"),
            ("suffix", '<tool_call>{"name":"system_info","arguments":{}}</tool_call> example', "insufficient_template_evidence"),
            ("markup_string", '{"example":"<tool_call>{\\"name\\":\\"system_info\\"}</tool_call>"}', "insufficient_template_evidence"),
            ("multiple", '<tool_call>{"name":"system_info","arguments":{}}</tool_call><tool_call>{"name":"system_info","arguments":{}}</tool_call>', "candidate_count_not_one"),
            ("unknown", '<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "canonical_validation_failed"),
            ("missing", '<tool_call>{"name":"fetch_url","arguments":{}}</tool_call>', "canonical_validation_failed"),
            ("wrong_type", '<tool_call>{"name":"system_info","arguments":{"include_cpu":"yes"}}</tool_call>', "canonical_validation_failed"),
            ("extra", '<tool_call>{"name":"system_info","arguments":{"extra":true}}</tool_call>', "canonical_validation_failed"),
            ("duplicate", '<tool_call>{"name":"system_info","arguments":{"include_cpu":true,"include_cpu":false}}</tool_call>', "duplicate_key"),
            ("alias", '<tool_call>{"tool":"system_info","arguments":{}}</tool_call>', "repair_not_whitelisted"),
            ("top_level", '<tool_call>{"name":"system_info","include_cpu":true}</tool_call>', "repair_not_whitelisted"),
        )
        for name, text, reason in cases:
            with self.subTest(name=name):
                repaired = self._repair(text)
                self.assertIsNone(repaired.tool_call)
                self.assertEqual(repaired.reason, reason)

        length = self._repair(
            '<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
            finish_reason="length",
        )
        self.assertEqual(length.reason, "incomplete_generation")
        permission = self._repair(
            '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
            allowed=("fetch_url",),
        )
        self.assertEqual(permission.reason, "canonical_validation_failed")
        policy = self._repair(
            '<tool_call>{"name":"exec_shell_full_command","arguments":{"command":"rm -f note.txt"}}</tool_call>',
            allowed=("exec_shell_full_command",),
            user_prompt="show note.txt",
        )
        self.assertEqual(policy.reason, "canonical_validation_failed")

    def test_opt_in_repair_property_preserves_atomic_values_and_is_idempotent(self) -> None:
        rng = random.Random(211)
        alphabet = string.ascii_letters + string.digits + " -_./?=&'"
        for _ in range(300):
            arguments = {
                "command": "printf %s " + json.dumps("".join(rng.choice(alphabet) for _ in range(24))),
                "timeout": rng.randint(1, 15),
                "max_output_size": rng.randint(1, 4096),
            }
            payload = {
                "name": "exec_shell_full_command",
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            }
            repaired = self._repair(f"<tool_call>{json.dumps(payload)}</tool_call>")
            self.assertIsNotNone(repaired.tool_call)
            function = repaired.tool_call["function"]
            self.assertEqual(function["name"], "exec_shell_full_command")
            self.assertEqual(json.loads(function["arguments"]), arguments)
            certificate = repaired.certificate
            self.assertIsNotNone(certificate)
            assert certificate is not None
            self.assertTrue(certificate.idempotent)
            self.assertTrue(certificate.argument_values_unchanged)

    def test_formal_repair_whitelist_is_exact_and_fail_closed(self) -> None:
        authorized = (
            '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
            '{"name":"system_info","arguments":{},}',
            '{"name":"fetch_url","arguments":"{\\"url\\":\\"https://example.com/a?x=1\\"}"}',
            '{"type":"function","function":{"name":"system_info","arguments":{}}}',
            '[TOOL_CALLS]system_info[ARGS]{}',
        )
        rejected = (
            ('{"tool":"system_info","arguments":{}}', None),
            ('{"name":"list_directory","path":"."}', None),
            ('<tool_call>{"name":"system_info","arguments":{}', None),
            ('{"name":"system_info","arguments":{},}', "length"),
            ('{"name":"system_info","arguments":{},}', "cancelled"),
            ('{"name":"system_info","arguments":{},}', "timeout"),
            ('{"name":"exec_shell_full_command","arguments":{"command":"rm -f note.txt"}}', None),
        )
        for text in authorized:
            with self.subTest(text=text):
                report = self._analyze(text)
                self.assertTrue(report.formal_repairable)
                self.assertEqual(report.formal_repair_reason, "formal_equivalence_proven")
        for text, finish_reason in rejected:
            with self.subTest(text=text):
                report = self._analyze(
                    text,
                    finish_reason=finish_reason,
                    user_prompt="show note.txt" if "rm -f" in text else "perform the operation",
                )
                self.assertFalse(report.formal_repairable)

        unavailable = self._analyze(
            '{"name":"system_info","arguments":{},}',
            allowed=("fetch_url",),
        )
        self.assertFalse(unavailable.formal_repairable)
        self.assertEqual(unavailable.validation_error, "tool_not_enabled")

    def test_formal_repair_property_preserves_random_atomic_values(self) -> None:
        rng = random.Random(148)
        alphabet = string.ascii_letters + string.digits + " {}[]\\\"'/?=&,"
        for _ in range(250):
            command = "printf %s " + json.dumps("".join(rng.choice(alphabet) for _ in range(40)))
            arguments = {"command": command, "timeout": rng.randint(1, 10)}
            canonical_text = json.dumps(
                {"name": "exec_shell_full_command", "arguments": arguments},
                separators=(",", ":"),
            )
            trailing = canonical_text[:-1] + ",}"
            encoded = json.dumps(
                {
                    "name": "exec_shell_full_command",
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
                separators=(",", ":"),
            )
            canonical = self._analyze(canonical_text)
            for repaired in (trailing, encoded, f"<tool_call>{canonical_text}</tool_call>"):
                report = self._analyze(repaired)
                self.assertTrue(report.formal_repairable, msg=repaired)
                self.assertEqual(report.normalized_tool_name_hash, canonical.normalized_tool_name_hash)
                self.assertEqual(report.normalized_arguments_hash, canonical.normalized_arguments_hash)
                self.assertEqual(report.formal_argument_count, len(arguments))

    def test_active_calls_use_same_strict_contract_observationally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            common = {
                "tool_definitions": tool_definitions(),
                "allowed_tool_names": default_tool_names(),
                "workdir": Path(tmp),
                "user_prompt": "inspect without modifying files",
            }
            valid = validate_active_tool_calls(
                [{"function": {"name": "system_info", "arguments": "{}"}}],
                **common,
            )
            extra = validate_active_tool_calls(
                [{"function": {"name": "system_info", "arguments": '{"extra":true}'}}],
                **common,
            )
            clamped = validate_active_tool_calls(
                [{"function": {"name": "list_directory", "arguments": '{"max_entries":2000}'}}],
                **common,
            )
            multiple = validate_active_tool_calls(
                [
                    {"function": {"name": "system_info", "arguments": "{}"}},
                    {"function": {"name": "system_info", "arguments": "{}"}},
                ],
                **common,
            )

        self.assertEqual(valid, ("valid", None))
        self.assertEqual(extra, ("rejected_validation", "additional_property"))
        self.assertEqual(clamped, ("rejected_validation", "limit_out_of_range"))
        self.assertEqual(multiple, ("rejected_validation", "multiple_candidates"))

    def test_valid_tool_call_is_detected_extracted_and_validated(self) -> None:
        report = self._analyze('<tool_call>{"name":"system_info","arguments":{}}</tool_call>')

        self.assertTrue(report.attempt_detected)
        self.assertTrue(report.candidate_extracted)
        self.assertEqual(report.tool_name, "system_info")
        self.assertEqual(report.outcome, "valid_shadow_candidate")

    def test_safe_formal_repairs_are_reported(self) -> None:
        cases = (
            (
                '<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
                "remove_trailing_comma",
            ),
            (
                '<tool_call>{"name":"system_info","arguments":{}',
                "close_json_structure",
            ),
            (
                'prefix <tool_call>{"name":"system_info","arguments":{}}</tool_call> suffix',
                None,
            ),
            ('{"tool":"system_info","arguments":{}}', "normalize_tool_field"),
            ('{"name":"list_directory","path":"."}', "wrap_top_level_arguments"),
            (
                '{"name":"fetch_url","arguments":"{\\"url\\":\\"https://example.com/?a=1\\"}"}',
                "decode_arguments_string",
            ),
            (
                '{"type":"function","function":{"name":"system_info","arguments":{}}}',
                "unwrap_function_object",
            ),
            ("[TOOL_CALLS]system_info[ARGS]{}", "unwrap_named_wrapper"),
            ("<|tool_call>call:system_info {}<tool_call|>", "unwrap_named_wrapper"),
        )
        for text, repair in cases:
            with self.subTest(text=text):
                report = self._analyze(text)
                self.assertEqual(report.outcome, "valid_shadow_candidate")
                if repair:
                    self.assertIn(repair, report.repairs)

    def test_formal_repairs_preserve_the_canonical_payload_and_are_idempotent(self) -> None:
        canonical = self._analyze('{"name":"fetch_url","arguments":{"url":"https://example.com/?a=1"}}')
        repaired_forms = (
            '<tool_call>{"name":"fetch_url","arguments":{"url":"https://example.com/?a=1"},}</tool_call>',
            '<tool_call>{"name":"fetch_url","arguments":{"url":"https://example.com/?a=1"}',
            '{"tool":"fetch_url","arguments":{"url":"https://example.com/?a=1"}}',
            '{"name":"fetch_url","url":"https://example.com/?a=1"}',
            '{"name":"fetch_url","arguments":"{\\"url\\":\\"https://example.com/?a=1\\"}"}',
        )
        self.assertEqual(canonical.repairs, ())
        for text in repaired_forms:
            with self.subTest(text=text):
                report = self._analyze(text)
                self.assertEqual(report.outcome, "valid_shadow_candidate")
                self.assertEqual(report.normalized_tool_name_hash, canonical.normalized_tool_name_hash)
                self.assertEqual(report.normalized_arguments_hash, canonical.normalized_arguments_hash)

    def test_unclosed_tag_with_complete_json_is_formally_normalized(self) -> None:
        report = self._analyze('<tool_call>{"name":"system_info","arguments":{}}')

        self.assertEqual(report.outcome, "valid_shadow_candidate")
        self.assertIn("unclosed_tool_call_tag", report.repairs)

    def test_schema_and_availability_failures_are_closed(self) -> None:
        cases = (
            ('{"name":"made_up","arguments":{}}', "tool_not_enabled"),
            ('{"name":"fetch_url","arguments":{}}', "missing_required"),
            ('{"name":"fetch_url","arguments":{"url":42}}', "type_mismatch"),
            (
                '{"name":"fetch_url","arguments":{"url":"https://example.com","extra":true}}',
                "additional_property",
            ),
            (
                '{"name":"fetch_url","arguments":{"url":"https://example.com"}}',
                "tool_not_enabled",
                ("system_info",),
            ),
            (
                '{"name":"fetch_url","arguments":{"url":"file:///etc/passwd"}}',
                "unsupported_url",
            ),
        )
        for case in cases:
            text, error, *allowed = case
            with self.subTest(text=text):
                report = self._analyze(text, allowed=allowed[0] if allowed else None)
                self.assertEqual(report.outcome, "rejected_validation")
                self.assertEqual(report.validation_error, error)

    def test_top_level_arguments_are_wrapped_only_for_exact_schema_keys(self) -> None:
        valid = self._analyze('{"name":"list_directory","path":".","recursive":true}')
        invalid = self._analyze('{"name":"list_directory","path":".","recursiv":true}')

        self.assertEqual(valid.outcome, "valid_shadow_candidate")
        self.assertIn("wrap_top_level_arguments", valid.repairs)
        self.assertEqual(invalid.outcome, "rejected_validation")
        self.assertEqual(invalid.validation_error, "additional_property")
        self.assertNotIn("wrap_top_level_arguments", invalid.repairs)
        self.assertEqual(invalid.validation_path, "arguments.<unknown>")

    def test_ambiguous_and_duplicate_structures_are_rejected(self) -> None:
        cases = (
            (
                '<tool_call>{"name":"system_info","arguments":{}}</tool_call>'
                '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
                "multiple_candidates",
            ),
            (
                '{"name":"system_info","arguments":{}} '
                '{"name":"system_info","arguments":{}}',
                "multiple_candidates",
            ),
            (
                '<tool_call>{"name":"system_info","name":"fetch_url","arguments":{}}</tool_call>',
                "duplicate_key",
            ),
            (
                '{"name":"system_info","tool":"system_info","arguments":{}}',
                "ambiguous_tool_identity",
            ),
            (
                '{"name":"system_info","arguments":{},"unexpected":1}',
                "ambiguous_wrapper",
            ),
            (
                '{"name":"system_info","arguments":[]}',
                "arguments_not_object",
            ),
        )
        for text, error in cases:
            with self.subTest(text=text):
                report = self._analyze(text)
                self.assertIn(report.outcome, {"rejected_parse", "rejected_validation"})
                self.assertEqual(report.parse_error or report.validation_error, error)

    def test_normal_text_and_json_citations_are_not_attempts(self) -> None:
        cases = (
            "The system_info tool can report hardware details.",
            'For example, JSON can contain {"name":"system_info","arguments":{}} in documentation.',
            'The field names are "name", "tool", "function", and "arguments".',
        )
        for text in cases:
            with self.subTest(text=text):
                report = self._analyze(text)
                self.assertFalse(report.attempt_detected)
                self.assertEqual(report.outcome, "no_attempt")

    def test_interrupted_output_with_unterminated_string_fails_closed(self) -> None:
        report = self._analyze(
            '<tool_call>{"name":"fetch_url","arguments":{"url":"https://example.com/incomplete'
        )

        self.assertEqual(report.outcome, "rejected_parse")
        self.assertEqual(report.parse_error, "unterminated_json_string")

    def test_dangerous_call_is_rejected_by_existing_runtime_policy(self) -> None:
        report = self._analyze(
            '{"name":"exec_shell_full_command","arguments":{"command":"rm -f note.txt"}}',
            user_prompt="show note.txt",
        )

        self.assertEqual(report.outcome, "rejected_validation")
        self.assertEqual(report.validation_error, "policy_read_only_mutation")

    def test_operational_limits_permissions_and_shell_syntax_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            cases = (
                (
                    '{"name":"exec_shell_full_command","arguments":{"command":"printf \\\"oops"}}',
                    "invalid_shell_syntax",
                ),
                (
                    '{"name":"exec_shell_full_command","arguments":{"command":"pwd","timeout":99}}',
                    "limit_out_of_range",
                ),
                (
                    '{"name":"list_directory","arguments":{"path":"../"}}',
                    "path_outside_workdir",
                ),
                (
                    '{"name":"list_directory","arguments":{"files_only":true,"dirs_only":true}}',
                    "contradictory_flags",
                ),
            )
            for text, error in cases:
                with self.subTest(text=text):
                    report = self._analyze(text, workdir=workdir)
                    self.assertEqual(report.outcome, "rejected_validation")
                    self.assertEqual(report.validation_error, error)

    def test_scanner_handles_nested_strings_escapes_and_braces(self) -> None:
        rng = random.Random(147)
        alphabet = string.ascii_letters + string.digits + '{}[]\\" /?=&'
        for _ in range(100):
            query = "".join(rng.choice(alphabet) for _ in range(32))
            url = "https://example.com/search?q=" + query
            text = "<tool_call>" + json.dumps(
                {"name": "fetch_url", "arguments": {"url": url}},
                ensure_ascii=False,
            ) + "</tool_call>"
            report = self._analyze(text)
            self.assertEqual(report.outcome, "valid_shadow_candidate", msg=text)

    def test_backend_structured_call_is_validated_without_rewriting(self) -> None:
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "system_info", "arguments": "{}"},
        }
        report = self._analyze("", tool_calls=[tool_call])

        self.assertEqual(report.outcome, "valid_shadow_candidate")
        self.assertEqual(report.tool_name, "system_info")
        self.assertIn("unwrap_function_object", report.repairs)
        self.assertIn("decode_arguments_string", report.repairs)

        duplicate = self._analyze(
            '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
            tool_calls=[tool_call],
        )
        self.assertEqual(duplicate.outcome, "rejected_parse")
        self.assertEqual(duplicate.parse_error, "multiple_candidates")

    def test_shadow_config_is_default_off_and_invalid_values_fail_closed(self) -> None:
        self.assertFalse(resolve_tool_call_healing_shadow({}).enabled)
        self.assertFalse(resolve_tool_call_healing_shadow({"ORBIT_TOOL_CALL_HEALING_SHADOW": "0"}).enabled)
        self.assertTrue(resolve_tool_call_healing_shadow({"ORBIT_TOOL_CALL_HEALING_SHADOW": "1"}).enabled)
        invalid = resolve_tool_call_healing_shadow({"ORBIT_TOOL_CALL_HEALING_SHADOW": "yes"})
        self.assertFalse(invalid.enabled)
        self.assertEqual(invalid.validation_error, "invalid_shadow_value")

    def test_diagnostic_is_bounded_and_does_not_store_complete_output(self) -> None:
        secret = "sensitive-route-output-" + ("x" * 300)
        text = '<tool_call>{"name":"exec_shell_full_command","arguments":{"command":"printf ' + secret + '"}}</tool_call>'
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "ORBIT_TOOL_CALL_HEALING_SHADOW": "1",
                    "ORBIT_KV_DIAG_FILE": str(log_path),
                },
                clear=False,
            ):
                report = observe_tool_attempt_shadow(
                    text=text,
                    tool_calls=[],
                    tool_definitions=tool_definitions(),
                    allowed_tool_names=default_tool_names(),
                    workdir=Path(tmp),
                    user_prompt="write the requested value",
                    finish_reason="stop",
                    output_tokens=20,
                )
            event = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertIsNotNone(report)
        self.assertEqual(event["event"], "kv_diag_tool_healing_shadow")
        self.assertLessEqual(len(event["candidate_excerpt"]), 96)
        self.assertEqual(len(event["candidate_hash"]), 16)
        self.assertEqual(len(event["output_hash"]), 16)
        self.assertEqual(event["output_chars"], len(text))
        self.assertNotIn("tool_name", event)
        self.assertIn("formal_repairable", event)
        self.assertIn("formal_repair_reason", event)
        self.assertEqual(event["original_tool_name_hash"], event["normalized_tool_name_hash"])
        self.assertNotIn("sensitive-route-output", event["candidate_excerpt"])
        self.assertNotIn(secret, json.dumps(event))
        self.assertNotIn(text, json.dumps(event))

    def test_excerpt_and_validation_path_do_not_expose_credentials_or_unknown_keys(self) -> None:
        secret = "sk-live-SECRET-123456789"
        report = self._analyze(
            '{"name":"fetch_url","arguments":{"url":"https://example.com/?token=' + secret + '"}}'
        )
        invalid = self._analyze('{"name":"system_info","arguments":{"api_key_' + secret + '":true}}')

        self.assertNotIn(secret, report.candidate_excerpt or "")
        self.assertNotIn("https://", report.candidate_excerpt or "")
        self.assertEqual(invalid.validation_path, "arguments.<unknown>")
        self.assertNotIn(secret, invalid.validation_path or "")

    def test_active_correlation_reports_exact_match_and_divergence_without_raw_values(self) -> None:
        shadow = self._analyze('{"name":"system_info","arguments":{}}')
        exact_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "system_info", "arguments": "{}"},
        }
        exact = correlate_tool_attempt_shadow(report=shadow, active_tool_calls=[exact_call], phase="tool_call")
        self.assertIsNotNone(exact)
        self.assertEqual(exact.agreement, "exact_match")

        no_active = correlate_tool_attempt_shadow(report=shadow, active_tool_calls=[], phase="tool_call")
        self.assertEqual(no_active.agreement, "shadow_only")

        name_divergence = correlate_tool_attempt_shadow(
            report=shadow,
            active_tool_calls=[
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "fetch_url", "arguments": '{"url":"https://example.com"}'},
                }
            ],
            phase="tool_call",
        )
        self.assertEqual(name_divergence.agreement, "tool_name_divergence")

        arguments_divergence = correlate_tool_attempt_shadow(
            report=shadow,
            active_tool_calls=[
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {"name": "system_info", "arguments": '{"include_gpu":true}'},
                }
            ],
            phase="tool_call",
        )
        self.assertEqual(arguments_divergence.agreement, "arguments_divergence")

    def test_disabled_shadow_emits_nothing_and_diagnostic_failure_is_observational(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            with mock.patch.dict(
                os.environ,
                {
                    "ORBIT_TOOL_CALL_HEALING_SHADOW": "0",
                    "ORBIT_KV_DIAG_FILE": str(log_path),
                },
                clear=False,
            ):
                disabled = observe_tool_attempt_shadow(
                    text='{"name":"system_info","arguments":{}}',
                    tool_calls=[],
                    tool_definitions=tool_definitions(),
                    allowed_tool_names=default_tool_names(),
                    workdir=Path(tmp),
                    user_prompt="show specs",
                    finish_reason="stop",
                    output_tokens=5,
                )
            self.assertIsNone(disabled)
            self.assertFalse(log_path.exists())

            with mock.patch.dict(os.environ, {"ORBIT_TOOL_CALL_HEALING_SHADOW": "1"}, clear=False):
                with mock.patch(
                    "orbit.runtime.tool_healing.emit_tool_healing_shadow",
                    side_effect=OSError("diagnostic sink unavailable"),
                ):
                    report = observe_tool_attempt_shadow(
                        text='{"name":"system_info","arguments":{}}',
                        tool_calls=[],
                        tool_definitions=tool_definitions(),
                        allowed_tool_names=default_tool_names(),
                        workdir=Path(tmp),
                        user_prompt="show specs",
                        finish_reason="stop",
                        output_tokens=5,
                    )
            self.assertIsNotNone(report)
            self.assertEqual(report.outcome, "valid_shadow_candidate")

    def test_shadow_mode_does_not_change_valid_tool_execution_or_model_calls(self) -> None:
        class ToolBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
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
                                    "arguments": '{"command":"printf shadow-equivalence"}',
                                },
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="specs returned",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=20,
                    completion_tokens=3,
                    cached_tokens=4,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            for enabled in ("0", "1"):
                backend = ToolBackend()
                tool_events: list[tuple[str, str]] = []
                tool_results: list[tuple[str, int, str, str]] = []
                with mock.patch.dict(
                    os.environ,
                    {
                        "ORBIT_TOOL_CALL_HEALING_SHADOW": enabled,
                        "ORBIT_TOOL_CALL_HEALING": "0",
                        "ORBIT_KV_DIAG_FILE": str(log_path),
                    },
                    clear=False,
                ):
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_with_tools(
                        "run printf shadow-equivalence",
                        temperature=0,
                        max_tokens=32,
                        workdir=Path(tmp),
                        tool_names=("exec_shell_full_command",),
                        on_tool_call=lambda name, arguments: tool_events.append((name, arguments)),
                        on_tool_result=lambda name, chars, source, content: tool_results.append(
                            (name, chars, source, content)
                        ),
                    )
                tool_messages = [message for message in runtime.messages if message.get("role") == "tool"]
                outcomes.append(
                    (
                        result.content,
                        result.finish_reason,
                        backend.calls,
                        tool_events,
                        tool_results,
                        [(message.get("name"), bool(message.get("content"))) for message in tool_messages],
                    )
                )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][:3], ("specs returned", "stop", 3))
        self.assertEqual(
            outcomes[0][3],
            [("exec_shell_full_command", '{"command":"printf shadow-equivalence"}')],
        )
        self.assertEqual(len(outcomes[0][4]), 1)
        self.assertEqual(outcomes[0][4], [("exec_shell_full_command", 18, "orbit", "shadow-equivalence")])
        self.assertEqual(outcomes[0][5], [("exec_shell_full_command", True)])
        correlation = next(event for event in events if event["event"] == "kv_diag_tool_healing_terminal")
        self.assertEqual(correlation["agreement"], "exact_match")
        self.assertEqual(correlation["active_outcome"], "executed")
        self.assertEqual(correlation["terminal_reason"], None)
        self.assertEqual(correlation["active_canonical_outcome"], "valid")
        self.assertIsNone(correlation["active_canonical_error"])
        self.assertNotIn("exec_shell_full_command", json.dumps(correlation))

    def test_shadow_repair_candidate_is_not_applied_to_execution(self) -> None:
        class MalformedBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    content='<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=8,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            for enabled in ("0", "1"):
                backend = MalformedBackend()
                tool_events: list[tuple[str, str]] = []
                with mock.patch.dict(
                    os.environ,
                    {
                        "ORBIT_TOOL_CALL_HEALING_SHADOW": enabled,
                        "ORBIT_TOOL_CALL_HEALING": "0",
                        "ORBIT_KV_DIAG_FILE": str(log_path),
                    },
                    clear=False,
                ):
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_with_tools(
                        "show system specs",
                        temperature=0,
                        max_tokens=32,
                        workdir=Path(tmp),
                        tool_names=("system_info",),
                        on_tool_call=lambda name, arguments: tool_events.append((name, arguments)),
                    )
                outcomes.append((result.content, result.finish_reason, backend.calls, tool_events))

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][1:], ("stop", 1, []))
        correlation = next(event for event in events if event["event"] == "kv_diag_tool_healing_terminal")
        shadow = next(event for event in events if event["event"] == "kv_diag_tool_healing_shadow")
        self.assertTrue(shadow["formal_repairable"])
        self.assertEqual(shadow["formal_repair_reason"], "formal_equivalence_proven")
        self.assertEqual(correlation["agreement"], "shadow_only")
        self.assertEqual(correlation["active_outcome"], "rejected_parse")
        self.assertEqual(correlation["terminal_reason"], "active_normalizer_rejected")

    def test_opt_in_repair_reenters_normal_canonical_execution_without_retry(self) -> None:
        class Backend:
            def __init__(self, *, malformed: bool) -> None:
                self.malformed = malformed
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    if self.malformed:
                        return ChatResult(
                            content='<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
                            model="fake", finish_reason="stop", tool_calls=[],
                            prompt_tokens=10, completion_tokens=8, cached_tokens=0,
                            prompt_tokens_per_second=None, generation_tokens_per_second=None,
                        )
                    return ChatResult(
                        content="", model="fake", finish_reason="tool_calls",
                        tool_calls=[{
                            "id": "normal-call", "type": "function",
                            "function": {"name": "system_info", "arguments": "{}"},
                        }],
                        prompt_tokens=10, completion_tokens=2, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="specs complete", model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=2, cached_tokens=4,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            for malformed, healing in ((False, "0"), (False, "1"), (True, "1")):
                backend = Backend(malformed=malformed)
                tool_events = []
                with mock.patch.dict(
                    os.environ,
                    {
                        "ORBIT_TOOL_CALL_HEALING": healing,
                        "ORBIT_TOOL_CALL_CANONICAL_GATE": "1",
                    },
                    clear=False,
                ), mock.patch(
                    "orbit.runtime.tool_backends.execute_tool",
                    return_value=ToolResult("system_info", "fixture specs"),
                ) as execute:
                    result = ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                        "show specs", temperature=0, max_tokens=32, workdir=Path(tmp),
                        tool_names=("system_info",),
                        on_tool_call=lambda name, arguments: tool_events.append((name, json.loads(arguments))),
                    )
                outcomes.append((result.content, result.finish_reason, backend.calls, tool_events, execute.call_count))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[1], outcomes[2])
        self.assertEqual(outcomes[0], ("specs complete", "stop", 2, [("system_info", {})], 1))

    def test_default_on_and_kill_switch_matrix_for_all_authorized_repairs(self) -> None:
        cases = (
            '<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
            '<tool_call>{"name":"system_info","arguments":{},}</tool_call>',
            '<tool_call>{"name":"system_info","arguments":"{}"}</tool_call>',
            '[TOOL_CALLS]system_info[ARGS]{}',
        )

        class Backend:
            def __init__(self, content: str) -> None:
                self.content = content
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content=self.content, model="fake", finish_reason="stop", tool_calls=[],
                        prompt_tokens=10, completion_tokens=8, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done", model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=1, cached_tokens=4,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            for content in cases:
                with self.subTest(content=content):
                    observed = []
                    for healing_env in ("0", None):
                        environment = {"ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}
                        if healing_env is not None:
                            environment["ORBIT_TOOL_CALL_HEALING"] = healing_env
                        backend = Backend(content)
                        tool_events: list[tuple[str, dict[str, object]]] = []
                        with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                            "orbit.runtime.tool_backends.execute_tool",
                            return_value=ToolResult("system_info", "fixture specs"),
                        ) as execute:
                            result = ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                                "show specs", temperature=0, max_tokens=32, workdir=Path(tmp),
                                tool_names=("system_info",),
                                on_tool_call=lambda name, arguments: tool_events.append((name, json.loads(arguments))),
                            )
                        observed.append((result.finish_reason, backend.calls, tool_events, execute.call_count))

                    self.assertEqual(observed[0], ("stop", 1, [], 0))
                    self.assertEqual(observed[1], ("stop", 2, [("system_info", {})], 1))

    def test_opt_in_order_is_repair_canonical_guardrail_executor_without_revalidation(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='<|tool_call>{"name":"system_info","arguments":{},}<tool_call|>',
                        model="fake", finish_reason="stop", tool_calls=[],
                        prompt_tokens=10, completion_tokens=8, cached_tokens=0,
                        prompt_tokens_per_second=None, generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="done", model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=1, cached_tokens=4,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        order: list[str] = []
        original_repair = build_tool_call_repair
        from orbit.runtime import tool_healing as healing_module
        original_validate = healing_module.validate_canonical_tool_call

        def repair_spy(**kwargs):
            order.append("repair")
            return original_repair(**kwargs)

        def canonical_spy(*args, **kwargs):
            order.append("canonical")
            return original_validate(*args, **kwargs)

        def guard_spy(*args, **kwargs):
            order.append("guardrail")
            return False

        def executor_spy(*args, **kwargs):
            order.append("executor")
            return ToolResult("system_info", "ok")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"ORBIT_TOOL_CALL_HEALING": "1", "ORBIT_TOOL_CALL_CANONICAL_GATE": "1"},
            clear=False,
        ), mock.patch(
            "orbit.runtime.tool_loop.build_tool_call_repair", side_effect=repair_spy
        ), mock.patch(
            "orbit.runtime.tool_healing.validate_canonical_tool_call", side_effect=canonical_spy
        ) as validate, mock.patch(
            "orbit.runtime.tool_loop.validate_canonical_tool_call_payload",
            side_effect=AssertionError("repaired call was validated twice in the loop"),
        ), mock.patch(
            "orbit.runtime.tool_backends.validate_canonical_tool_call",
            side_effect=AssertionError("repaired call was validated twice in the executor"),
        ), mock.patch(
            "orbit.runtime.tool_loop._should_guard_existing_file_rewrite", side_effect=guard_spy
        ), mock.patch(
            "orbit.runtime.tool_backends.execute_tool", side_effect=executor_spy
        ):
            backend = Backend()
            result = ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                "show specs", temperature=0, max_tokens=32, workdir=Path(tmp),
                tool_names=("system_info",),
            )

        self.assertEqual((result.content, result.finish_reason, backend.calls), ("done", "stop", 2))
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(order, ["repair", "canonical", "guardrail", "executor"])

    def test_rejected_repairs_do_not_reach_guardrails_events_or_executor(self) -> None:
        class Backend:
            def __init__(self, content: str) -> None:
                self.content = content
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                return ChatResult(
                    content=self.content, model="fake", finish_reason="stop", tool_calls=[],
                    prompt_tokens=10, completion_tokens=8, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        cases = (
            (
                "schema",
                '<|tool_call>{"name":"system_info","arguments":{"extra":true},}<tool_call|>',
                ("system_info",),
                "show specs",
            ),
            (
                "permission",
                '<|tool_call>{"name":"fetch_url","arguments":{"url":"https://example.invalid"},}<tool_call|>',
                ("system_info",),
                "fetch the URL",
            ),
            (
                "policy",
                '[TOOL_CALLS]exec_shell_full_command[ARGS]{"command":"rm -f note.txt",}',
                ("exec_shell_full_command",),
                "show note.txt",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for label, content, allowed, prompt in cases:
                with self.subTest(label=label), mock.patch.dict(
                    os.environ,
                    {"ORBIT_TOOL_CALL_HEALING": "1", "ORBIT_TOOL_CALL_CANONICAL_GATE": "1"},
                    clear=False,
                ), mock.patch(
                    "orbit.runtime.tool_loop._should_guard_existing_file_rewrite"
                ) as guard, mock.patch(
                    "orbit.runtime.tool_backends.execute_tool"
                ) as execute:
                    backend = Backend(content)
                    tool_events: list[tuple[str, str]] = []
                    result = ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                        prompt, temperature=0, max_tokens=32, workdir=Path(tmp),
                        tool_names=allowed,
                        on_tool_call=lambda name, arguments: tool_events.append((name, arguments)),
                    )
                self.assertEqual((backend.calls, result.finish_reason), (1, "stop"))
                self.assertEqual(tool_events, [])
                guard.assert_not_called()
                execute.assert_not_called()

    def test_opt_in_repair_kill_switch_and_ambiguous_text_never_execute(self) -> None:
        class Backend:
            def __init__(self, content: str, finish_reason: str = "stop") -> None:
                self.content = content
                self.finish_reason = finish_reason
                self.calls = 0

            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls += 1
                return ChatResult(
                    content=self.content, model="fake", finish_reason=self.finish_reason, tool_calls=[],
                    prompt_tokens=10, completion_tokens=8, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        cases = (
            ("gate_off", '<tool_call>{"name":"system_info","arguments":{},}</tool_call>', "stop", "0"),
            ("example", 'Example: <tool_call>{"name":"system_info","arguments":{}}</tool_call>', "stop", "1"),
            ("markdown", '```json\n<tool_call>{"name":"system_info","arguments":{}}</tool_call>\n```', "stop", "1"),
            ("multiple", '<tool_call>{"name":"system_info","arguments":{}}</tool_call><tool_call>{"name":"system_info","arguments":{}}</tool_call>', "stop", "1"),
            ("length", '<tool_call>{"name":"system_info","arguments":{},}</tool_call>', "length", "1"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, content, finish_reason, gate in cases:
                with self.subTest(name=name), mock.patch.dict(
                    os.environ,
                    {
                        "ORBIT_TOOL_CALL_HEALING": "1",
                        "ORBIT_TOOL_CALL_CANONICAL_GATE": gate,
                    },
                    clear=False,
                ), mock.patch("orbit.runtime.tool_backends.execute_tool") as execute:
                    backend = Backend(content, finish_reason)
                    ChatRuntime(backend=backend, system_prompt=None).ask_with_tools(
                        "show specs", temperature=0, max_tokens=32, workdir=Path(tmp),
                        tool_names=("system_info",),
                    )
                execute.assert_not_called()

    def test_opt_in_repair_timeout_cancel_and_reset_remain_bounded(self) -> None:
        class TimeoutBackend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                raise TimeoutError("healing timeout")

        class CancelBackend:
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                return ChatResult(
                    content='<tool_call>{"name":"system_info","arguments":{}}</tool_call>',
                    model="fake", finish_reason="cancelled", tool_calls=[],
                    prompt_tokens=10, completion_tokens=1, cached_tokens=0,
                    prompt_tokens_per_second=None, generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"ORBIT_TOOL_CALL_HEALING": "1", "ORBIT_TOOL_CALL_CANONICAL_GATE": "1"},
            clear=False,
        ), mock.patch("orbit.runtime.tool_backends.execute_tool") as execute:
            timeout_runtime = ChatRuntime(backend=TimeoutBackend(), system_prompt=None)
            with self.assertRaisesRegex(TimeoutError, "healing timeout"):
                timeout_runtime.ask_with_tools(
                    "show specs", temperature=0, max_tokens=32, workdir=Path(tmp), tool_names=("system_info",)
                )
            timeout_runtime.reset()
            cancel_runtime = ChatRuntime(backend=CancelBackend(), system_prompt=None)
            cancelled = cancel_runtime.ask_with_tools(
                "show specs", temperature=0, max_tokens=32, workdir=Path(tmp), tool_names=("system_info",)
            )
            cancel_runtime.reset()

        execute.assert_not_called()
        self.assertEqual(timeout_runtime.messages, [])
        self.assertEqual((cancelled.finish_reason, cancel_runtime.messages), ("cancelled", []))

    def test_active_executor_outcomes_reuse_the_real_guardrails_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"},
            clear=False,
        ):
            workdir = Path(tmp)
            executor = HybridToolExecutor(
                backend=None,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command", "list_directory", "system_info"),
                user_prompt="inspect the current files without changing them",
            )
            for index in range(1002):
                (workdir / f"entry-{index:04d}.txt").touch()

            success = executor.execute("system_info", {"ignored_extra": True}, chunk_budget={})
            clamped = executor.execute("list_directory", {"max_entries": 2000}, chunk_budget={})
            parse = executor.execute("system_info", "{", chunk_budget={})
            permission = HybridToolExecutor(
                backend=None,
                workdir=workdir,
                allowed_tool_names=(),
            ).execute("system_info", {}, chunk_budget={})
            policy = HybridToolExecutor(
                backend=None,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="show note.txt",
            ).execute(
                "exec_shell_full_command",
                {"command": "rm -f note.txt"},
                chunk_budget={},
            )
            guardrail = HybridToolExecutor(
                backend=None,
                workdir=workdir,
                allowed_tool_names=("exec_shell_full_command",),
                user_prompt="analyze source.py for vulnerabilities",
            ).execute("exec_shell_full_command", {"command": "ls"}, chunk_budget={})
            runtime_error = executor.execute(
                "exec_shell_full_command",
                {"command": "sh -c 'exit 7'"},
                chunk_budget={},
            )

        self.assertEqual((success.terminal_outcome, success.terminal_reason), ("executed", None))
        self.assertEqual((clamped.terminal_outcome, clamped.terminal_reason), ("executed", None))
        self.assertIn("shown=1000 total_seen=1002 truncated=true", clamped.result.content)
        self.assertEqual(parse.terminal_outcome, "rejected_parse")
        self.assertEqual(permission.terminal_outcome, "rejected_permission")
        self.assertEqual(policy.terminal_outcome, "rejected_policy")
        self.assertEqual(guardrail.terminal_outcome, "rejected_guardrail")
        self.assertEqual(runtime_error.terminal_outcome, "runtime_error")

        extra_shadow = self._analyze('{"name":"system_info","arguments":{"ignored_extra":true}}')
        range_shadow = self._analyze('{"name":"list_directory","arguments":{"max_entries":2000}}')
        self.assertEqual(extra_shadow.validation_error, "additional_property")
        self.assertEqual(range_shadow.validation_error, "limit_out_of_range")

    def test_retry_results_have_unique_attempt_ids_and_superseded_terminal_state(self) -> None:
        class RetryBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
                    return ChatResult(
                        content='<tool_call>{"name":"system_info"',
                        model="fake",
                        finish_reason="length",
                        tool_calls=[],
                        prompt_tokens=10,
                        completion_tokens=8,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                if self.calls == 2:
                    return ChatResult(
                        content="",
                        model="fake",
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call-retry",
                                "type": "function",
                                "function": {"name": "system_info", "arguments": "{}"},
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="system information returned",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=3,
                    cached_tokens=4,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "retry-shadow.jsonl"
            with mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_HEALING_SHADOW": "1", "ORBIT_KV_DIAG_FILE": str(log_path)},
                clear=False,
            ):
                backend = RetryBackend()
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                result = runtime.ask_with_tools(
                    "show system specs",
                    temperature=0,
                    max_tokens=32,
                    workdir=Path(tmp),
                    tool_names=("system_info",),
                )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        shadows = [event for event in events if event["event"] == "kv_diag_tool_healing_shadow"]
        terminals = [event for event in events if event["event"] == "kv_diag_tool_healing_terminal"]
        self.assertEqual((result.content, result.finish_reason), ("system information returned", "stop"))
        self.assertEqual(backend.calls, 3)
        self.assertEqual(len(shadows), 2)
        self.assertEqual(len(terminals), 2)
        self.assertEqual(len({event["attempt_id"] for event in shadows}), 2)
        self.assertEqual({event["attempt_id"] for event in shadows}, {event["attempt_id"] for event in terminals})
        self.assertEqual(terminals[0]["active_outcome"], "superseded")
        self.assertEqual(terminals[0]["terminal_reason"], "length_retry")
        self.assertEqual(terminals[1]["active_outcome"], "executed")
        self.assertEqual(terminals[1]["agreement"], "exact_match")
        self.assertEqual(terminals[1]["active_canonical_outcome"], "valid")
        self.assertIsNone(terminals[1]["active_canonical_error"])

    def test_terminal_diagnostic_failure_cannot_change_execution(self) -> None:
        class ToolBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                if self.calls == 1:
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
                                    "arguments": '{"command":"printf terminal-safe"}',
                                },
                            }
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                        cached_tokens=0,
                        prompt_tokens_per_second=None,
                        generation_tokens_per_second=None,
                    )
                return ChatResult(
                    content="terminal safe",
                    model="fake",
                    finish_reason="stop",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=2,
                    cached_tokens=4,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "ORBIT_TOOL_CALL_HEALING_SHADOW": "1",
                    "ORBIT_KV_DIAG_FILE": str(Path(tmp) / "terminal-failure.jsonl"),
                },
                clear=False,
            ):
                with mock.patch(
                    "orbit.runtime.kv_diag.emit_tool_healing_terminal",
                    side_effect=OSError("terminal sink unavailable"),
                ), mock.patch(
                    "orbit.runtime.tool_healing.validate_active_tool_calls",
                    side_effect=RuntimeError("validator unavailable"),
                ):
                    backend = ToolBackend()
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_with_tools(
                        "print terminal-safe",
                        temperature=0,
                        max_tokens=32,
                        workdir=Path(tmp),
                        tool_names=("exec_shell_full_command",),
                    )

        self.assertEqual((result.content, result.finish_reason), ("terminal safe", "stop"))
        self.assertEqual(backend.calls, 3)
        self.assertTrue(any(message.get("role") == "tool" for message in runtime.messages))

    def test_shadow_off_on_timeout_cancel_and_reset_are_equivalent(self) -> None:
        class TimeoutBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                raise TimeoutError("synthetic timeout")

        class CancelBackend:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, messages: list[Message], *, temperature: float, max_tokens: int, tools=None) -> ChatResult:
                self.calls += 1
                return ChatResult(
                    content="",
                    model="fake",
                    finish_reason="cancelled",
                    tool_calls=[],
                    prompt_tokens=10,
                    completion_tokens=0,
                    cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        timeout_outcomes = []
        cancel_outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "lifecycle-shadow.jsonl"
            for enabled in ("0", "1"):
                with mock.patch.dict(
                    os.environ,
                    {
                        "ORBIT_TOOL_CALL_HEALING_SHADOW": enabled,
                        "ORBIT_KV_DIAG_FILE": str(log_path),
                    },
                    clear=False,
                ):
                    timeout_backend = TimeoutBackend()
                    timeout_runtime = ChatRuntime(backend=timeout_backend, system_prompt=None)
                    with self.assertRaisesRegex(TimeoutError, "synthetic timeout"):
                        timeout_runtime.ask_with_tools(
                            "show specs",
                            temperature=0,
                            max_tokens=32,
                            workdir=Path(tmp),
                            tool_names=("system_info",),
                        )
                    timeout_runtime.reset()
                    timeout_outcomes.append((timeout_backend.calls, timeout_runtime.messages))

                    cancel_backend = CancelBackend()
                    cancel_runtime = ChatRuntime(backend=cancel_backend, system_prompt=None)
                    cancelled = cancel_runtime.ask_with_tools(
                        "show specs",
                        temperature=0,
                        max_tokens=32,
                        workdir=Path(tmp),
                        tool_names=("system_info",),
                    )
                    cancel_runtime.reset()
                    cancel_outcomes.append(
                        (cancel_backend.calls, cancelled.content, cancelled.finish_reason, cancel_runtime.messages)
                    )

            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(timeout_outcomes, [(1, []), (1, [])])
        self.assertEqual(cancel_outcomes, [(1, "", "cancelled", []), (1, "", "cancelled", [])])
        terminal_events = [
            event
            for event in events
            if event["event"] == "kv_diag_tool_healing_terminal"
        ]
        self.assertEqual([event["active_outcome"] for event in terminal_events], ["timeout", "cancelled"])
        self.assertEqual(terminal_events[0]["agreement"], "uncorrelated")


if __name__ == "__main__":
    unittest.main()

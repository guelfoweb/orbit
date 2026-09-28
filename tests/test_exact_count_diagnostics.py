"""Admission failures retain bounded causes without changing permission/budgets."""
import copy
from dataclasses import replace
import unittest
from unittest import mock

from orbit.backend.base import TokenCount
from orbit.backend.llama_server import LlamaServerError
from orbit.runtime.context_manager import plan_exact_context
from tests.test_context_manager import _tool_turn


class ExactCountDiagnosticTests(unittest.TestCase):
    def plan(self, counter, messages=None, **kwargs):
        return plan_exact_context(
            messages or [{"role": "user", "content": "safe fixture"}],
            backend=object(), count_chat_override=counter,
            output_reserve=64, next_action_reserve=32, safety_margin=16,
            configured_context_tokens=1024, tools=[], thinking=False, **kwargs,
        )

    def count(self, tokens=100, digest="a"):
        return TokenCount(tokens, 1024, digest * 64, digest * 64)

    def test_absent_or_unavailable_counter_is_not_identity_change(self):
        for counter in (None, mock.Mock(return_value=None)):
            with self.subTest(counter=counter):
                plan = self.plan(counter)
                self.assertFalse(plan.admitted)
                self.assertEqual(plan.reason, "exact-token-count-unavailable")
                self.assertIsNone(plan.tokens_before)

    def test_capability_cause_from_either_attestation_survives(self):
        error = LlamaServerError("private backend detail")
        error.diagnostic_code = "required-tool-decoding-unavailable"
        sequences = [(error, error)]
        for other in (self.count(), None, ValueError("private"), {}):
            sequences.extend(((other, error), (error, other)))
        for sequence in sequences:
            with self.subTest(sequence=sequence):
                counter = mock.Mock(side_effect=sequence)
                plan = self.plan(counter)
                self.assertFalse(plan.admitted)
                self.assertEqual(plan.reason, "required-tool-decoding-unavailable")
                self.assertEqual(counter.call_count, 2)
                self.assertIsNone(plan.tokens_after)

    def test_unexpected_and_malformed_errors_fail_closed_without_text_leak(self):
        for code in (None, "required-context-does-not-fit", "secret\n\x1b[31m", {}, [], True):
            error = LlamaServerError("secret\n\x1b[31m")
            error.diagnostic_code = code
            with self.subTest(code=code):
                plan = self.plan(mock.Mock(side_effect=error))
                self.assertFalse(plan.admitted)
                self.assertEqual(plan.reason, "exact-token-count-failed")
        error = RuntimeError("server does not support required tool decoding")
        error.diagnostic_code = "required-tool-decoding-unavailable"
        plan = self.plan(mock.Mock(side_effect=error))
        self.assertEqual(plan.reason, "exact-token-count-failed")

    def test_invalid_return_is_not_a_count_or_diagnostic_code(self):
        values = [{"tokens": 100}, "required-tool-decoding-unavailable", False]
        for fields in ({"tokens": -1}, {"context_tokens": None},
                       {"rendered_hash": None}, {"token_hash": "short"}):
            values.append(replace(self.count(), **fields))
        for value in values:
            with self.subTest(value=value):
                plan = self.plan(mock.Mock(return_value=value))
                self.assertFalse(plan.admitted)
                self.assertEqual(plan.reason, "exact-token-count-invalid")

    def test_real_overflow_and_valid_count_keep_budget_and_messages(self):
        messages = [{"role": "user", "content": "same request"}]
        for tokens, admitted, reason in ((912, True, None), (913, False, "required-context-does-not-fit")):
            with self.subTest(tokens=tokens):
                counter = mock.Mock(return_value=self.count(tokens))
                plan = self.plan(counter, messages)
                self.assertEqual(plan.admitted, admitted)
                self.assertEqual(plan.reason, reason)
                self.assertEqual(plan.input_limit, 912)
                self.assertEqual(plan.tokens_before, tokens)
                self.assertEqual(list(plan.messages), messages)
                self.assertEqual(counter.call_count, 2)

    def test_valid_but_changed_attestations_still_report_identity_change(self):
        plan = self.plan(mock.Mock(side_effect=[self.count(), self.count(digest="b")]))
        self.assertFalse(plan.admitted)
        self.assertEqual(plan.reason, "tokenizer-template-or-context-changed")

    def test_compaction_failure_retains_cause_and_existing_projection(self):
        messages = [*_tool_turn(1), {"role": "user", "content": "next"}]
        original = copy.deepcopy(messages)
        error = LlamaServerError("private backend detail")
        error.diagnostic_code = "required-tool-decoding-unavailable"
        for failure, reason in ((error, "required-tool-decoding-unavailable"),
                                (None, "exact-token-count-unavailable")):
            with self.subTest(reason=reason):
                counter = mock.Mock(side_effect=[self.count(1500), self.count(1500), failure, failure])
                plan = self.plan(counter, messages, available_evidence_ids={"ev-1"},
                                 covered_evidence_ids={"ev-1"})
                self.assertFalse(plan.admitted)
                self.assertEqual(plan.reason, reason)
                self.assertEqual(plan.compacted_turns, 1)
                self.assertEqual(plan.externalized_evidence_ids, ("ev-1",))
                self.assertEqual(plan.input_limit, 912)
                self.assertEqual(plan.tokens_before, 1500)
                self.assertIsNone(plan.tokens_after)
                self.assertEqual(counter.call_count, 4)
                self.assertEqual(messages, original)

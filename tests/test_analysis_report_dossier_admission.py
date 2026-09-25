"""Optional REPORT quotes must fit beside the complete required dossier.

Synthetic observations, real recording/rendering/admission/dispatch paths.
The backend models non-additive template cost; the retained replay separately
uses the production Ornith tokenizer without running inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from unittest import mock

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime import analysis_runtime as ar
from tests.test_analysis_report_evidence_allocation import _Base


class Backend:
    thinking = False

    def __init__(self):
        self.sent = []
        self.context = 8192
        self.template_cost = 30

    def supports_exact_context_admission(self):
        return True

    def count_text_tokens(self, text):
        # UTF-8 fixture characters can cost several tokens; never len/constant.
        return TokenCount((len(text) + 3) // 4 + text.count('界') * 3, self.context)

    def count_chat_tokens(self, messages, **kwargs):
        identity = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        tokens = self.template_cost + sum(
            self.count_text_tokens(m.get('content', '')).tokens for m in messages)
        return TokenCount(tokens, self.context, identity, identity)

    def chat_stream(self, messages, **kwargs):
        self.sent.append((copy.deepcopy(messages), kwargs))
        return ChatResult(
            content='A bounded answer from the evidence.', model='test',
            finish_reason='stop', tool_calls=[], prompt_tokens=None,
            completion_tokens=None, cached_tokens=None,
            prompt_tokens_per_second=None, generation_tokens_per_second=None)


class CompleteReportBudgetTests(_Base):
    def setUp(self):
        super().setUp()
        self.backend = Backend()
        self.runtime.backend = self.backend
        self.runtime.context_tokens = 8192
        for _ in range(4):
            self._record(self._big_with_trailing_chain())
        self.records = self.runtime._reportable_records()
        self.asked = ('Original questions and proposed answers remain verbatim.\n'
                      + ('unverified detail. ' * 850).rstrip())

    def _record(self, stdout):
        self._call += 1
        result = ar.AnalysisResult(
            status='ok', code_sha256='c' * 64, input_sha256=self.source.sha256,
            stdout=stdout, stderr='', exit_status=0, duration_seconds=0.0)
        observation, truncated, full_chars = ar._bounded_observation(result)
        call = {'id': f'call_{self._call}',
                'function': {'name': 'execute_analysis', 'arguments': '{}'}}
        return self.runtime._record_action_evidence(
            call, result, observation, truncated=truncated, full_chars=full_chars)

    def state(self):
        return copy.deepcopy((self.runtime.messages, self.runtime._report_runs,
                              [asdict(r) for r in self.store.records.values()]))

    def assert_budget(self):
        messages, kwargs = self.backend.sent[-1]
        self.assertLessEqual(
            self.backend.count_chat_tokens(messages).tokens + kwargs['max_tokens'] + 256,
            min(self.backend.context, self.runtime.context_tokens))
        self.assertEqual(kwargs['tools'], [])
        return messages

    def test_long_dossier_uses_full_request_budget_and_still_generates(self):
        before = self.state()
        report = self.runtime._report_narrative(self.asked)
        self.assertEqual(report.model_calls, 1)
        messages = self.assert_budget()
        self.assertEqual(self.backend.sent[0][1]['max_tokens'], 2048)
        self.assertTrue(messages[-1]['content'].endswith(self.asked))
        for record in self.records:
            self.assertIn(record.evidence_id, messages[-1]['content'])
            self.assertIn(record.raw_sha256[:16], messages[-1]['content'])
        self.assertIn(ar.EVIDENCE_CONTENT_NOT_SHOWN, messages[-1]['content'])
        self.assertEqual(self.state(), before)

    def test_short_dossier_keeps_supported_quotes(self):
        self.runtime._report_narrative('Report on this evidence.')
        self.assertEqual(len(self.backend.sent), 1)
        self.assertIn('delta.Invoke', str(self.assert_budget()))

    def test_even_provenance_not_fitting_is_refused_without_dispatch(self):
        before = self.state()
        with self.assertRaises(ar.ContextAdmissionError):
            self.runtime._report_narrative(self.asked * 3)
        self.assertEqual(self.backend.sent, [])
        self.assertEqual(self.state(), before)
        self.assertEqual(self.runtime.model_calls, 0)

    def test_explicit_smaller_context_and_output_remain_binding(self):
        self.runtime.context_tokens = 6144
        self.runtime.max_tokens = 1024
        self.runtime._report_narrative('Original dossier. ' * 700)
        self.assertEqual(len(self.backend.sent), 1)
        self.assertEqual(self.backend.sent[0][1]['max_tokens'], 1024)
        self.assert_budget()

    def test_complete_template_count_controls_quotes_not_piece_sum(self):
        self.backend.template_cost += 800
        self.runtime._report_narrative(self.asked)
        self.assertEqual(len(self.backend.sent), 1)
        self.assert_budget()

    def test_multiple_short_results_can_be_cheaper_than_provenance_floors(self):
        for record in self.records:
            self.store.discard(record.evidence_id)
        for i in range(8):
            self._record('Measured value: ' + str(i))
        records = self.runtime._reportable_records()
        with mock.patch.object(self.runtime, '_evidence_cards', return_value=[
                self.runtime._evidence_card(r) for r in records]):
            complete = self.runtime._report_messages(self.asked, records)
        limit = self.backend.count_chat_tokens(complete).tokens + 10
        self.backend.context = self.runtime.context_tokens = limit + 2048 + 256
        self.runtime._report_narrative(self.asked)
        sent = self.assert_budget()
        for record in records:
            self.assertIn(record.evidence_id, str(sent))
            self.assertIn(self.store.reattest_exact(record.evidence_id), sent[-1]['content'])

    def test_no_excerpt_and_exact_boundary_do_not_drop_required_context(self):
        floors = [self.runtime._provenance_card(r) for r in self.records]
        # A real unavailable body uses the existing provenance representation.
        with mock.patch.object(self.runtime, '_evidence_card', side_effect=self.runtime._provenance_card):
            minimum = self.runtime._report_messages(self.asked, self.records)
            self.assertTrue(all(f in minimum[-1]['content'] for f in floors))
            limit = self.backend.count_chat_tokens(minimum).tokens
            self.backend.context = self.runtime.context_tokens = limit + 2048 + 256
            self.runtime._report_narrative(self.asked)
            self.assertEqual(self.assert_budget(), minimum)
            self.backend.sent.clear()
            self.runtime.context_tokens -= 1
            with self.assertRaises(ar.ContextAdmissionError):
                self.runtime._report_narrative(self.asked)
            self.assertEqual(self.backend.sent, [])

    def test_large_optional_excerpt_is_demoted_without_rehydration(self):
        full = self.runtime._evidence_card
        def huge(record):
            return full(record) + '\n' + '界' * 10000
        before = self.state()
        with mock.patch.object(self.runtime, '_evidence_card', side_effect=huge), \
                mock.patch.object(self.runtime, '_with_evidence_rehydration',
                                  side_effect=AssertionError('REPORT references are not requests')):
            self.runtime._report_narrative(self.asked)
        sent = self.assert_budget()
        self.assertNotIn('界', str(sent))
        for record in self.records:
            self.assertIn(self.runtime._provenance_card(record), sent[-1]['content'])
        self.assertEqual(self.state(), before)

    def test_non_character_tokenization_preserves_full_dossier(self):
        asked = 'Unmodified required data: ' + '界' * 1100
        self.assertGreater(self.backend.count_text_tokens(asked).tokens, len(asked))
        self.runtime._report_narrative(asked)
        self.assertTrue(self.assert_budget()[-1]['content'].endswith(asked))

    def test_optional_failure_preserves_canonical_document(self):
        before = self.state()
        expected = self.runtime.report(self.asked * 3, generate_narrative=False)
        actual = self.runtime.report(self.asked * 3)
        self.assertEqual(actual.narrative_status, 'unavailable:ContextAdmissionError')
        self.assertEqual(actual.evidence_ids, expected.evidence_ids)
        self.assertEqual(actual.document_complete, expected.document_complete)
        # Only optional narrative status differs, not the structured dossier.
        self.assertEqual(actual.dossier_text.replace(actual.narrative_status, expected.narrative_status),
                         expected.dossier_text)
        self.assertEqual(self.backend.sent, [])
        self.assertEqual(self.state(), before)

    def test_unknown_exact_token_count_fails_closed(self):
        self.backend.count_chat_tokens = lambda *a, **k: None
        with self.assertRaises(ar.ContextAdmissionError):
            self.runtime._report_narrative(self.asked)
        self.assertEqual(self.backend.sent, [])

"""REPORT's optional quotes must fit beside its full question/dossier."""
import copy
import hashlib
import json
from dataclasses import replace
from unittest import mock
from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime import analysis_runtime as ar
from tests.test_analysis_report_evidence_allocation import _Base

class Backend:
    thinking = False
    def __init__(self): self.sent = []; self.context = 8192
    def supports_exact_context_admission(self): return True
    def count_text_tokens(self, text): return TokenCount((len(text)+3)//4, self.context)
    def count_chat_tokens(self, messages, **kwargs):
        identity=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
        return TokenCount(30 + sum((len(m.get('content',''))+3)//4 for m in messages), self.context, identity, identity)
    def chat_stream(self, messages, **kwargs):
        self.sent.append((copy.deepcopy(messages), kwargs))
        return ChatResult(content='A bounded answer from the evidence.', model='test', finish_reason='stop', tool_calls=[], prompt_tokens=None, completion_tokens=None, cached_tokens=None, prompt_tokens_per_second=None, generation_tokens_per_second=None)

class CompleteReportBudgetTests(_Base):
    def setUp(self):
        super().setUp()
        self.backend=Backend(); self.runtime.backend=self.backend; self.runtime.context_tokens=8192
        for _ in range(4): self._record(self._big_with_trailing_chain())
        self.records=self.runtime._reportable_records()
        self.asked='Original questions and proposed answers remain verbatim.\n' + 'unverified detail. ' * 850

    def test_long_dossier_uses_full_request_budget_and_still_generates(self):
        expected=self.asked.strip()
        report=self.runtime._report_narrative(self.asked)
        self.assertEqual(report.model_calls,1)
        messages,kwargs=self.backend.sent[0]
        count=self.backend.count_chat_tokens(messages).tokens
        self.assertLessEqual(count+kwargs['max_tokens']+256,8192)
        self.assertEqual(kwargs['max_tokens'],2048)
        self.assertTrue(messages[-1]['content'].endswith(expected))
        for record in self.records: self.assertIn(record.evidence_id,messages[-1]['content'])
        self.assertIn(ar.EVIDENCE_CONTENT_NOT_SHOWN,messages[-1]['content'])

    def test_short_dossier_keeps_supported_quotes(self):
        self.runtime._report_narrative('Report on this evidence.')
        self.assertEqual(len(self.backend.sent),1)
        self.assertIn('delta.Invoke',str(self.backend.sent[0][0]))

    def test_even_provenance_not_fitting_is_refused_without_dispatch(self):
        with self.assertRaises(ar.ContextAdmissionError):
            self.runtime._report_narrative(self.asked * 3)
        self.assertEqual(self.backend.sent,[])
        self.assertEqual(len(self.runtime._reportable_records()),len(self.records))

    def test_explicit_smaller_context_and_output_remain_binding(self):
        self.runtime.context_tokens=6144
        self.runtime.max_tokens=1024
        self.runtime._report_narrative('Original dossier. ' * 700)
        self.assertEqual(len(self.backend.sent),1)
        messages,kwargs=self.backend.sent[0]
        self.assertEqual(kwargs['max_tokens'],1024)
        self.assertLessEqual(self.backend.count_chat_tokens(messages).tokens+1024+256,6144)

    def test_complete_render_not_sum_of_piece_counts_controls_quotes(self):
        original=self.backend.count_chat_tokens
        def counted(messages,**kw):
            result=original(messages,**kw)
            # Model-template cost not visible to count_text_tokens.
            return replace(result,tokens=result.tokens+800)
        self.backend.count_chat_tokens=counted
        self.runtime._report_narrative(self.asked)
        self.assertEqual(len(self.backend.sent),1)
        self.assertLessEqual(counted(self.backend.sent[0][0]).tokens+2048+256,8192)

    def test_multiple_short_complete_results_can_be_cheaper_than_their_floors(self):
        for record in self.records:
            self.runtime.evidence_store.discard(record.evidence_id)
        for i in range(8):
            self._record('Measured value: ' + str(i))
        records=self.runtime._reportable_records()
        with mock.patch.object(self.runtime,'_evidence_cards',return_value=[self.runtime._evidence_card(r) for r in records]):
            complete=self.runtime._report_messages(self.asked,records)
        limit=self.backend.count_chat_tokens(complete).tokens+10
        self.backend.context=self.runtime.context_tokens=limit+2048+256
        self.runtime._report_narrative(self.asked)
        self.assertEqual(len(self.backend.sent),1)
        sent=self.backend.sent[0][0]
        self.assertLessEqual(self.backend.count_chat_tokens(sent).tokens,limit)
        for record in records:
            self.assertIn(record.evidence_id,str(sent))

"""Exact bodies must reach the admitted phase, not merely exist in the store."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import TokenCount
from orbit.runtime.analysis_controller import AnalysisController
from orbit.runtime.analysis_runtime import (AnalysisRuntime, acquire_analysis_source,
    ANALYSIS_TOOL_SCHEMA, FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA, _control_context)
from orbit.runtime.evidence import EvidenceStore
from tests.test_analysis_runtime import ScriptedBackend, prose_response
from tests.test_analysis_evidence_first import decodable, encode, SECRET
from unittest import mock
from dataclasses import replace


class Backend(ScriptedBackend):
    thinking = False
    context = 16384

    def supports_exact_context_admission(self):
        return True

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        rendered = [{k:v for k,v in m.items() if k in ('role','content','tool_calls','tool_call_id','name')} for m in messages]
        raw = json.dumps([rendered, tools, thinking], sort_keys=True)
        h = hashlib.sha256(raw.encode()).hexdigest()
        return TokenCount(tokens=len(raw), context_tokens=self.context,
                          rendered_hash=h, token_hash=h)


class EvidenceDeliveryTests(unittest.TestCase):
    def runtime(self, text=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        p = Path(temp.name)
        (p/'sample.js').write_text(text or decodable(), encoding='utf-8')
        b = Backend(prose_response('No action necessary.'))
        rt = AnalysisRuntime(backend=b, source=acquire_analysis_source(p/'sample.js', p/'snapshot'),
                             evidence_store=EvidenceStore(p/'evidence'))
        self.addCleanup(rt.close)
        return rt, b

    def test_plan_step_finish_receive_exact_body_without_user_request(self):
        rt, b = self.runtime()
        canonical = copy.deepcopy(rt.messages)
        controller = AnalysisController()
        rt.plan_analysis(controller, 'Inspect the artifact.', max_calls=1)
        question = controller.activate_next()
        rt.step('Continue', controller_messages=rt._resolve_messages(controller, question))
        record = rt.transform_stages[0][1]
        rt.finish_question(controller, question, 'No new observation.', record.evidence_id, max_calls=1)
        for messages in b.seen_messages:
            self.assertIn(SECRET, '\n'.join(str(m.get('content','')) for m in messages))
        self.assertEqual(rt.messages[:len(canonical)], canonical)
        self.assertEqual(rt.actions_executed, 0)

    def test_reference_only_instruction_does_not_claim_assistant_can_read(self):
        rt, _ = self.runtime()
        self.assertNotIn('Name an id as', rt.messages[-1]['content'])

    def test_report_facts_precede_unverified_answers(self):
        rt, _ = self.runtime()
        doc = rt.report(generate_narrative=False).text
        self.assertLess(doc.index(SECRET), doc.index('## Investigation and original questions'))
        self.assertIn('not investigation success', doc)

    def admit(self, rt, messages=None, tool=ANALYSIS_TOOL_SCHEMA):
        return rt._admit(messages or rt.messages, max_tokens=2048,
                         tools=[tool], next_action_reserve=0)

    def receipts(self, messages):
        return [d for m in messages for d in m.get('analysis_evidence_delivery', [])]

    def test_repeated_requests_preserve_history_store_and_exact_ranges(self):
        rt, _ = self.runtime(decodable('café\r\n終'))
        history = copy.deepcopy(rt.messages)
        records = copy.deepcopy(rt.evidence_store.records)
        first = self.admit(rt)
        self.assertEqual(first, self.admit(rt))
        self.assertEqual(rt.messages, history)
        self.assertEqual(rt.evidence_store.records, records)
        stage, record = rt.transform_stages[0]
        d = self.receipts(first)[0]
        self.assertEqual(d['byte_range'], [0, len(stage.output.encode('utf-8'))])
        self.assertEqual(d['sha256'], hashlib.sha256(stage.output.encode('utf-8')).hexdigest())
        body_line = next(line for m in first for line in m.get('content','').splitlines() if line.startswith('  exact_output: '))
        self.assertEqual(json.loads(body_line.partition(': ')[2]).encode('utf-8'), stage.output.encode('utf-8'))

    def test_absent_withdrawn_altered_or_wrong_snapshot_never_delivered(self):
        for fault in ('missing', 'withdrawn', 'altered', 'snapshot', 'provenance'):
            with self.subTest(fault=fault):
                rt, _ = self.runtime()
                _, r = rt.transform_stages[0]
                sidecar = rt.evidence_store.root / (r.evidence_id+'.txt')
                if fault == 'missing': sidecar.unlink()
                if fault == 'withdrawn': rt.evidence_store.discard(r.evidence_id)
                if fault == 'altered': sidecar.write_text('forged')
                if fault == 'snapshot':
                    rt.source.snapshot_path.chmod(0o600)
                    rt.source.snapshot_path.write_bytes(b'wrong source')
                if fault == 'provenance':
                    rt.evidence_store.records[r.evidence_id] = replace(r, metadata={**r.metadata, 'analysis_source_sha256':'wrong'})
                sent = self.admit(rt)
                self.assertNotIn(SECRET, '\n'.join(m.get('content','') for m in sent))
                self.assertEqual(self.receipts(sent)[0]['status'], 'not_delivered')
                self.assertIn('output not supplied', str(sent))
                self.assertNotIn('not missing evidence', str(sent))

    def test_large_computed_result_whole_or_explicitly_undelivered(self):
        value = 'Z' * 5000
        rt, b = self.runtime(decodable(value))
        b.context = 6000
        sent = self.admit(rt)
        self.assertNotIn('Z'*100, str(sent))
        self.assertEqual(self.receipts(sent)[0]['status'], 'not_delivered')
        self.assertLessEqual(rt.last_context_plan.tokens_after, rt.last_context_plan.input_limit)
        self.assertEqual(rt.evidence_store.reattest_exact(rt.transform_stages[0][1].evidence_id), value)

    def test_duplicate_index_and_explicit_request_are_not_falsely_withheld(self):
        rt, _ = self.runtime()
        eid = rt.transform_stages[0][1].evidence_id
        messages = [*rt.messages, copy.deepcopy(rt.messages[-1]), {'role':'user','content':'Continue.'}]
        sent = self.admit(rt, messages)
        self.assertEqual('\n'.join(m.get('content','') for m in sent).count(SECRET), 1)
        self.assertEqual([d['status'] for d in self.receipts(sent)], ['complete','supplied_elsewhere'])
        messages[-1]['content'] = 'Read evidence:'+eid
        sent = self.admit(rt, messages)
        self.assertEqual('\n'.join(m.get('content','') for m in sent).count(SECRET), 1)
        self.assertTrue(all(d['status']=='supplied_elsewhere' for d in self.receipts(sent)))
        self.assertNotIn('output not supplied', str(sent))

    def test_unrelated_citation_does_not_become_automatic_read(self):
        rt, _ = self.runtime()
        r = rt.evidence_store.add('execute_analysis','UNRELATED-SECRET', metadata={
            'tool_call_id':'unrelated','user_turn_id':'turn_x','produced_by_phase':'analysis',
            'analysis_source_sha256':rt.source.sha256})
        for tail in ([], [{'role':'user','content':'Continue.'}]):
            sent = self.admit(rt, [*rt.messages, {'role':'assistant','content':'Cited evidence:'+r.evidence_id}, *tail])
            self.assertNotIn('UNRELATED-SECRET', str(sent))
            self.assertIn(SECRET, str(sent))

    def test_multiple_explicit_requests_deliver_each_body_once(self):
        rt, _ = self.runtime(decodable() + f'dec("{encode("OTHER", 19, ",")}", 19, ",");')
        self.assertEqual(len(rt.transform_stages), 2)
        request = 'Read ' + ' '.join('evidence:'+r.evidence_id for _, r in rt.transform_stages)
        sent = self.admit(rt, [*rt.messages, {'role':'user','content':request}])
        bodies = '\n'.join(m.get('content', '') for m in sent)
        for value in (SECRET, 'OTHER'):
            self.assertEqual(bodies.count(value), 1)
        self.assertTrue(all(d['status']=='supplied_elsewhere' for d in self.receipts(sent)))

    def test_windowed_source_does_not_mark_withheld_tail_as_delivered(self):
        rt, b = self.runtime()
        module = rt.evidence_store.add('extract_office_vba', 'Module source line\n'*2000,
            metadata={'office_module_name':'Module', 'analysis_source_sha256':rt.source.sha256,
                      'tool_call_id':'office_vba_0','user_turn_id':'turn_0',
                      'produced_by_phase':'analysis_office_preflight'})
        eid = rt.transform_stages[0][1].evidence_id
        b.context = 8000
        sent = self.admit(rt, [*rt.messages, {'role':'user','content':
            f'Read evidence:{module.evidence_id} and evidence:{eid}'}])
        self.assertIn(eid, [entry[0] for entry in rt.last_rehydration_diag['windowed']])
        delivery = self.receipts(sent)[0]
        self.assertNotEqual(delivery['status'], 'supplied_elsewhere')
        self.assertEqual('\n'.join(m.get('content','') for m in sent).count(SECRET),
                         1 if delivery['status']=='complete' else 0)

    def test_another_sessions_index_cannot_deliver_its_body(self):
        first, _ = self.runtime(decodable('FIRST-SESSION'))
        second, _ = self.runtime(decodable('SECOND-SESSION'))
        sent = self.admit(second, [*second.messages, copy.deepcopy(first.messages[-1])])
        self.assertNotIn('FIRST-SESSION', str(sent))
        self.assertIn('SECOND-SESSION', str(sent))

    def test_non_investigative_request_does_not_enable_delivery(self):
        rt, _ = self.runtime()
        sent = rt._admit(rt.messages, max_tokens=2048, tools=None, next_action_reserve=0)
        self.assertNotIn(SECRET, str(sent))
        self.assertFalse(self.receipts(sent))

    def test_optional_delivery_does_not_starve_finish_output_or_lower_explicit_cap(self):
        rt, b = self.runtime(decodable('K'*4000))
        c=AnalysisController(); c.adopt_plan([{'question':'Original question','missing_fact':'Original requirement'}])
        q=c.activate_next()
        messages=rt._finish_messages(q,'bounded observation',rt.transform_stages[0][1].evidence_id)
        # An ordinary request fits; body would fit only by spending output reserve.
        b.context=b.count_chat_tokens(_control_context(messages),tools=[FINISH_TOOL_SCHEMA]).tokens+2500
        sent, cap=rt._admit_finish(messages,FINISH_TOOL_SCHEMA)
        self.assertEqual(cap,2048)
        self.assertNotIn('K'*100,str(sent))
        self.assertLessEqual(rt.last_context_plan.tokens_after+cap+256,b.context)
        rt.max_tokens = 512
        sent, cap=rt._admit_finish(messages,FINISH_TOOL_SCHEMA)
        self.assertEqual(cap,512)
        self.assertLessEqual(rt.last_context_plan.tokens_after+cap+256,b.context)

    def test_finish_repair_receives_bodies_without_changing_question_or_decision(self):
        rt,b=self.runtime()
        c=AnalysisController(); c.adopt_plan([{'question':'Original question','missing_fact':'Original requirement'}]);q=c.activate_next()
        eid=rt.transform_stages[0][1].evidence_id
        b._finish_decisions=[{'status':'invalid'}, {'status':'resolved','answer_summary':'Supported proposal','evidence_ids':[eid]}]
        rt.finish_question(c,q,'The exact transform is available.',eid)
        finish=[ms for ms,ts in zip(b.seen_messages,b.seen_tools) if 'finish_analysis_question' in ts]
        self.assertEqual(len(finish),2)
        for ms in finish:
            self.assertIn(SECRET,str(ms));self.assertIn('Original requirement',str(ms))
        self.assertEqual(c.states[q.id].status,'answered_unverified')
        self.assertEqual(c.repairs,1)
        self.assertEqual(rt.actions_executed,0)

    def test_report_with_withdrawn_body_is_incomplete_and_does_not_publish_fact(self):
        rt,_=self.runtime();rt.evidence_store.discard(rt.transform_stages[0][1].evidence_id)
        doc=rt.report(generate_narrative=False)
        self.assertFalse(doc.document_complete)
        self.assertNotIn(SECRET,doc.text)
        self.assertIn('Missing referenced evidence',doc.text)


if __name__ == '__main__':
    unittest.main()

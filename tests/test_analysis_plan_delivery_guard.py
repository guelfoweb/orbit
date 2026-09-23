"""Call-owned byte delivery: no question meaning is inferred by the runtime."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.backend.base import ChatResult, ToolCallParseError
from orbit.runtime.analysis_controller import AnalysisController, ControlError, PHASE_REPORT
from orbit.runtime.analysis_runtime import AnalysisRuntime, PLAN_TOOL_SCHEMA, acquire_analysis_source, _cover_message
from orbit.runtime.analysis_coverage import SourceCoverage, COVERAGE_COMPLETE
from orbit.runtime.evidence import EvidenceStore
from tests.test_analysis_evidence_delivery import Backend
from tests.test_analysis_evidence_first import decodable


def entry(data=None, text='What remains to establish?'):
    return {'question': text, 'missing_fact': 'Original requirement', 'data_request': data}


class Plans(Backend):
    def __init__(self):
        super().__init__()
        self.answers=[]
        self.requests=[]
        self.hook=None

    def chat_stream(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        if self.hook:self.hook()
        answer=self.answers.pop(0)
        if isinstance(answer, Exception):raise answer
        reason='stop'
        if isinstance(answer,tuple):answer,reason=answer
        return ChatResult(content='',model='fixture',finish_reason=reason,
            prompt_tokens=1,completion_tokens=1,cached_tokens=0,
            prompt_tokens_per_second=None,generation_tokens_per_second=None,
            tool_calls=[{'type':'function','id':'plan','function':{'name':'submit_analysis_plan',
                         'arguments':json.dumps({'questions':answer})}}])


class DeliveryGuardTests(unittest.TestCase):
    def runtime(self,text=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);p=root/'sample.js';p.write_bytes((text or decodable('café\r\n終')).encode())
        b=Plans();rt=AnalysisRuntime(backend=b,source=acquire_analysis_source(p,root/'snapshot'),
                                  evidence_store=EvidenceStore(root/'evidence'))
        self.addCleanup(rt.close);return rt,b

    def data(self,rt,index=0,start=0,end=None):
        stage,r=rt.transform_stages[index]
        return {'ref':r.raw_ref,'start':start,'end':len(stage.output.encode()) if end is None else end}

    def dispatch(self,rt,b,messages=None):
        b.answers=[[]]
        rt._control_call(messages or rt.messages,PLAN_TOOL_SCHEMA,repair_budget=0)

    def test_supplied_request_rejected_before_adoption_then_interpretation_repair(self):
        rt,b=self.runtime();q=entry(self.data(rt));history=copy.deepcopy(rt.messages);records=copy.deepcopy(rt.evidence_store.records)
        b.answers=[[q],[entry(None,q['question'])]];c=AnalysisController()
        rt.plan_analysis(c,'Inspect it',max_calls=2)
        self.assertEqual(rt.control_attempts,2);self.assertEqual(c.repairs,1)
        self.assertEqual(c.order,['Q1']);self.assertEqual(c.questions['Q1'].question,q['question'])
        self.assertIn('already supplied',b.requests[1][-1]['content'])
        self.assertEqual(c.states['Q1'].status,'open');self.assertEqual(c.states['Q1'].actions,0)
        self.assertEqual(rt.messages,history);self.assertEqual(rt.evidence_store.records,records)
        self.assertEqual(rt.actions_executed,0)

    def test_repeated_contradiction_is_atomic_and_bounded(self):
        rt,b=self.runtime();q=entry(self.data(rt));b.answers=[[entry(None,'Keep original'),q],[q]];c=AnalysisController()
        rt.plan_analysis(c,'Inspect',max_calls=2)
        self.assertTrue(c.unsupported);self.assertEqual(c.phase,PHASE_REPORT)
        self.assertEqual(c.questions,{});self.assertEqual(c.states,{});self.assertEqual(rt.control_attempts,2)

    def test_no_repair_when_only_one_call_allowed(self):
        rt,b=self.runtime();b.answers=[[entry(self.data(rt))]];c=AnalysisController()
        rt.plan_analysis(c,'Inspect',max_calls=1)
        self.assertTrue(c.unsupported);self.assertEqual(c.repairs,0);self.assertEqual(rt.control_attempts,1)

    def test_subrange_utf8_bytes_and_explicit_interpretation(self):
        rt,b=self.runtime();self.dispatch(rt,b)
        with self.assertRaisesRegex(ControlError,'already supplied'):
            rt._validate_plan_delivery([entry(self.data(rt,start=1,end=4))])
        rt._validate_plan_delivery([entry(None)])
        with self.assertRaisesRegex(ControlError,'already supplied'):
            rt._validate_plan_delivery([entry({'ref':self.data(rt)['ref']})])
        self.assertEqual(self.data(rt)['end'],len('café\r\n終'.encode()))

    def test_current_call_not_previous_call_or_other_session(self):
        rt,b=self.runtime();self.dispatch(rt,b);receipt=rt._plan_delivery
        rt.control_attempts+=1
        with self.assertRaisesRegex(ControlError,'session/call'):rt._validate_plan_delivery([entry(self.data(rt))])
        other,ob=self.runtime();self.dispatch(other,ob);other._plan_delivery=receipt;other.source=rt.source;other.control_attempts=receipt[2]
        with self.assertRaisesRegex(ControlError,'session/call'):other._validate_plan_delivery([entry(self.data(other))])

    def test_wrong_snapshot_identity_even_if_store_and_attempt_match(self):
        rt,b=self.runtime();self.dispatch(rt,b);other,_=self.runtime()
        rt.source=other.source
        with self.assertRaisesRegex(ControlError,'session/call'):rt._validate_plan_delivery([entry(self.data(rt))])

    def test_retired_changed_or_missing_evidence_never_counts_as_delivered(self):
        for fault in ['discard','missing','changed','snapshot','record']:
            with self.subTest(fault=fault):
                rt,b=self.runtime();self.dispatch(rt,b);data=self.data(rt);r=rt.transform_stages[0][1]
                if fault=='discard':rt.evidence_store.discard(r.evidence_id)
                if fault=='missing':(rt.evidence_store.root/(r.evidence_id+'.txt')).unlink()
                if fault=='changed':(rt.evidence_store.root/(r.evidence_id+'.txt')).write_text('changed')
                if fault=='snapshot':rt.source.snapshot_path.chmod(0o600);rt.source.snapshot_path.write_bytes(b'x'*rt.source.size_bytes)
                if fault=='record':rt.evidence_store.records.pop(r.evidence_id)
                with self.assertRaises(ControlError) as exc:rt._validate_plan_delivery([entry(data)])
                self.assertNotIn('already supplied',str(exc.exception))

    def test_changed_same_size_source_cannot_use_old_receipt(self):
        rt,b=self.runtime('a'*6000)
        self.dispatch(rt,b,[*rt.messages,{'role':'user','content':rt._oversized_source_overview()}])
        rt.source.snapshot_path.chmod(0o600);rt.source.snapshot_path.write_bytes(b'b'*6000)
        with self.assertRaisesRegex(ControlError,'snapshot is unavailable or changed'):
            rt._validate_plan_delivery([entry({'ref':'source:'+rt.source.sha256,'start':0,'end':1200})])

    def test_compaction_cannot_turn_undelivered_large_body_into_receipt(self):
        rt,b=self.runtime(decodable('Z'*5000));b.context=6500
        self.dispatch(rt,b)
        self.assertNotIn('Z'*100,str(b.requests[-1]))
        rt._validate_plan_delivery([entry(self.data(rt))])
        self.assertLessEqual(rt.last_context_plan.tokens_after,rt.last_context_plan.input_limit)

    def test_no_receipt_or_body_no_claim_of_delivery(self):
        for fault in ['receipt','body','sha','source_sha','status','range']:
            rt,b=self.runtime();admitted=rt._admit(rt.messages,max_tokens=2048,tools=[PLAN_TOOL_SCHEMA],next_action_reserve=0)
            for m in admitted:
                if not m.get('analysis_evidence_delivery'):continue
                if fault=='receipt':m.pop('analysis_evidence_delivery');continue
                if fault=='body':m['content']='body withheld';continue
                r=m['analysis_evidence_delivery'][0]
                r[{'sha':'sha256','source_sha':'source_sha256','status':'status','range':'byte_range'}[fault]]={'status':'not_delivered','range':[0,1]}.get(fault,'wrong')
            with patch.object(rt,'_admit',return_value=admitted):self.dispatch(rt,b)
            rt._validate_plan_delivery([entry(self.data(rt))])

    def test_new_call_withheld_does_not_use_previous_receipts(self):
        rt,b=self.runtime();self.dispatch(rt,b)
        with patch.object(rt,'_admit',return_value=[{'role':'user','content':'Inspect'}]):self.dispatch(rt,b)
        rt._validate_plan_delivery([entry(self.data(rt))])

    def test_other_registered_evidence_not_delivered(self):
        rt,b=self.runtime(decodable('first')+'\n'+decodable('second').replace('MMGCLZ','OTHER'))
        # Use a legitimate current transform but omit all delivery from this call.
        with patch.object(rt,'_admit',return_value=[{'role':'user','content':'Inspect'}]):self.dispatch(rt,b)
        rt._validate_plan_delivery([entry(self.data(rt))])

    def test_source_full_and_partial_ranges(self):
        rt,b=self.runtime('a'*6000);overview=rt._oversized_source_overview()
        self.dispatch(rt,b,[*rt.messages,{'role':'user','content':overview}])
        d={'ref':'source:'+rt.source.sha256,'start':0,'end':1200}
        with self.assertRaisesRegex(ControlError,'already supplied'):rt._validate_plan_delivery([entry(d)])
        rt._validate_plan_delivery([entry({**d,'end':1201})])
        rt._validate_plan_delivery([entry({**d,'start':1199,'end':4801})])
        covered=SourceCoverage('a'*6000,COVERAGE_COMPLETE,rt.source.sha256,6000)
        self.dispatch(rt,b,[*rt.messages,{'role':'user','source_covered':True,'content':_cover_message(covered,rt.source)}])
        with self.assertRaisesRegex(ControlError,'already supplied'):rt._validate_plan_delivery([entry({**d,'end':6000})])

    def test_overview_utf8_replacement_does_not_prove_bytes(self):
        rt,b=self.runtime('a'*1199+'終'+'b'*5000);self.dispatch(rt,b,[*rt.messages,{'role':'user','content':rt._oversized_source_overview()}])
        rt._validate_plan_delivery([entry({'ref':'source:'+rt.source.sha256,'start':0,'end':1200})])

    def test_invalid_reference_range_and_missing_discriminator(self):
        rt,b=self.runtime();self.dispatch(rt,b);d=self.data(rt)
        for v in [[],{}, {'ref':'/etc/passwd','start':0,'end':1},{**d,'ref':'evidence:ev_not_real'},
                  {**d,'ref':'source:'+'0'*64},{**d,'start':True},{**d,'start':-1},
                  {**d,'start':1,'end':1},{**d,'end':100000},{**d,'extra':1}]:
            with self.subTest(v=v),self.assertRaises(ControlError):rt._validate_plan_delivery([entry(v)])
        with self.assertRaises(ControlError):rt._validate_plan_delivery([{'question':'Q','missing_fact':'M'}])

    def test_parse_repair_and_delivery_repair_share_two_dispatches(self):
        rt,b=self.runtime();b.answers=[ToolCallParseError('bad control'),[entry(self.data(rt))]];c=AnalysisController()
        rt.plan_analysis(c,'Inspect',max_calls=2)
        self.assertEqual(rt.control_attempts,2);self.assertTrue(c.unsupported);self.assertEqual(c.order,[])

    def test_binary_snapshot_transform_delivery(self):
        rt,b=self.runtime('binary fixture')
        p=rt.source.snapshot_path.parent/'binary.doc';p.write_bytes(b'\xd0\xcf\x11\xe0\xff\x00')
        rt.source=acquire_analysis_source(p,p.parent/'binary_snapshot')
        rt._ingest_transform_stages(decodable('extracted value'),origin='VBA/Module1')
        self.assertIsNone(rt._snapshot_text())
        self.dispatch(rt,b)
        with self.assertRaisesRegex(ControlError,'already supplied'):
            rt._validate_plan_delivery([entry(self.data(rt))])
        with patch.object(rt,'_admit',return_value=[{'role':'user','content':'Inspect'}]):
            self.dispatch(rt,b)
        rt._validate_plan_delivery([entry(self.data(rt))])
        rt._validate_plan_delivery([entry({'ref':'source:'+rt.source.sha256})])

    def test_empty_whole_source_is_not_delivery_without_a_receipt(self):
        rt,b=self.runtime('safe')
        p=rt.source.snapshot_path.parent/'empty';p.write_bytes(b'')
        rt.source=acquire_analysis_source(p,p.parent/'empty_snapshot')
        self.dispatch(rt,b)
        with self.assertRaisesRegex(ControlError,'range is empty'):
            rt._validate_plan_delivery([entry({'ref':'source:'+rt.source.sha256})])

    def test_parse_repaired_empty_plan_has_no_phantom_reask(self):
        rt,b=self.runtime();b.answers=[ToolCallParseError('bad control'),[]];c=AnalysisController()
        rt.plan_analysis(c,'Inspect',max_calls=5)
        self.assertEqual(rt.control_attempts,2);self.assertEqual(rt.control_repairs,1)
        self.assertFalse(c.unsupported);self.assertEqual(c.phase,PHASE_REPORT)
        self.assertEqual(c.repairs,0);self.assertFalse(rt._empty_plan_re_asked)

    def test_truncated_parseable_plan_never_adopted(self):
        rt,b=self.runtime();b.answers=[([entry(None)],'length')];c=AnalysisController()
        rt.plan_analysis(c,'Inspect',max_calls=1)
        self.assertEqual(c.order,[]);self.assertTrue(c.unsupported)

    def test_fifteen_retained_questions_no_semantic_inference(self):
        rows=json.loads((Path(__file__).parent/'fixtures/analysis_plan_delivery_questions.json').read_text())
        self.assertEqual(len(rows),15)
        samples=Path(os.environ.get('ORBIT_TEST_CORPUS', str(Path(__file__).resolve().parents[1]/'workdir/samples')))
        for row in rows:
            name={'iban':'IBAN.js','fattura':'Fattura981033956.js','mine':'mine.hta'}[row['case'].split('_')[0]]
            if not (samples/name).exists():self.skipTest('machine-local frozen corpus absent')
            with self.subTest(case=row['case'],question=row['ordinal']):
                rt,b=self.runtime((samples/name).read_bytes().decode());self.dispatch(rt,b)
                target=row['data_target']
                d=None if target is None else ({'ref':'source:'+rt.source.sha256,'start':0,'end':rt.source.size_bytes}
                    if target=='source' else self.data(rt,index=target))
                e={k:row[k] for k in ['question','missing_fact']};e['data_request']=d
                if row['classification']=='ALREADY_PRESENT':
                    with self.assertRaisesRegex(ControlError,'already supplied'):rt._validate_plan_delivery([e])
                else:
                    rt._validate_plan_delivery([e]);c=AnalysisController();c.adopt_plan([e])
                    self.assertEqual(c.questions['Q1'].question,row['question']);self.assertEqual(c.states['Q1'].status,'open')

if __name__=='__main__':unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.backend.base import ChatResult, TokenCount
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.completion_budget import reasoning_final_budget
from orbit.runtime.context_manager import ContextAdmissionError
from orbit.runtime.environments import FinalFromToolEnvironment, TransportEnvironment
from orbit.runtime.final_policy import FinalToolPolicy, FinalAnswerCompleteness


def result(text='alpha.txt', finish='stop'):
    return ChatResult(text,'test',finish,[],1,1,0,None,None)


class Backend:
    def __init__(self, required=True, responses=None, context=8192):
        self.required=required
        self.responses=list(responses or [result()])
        self.calls=[]
        self.context=context
        self.thinking=False

    def requires_native_reasoning(self):
        return self.required

    def count_chat_tokens(self,messages,*,tools=None,thinking=False):
        return TokenCount(10,self.context,'a'*64,'b'*64)

    def chat(self,messages,*,temperature,max_tokens,tools=None):
        self.calls.append(max_tokens)
        return self.responses.pop(0)

    def chat_stream(self,messages,*,temperature,max_tokens,on_delta,on_progress,tools=None):
        answer=self.chat(messages,temperature=temperature,max_tokens=max_tokens,tools=tools)
        if answer.content:on_delta(answer.content)
        return answer


class ReasoningFinalBudgetTests(unittest.TestCase):
    def run_final(self,backend,*,stream=False,incomplete=False,retry=True):
        runtime=ChatRuntime(backend=backend,system_prompt=None,context_tokens=backend.context)
        env=FinalFromToolEnvironment(runtime,TransportEnvironment(runtime))
        policy=FinalToolPolicy([{'role':'user','content':'Summarize the tool result.'}],96,retry,incomplete,False,False)
        with patch('orbit.runtime.environments.build_final_tool_policy',return_value=policy):
            return env.answer(temperature=0,max_tokens=2048,on_final_delta=(lambda text:None) if stream else None,
                              on_progress=None,on_model_step=None,on_phase_start=None,loop=1,use_tool_prompt=True)

    def test_capability_is_strict_and_reserve_is_bounded_once(self):
        for value in (False,None,1,'always'):
            self.assertEqual(reasoning_final_budget(96,Backend(value)),96)
        self.assertEqual(reasoning_final_budget(96,object()),96)
        self.assertEqual(reasoning_final_budget(96,Backend(True)),1120)

    def test_real_final_wiring_preserves_ordinary_caps_in_both_transports(self):
        for stream in (False,True):
            for required,expected in ((True,1120),(False,96)):
                with self.subTest(stream=stream,required=required):
                    backend=Backend(required)
                    out=self.run_final(backend,stream=stream)
                    self.assertEqual(out.result.content,'alpha.txt')
                    self.assertEqual(backend.calls,[expected])

    def test_existing_length_retry_receives_reserve_without_more_attempts(self):
        for stream in (False,True):
            for required,expected in ((True,[1120,1216]),(False,[96,192])):
                backend=Backend(required,[result('', 'length'),result()])
                out=self.run_final(backend,stream=stream)
                self.assertEqual(backend.calls,expected)
                self.assertTrue(out.used_retry_or_repair_pass)

    def test_existing_completion_repair_receives_reserve(self):
        for stream in (False,True):
            for required,expected in ((True,[1120,1184]),(False,[96,160])):
                backend=Backend(required,[result('unfinished'),result('Finished answer.')])
                with patch('orbit.runtime.environments.classify_final_answer_completeness',side_effect=[FinalAnswerCompleteness('incomplete'),FinalAnswerCompleteness('complete')]):
                    self.run_final(backend,stream=stream,incomplete=True,retry=False)
                self.assertEqual(backend.calls,expected)

    def test_existing_compact_repair_receives_reserve(self):
        for stream in (False,True):
            for required,expected in ((True,[1120,1184]),(False,[96,160])):
                backend=Backend(required,[result('A long answer.'),result('Short answer.')])
                with patch('orbit.runtime.environments.final_from_tool_compact_retry_reason',return_value='verbose'):
                    self.run_final(backend,stream=stream,retry=False)
                self.assertEqual(backend.calls,expected)

    def test_combined_budget_is_refused_before_any_generation_when_it_cannot_fit(self):
        for stream in (False,True):
            backend=Backend(True,context=1024)
            with self.assertRaises(ContextAdmissionError):self.run_final(backend,stream=stream)
            self.assertEqual(backend.calls,[])

    def test_directory_tool_path_reaches_real_final_policy(self):
        class RouteBackend(Backend):
            def chat(self,messages,*,temperature,max_tokens,tools=None):
                self.calls.append(max_tokens)
                return result('{"path":".","recursive":false}' if len(self.calls)==1 else 'alpha.txt')
        for required,expected in ((True,[1088,1120]),(False,[64,96])):
            backend=RouteBackend(required)
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/'alpha.txt').write_text('fixture')
                runtime=ChatRuntime(backend=backend,system_prompt=None)
                out=runtime.ask_auto('List the current directory.',temperature=0,max_tokens=2048,workdir=Path(directory))
            self.assertEqual(out.content,'alpha.txt')
            self.assertEqual(backend.calls,expected)

class GuiTurnNamespaceTests(unittest.TestCase):
    def test_namespace_is_explicit_and_legacy_turn_ids_stay_unchanged(self):
        legacy=ChatRuntime(backend=Backend(False),system_prompt=None)
        first=ChatRuntime(backend=Backend(False),system_prompt=None,user_turn_namespace="gui_first")
        second=ChatRuntime(backend=Backend(False),system_prompt=None,user_turn_namespace="gui_second")
        self.assertEqual(legacy._begin_user_turn(),"turn_1")
        self.assertEqual(first._begin_user_turn(),"turn_gui_first_1")
        self.assertEqual(first._begin_user_turn(),"turn_gui_first_2")
        self.assertEqual(second._begin_user_turn(),"turn_gui_second_1")

    def test_namespace_rejects_invalid_identifiers(self):
        for namespace in ("",123,"a/b","a b","x"*65):
            with self.subTest(namespace=namespace),self.assertRaises(ValueError):
                ChatRuntime(backend=Backend(False),user_turn_namespace=namespace)

    def test_shared_store_cannot_cite_a_previous_conversation_first_turn(self):
        from orbit.runtime.evidence import EvidenceStore, citable_evidence_ids
        with tempfile.TemporaryDirectory() as directory:
            store=EvidenceStore(Path(directory)/"evidence")
            old=ChatRuntime(backend=Backend(False),system_prompt=None,evidence_store=store,user_turn_namespace="old")
            old._begin_user_turn()
            record=store.add("exec_shell_full_command","old private bytes",metadata={"user_turn_id":old.current_user_turn_id,"tool_call_id":"old-call","produced_by_phase":"final_from_tool"})
            new=ChatRuntime(backend=Backend(False),system_prompt=None,evidence_store=store,user_turn_namespace="new")
            new._begin_user_turn()
            self.assertIn(record.evidence_id,citable_evidence_ids(store,user_turn_id=old.current_user_turn_id))
            self.assertNotIn(record.evidence_id,citable_evidence_ids(store,user_turn_id=new.current_user_turn_id))
            rendered=new._render_cited_evidence(result("evidence:"+record.evidence_id))
            self.assertNotIn("old private bytes",rendered.content)

if __name__=='__main__':unittest.main()

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from orbit.reasoning import REASONING_EFFORT_LEVELS, parse_reasoning_effort, current_reasoning_effort, reasoning_effort_scope
from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload
from orbit.backend.llama_server import LlamaServerBackend
from orbit.native_server.protocol import parse_chat_request, parse_continue_request
from orbit.native_server.app import OrbitNativeServer
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.bindings import ChatBridgeLibrary
from orbit.native_llama.model_profiles import INKLING2_PROFILE_ID
from tests.test_native_server_think import _FakeClient, _FakeCompletion
from tests import test_aligned_prefix_cache as aligned_tests


class ReasoningEffortTests(unittest.TestCase):
    def test_all_named_levels_and_strict_rejection(self):
        for value in REASONING_EFFORT_LEVELS:
            self.assertEqual(parse_reasoning_effort(value), value)
        for value in ('LOW','', 'unknown',0,0.2,True,[],{}):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_reasoning_effort(value)
        self.assertIsNone(parse_reasoning_effort(None))

    def test_scopes_are_nested_exception_safe_and_thread_isolated(self):
        self.assertIsNone(current_reasoning_effort())
        with reasoning_effort_scope('low'):
            with self.assertRaises(RuntimeError):
                with reasoning_effort_scope('high'):
                    self.assertEqual(current_reasoning_effort(),'high');raise RuntimeError('failed')
            self.assertEqual(current_reasoning_effort(),'low')
            with ThreadPoolExecutor(max_workers=3) as pool:
                def probe(value):
                    with reasoning_effort_scope(value):return current_reasoning_effort()
                self.assertEqual(list(pool.map(probe,REASONING_EFFORT_LEVELS)),list(REASONING_EFFORT_LEVELS))
        self.assertIsNone(current_reasoning_effort())

    def test_payload_protocol_roundtrip_and_conflicts(self):
        for value in REASONING_EFFORT_LEVELS:
            payload=build_chat_payload(ChatPayloadOptions(model='m',messages=[{'role':'user','content':'Hi'}],temperature=0,max_tokens=32,reasoning_effort=value))
            self.assertEqual(payload['chat_template_kwargs']['reasoning_effort'],value)
            self.assertEqual(parse_chat_request(payload).reasoning_effort,value)
            payload['chat_template_kwargs']['reasoning_effort']='minimal' if value!='minimal' else 'none'
            with self.assertRaises(ValueError):parse_chat_request(payload)
        old=build_chat_payload(ChatPayloadOptions(model='m',messages=[{'role':'user','content':'Hi'}],temperature=0,max_tokens=32))
        self.assertNotIn('reasoning_effort',old)
        self.assertIsNone(parse_chat_request(old).reasoning_effort)
        self.assertEqual(parse_chat_request({'messages':[{'role':'user','content':'Hi'}],'chat_template_kwargs':{'reasoning_effort':'low'}}).reasoning_effort,'low')

    def test_required_and_artifact_requests_refuse_incompatible_effort(self):
        with self.assertRaises(ValueError):parse_chat_request({'messages':[{'role':'user','content':'Hi'}],'tool_choice':'required','tools':[{'type':'function','function':{'name':'t'}}],'reasoning_effort':'high'})
        fake=_FakeClient(False);fake.validate_reasoning_effort=Mock()
        with self.assertRaises(ValueError):OrbitNativeServer(client=fake,model_alias='m').complete(parse_chat_request({'messages':[{'role':'user','content':'Hi'}],'artifact_content':True,'reasoning_effort':'none'}))
        self.assertFalse(fake.calls)
        with self.assertRaises(ValueError):parse_continue_request({"reasoning_effort":"low"})

    def test_server_generation_and_exact_count_share_request_effort(self):
        fake=_FakeClient(False);fake.validate_reasoning_effort=Mock();seen=[]
        def generate(*args,**kwargs):seen.append(current_reasoning_effort());return _FakeCompletion()
        def count(*args,**kwargs):seen.append(current_reasoning_effort());return 1,'render','tokens'
        fake.complete_chat_text=generate;fake.inspect_chat_tokens=count
        server=OrbitNativeServer(client=fake,model_alias='m')
        for value in REASONING_EFFORT_LEVELS:
            server.complete(parse_chat_request({'messages':[{'role':'user','content':'Hi'}],'reasoning_effort':value}))
            response=server.count_chat_tokens([],tools=[],thinking=False,reasoning_effort=value)
            self.assertEqual(response['reasoning_effort'],value)
            self.assertEqual(seen[-2:],[value,value]);self.assertIsNone(current_reasoning_effort())
        fake.complete_chat_text=Mock(side_effect=RuntimeError('failed'))
        with self.assertRaises(RuntimeError):server.complete(parse_chat_request({'messages':[{'role':'user','content':'Hi'}],'reasoning_effort':'max'}))
        self.assertIsNone(current_reasoning_effort())

    def test_backend_transmits_selection_in_chat_stream_and_token_count(self):
        b=LlamaServerBackend(base_url='http://localhost:1',timeout=1)
        b._is_orbit_native_backend=Mock(return_value=True);b._props_or_empty=Mock(return_value={})
        b._post_native_stream=Mock(return_value=_FakeCompletion());b._observe_call=lambda fn:fn()
        b._post_json=Mock(return_value={'tokens':1,'context_tokens':8192})
        for value in REASONING_EFFORT_LEVELS:
            b.reasoning_effort=value
            b.chat([],temperature=0,max_tokens=16)
            self.assertEqual(b._post_native_stream.call_args.args[1]['reasoning_effort'],value)
            b.chat_stream([],temperature=0,max_tokens=16,on_delta=lambda t:None)
            self.assertEqual(b._post_native_stream.call_args.args[1]['reasoning_effort'],value)
            b.count_chat_tokens([])
            self.assertEqual(b._post_json.call_args.args[1]['reasoning_effort'],value)

    def test_native_bridge_uses_additive_entrypoint_and_legacy_fallback(self):
        bridge=object.__new__(ChatBridgeLibrary);bridge.lib=SimpleNamespace(orbit_chat_bridge_render=object(),orbit_chat_bridge_render_effort=object())
        bridge._json_call=Mock(return_value={'prompt':'valid'})
        bridge.render(1,[],[],thinking=False,reasoning_effort='low')
        self.assertIs(bridge._json_call.call_args.args[0],bridge.lib.orbit_chat_bridge_render_effort)
        self.assertEqual(bridge._json_call.call_args.args[-2:],(b'low',False))
        bridge.render(1,[],[],thinking=False)
        self.assertIs(bridge._json_call.call_args.args[0],bridge.lib.orbit_chat_bridge_render)
        del bridge.lib.orbit_chat_bridge_render_effort
        with self.assertRaises(RuntimeError):bridge.render(1,[],[],thinking=False,reasoning_effort='low')

    def test_only_verified_inkling_and_upgraded_bridge_admit_levels(self):
        client=object.__new__(NativeLlamaClient);client.model_profile=SimpleNamespace(profile_id=INKLING2_PROFILE_ID,verified=True);client.chat_bridge=SimpleNamespace(supports_reasoning_effort=True)
        self.assertEqual(client.supported_reasoning_effort_levels(),REASONING_EFFORT_LEVELS)
        for value in REASONING_EFFORT_LEVELS:client.validate_reasoning_effort(value)
        for owner,key,value in [(client.model_profile,'verified',False),(client.model_profile,'profile_id','other'),(client.chat_bridge,'supports_reasoning_effort',False)]:
            old=getattr(owner,key);setattr(owner,key,value)
            with self.assertRaises(ValueError):client.validate_reasoning_effort('none')
            client.validate_reasoning_effort(None);setattr(owner,key,old)

    def test_http_props_publish_actual_supported_effort_levels(self):
        import threading
        from http.server import ThreadingHTTPServer
        import json, urllib.request
        from orbit.native_server.app import OrbitNativeHandler
        from tests.test_server_build_identity import _fake_state
        state=_fake_state();runtime=state.runtime_info()
        runtime["reasoning_effort_levels"]=list(REASONING_EFFORT_LEVELS)
        state.runtime_info=lambda:runtime
        with ThreadingHTTPServer(("127.0.0.1",0),OrbitNativeHandler) as server:
            server.orbit_state=state
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/props",timeout=2) as response:props=json.load(response)
                self.assertEqual(props["reasoning_effort_levels"],list(REASONING_EFFORT_LEVELS))
            finally:server.shutdown();thread.join(timeout=3)

    def test_client_template_receives_explicit_scope_without_changing_boolean(self):
        client=object.__new__(NativeLlamaClient)
        client.model_profile=SimpleNamespace(profile_id=INKLING2_PROFILE_ID,verified=True,uses_native_chat_bridge=True)
        client.chat_bridge=SimpleNamespace(supports_reasoning_effort=True,render=Mock(return_value={"prompt":"valid"}))
        client._model=1;client._chat_bridge_context=1;client._thinking_enabled=lambda value:False
        client._serialize_profile_messages=lambda messages:messages
        for value in REASONING_EFFORT_LEVELS:
            with reasoning_effort_scope(value):self.assertEqual(client.apply_chat_template([{"role":"user","content":"Hi"}],thinking=False),"valid")
            self.assertEqual(client.chat_bridge.render.call_args.kwargs["reasoning_effort"],value)
            self.assertFalse(client.chat_bridge.render.call_args.kwargs["thinking"])
        client.apply_chat_template([],thinking=False)
        self.assertNotIn("reasoning_effort",client.chat_bridge.render.call_args.kwargs)

    def test_aligned_cache_remains_qualified_only_at_none_effort(self):
        fixture=aligned_tests.AlignedPrefixCacheTests();fixture.setUp()
        try:
            client=fixture.client()
            for value in REASONING_EFFORT_LEVELS:
                with reasoning_effort_scope(value):self.assertEqual(client._aligned_prefix_eligible(thinking=False),value=='none')
        finally:fixture.doCleanups()

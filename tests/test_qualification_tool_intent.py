from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult, TokenCount
from orbit.qualification import tool_intent as gate
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime import tool_backends
from orbit.runtime.kv_diag import model_call_context
from orbit.terminal.tool_mode import allowed_tool_names_for_spec

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "qualification/fixtures/tool-intent-v1.json"
REPLAY = ROOT / "qualification/fixtures/tool-intent-recorded-v1.json"


def result(text='{"route":"CHAT"}', *, finish="stop", tool=None, arguments=None):
    calls = [] if tool is None else [{"id": "fixture-call", "type": "function", "function": {
        "name": tool, "arguments": json.dumps(arguments or {})}}]
    return ChatResult(text, "scripted", finish, calls, 100, 8, 0, None, None)


class Scripted:
    thinking = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def supports_exact_context_admission(self):
        return True

    def count_chat_tokens(self, messages, **kwargs):
        return TokenCount(100, 8192, "a" * 64, "b" * 64)

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)

    def chat_stream(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), dict(kwargs)))
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if response.content and kwargs.get("on_delta"):
            kwargs["on_delta"](response.content)
        return response


class ToolIntentTests(unittest.TestCase):
    def setUp(self):
        self.corpus = gate.load_corpus(CORPUS)
        self.cases = {c["id"]: c for c in self.corpus["cases"]}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        (self.directory / "notes.txt").write_text("Safe routing fixture.\n")
        self.real_executor = mock.patch.object(HybridToolExecutor, "execute",
                                               side_effect=AssertionError("real executor reached")).start()
        self.low_level = mock.patch.object(tool_backends, "execute_tool",
                                           side_effect=AssertionError("real tool reached")).start()
        self.addCleanup(mock.patch.stopall)

    def observe(self, case_id, responses):
        backend = Scripted(responses)
        obs = gate.observe_case(self.cases[case_id], backend, self.directory)
        self.real_executor.assert_not_called()
        self.low_level.assert_not_called()
        return obs, gate.score_case(self.cases[case_id], obs), backend

    def test_corpus_is_exactly_frozen_bilingual_24_cases(self):
        self.assertEqual(hashlib.sha256(CORPUS.read_bytes()).hexdigest(), gate.CORPUS_SHA256)
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(sum(c['expected'] == 'content' for c in self.cases.values()), 12)
        self.assertEqual({c['language'] for c in self.cases.values()}, {'it', 'en'})
        self.assertEqual(self.cases['recap']['history'], self.cases['refresh']['history'])
        self.assertEqual(self.cases['write_prose']['expected'], 'content')

    def test_modified_oracle_is_rejected(self):
        path = self.directory / 'corpus.json'
        data = copy.deepcopy(self.corpus)
        data['cases'][0]['expected'] = 'tool'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            gate.load_corpus(path)

    def test_chat_and_original_code_request_pass_without_executor(self):
        obs, scored, backend = self.observe('original', [result()])
        self.assertEqual(scored['status'], 'PASS')
        self.assertEqual(scored['observed_mode'], 'CHAT')
        self.assertEqual(scored['model_calls'], 1)
        self.assertFalse(scored['false_execution'])
        messages, params = backend.requests[0]
        self.assertEqual(messages[0]['content'], ROUTE_SYSTEM_PROMPT)
        self.assertEqual(messages[-1]['content'], self.cases['original']['prompt'])
        self.assertEqual(params['max_tokens'], 64)
        self.assertIsNone(params.get('tools'))

    def test_valid_authorized_execution_is_intercepted_and_passes(self):
        _, scored, _ = self.observe('actual_run', [result('', tool='exec_shell_full_command',
                                  arguments={'command': 'python3 -c "print(6 * 7)"'})])
        self.assertEqual(scored['status'], 'PASS')
        self.assertEqual(scored['tool'], 'exec_shell_full_command')
        self.assertEqual(scored['details']['canonical']['terminal_decision'], 'accepted')
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ['notes.txt'])

    def test_false_execution_fails_even_when_canonical_contract_allows_it(self):
        _, scored, _ = self.observe('explain_concept', [result('', tool='list_directory', arguments={'path': '.'})])
        self.assertTrue(scored['false_execution'])
        self.assertEqual(scored['status'], 'FAIL')

    def test_artifact_publication_is_intercepted(self):
        args = {'path': 'email.txt', 'overwrite': False, 'create_parents': True}
        with mock.patch.object(gate.tool_loop, 'begin_artifact_generation', side_effect=AssertionError('artifact generated')) as artifact:
            _, scored, _ = self.observe('write_prose', [result('', tool='write_artifact', arguments=args)])
        artifact.assert_not_called()
        self.assertTrue(scored['false_execution'])
        self.assertFalse((self.directory / 'email.txt').exists())

    def test_missed_explicit_execution_fails(self):
        _, scored, _ = self.observe('actual_run', [result()])
        self.assertTrue(scored['missed_execution'])
        self.assertEqual(scored['status'], 'FAIL')

    def test_complete_direct_prose_survives_arbitrary_stream_fragmentation(self):
        # No oracle or prompt words participate in the transport decision.
        for text in ('Lima', '/var/tmp/example', 'Una risposta completa.', '答案。'):
            for width in (1, 3, len(text)):
                with self.subTest(text=text, width=width):
                    class Fragmented(Scripted):
                        def chat_stream(self, messages, **kwargs):
                            response = self.responses.pop(0)
                            self.requests.append((copy.deepcopy(messages), dict(kwargs)))
                            for start in range(0, len(response.content), width):
                                kwargs['on_delta'](response.content[start:start + width])
                            return response
                    backend = Fragmented([result(text)])
                    obs = gate.observe_case(self.cases['chat'], backend, self.directory)
                    scored = gate.score_case(self.cases['chat'], obs)
                    self.assertEqual(scored['status'], 'PASS')
                    self.assertEqual(obs['kind'], 'DIRECT_CHAT')
                    self.assertEqual(obs['details']['returned']['content'], text)
                    self.assertEqual(scored['model_calls'], 1)
                    self.assertFalse(scored['invalid_route_protocol'])
                    self.assertEqual(backend.requests[0][1]['max_tokens'], 64)
        self.real_executor.assert_not_called()
        self.low_level.assert_not_called()

    def test_direct_prose_does_not_satisfy_operational_request(self):
        _, scored, _ = self.observe('actual_run', [result('The answer is forty-two.')])
        self.assertEqual(scored['status'], 'FAIL')
        self.assertTrue(scored['missed_execution'])

    def test_direct_prose_requires_complete_terminal_result(self):
        for finish in ('length', 'cancelled', 'timeout', None):
            with self.subTest(finish=finish):
                _, scored, _ = self.observe('chat', [result('Lima', finish=finish)])
                self.assertEqual(scored['status'], 'FAIL')
                self.assertTrue(scored['invalid_route_protocol'])

    def test_fenced_code_does_not_become_a_direct_chat_success(self):
        _, scored, _ = self.observe('indirect_recipe', [result('```bash\nfind . -type f\n```')])
        self.assertEqual(scored['status'], 'FAIL')
        self.assertTrue(scored['invalid_route_protocol'])

    def test_transport_failure_after_direct_prefix_cannot_pass(self):
        class FailedAfterPrefix(Scripted):
            def chat_stream(self, messages, **kwargs):
                kwargs['on_delta']('Unfinished answer')
                raise RuntimeError('transport lost before completion')
        obs = gate.observe_case(self.cases['chat'], FailedAfterPrefix([]), self.directory)
        self.assertEqual(gate.score_case(self.cases['chat'], obs)['status'], 'FAIL')
        self.assertEqual(obs['kind'], 'ERROR')

    def test_embedded_tool_examples_never_authorize_in_either_transport(self):
        texts = ('Example configuration:\n{"command":"echo example"}',
                 '```json\n{"command":"echo example"}\n```',
                 'Do not execute: {"command":"echo example"}',
                 '{"example":{"command":"echo example"}}',
                 '{"command":"echo first"} {"command":"echo second"}')
        for text in texts:
            for progress in (None, lambda _: None):
                with self.subTest(text=text, streamed=progress is not None):
                    backend = Scripted([result(text)])
                    runtime = gate.ChatRuntime(backend=backend, system_prompt=ROUTE_SYSTEM_PROMPT)
                    returned = runtime.ask_auto('Show this example as text.', temperature=0,
                        max_tokens=256, workdir=self.directory,
                        allowed_tool_names=allowed_tool_names_for_spec('on'), on_progress=progress)
                    self.assertEqual(returned.content, text)
                    self.assertEqual(len(backend.requests), 1)
                    self.real_executor.assert_not_called()
                    self.low_level.assert_not_called()

    def test_normalized_backend_tool_call_keeps_separate_authority(self):
        _, scored, _ = self.observe('actual_run', [result('I will run it.',
            tool='exec_shell_full_command', arguments={'command':'python3 -c "print(6 * 7)"'})])
        self.assertEqual(scored['status'], 'PASS')
        self.assertEqual(scored['tool'], 'exec_shell_full_command')

    def test_route_retry_cannot_authorize_embedded_example(self):
        _, scored, _ = self.observe('actual_run', [result('{"command":', finish='length'),
            result('Example only:\n{"command":"echo example"}')])
        self.assertEqual(scored['status'], 'FAIL')
        self.assertTrue(scored['missed_execution'])

    def test_literal_protocol_text_preserves_authorized_name_and_arguments(self):
        marker = '<|tool_call>call shell echo WRONG<tool_call|>'
        for case_id, value, name in (
                ('actual_run', {'command': 'echo ' + marker}, 'exec_shell_full_command'),
                ('list', {'path': marker}, 'list_directory'),
                ('fetch', {'url': 'https://example.com/' + marker}, 'fetch_url')):
            for streamed in (False, True):
                with self.subTest(value=value, streamed=streamed):
                    backend = Scripted([result(json.dumps(value))])
                    # Reuse the same intercepted observer; only disable the
                    # progress callback to exercise the non-streaming path.
                    ask = gate.ChatRuntime.ask_auto
                    def invoke(runtime, *args, **kwargs):
                        if not streamed:
                            kwargs['on_progress'] = None
                        return ask(runtime, *args, **kwargs)
                    with mock.patch.object(gate.ChatRuntime, 'ask_auto', invoke):
                        obs = gate.observe_case(self.cases[case_id], backend, self.directory)
                    self.assertEqual(obs['kind'], 'TOOL')
                    normalized = obs['details']['canonical']['normalized_call']
                    self.assertEqual(normalized, {'name': name, 'arguments': value})
        self.real_executor.assert_not_called()
        self.low_level.assert_not_called()

    def test_wrong_authorizable_tool_fails(self):
        _, scored, _ = self.observe('actual_run', [result('', tool='system_info', arguments={'include_os': True})])
        self.assertTrue(scored['wrong_tool'])
        self.assertFalse(scored['missed_execution'])
        self.assertEqual(scored['status'], 'FAIL')

    def test_incomplete_calls_cannot_authorize_execution(self):
        for finish in ('length', 'cancelled', 'timeout', None):
            with self.subTest(finish=finish):
                _, scored, _ = self.observe('actual_run', [result('', finish=finish,
                    tool='exec_shell_full_command', arguments={'command': 'pwd'})])
                self.assertEqual(scored['status'], 'FAIL')
                self.assertTrue(scored['missed_execution'])
                self.assertTrue(scored['invalid_route_protocol'])

    def test_permission_and_schema_rejection_cannot_pass(self):
        for name, args in [('not_offered', {}), ('exec_shell_full_command', {'unknown': 'pwd'})]:
            with self.subTest(name=name):
                _, scored, _ = self.observe('actual_run', [result('', tool=name, arguments=args)])
                self.assertEqual(scored['status'], 'FAIL')
                self.assertTrue(scored['missed_execution'])

    def test_policy_rejection_is_not_authorized_execution(self):
        _, scored, _ = self.observe('fetch', [result('{"command":"orbit-web-search https://example.com"}')])
        self.assertEqual(scored['observed_mode'], 'REJECTED')
        self.assertEqual(scored['status'], 'FAIL')
        self.assertTrue(scored['missed_execution'])

    def test_runtime_rejection_is_not_a_chat_success(self):
        for response in [result('{"route":"ANALYSIS","artifact":"notes.txt"}'),
                         result('', tool='not_offered', arguments={})]:
            with self.subTest(response=response):
                _, scored, _ = self.observe('original', [response])
                self.assertEqual(scored['status'], 'FAIL')
                self.assertNotEqual(scored['observed_mode'], 'CHAT')

    def test_malformed_stopped_route_is_not_direct_prose(self):
        for text in ('{"route":', '{"command":', '[', '```json\n{"route":'):
            with self.subTest(text=text):
                _, scored, _ = self.observe('original', [result(text)])
                self.assertEqual(scored['status'], 'FAIL')
                self.assertTrue(scored['invalid_route_protocol'])

    def test_tool_generation_and_existing_retry_phases_are_observed(self):
        probe = gate._ProbeBackend(Scripted([result(), result()]))
        for phase in ('tool_call', 'tool_call_retry'):
            with model_call_context(phase=phase, tools_mode='on'):
                probe.chat([], temperature=0, max_tokens=64)
        self.assertEqual([c['phase'] for c in probe.calls], ['tool_call', 'tool_call_retry'])

    def test_document_acquisition_is_not_run_by_gate(self):
        case = copy.deepcopy(self.cases['inspect_file'])
        case['prompt'] = 'Read the entire file notes.txt and explain it.'
        backend = Scripted([result('{"command":"cat notes.txt"}')])
        with mock.patch('orbit.runtime.file_tools.read_full_document_snapshot',
                        side_effect=AssertionError('real read')) as read:
            obs = gate.observe_case(case, backend, self.directory)
        read.assert_not_called()
        self.assertEqual(obs['kind'], 'DEFERRED')
        self.assertEqual(gate.score_case(case, obs)['status'], 'FAIL')

    def test_repair_metrics_preserve_initial_invalid_attempt(self):
        _, scored, backend = self.observe('actual_run', [result('{"command":', finish='length'),
            result('', tool='exec_shell_full_command', arguments={'command': 'pwd'})])
        self.assertEqual(scored['model_calls'], 2)
        self.assertEqual(scored['retry'], 1)
        self.assertTrue(scored['invalid_route_protocol'])
        self.assertEqual(scored['status'], 'PASS')
        self.assertTrue(scored['recovered'])
        self.assertEqual(len(backend.requests), 2)

    def test_stopped_malformed_route_recovery_keeps_protocol_failure(self):
        _, scored, _ = self.observe('inspect_file', [result('{"route":'),
            result('', tool='exec_shell_full_command', arguments={'command': 'cat notes.txt'})])
        self.assertEqual(scored['status'], 'PASS')
        self.assertEqual(scored['retry'], 1)
        self.assertTrue(scored['invalid_route_protocol'])
        self.assertTrue(scored['recovered'])

    def test_history_preserved_and_not_reused_between_cases(self):
        _, _, backend = self.observe('recap', [result()])
        self.assertEqual(backend.requests[0][0][1:-1], self.cases['recap']['history'])
        _, _, other = self.observe('chat', [result()])
        self.assertEqual(len(other.requests[0][0]), 2)

    def test_production_executor_restored_after_probe(self):
        self.observe('original', [result()])
        self.assertIs(gate.tool_loop.HybridToolExecutor, HybridToolExecutor)

    def test_cancellation_restores_executor_and_does_not_run_it(self):
        backend = Scripted([])
        with mock.patch.object(backend, 'chat_stream', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                gate.observe_case(self.cases['original'], backend, self.directory)
        self.assertIs(gate.tool_loop.HybridToolExecutor, HybridToolExecutor)
        self.real_executor.assert_not_called()

    def test_probe_ceiling_does_not_add_retries(self):
        probe = gate._ProbeBackend(Scripted([result()] * (gate.MAX_MODEL_CALLS + 1)))
        with model_call_context(phase='route', tools_mode='on'):
            for _ in range(gate.MAX_MODEL_CALLS):
                probe.chat([], temperature=0, max_tokens=64)
            with self.assertRaises(gate._Boundary) as caught:
                probe.chat([], temperature=0, max_tokens=64)
        self.assertEqual(caught.exception.details['reason'], 'harness_call_limit')
        self.assertEqual(len(probe.calls), gate.MAX_MODEL_CALLS)

    def test_canonical_kill_switch_is_fail_closed(self):
        with mock.patch.dict('os.environ', {'ORBIT_TOOL_CALL_CANONICAL_GATE': '0'}):
            obs, scored, backend = self.observe('original', [result()])
        self.assertEqual(obs['details']['reason'], 'canonical_gate_disabled')
        self.assertEqual(scored['status'], 'FAIL')
        self.assertEqual(backend.requests, [])

    def test_missing_metrics_are_null_not_zero(self):
        res = asdict(result());res.update(prompt_tokens=None, completion_tokens=None)
        _, scored, _ = self.observe('original', [ChatResult(**res)])
        self.assertIsNone(scored['input_tokens'])
        self.assertIsNone(scored['output_tokens'])

    def test_recorded_A_replays_all_24_through_runtime(self):
        replay = gate.load_replay(REPLAY, self.corpus)
        observations = []
        for row in replay['cases']:
            case = self.cases[row['case_id']]
            backend = gate.ReplayBackend(row['calls'])
            obs = gate.observe_case(case, backend, self.directory)
            self.assertFalse(backend.pending)
            observations.append({'id': case['id'], **obs})
        report = gate.summarize(self.corpus, observations, provenance=replay['provenance'])
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['metrics']['completed'], 24)
        self.assertEqual(report['metrics']['model_calls'], 24)
        self.assertEqual(report['metrics']['false_execution'], 0)
        self.assertEqual(report['metrics']['missed_execution'], 11)
        # The frozen recording is unchanged. Completed direct answers now
        # survive the streaming consumer, as they already did without progress.
        # Historical qualification remains 5/24; this is a new runtime replay.
        self.assertEqual(report['metrics']['passed'], 8)
        self.assertEqual(sum(r['observed_mode'] == 'CHAT' for r in report['cases']), 10)
        self.assertEqual(report['metrics']['invalid_route_protocol'], 13)
        self.assertEqual({r['id'] for r in report['cases'] if r['status'] == 'PASS'},
                         {'indirect_recipe', 'indirect_write', 'explain_code',
                          'explain_concept', 'explain_os', 'host', 'chat', 'recap'})
        self.assertEqual(sum(r['input_tokens'] for r in report['cases']), 25714)
        self.assertEqual(sum(r['output_tokens'] for r in report['cases']), 631)
        self.real_executor.assert_not_called()
        self.low_level.assert_not_called()

    def test_positive_scripted_gate_traverses_real_runtime_for_all_cases(self):
        args = {'exec_shell_full_command': {'command': 'cat notes.txt'},
                'system_info': {'include_os': True}, 'list_directory': {'path': '.'},
                'fetch_url': {'url': 'https://example.com'},
                'write_artifact': {'path': 'greeting.txt', 'overwrite': False, 'create_parents': True}}
        observations = []
        for case in self.corpus['cases']:
            response = result() if case['expected'] == 'content' else result(
                '', tool=case['allowed_tools'][0], arguments=args[case['allowed_tools'][0]])
            obs, scored, _ = self.observe(case['id'], [response])
            self.assertEqual(scored['status'], 'PASS', case['id'])
            observations.append({'id': case['id'], **obs})
        self.assertEqual(gate.summarize(self.corpus, observations, provenance={})['status'], 'PASS')

    def test_replay_cannot_be_rebound_to_changed_request(self):
        replay = gate.load_replay(REPLAY, self.corpus)
        backend = gate.ReplayBackend(replay['cases'][0]['calls'])
        changed = copy.deepcopy(self.cases['original']);changed['prompt'] += ' changed'
        obs = gate.observe_case(changed, backend, self.directory)
        self.assertEqual(obs['kind'], 'ERROR')
        self.assertEqual(len(backend.pending), 1)
        self.assertEqual(obs['calls'], [])

    def test_replay_rejects_missing_duplicate_or_mismatched_corpus(self):
        raw = json.loads(REPLAY.read_text())
        variants = [copy.deepcopy(raw) for _ in range(4)]
        variants[0]['cases'].pop()
        variants[1]['cases'][1] = variants[1]['cases'][0]
        variants[2]['corpus_sha256'] = '0' * 64
        variants[3]['cases'][0]['calls'][0]['result']['completion_tokens'] = -1
        path = self.directory / 'tape.json'
        for value in variants:
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                gate.load_replay(path, self.corpus)

    def test_scoring_requires_all_cases_and_rejects_one_false_or_missed_or_wrong(self):
        observations = []
        for case in self.corpus['cases']:
            kind = 'CHAT_BOUNDARY' if case['expected'] == 'content' else 'TOOL'
            details = {} if kind == 'CHAT_BOUNDARY' else {
                'tool': case['allowed_tools'][0], 'canonical': {'terminal_decision': 'accepted'}}
            observations.append({'id': case['id'], 'kind': kind, 'details': details, 'wall_seconds': .1,
                                 'calls': [{'phase': 'route', 'result': asdict(result())}]})
        self.assertEqual(gate.summarize(self.corpus, observations, provenance={})['status'], 'PASS')
        for change in ('false', 'missed', 'wrong'):
            rows = copy.deepcopy(observations)
            if change == 'false':
                rows[0].update(kind='TOOL', details={'tool': 'system_info', 'canonical': {'terminal_decision': 'accepted'}})
            elif change == 'missed':
                rows[1].update(kind='CHAT_BOUNDARY', details={})
            else:
                rows[1]['details']['tool'] = 'system_info'
            self.assertEqual(gate.summarize(self.corpus, rows, provenance={})['status'], 'FAIL')
        with self.assertRaises(ValueError):
            gate.summarize(self.corpus, observations[:-1], provenance={})
        with self.assertRaises(ValueError):
            gate.summarize(self.corpus, observations[:-1] + [observations[0]], provenance={})

    def test_cli_writes_machine_and_human_report_offline(self):
        path = self.directory / 'report.json'
        child = subprocess.run([sys.executable, str(ROOT/'scripts/orbit_qualify_tool_intent.py'),
            '--replay', str(REPLAY), '--output', str(path)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 1, child.stderr)
        report = json.loads(path.read_text())
        self.assertEqual(report['provenance']['inference_calls_this_run'], 0)
        self.assertEqual(report['metrics']['completed'], 24)
        self.assertIn('Tool-intent qualification: FAIL', path.with_suffix('.md').read_text())

    def test_live_cli_requires_verified_idle_profile_and_resets_each_case(self):
        spec = importlib.util.spec_from_file_location('tool_intent_cli', ROOT/'scripts/orbit_qualify_tool_intent.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        backend = Scripted([result()] * 24)
        props = {'ctx_size': 8192, 'in_flight': False,
                 'model_compatibility': {'verified': True, 'compatibility_profile': 'test-profile'}}
        backend.backend_props = mock.Mock(return_value=props)
        backend.reset_static_analysis_session = mock.Mock(return_value=None)
        path = self.directory/'live.json'
        argv = ['--base-url', 'http://127.0.0.1:1', '--profile', 'test-profile', '--output', str(path)]
        with mock.patch.object(cli, 'LlamaServerBackend', return_value=backend), mock.patch('builtins.print'):
            self.assertEqual(cli.main(argv), 1)  # CHAT on every case misses 12 executions.
            self.assertEqual(backend.reset_static_analysis_session.call_count, 24)
            props['model_compatibility']['verified'] = False
            with self.assertRaisesRegex(ValueError, 'verified'):
                cli.main(argv)
            props['model_compatibility']['verified'] = True
            props['in_flight'] = True
            with self.assertRaisesRegex(ValueError, 'idle'):
                cli.main(argv)
        self.real_executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orbit.backend.base import ChatResult, TokenCount
from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.context_manager import ContextAdmissionError
from orbit.runtime.completion_budget import reasoning_decision_budget


class Backend:
    def __init__(self, required=True):
        self.required = required
        self.calls = []

    def requires_native_reasoning(self):
        return self.required

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.calls.append(max_tokens)
        return ChatResult('{"route":"CHAT"}' if len(self.calls) == 1 else 'Hello.',
                          'test', 'stop', [], 1, 1, 0, None, None)


class ReasoningDecisionBudgetTests(unittest.TestCase):
    def test_required_reasoning_adds_a_fixed_bounded_internal_reserve(self):
        backend = Backend()
        for base in (1, 32, 64, 96, 256, 2048):
            with self.subTest(base=base):
                self.assertEqual(reasoning_decision_budget(base, backend), base + 1024)

    def test_unknown_optional_and_malformed_capabilities_keep_exact_caps(self):
        for backend in (object(), Backend(False), Backend(None), Backend(1), Backend('always')):
            self.assertEqual(reasoning_decision_budget(64, backend), 64)
            self.assertEqual(reasoning_decision_budget(96, backend), 96)

    def test_real_route_wiring_and_user_visible_budget_are_separate(self):
        for required, expected in ((True, 1088), (False, 64)):
            backend = Backend(required)
            with tempfile.TemporaryDirectory() as directory:
                result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                    'Say hello.', temperature=0, max_tokens=256, workdir=Path(directory),
                )
            self.assertEqual(result.content, 'Hello.')
            self.assertEqual(backend.calls, [expected, 256])

    def test_tool_call_budget_uses_capability_before_model_request(self):
        class StopBeforeGeneration(BaseException):
            pass
        class ToolBackend(Backend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls.append(max_tokens)
                raise StopBeforeGeneration
        for required, expected in ((True, 1120), (False, 96)):
            backend = ToolBackend(required)
            with tempfile.TemporaryDirectory() as directory:
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                with self.assertRaises(StopBeforeGeneration):
                    runtime.ask_with_tools('Inspect local state.', temperature=0, max_tokens=256,
                                           workdir=Path(directory), tool_names=('system_info',))
            self.assertEqual(backend.calls, [expected])

    def test_length_retry_preserves_reserve_without_changing_legacy_retry(self):
        class StopAfterRetry(BaseException):
            pass
        class RetryBackend(Backend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                self.calls.append(max_tokens)
                if len(self.calls) == 1:
                    return ChatResult('', 'test', 'length', [], 1, max_tokens, 0, None, None)
                raise StopAfterRetry
        for required, expected in ((True, [1120, 1120]), (False, [96, 256])):
            backend = RetryBackend(required)
            with tempfile.TemporaryDirectory() as directory:
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                with self.assertRaises(StopAfterRetry):
                    runtime.ask_with_tools('Inspect local state.', temperature=0, max_tokens=256,
                                           workdir=Path(directory), tool_names=('system_info',))
            self.assertEqual(backend.calls, expected)

    def test_expanded_total_is_refused_before_generation_when_context_is_too_small(self):
        class ExactBackend(Backend):
            def count_chat_tokens(self, messages, *, tools=None, thinking=False):
                return TokenCount(10, 1024, "rendered", "tokens")
        backend = ExactBackend()
        with tempfile.TemporaryDirectory() as directory:
            runtime = ChatRuntime(backend=backend, system_prompt=None, context_tokens=1024)
            with self.assertRaises(ContextAdmissionError):
                runtime.ask_auto('Say hello.', temperature=0, max_tokens=256, workdir=Path(directory))
        self.assertEqual(backend.calls, [])

    def test_backend_requires_explicit_native_always_separated_contract(self):
        backend = LlamaServerBackend(base_url='http://localhost', timeout=1)
        good = {'backend': 'orbit-native', 'model_compatibility': {
            'reasoning_behavior': 'always', 'separated_reasoning': True, 'verified': False,
        }}
        import copy
        with patch.object(backend, '_props_or_empty', return_value=good):
            self.assertTrue(backend.requires_native_reasoning())
            self.assertFalse(backend.uses_native_separated_reasoning())  # no authority granted
        for key, value in (('reasoning_behavior', 'optional'), ('separated_reasoning', 1),
                           ('separated_reasoning', False)):
            props = copy.deepcopy(good)
            props['model_compatibility'][key] = value
            with patch.object(backend, '_props_or_empty', return_value=props):
                self.assertFalse(backend.requires_native_reasoning())
        for props in ({}, {'backend': 'other', 'model_compatibility': good['model_compatibility']},
                      {'backend': 'orbit-native', 'model_compatibility': None}):
            with patch.object(backend, '_props_or_empty', return_value=props):
                self.assertFalse(backend.requires_native_reasoning())


if __name__ == '__main__':
    unittest.main()

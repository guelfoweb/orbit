"""ANALYSIS must plan every model call through Orbit's exact context admission.

MANUAL-TEST-DIAG-2 measured the failure end to end. ANALYSIS had no admission at
all: prompts grew 581 -> 2898 -> 5241 -> 6105 -> 6991 against a budget of
`8192 - 2048 - 256 - 256 = 5632`, the two over-budget calls were submitted
anyway, and the resident sequence then reached the context wall -- generation
died at iteration 263 with `llama_decode == 1` ("could not find a KV slot") at
physical frontier 8192 of 8192, one token short.

The cause was logical over-admission, so these tests pin the logical fix. The
token counts below are the ones actually measured, not invented boundaries.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime.analysis_runtime import AnalysisRuntime  # noqa: E402
from orbit.runtime.context_manager import ContextAdmissionError  # noqa: E402

CTX = 8192
ANALYSIS_BUDGET = 5632  # 8192 - 2048 output - 256 next-action - 256 margin


def _result() -> ChatResult:
    return ChatResult(
        content="ok", model="m", finish_reason="stop", tool_calls=[],
        prompt_tokens=1, completion_tokens=1, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


class _Backend:
    """An orbit-native backend whose exact token count the test controls."""

    thinking = False

    def __init__(self, tokens: int, *, context_tokens: int = CTX) -> None:
        self._count = TokenCount(
            tokens=tokens, context_tokens=context_tokens,
            rendered_hash="a" * 64, token_hash="b" * 64,
        )
        self.chat_stream_calls: list[list[dict]] = []
        self.count_calls = 0

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.count_calls += 1
        return self._count

    def chat_stream(self, messages, **kwargs):
        self.chat_stream_calls.append(list(messages))
        return _result()

    def chat(self, messages, **kwargs):
        return _result()


def _runtime(backend) -> AnalysisRuntime:
    runtime = object.__new__(AnalysisRuntime)
    object.__setattr__(runtime, "backend", backend)
    object.__setattr__(runtime, "messages", [
        {"role": "system", "content": "analysis system"},
        {"role": "user", "content": "artifact"},
    ])
    object.__setattr__(runtime, "context_compactions", 0)
    object.__setattr__(runtime, "last_context_plan", None)
    return runtime


class MeasuredPromptSizeTests(unittest.TestCase):
    """The five prompt sizes the failing run actually produced."""

    def test_the_three_in_budget_prompts_are_admitted(self) -> None:
        for tokens in (581, 2898, 5241):
            with self.subTest(tokens=tokens):
                backend = _Backend(tokens)
                admitted = _runtime(backend)._admit(
                    [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
                )
                self.assertTrue(admitted)
                self.assertLess(tokens, ANALYSIS_BUDGET)

    def test_the_6105_token_call_is_refused_before_the_backend(self) -> None:
        """The first over-budget call of the measured run."""
        backend = _Backend(6105)
        with self.assertRaises(ContextAdmissionError):
            _runtime(backend)._admit(
                [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
            )
        self.assertEqual(
            backend.chat_stream_calls, [],
            "an over-budget request must never reach the backend",
        )

    def test_the_6991_token_call_is_refused_before_the_backend(self) -> None:
        """The last call before the physical context wall was hit."""
        backend = _Backend(6991)
        with self.assertRaises(ContextAdmissionError):
            _runtime(backend)._admit(
                [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
            )
        self.assertEqual(backend.chat_stream_calls, [])

    def test_the_budget_boundary_is_the_shared_one(self) -> None:
        """Exactly at the limit admits; one token past it does not.

        Pins that ANALYSIS uses the shared arithmetic rather than a local copy:
        a duplicated formula could drift and this would catch it.
        """
        ok = _Backend(ANALYSIS_BUDGET)
        self.assertTrue(_runtime(ok)._admit([{"role": "user", "content": "x"}],
                                            max_tokens=2048, tools=[]))
        over = _Backend(ANALYSIS_BUDGET + 1)
        with self.assertRaises(ContextAdmissionError):
            _runtime(over)._admit([{"role": "user", "content": "x"}],
                                  max_tokens=2048, tools=[])


class AdmissionUsesAnalysisHistoryTests(unittest.TestCase):
    """The history planned must be ANALYSIS's own, never chat's."""

    def test_the_planned_messages_are_the_ones_passed_in(self) -> None:
        backend = _Backend(100)
        runtime = _runtime(backend)
        own = [{"role": "user", "content": "analysis-owned-history"}]
        admitted = runtime._admit(own, max_tokens=2048, tools=[])
        self.assertEqual(
            [m["content"] for m in admitted], ["analysis-owned-history"],
            "admission must plan the caller's history, not some other list",
        )

    def test_the_context_size_comes_from_the_backend(self) -> None:
        """Not a constant here: ANALYSIS must not drift from the loaded model."""
        backend = _Backend(100)
        self.assertEqual(_runtime(backend)._context_tokens(), CTX)


class ConfiguredContextIsAppliedTests(unittest.TestCase):
    """The configured cap must constrain admission, not just the backend's."""

    def test_the_smaller_configured_context_wins(self) -> None:
        """A backend claiming a larger window must not widen the budget.

        `plan_exact_context` takes `min(attested, configured)`. Without the
        configured value the planner would trust the backend alone, so a
        backend reporting 32k would admit a prompt that cannot fit the 8k
        context the model was actually loaded with -- exactly the class of
        over-admission that caused the KV exhaustion.
        """
        class Roomy(_Backend):
            def model_info(self):
                class _Info:
                    context_length = CTX  # what the model was really loaded with
                return _Info()

        # The backend attests a much larger window than it was loaded with.
        backend = Roomy(6991, context_tokens=32768)
        with self.assertRaises(ContextAdmissionError):
            _runtime(backend)._admit(
                [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
            )
        self.assertEqual(backend.chat_stream_calls, [])


class AdmissionCapabilityTests(unittest.TestCase):
    """Behaviour mirrors CHAT for backends that cannot attest exact tokens."""

    def test_a_non_attesting_backend_skips_admission_rather_than_guessing(self) -> None:
        class Plain(_Backend):
            def supports_exact_context_admission(self) -> bool:
                return False

        backend = Plain(9999)
        messages = [{"role": "user", "content": "x"}]
        self.assertEqual(
            _runtime(backend)._admit(messages, max_tokens=2048, tools=[]), messages
        )

    def test_an_unknown_capability_fails_closed(self) -> None:
        class Unknown(_Backend):
            def supports_exact_context_admission(self):
                return None

        with self.assertRaises(ContextAdmissionError):
            _runtime(Unknown(100))._admit(
                [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
            )


class SinglePlanningTests(unittest.TestCase):
    """Admission must happen once per request, not stack with another layer."""

    def test_one_admission_plans_the_request_once(self) -> None:
        backend = _Backend(100)
        _runtime(backend)._admit([{"role": "user", "content": "x"}],
                                 max_tokens=2048, tools=[])
        # plan_exact_context counts twice by design (it re-counts to prove the
        # render is stable); more than that means a second admission layer.
        self.assertLessEqual(
            backend.count_calls, 2,
            "a stacked second admission would re-count the same request",
        )


class CallSiteWiringTests(unittest.TestCase):
    """Both streaming ANALYSIS call sites go through admission."""

    def test_step_and_report_both_admit_before_calling_the_backend(self) -> None:
        import ast

        source = (ROOT / "src/orbit/runtime/analysis_runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        admitting: set[str] = set()
        streaming: set[str] = set()
        for parent in ast.walk(tree):
            if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(parent)
            if "self._admit(" in body:
                admitting.add(parent.name)
            if "self.backend.chat_stream(" in body:
                streaming.add(parent.name)

        self.assertTrue(
            streaming.issubset(admitting),
            f"every chat_stream site must admit first; missing {streaming - admitting}",
        )


if __name__ == "__main__":
    unittest.main()

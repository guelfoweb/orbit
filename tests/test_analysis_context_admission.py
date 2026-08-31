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
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    acquire_analysis_source,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402
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
        self.counted_thinking: list[bool] = []
        self.thinking = False

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        self.count_calls += 1
        self.counted_thinking.append(bool(thinking))
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


class AdmitOrRefuseContractTests(unittest.TestCase):
    """ANALYSIS admission is admit-or-refuse, and that is deliberate."""

    def test_analysis_history_is_never_compacted_only_refused(self) -> None:
        """Pins the real contract so a future reader is not misled.

        `plan_context` externalises only tool turns whose evidence is available
        AND covered; analysis tool messages carry no `evidence_id`, so no turn
        is eligible. The planner can return "unchanged" or "blocked" and never
        "compacted". This is not a lost opportunity being hidden -- it is the
        shape of analysis history -- but it must not be described as compaction.
        """
        from orbit.runtime.context_manager import ContextBudget, plan_context

        budget = ContextBudget(context_tokens=CTX, output_reserve=2048,
                               next_action_reserve=256, safety_margin=256)
        analysis_shaped = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "artifact"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1",
             "name": "execute_analysis", "content": "result"},
            {"role": "assistant", "content": "finding"},
            {"role": "user", "content": "continue"},
        ]
        plan = plan_context(analysis_shaped, budget=budget, count_tokens=lambda m: 6991)

        self.assertEqual(plan.status, "blocked")
        self.assertEqual(plan.compacted_turns, 0)
        self.assertEqual(plan.externalized_evidence_ids, ())


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


class CountedRenderMatchesSubmittedRenderTests(unittest.TestCase):
    """The render counted must be the render submitted."""

    def test_thinking_mode_is_taken_from_the_backend_not_assumed(self) -> None:
        """`chat_stream` sends `backend.thinking`; admission must count it.

        The REPL sets `backend.thinking` from `--think`, and ANALYSIS shares that
        backend object. Counting with `thinking=False` while submitting a
        thinking template under-counts in the permissive direction -- which is
        the over-admission this whole mechanism exists to prevent.
        """
        backend = _Backend(100)
        backend.thinking = True
        _runtime(backend)._admit(
            [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
        )
        self.assertEqual(
            set(backend.counted_thinking), {True},
            "admission counted a different render than chat_stream would send",
        )

    def test_thinking_off_is_counted_as_off(self) -> None:
        backend = _Backend(100)
        backend.thinking = False
        _runtime(backend)._admit(
            [{"role": "user", "content": "x"}], max_tokens=2048, tools=[]
        )
        self.assertEqual(set(backend.counted_thinking), {False})


class AdmittedMessagesReachTheBackendTests(unittest.TestCase):
    """Built with the real constructor: what `_admit` returns is what is sent.

    The AST wiring test proves `_admit` and `chat_stream` co-occur; it cannot
    prove the admitted result is the object actually submitted. Two mutations --
    sending `list(self.messages)` instead of `admitted`, and returning the input
    instead of `plan.messages` -- are equivalent while analysis can never
    compact, and would become silent defects the moment it can. This closes that
    data-flow gap behaviourally.
    """

    def setUp(self) -> None:
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="orbit-admission-rt-")
        self.addCleanup(self._dir.cleanup)
        tmp = pathlib.Path(self._dir.name)
        artifact = tmp / "artifact.txt"
        artifact.write_text("payload", encoding="utf-8")
        self.source = acquire_analysis_source(artifact, tmp / "owned")
        self.store = EvidenceStore(root=tmp / "evidence")

    def test_the_object_returned_by_admission_is_the_one_submitted(self) -> None:
        marker = {"role": "user", "content": "ADMITTED-PROJECTION-MARKER"}

        class Backend(_Backend):
            def __init__(self) -> None:
                super().__init__(100)

        backend = Backend()
        runtime = AnalysisRuntime(
            backend=backend, source=self.source, evidence_store=self.store
        )
        self.addCleanup(runtime.close)
        # Force admission to return a distinguishable projection.
        runtime._admit = lambda messages, **kwargs: [marker]  # type: ignore[assignment]

        runtime.step("go")

        self.assertTrue(backend.chat_stream_calls, "the backend must be called")
        self.assertEqual(
            backend.chat_stream_calls[-1], [marker],
            "the messages sent must be exactly what admission returned",
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

"""An empty plan, asked once more when the runtime already decoded something.

An empty plan is a legitimate answer and stays one. What these tests pin is the
narrower case: the preflight decoded evidence before the model was asked
anything, and the plan still came back empty. That case is induced rather than
random -- the opening instruction ends with "If it is sufficient, report now",
and for an artifact whose whole payload is one decoded stage the rehydrated
evidence can genuinely look sufficient -- so the runtime asks once, states what
it holds, and offers the same empty plan back as a first-class answer.

The boundary that matters most is the one in the other direction: an artifact
with nothing decoded behind it must still close honestly on an empty plan,
without a second call and without inventing work.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_controller import (  # noqa: E402
    PHASE_PLAN,
    PHASE_REPORT,
    PHASE_RESOLVE,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    EMPTY_PLAN_WITH_EVIDENCE_MESSAGE,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
    STOP_LEDGER_EXHAUSTED,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192

# A fromCharCode-offset artifact: the shape whose whole payload is one decoded
# stage. Written here rather than read from the corpus so the test is about the
# transform family, not about one sample.
BASE = 177904575
PAYLOAD = 'var x = new ActiveXObject("WScript.Shell");x.Run("calc.exe");'
DECODED_SOURCE = (
    f"var K={BASE}\n"
    "var DEC = String.fromCharCode("
    + ",".join(f"{BASE + ord(ch)}-K" for ch in PAYLOAD)
    + ")\neval(DEC)\n"
)
# Nothing to decode, no container, no macro: the honest-closure side.
INERT_SOURCE = "def add(a, b):\n    return a + b\n"


class _Model:
    """Scripted by call type. `plans` is consumed one per PLAN call."""

    prose = ""

    def __init__(self, plans, code="print(1)") -> None:
        self.plans = list(plans)
        self.code = code
        self.plan_calls = 0
        self.actions = 0
        self.plan_messages: list[list[dict]] = []

    def reply(self, tools, messages):
        names = [t["function"]["name"] for t in (tools or [])]
        if PLAN_TOOL_NAME in names:
            self.plan_messages.append([dict(m) for m in messages])
            plan = (
                self.plans[self.plan_calls]
                if self.plan_calls < len(self.plans)
                else []
            )
            self.plan_calls += 1
            return self._call(PLAN_TOOL_NAME, {"questions": plan})
        if FINISH_TOOL_NAME in names:
            return self._call(
                FINISH_TOOL_NAME,
                {"status": "resolved", "answer_summary": "done"},
            )
        if ANALYSIS_TOOL_NAME in names:
            self.actions += 1
            return self._call(
                ANALYSIS_TOOL_NAME,
                {"code": f"# action {self.actions}\n{self.code}"},
            )
        return None

    def _call(self, name, arguments):
        if name == PLAN_TOOL_NAME and isinstance(arguments.get("questions"), list):
            arguments = {**arguments, "questions": [
                {"data_request": None, **q} if isinstance(q, dict) else q
                for q in arguments["questions"]]}
        return [{
            "id": f"c{name}{self.plan_calls}{self.actions}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]


class _Backend:
    thinking = False

    def __init__(self, model: _Model) -> None:
        self.model = model
        self.chat_calls: list[dict] = []

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def count_text_tokens(self, text: str):
        return TokenCount(tokens=len(text.split()), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        tools = kwargs.get("tools")
        self.chat_calls.append({"tools": tools, "messages": list(messages)})
        calls = self.model.reply(tools, messages) or []
        return ChatResult(
            content=self.model.prose, model="m", finish_reason="stop",
            tool_calls=calls, prompt_tokens=1, completion_tokens=1,
            cached_tokens=0, prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def _question(text: str) -> dict:
    return {"question": text, "missing_fact": "needs execution"}


class _Case(unittest.TestCase):
    def _runtime(self, model: _Model, data: bytes, name="artifact.js"):
        self.backend = _Backend(model)
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / name
        path.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=self.backend,
            source=AnalysisSource(
                snapshot_path=path, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        return runtime

    def _run(self, runtime, **kwargs):
        counter = {"n": 0}

        def distinct(**_kwargs):
            counter["n"] += 1
            return AnalysisResult(
                status="ok", code_sha256=f"{counter['n']:064d}",
                input_sha256="i" * 64, stdout=f"FINDING {counter['n']}",
                stderr="", exit_status=0, duration_seconds=0.1,
            )

        with mock.patch.object(module, "execute_analysis", distinct):
            return runtime.run_autonomous("Analyse it.", finalize=False, **kwargs)

    def _plan_prompts(self, model):
        return [
            "\n".join(str(m.get("content") or "") for m in msgs)
            for msgs in model.plan_messages
        ]


class DecodedEvidenceTests(_Case):
    """T1. Deterministic evidence + an empty plan is asked once more."""

    def test_the_fixture_really_decodes(self) -> None:
        """T11. If this stops decoding, every other test here is vacuous."""
        runtime = self._runtime(_Model([[]]), DECODED_SOURCE.encode())
        self.assertEqual(len(runtime.transform_stages), 1)
        stage, record = runtime.transform_stages[0]
        self.assertEqual(stage.kind, "js_fromcharcode_offset")
        self.assertEqual(stage.output, PAYLOAD)
        self.assertEqual(record.produced_by_phase, "analysis_transform")

    def test_an_empty_plan_with_evidence_is_re_asked_once(self) -> None:
        model = _Model([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime)
        self.assertEqual(model.plan_calls, 2)
        self.assertEqual(run.plan_calls, 2)
        # Still empty the second time, so the run still closes honestly.
        self.assertEqual(run.initial_questions, 0)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_the_re_ask_states_what_the_runtime_holds(self) -> None:
        model = _Model([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        self._run(runtime)
        second = self._plan_prompts(model)[1]
        self.assertIn("already holds deterministic evidence", second)
        self.assertIn("1 deterministic transformation", second)
        self.assertIn("js_fromcharcode_offset", second)
        # It offers the empty plan back rather than demanding questions.
        self.assertIn("with an empty list", second)

    def test_a_re_asked_plan_with_questions_is_worked(self) -> None:
        """The recovery is worth having only if the second plan can run."""
        model = _Model([[], [_question("what does the decoded stage run?")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime)
        self.assertEqual(run.plan_calls, 2)
        self.assertEqual(run.initial_questions, 1)
        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.answered_unverified_questions, ("Q1",))

    def test_question_ids_start_at_Q1_after_an_empty_adoption(self) -> None:
        """An empty adoption records nothing, so the re-ask is a clean slate."""
        model = _Model([[], [_question("a"), _question("b")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime)
        self.assertEqual(set(run.answered_unverified_questions), {"Q1", "Q2"})


class HonestClosureTests(_Case):
    """T2/T8. Nothing decoded means nothing to re-ask about."""

    def test_an_inert_artifact_is_not_re_asked(self) -> None:
        model = _Model([[]])
        runtime = self._runtime(model, INERT_SOURCE.encode(), name="a.py")
        self.assertEqual(runtime.transform_stages, [])
        run = self._run(runtime)
        self.assertEqual(model.plan_calls, 1)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.actions_executed, 0)
        self.assertEqual(run.stop_reason, STOP_LEDGER_EXHAUSTED)

    def test_the_summary_is_empty_without_preflight_evidence(self) -> None:
        runtime = self._runtime(_Model([[]]), INERT_SOURCE.encode(), name="a.py")
        self.assertEqual(runtime._deterministic_evidence_summary(), "")

    def test_the_summary_describes_evidence_without_quoting_it(self) -> None:
        """T10. Counts and kinds, never the decoded bytes."""
        runtime = self._runtime(_Model([[]]), DECODED_SOURCE.encode())
        summary = runtime._deterministic_evidence_summary()
        self.assertIn("1 deterministic transformation", summary)
        self.assertIn("js_fromcharcode_offset", summary)
        self.assertNotIn(PAYLOAD, summary)
        self.assertNotIn("calc.exe", summary)


class EvidenceSummaryTests(_Case):
    """T7. The Office/VBA half of the summary, and how the two combine.

    Driven by assigning the runtime's own preflight fields rather than by
    parsing a real container: the summary is a function of those fields, and a
    `.doc` fixture would test the OLE extractor instead.
    """

    def _runtime_for_summary(self):
        return self._runtime(_Model([[]]), INERT_SOURCE.encode(), name="a.py")

    def test_office_modules_alone_are_summarised(self) -> None:
        runtime = self._runtime_for_summary()
        runtime.office_modules = [(object(), object())]
        self.assertEqual(
            runtime._deterministic_evidence_summary(),
            "1 extracted Office/VBA module",
        )

    def test_office_modules_are_pluralised(self) -> None:
        runtime = self._runtime_for_summary()
        runtime.office_modules = [(object(), object()), (object(), object())]
        self.assertEqual(
            runtime._deterministic_evidence_summary(),
            "2 extracted Office/VBA modules",
        )

    def test_both_kinds_of_evidence_are_joined(self) -> None:
        runtime = self._runtime(_Model([[]]), DECODED_SOURCE.encode())
        runtime.office_modules = [(object(), object())]
        summary = runtime._deterministic_evidence_summary()
        self.assertEqual(
            summary,
            "1 deterministic transformation (js_fromcharcode_offset) "
            "and 1 extracted Office/VBA module",
        )

    def test_repeated_kinds_are_named_once(self) -> None:
        runtime = self._runtime(_Model([[]]), DECODED_SOURCE.encode())
        stage, record = runtime.transform_stages[0]
        runtime.transform_stages = [(stage, record), (stage, record)]
        summary = runtime._deterministic_evidence_summary()
        self.assertEqual(
            summary,
            "2 deterministic transformations (js_fromcharcode_offset)",
        )

    def test_an_office_only_artifact_is_re_asked(self) -> None:
        """The Office branch drives the recovery, not just the sentence."""
        model = _Model([[], []])
        runtime = self._runtime(model, INERT_SOURCE.encode(), name="a.py")
        runtime.office_modules = [(object(), object())]
        controller = module.AnalysisController()
        runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        self.assertEqual(model.plan_calls, 2)


class BoundednessTests(_Case):
    """T3/T8. One extra call, once, inside the existing ceiling."""

    def test_the_re_ask_happens_at_most_once(self) -> None:
        # Empty every time: the runtime must not keep asking.
        model = _Model([[], [], [], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime)
        self.assertEqual(model.plan_calls, 2)
        self.assertEqual(run.plan_calls, 2)

    def test_a_one_call_ceiling_denies_the_re_ask(self) -> None:
        """It shares the plan budget rather than adding to it."""
        model = _Model([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        calls = runtime.plan_analysis(controller, "Analyse it.", max_calls=1)
        self.assertEqual(calls, 1)
        self.assertEqual(model.plan_calls, 1)
        self.assertEqual(controller.phase, PHASE_REPORT)

    def test_the_re_ask_leaves_a_call_to_investigate_with(self) -> None:
        """Two calls is enough to re-ask and not enough to act on the answer.

        Spending the last call replacing an honest empty close with a question
        nothing can work is strictly worse than not asking, so the re-ask wants
        a spare call exactly as COVER does.
        """
        model = _Model([[], [_question("a")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        calls = runtime.plan_analysis(controller, "Analyse it.", max_calls=2)
        self.assertEqual(calls, 1)
        self.assertEqual(model.plan_calls, 1)
        self.assertEqual(controller.order, [])

    def test_an_empty_plan_after_a_protocol_repair_is_not_re_asked(self) -> None:
        """The last iteration cannot arm a re-ask it has no call to send.

        The protocol repair reaches attempt 1 without touching the re-ask
        flags: a reply with no tool call, then a valid empty plan. Arming the
        re-ask there would build a message the exhausted loop never sends --
        counting a repair that never reached the model, burning the run's one
        re-ask on a phantom, and leaving the phase at PHASE_PLAN.
        """
        class _Mute(_Model):
            def reply(self, tools, messages):
                names = [t["function"]["name"] for t in (tools or [])]
                if PLAN_TOOL_NAME in names and self.plan_calls == 0:
                    self.plan_calls += 1
                    return []            # no tool call: drives the repair
                return super().reply(tools, messages)

        model = _Mute([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        self.assertFalse(runtime._empty_plan_re_asked)
        # One repair: the protocol one, which really was dispatched.
        self.assertEqual(controller.repairs, 1)
        self.assertEqual(controller.phase, PHASE_REPORT)
        self.assertEqual(controller.order, [])

    def test_a_withdrawn_evidence_message_does_not_buy_a_second_re_ask(
        self,
    ) -> None:
        """PLAN runs twice when admission withdraws the evidence-first line.

        Both that withdrawal and the re-ask require deterministic evidence, so
        they coincide by construction. The guard is therefore run-scoped, and
        a second `plan_analysis` must not re-ask again.
        """
        model = _Model([[], [], [], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        first = module.AnalysisController()
        runtime.plan_analysis(first, "Analyse it.", max_calls=3)
        self.assertEqual(model.plan_calls, 2)
        self.assertTrue(runtime._empty_plan_re_asked)
        # The withdrawal retry: a fresh controller, the same runtime.
        second = module.AnalysisController()
        runtime.plan_analysis(second, "Analyse it.", max_calls=3)
        self.assertEqual(model.plan_calls, 3)      # one dispatch, no re-ask
        self.assertEqual(second.repairs, 0)

    def test_the_re_ask_is_counted_as_a_repair(self) -> None:
        model = _Model([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        self.assertEqual(controller.repairs, 1)

    def test_the_model_call_ceiling_still_bounds_the_run(self) -> None:
        model = _Model([[], []])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime, max_model_calls=2)
        self.assertLessEqual(run.model_calls, 2)


class UnchangedPlanTests(_Case):
    """T4. A plan with questions never sees any of this."""

    def test_a_non_empty_plan_is_adopted_on_the_first_call(self) -> None:
        model = _Model([[_question("a"), _question("b")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        run = self._run(runtime)
        self.assertEqual(model.plan_calls, 1)
        self.assertEqual(run.plan_calls, 1)
        self.assertEqual(run.initial_questions, 2)
        self.assertEqual(run.actions_executed, 2)

    def test_a_non_empty_plan_is_not_counted_as_a_repair(self) -> None:
        model = _Model([[_question("a")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        runtime.plan_analysis(controller, "Analyse it.", max_calls=2)
        self.assertEqual(controller.repairs, 0)


class ProtocolFailureTests(_Case):
    """The re-ask must not be mistaken for a protocol failure."""

    def test_an_unusable_re_ask_keeps_the_adopted_empty_plan(self) -> None:
        class _Silent(_Model):
            def reply(self, tools, messages):
                names = [t["function"]["name"] for t in (tools or [])]
                if PLAN_TOOL_NAME in names and self.plan_calls >= 1:
                    self.plan_calls += 1
                    return []          # no call at all on the re-ask
                return super().reply(tools, messages)

        model = _Silent([[]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        calls = runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        # The empty plan from attempt one still stands: not `unsupported`,
        # which would report a protocol the model never failed to produce.
        self.assertFalse(controller.unsupported)
        self.assertEqual(controller.phase, PHASE_REPORT)
        # One call returned into planning -- the first. The re-ask did not.
        self.assertEqual(calls, 1)

    def test_the_phase_is_restored_so_the_re_asked_plan_is_adopted(self) -> None:
        """The restore is what lets the second plan be taken at all.

        Asserting only that the phase is not PHASE_PLAN afterwards would pass
        with the recovery deleted, because the first empty adoption already
        leaves PHASE_REPORT. What proves the restore happened is the re-asked
        plan being adopted -- `adopt_plan` refuses outright from any other
        phase.
        """
        model = _Model([[], [_question("a")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        self.assertEqual(controller.phase, PHASE_PLAN)
        runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        self.assertEqual(controller.order, ["Q1"])
        self.assertEqual(controller.phase, PHASE_RESOLVE)

    def test_a_refused_re_ask_leaves_no_question_behind(self) -> None:
        """A plan the controller REFUSES must leave no trace of itself.

        The re-ask returns one usable question and one duplicate. `adopt_plan`
        rejects the plan as a whole, so the run must end with the empty plan it
        already had -- not working a question out of a plan that was refused.
        """
        model = _Model([[], [_question("a"), _question("a")]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        calls = runtime.plan_analysis(controller, "Analyse it.", max_calls=3)
        self.assertEqual(controller.questions, {})
        self.assertEqual(controller.order, [])
        self.assertEqual(controller.open_ids, [])
        self.assertEqual(controller.phase, PHASE_REPORT)
        self.assertFalse(controller.unsupported)
        # Exactly one call returned into planning, and it is reported as one.
        self.assertEqual(calls, 1)


class NoSampleSpecificsTests(unittest.TestCase):
    """T10. Nothing here knows about any particular artifact."""

    def test_the_message_names_no_sample(self) -> None:
        text = EMPTY_PLAN_WITH_EVIDENCE_MESSAGE.lower()
        for token in ("iban", "j7f", "mmgclz", "fattura", "hta", "productoslili"):
            self.assertNotIn(token, text)

    def test_the_message_asserts_no_finding(self) -> None:
        """It states what exists, never what the artifact does."""
        text = EMPTY_PLAN_WITH_EVIDENCE_MESSAGE.lower()
        for token in ("malicious", "malware", "dropper", "payload", "c2"):
            self.assertNotIn(token, text)


class CancellationTests(_Case):
    """T9. A cancel during the recovery propagates, it is not swallowed."""

    def test_cancellation_during_the_re_ask_propagates(self) -> None:
        class _Cancel(_Model):
            def reply(self, tools, messages):
                names = [t["function"]["name"] for t in (tools or [])]
                if PLAN_TOOL_NAME in names and self.plan_calls >= 1:
                    raise KeyboardInterrupt
                return super().reply(tools, messages)

        model = _Cancel([[]])
        runtime = self._runtime(model, DECODED_SOURCE.encode())
        controller = module.AnalysisController()
        with self.assertRaises(KeyboardInterrupt):
            runtime.plan_analysis(controller, "Analyse it.", max_calls=3)


if __name__ == "__main__":
    unittest.main()

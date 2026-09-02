"""One repair opportunity after an execution that ran and raised.

A program that failed on its own defect is not the same situation as one that
produced nothing useful: the model already holds the source it submitted and
the interpreter's verbatim reason. Observed behaviour without an invitation to
use them is to abandon the attempt and resume reading source, which spends the
budget re-establishing what was already known.

These cover the offer and, more importantly, its bounds: exactly one per
failure, never two in a row, never for a failure resubmitting cannot fix, and
never at the cost of the ceilings that make a run terminate.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.backend.base import ChatResult
from orbit.runtime.analysis_runtime import (
    ANALYSIS_TOOL_NAME,
    AUTONOMOUS_CONTINUATION_MESSAGE,
    AUTONOMOUS_REPAIR_MESSAGE,
    AUTONOMOUS_REPLAN_MESSAGE,
    MAX_AUTONOMOUS_ACTIONS,
    MAX_AUTONOMOUS_MODEL_CALLS,
    AnalysisRuntime,
    _is_locally_repairable,
    acquire_analysis_source,
)
from orbit.runtime.analysis_sandbox import AnalysisResult
from orbit.runtime.evidence import EvidenceStore

SECRET = "STAGE-TWO-COMMAND"
ENCODED = "-".join(str(ord(c) ^ 7) for c in SECRET)
SOURCE = f"var x = 1;\npayload={ENCODED}\n"

# The real shape of the observed failure: JScript coerces a string operand,
# Python does not. The corrected form differs only by the int() call.
BROKEN = (
    "import orbit_tools\n"
    "src = orbit_tools.read_file('/workspace/input')\n"
    "blob = src.split('payload=')[1].strip()\n"
    "print(''.join(chr(t ^ 7) for t in blob.split('-')))\n"
)
FIXED = BROKEN.replace("chr(t ^ 7)", "chr(int(t) ^ 7)")
READ = "import orbit_tools; print(orbit_tools.read_file('/workspace/input')[:60])"


def _tool_call(code: str, *, call_id: str = "call_1") -> ChatResult:
    return ChatResult(
        content="", model="m", finish_reason="stop",
        tool_calls=[{
            "id": call_id, "type": "function",
            "function": {"name": ANALYSIS_TOOL_NAME, "arguments": json.dumps({"code": code})},
        }],
        prompt_tokens=10, completion_tokens=5, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


def _prose(text: str) -> ChatResult:
    return ChatResult(
        content=text, model="m", finish_reason="stop", tool_calls=[],
        prompt_tokens=10, completion_tokens=5, cached_tokens=0,
        prompt_tokens_per_second=None, generation_tokens_per_second=None,
    )


class RecordingBackend:
    """Serves scripted responses and records the analyst line that drove each."""

    def __init__(self, *responses: ChatResult) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.instructions: list[str] = []

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta, on_progress=None):
        if self.calls >= len(self._responses):
            raise AssertionError(
                f"model invoked {self.calls + 1} times; only {len(self._responses)} scripted"
            )
        users = [m for m in messages if m.get("role") == "user"]
        self.instructions.append(users[-1]["content"] if users else "")
        response = self._responses[self.calls]
        self.calls += 1
        if response.content:
            on_delta(response.content)
        return response


class RepairTestBase(unittest.TestCase):
    SOURCE_TEXT = SOURCE

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-repair-")
        self.addCleanup(self._dir.cleanup)
        self.tmp = Path(self._dir.name)
        artifact = self.tmp / "artifact.js"
        artifact.write_text(self.SOURCE_TEXT, encoding="utf-8")
        self.source = acquire_analysis_source(artifact, self.tmp / "owned")
        self.store = EvidenceStore(root=self.tmp / "evidence")

    def runtime(self, backend) -> AnalysisRuntime:
        built = AnalysisRuntime(
            backend=backend, source=self.source, evidence_store=self.store
        )
        self.addCleanup(built.close)
        return built


def _result(status: str, *, stderr: str = "", stdout: str = "") -> AnalysisResult:
    return AnalysisResult(
        status=status, code_sha256="c", input_sha256="i",
        stdout=stdout, stderr=stderr,
        exit_status=0 if status == "ok" else 1, duration_seconds=0.1,
    )


class _Step:
    """The fields `_is_locally_repairable` reads, and nothing else."""

    def __init__(self, *, executed=True, result=None, suppressed=None):
        self.action_executed = executed
        self.result = result
        self.suppressed_duplicate_of = suppressed


class EligibilityTests(unittest.TestCase):
    def test_an_execution_that_ran_and_raised_is_repairable(self) -> None:
        step = _Step(result=_result("error", stderr="Traceback...\nTypeError: x"))
        self.assertTrue(_is_locally_repairable(step))

    def test_success_is_not_repairable(self) -> None:
        self.assertFalse(_is_locally_repairable(_Step(result=_result("ok", stdout="hi"))))

    def test_resource_ceilings_are_not_repairable(self) -> None:
        """The same program is the wrong answer; a retry re-hits the wall."""
        for status in ("timeout", "bounded"):
            with self.subTest(status=status):
                step = _Step(result=_result(status, stderr="killed"))
                self.assertFalse(_is_locally_repairable(step))

    def test_a_step_that_never_executed_is_not_repairable(self) -> None:
        """No traceback to reason from.

        Checked with a result attached as well as without: a refused step is
        ineligible because nothing ran, not merely because `result` happened
        to be None. The refusal path can carry a prior result object, and
        keying on its absence would offer a repair for a program the sandbox
        never executed.
        """
        self.assertFalse(_is_locally_repairable(_Step(executed=False, result=None)))
        self.assertFalse(
            _is_locally_repairable(
                _Step(executed=False, result=_result("error", stderr="Traceback\nboom"))
            )
        )

    def test_a_suppressed_duplicate_is_not_repairable(self) -> None:
        step = _Step(result=_result("error", stderr="boom"), suppressed="ev_1")
        self.assertFalse(_is_locally_repairable(step))

    def test_a_silent_failure_is_not_repairable(self) -> None:
        """"Try again" with no diagnosis is exactly what this avoids."""
        self.assertFalse(_is_locally_repairable(_Step(result=_result("error", stderr="   "))))


class GenericContractTests(unittest.TestCase):
    """The offer must not learn anything about what it is analysing."""

    FORBIDDEN = (
        "xor", "base64", "hex", "decode", "decoder", "decrypt", "malware",
        "javascript", "jscript", "powershell", "int(", "typeerror",
        "nameerror", "valueerror", "attributeerror", "syntaxerror",
    )

    def test_the_repair_message_names_no_technique_or_error_class(self) -> None:
        lowered = AUTONOMOUS_REPAIR_MESSAGE.lower()
        for term in self.FORBIDDEN:
            with self.subTest(term=term):
                self.assertNotIn(term, lowered)

    def test_eligibility_does_not_inspect_the_error_text(self) -> None:
        """Any diagnosed failure qualifies, whatever the exception class.

        Keying on a particular exception would make the runtime an expert on
        the artifact rather than on its own execution, and would silently stop
        offering repairs the day a program failed some other way.
        """
        for stderr in (
            "Traceback (most recent call last):\nTypeError: bad operand",
            "Traceback (most recent call last):\nValueError: bad literal",
            "Traceback (most recent call last):\nKeyError: 'k'",
            "Traceback (most recent call last):\nZeroDivisionError: division by zero",
            "SyntaxError: invalid syntax",
            "some non-python diagnostic on stderr",
        ):
            with self.subTest(stderr=stderr.splitlines()[-1]):
                self.assertTrue(
                    _is_locally_repairable(_Step(result=_result("error", stderr=stderr)))
                )


class RepairFlowTests(RepairTestBase):
    def test_failure_then_repair_then_success(self) -> None:
        backend = RecordingBackend(
            _tool_call(BROKEN), _tool_call(FIXED, call_id="call_2"), _prose("decoded")
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 1, "exactly one repair opportunity")
        self.assertEqual(backend.instructions[1], AUTONOMOUS_REPAIR_MESSAGE)
        # No observational detour between the failure and the correction.
        self.assertNotIn(AUTONOMOUS_CONTINUATION_MESSAGE, backend.instructions[1])
        self.assertNotIn(AUTONOMOUS_REPLAN_MESSAGE, backend.instructions[1])

        corrected = run.steps[1]
        self.assertTrue(corrected.action_executed)
        self.assertIsNone(corrected.suppressed_duplicate_of)
        self.assertIn(SECRET, self.store.reattest_exact(corrected.evidence.evidence_id))

    def test_the_repair_call_sees_the_failed_code_and_its_traceback(self) -> None:
        """Both halves of a fix must still be in front of the model."""
        backend = RecordingBackend(
            _tool_call(BROKEN), _tool_call(FIXED, call_id="call_2"), _prose("ok")
        )
        runtime = self.runtime(backend)
        runtime.run_autonomous("analyse", finalize=False)

        history = "\n".join(
            str(m.get("content", "")) for m in runtime.messages
        ) + json.dumps([
            m.get("tool_calls") for m in runtime.messages if m.get("tool_calls")
        ])
        self.assertIn("chr(t ^ 7)", history, "the submitted code is preserved")
        self.assertIn("TypeError", history, "the traceback is preserved")

    def test_the_correction_adds_state_and_the_offer_does_not_depend_on_ERROR(self) -> None:
        """A raised program still writes its traceback, so the ledger calls it
        NEW_CONTENT. The offer keys off the sandbox status, not the
        classification, which is why it fires here at all -- and the
        correction adds state of its own rather than restating the failure.
        """
        backend = RecordingBackend(
            _tool_call(BROKEN), _tool_call(FIXED, call_id="call_2"), _prose("done")
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 1)
        self.assertTrue(run.progress[1].is_new_content)
        self.assertNotEqual(
            run.steps[0].evidence.raw_sha256, run.steps[1].evidence.raw_sha256
        )

    def test_a_successful_execution_is_never_offered_a_repair(self) -> None:
        backend = RecordingBackend(
            _tool_call(FIXED), _tool_call(READ, call_id="call_2"), _prose("done")
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 0)
        self.assertNotIn(AUTONOMOUS_REPAIR_MESSAGE, backend.instructions)


class RepairBoundTests(RepairTestBase):
    def test_a_repair_that_fails_again_gets_no_third_attempt(self) -> None:
        """One offer per failure, and never two consecutively."""
        broken_two = BROKEN.replace("[:60]", "[:61]") + "# variant\n"
        backend = RecordingBackend(
            _tool_call(BROKEN),
            _tool_call(broken_two, call_id="call_2"),
            _prose("giving up"),
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 1, "the second failure earns no new offer")
        self.assertEqual(backend.instructions.count(AUTONOMOUS_REPAIR_MESSAGE), 1)

    def test_a_failed_repair_hands_back_to_the_ordinary_loop(self) -> None:
        """One instruction per step, and the repair is spent exactly once.

        A second failing program whose stderr matches the first adds no new
        state, so the ledger calls it NO_PROGRESS while the sandbox still
        reports a diagnosed error -- a stall and a diagnosed failure at once.
        Because that step IS the repair, it earns no second offer, and the
        ordinary replan takes over: the model is asked for a different
        strategy rather than a third attempt at the same program.
        """
        fail = (
            "import sys\n"
            "sys.stderr.write('Traceback\\nBoomError: x\\n')\n"
            "sys.exit(1)\n"
        )
        backend = RecordingBackend(
            _tool_call(fail),
            _tool_call(fail + "# variant\n", call_id="call_2"),
            _prose("stopping"),
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 1, "the repair is offered exactly once")
        self.assertEqual(backend.instructions[1], AUTONOMOUS_REPAIR_MESSAGE)
        # Handed back: the failed repair gets the ordinary replan, not a
        # second repair, and never both directives for one step.
        self.assertEqual(backend.instructions[2], AUTONOMOUS_REPLAN_MESSAGE)
        self.assertEqual(backend.instructions.count(AUTONOMOUS_REPAIR_MESSAGE), 1)

    def test_a_dropped_replan_is_not_counted_as_one_sent(self) -> None:
        """`replans` counts messages, not intentions.

        A failing program writes its traceback as evidence, so the first such
        failure is NEW_CONTENT -- but a later failure with byte-identical
        stderr yields the same evidence hash, adds no state, and is classified
        NO_PROGRESS while the sandbox still reports a diagnosed error. Both
        the replan and the repair arm on that step; the repair wins and the
        replan is dropped. The analyst reads this figure, so a counter that
        recorded the intention would report a replan nobody was sent.
        """
        fail = (
            "import sys\n"
            "sys.stderr.write('Traceback\\nBoomError: x\\n')\n"
            "sys.exit(1)\n"
        )
        backend = RecordingBackend(
            _tool_call(fail),                                # NEW_CONTENT, repairable
            _tool_call("print('p1')\n", call_id="call_2"),   # the repair: progress
            _tool_call(fail + "#3\n", call_id="call_3"),     # same stderr: NO_PROGRESS *and* repairable
            _tool_call("print('p2')\n", call_id="call_4"),
            _prose("done"),
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        # The overlap really occurs: a repairable step classified NO_PROGRESS,
        # with no repair in flight, so both the replan and the repair arm.
        self.assertEqual(run.progress[2].classification, "NO_PROGRESS")
        # The repair wins and the replan is dropped -- not queued behind it.
        self.assertEqual(backend.instructions[3], AUTONOMOUS_REPAIR_MESSAGE)
        self.assertNotIn(AUTONOMOUS_REPLAN_MESSAGE, backend.instructions)
        self.assertEqual(
            run.replans,
            backend.instructions.count(AUTONOMOUS_REPLAN_MESSAGE),
            "replans must count messages actually sent, not intentions",
        )
        self.assertEqual(run.replans, 0)

    def test_an_offer_is_never_carried_across_an_intervening_step(self) -> None:
        """Freshly decided each iteration, never accumulated.

        An offer armed by step N and consumed at step N+2 would invite a fix
        to a program the model has since moved on from.
        """
        fail = (
            "import sys\n"
            "sys.stderr.write('Traceback\\nBoomError: y\\n')\n"
            "sys.exit(1)\n"
        )
        backend = RecordingBackend(
            _tool_call(fail),
            _tool_call("print('unrelated observation')\n", call_id="call_2"),
            _tool_call("print('another observation')\n", call_id="call_3"),
            _prose("done"),
        )
        runtime = self.runtime(backend)
        runtime.run_autonomous("analyse", finalize=False)

        # Offered once, immediately after the failure, and never again.
        self.assertEqual(backend.instructions[1], AUTONOMOUS_REPAIR_MESSAGE)
        self.assertNotIn(AUTONOMOUS_REPAIR_MESSAGE, backend.instructions[2:])

    def test_repair_does_not_raise_the_ceilings(self) -> None:
        """It spends an existing model call and no extra action budget."""
        backend = RecordingBackend(
            _tool_call(BROKEN), _tool_call(FIXED, call_id="call_2"), _prose("done")
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        # The ceilings themselves, not just conformance to them: a repair
        # mechanism that quietly bought headroom would still satisfy
        # `<= ceiling`, so the values are pinned here.
        self.assertEqual(MAX_AUTONOMOUS_ACTIONS, 12)
        self.assertEqual(MAX_AUTONOMOUS_MODEL_CALLS, 18)
        self.assertLessEqual(run.model_calls, MAX_AUTONOMOUS_MODEL_CALLS)
        self.assertLessEqual(run.actions_executed, MAX_AUTONOMOUS_ACTIONS)
        # The correction is an ordinary action, not a free one.
        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(run.model_calls, 3)

    def test_the_runtime_never_re_runs_the_failed_code_itself(self) -> None:
        """Every execution is one the model submitted."""
        backend = RecordingBackend(
            _tool_call(BROKEN), _tool_call(FIXED, call_id="call_2"), _prose("done")
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(backend.calls, 3)
        self.assertEqual(len(run.steps), 3)

    def test_a_later_failure_after_progress_earns_its_own_repair(self) -> None:
        """The bound is per failure, not per run."""
        backend = RecordingBackend(
            _tool_call(BROKEN),
            _tool_call(FIXED, call_id="call_2"),
            _tool_call(BROKEN.replace("[1].strip()", "[1].strip()  # second"), call_id="call_3"),
            _tool_call(READ, call_id="call_4"),
            _prose("done"),
        )
        runtime = self.runtime(backend)
        run = runtime.run_autonomous("analyse", finalize=False)

        self.assertEqual(run.repairs, 2)
        self.assertEqual(backend.instructions.count(AUTONOMOUS_REPAIR_MESSAGE), 2)


if __name__ == "__main__":
    unittest.main()

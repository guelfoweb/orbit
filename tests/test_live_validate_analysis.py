"""The live-validation recorder, tested by running it.

The previous version of this file asserted almost entirely on the harness's
SOURCE TEXT, and was worthless: one guard filtered lines by a predicate that
matched none of them, so its assertion never executed at all, and `main()` --
where every field access, the output write, the exit code and the cleanup
live -- was executed by no test. Two field-name bugs shipped underneath that
green suite, either of which would have destroyed a live run on a real sample.

So the instrument here is `_run_main`: it drives the real `main()` against a
stub backend and returns the JSON that was actually written to disk. Assertions
are about observable behaviour -- the file, the exit code, the record, the
workspace directory. Source-text guards appear only where they protect
something a run cannot show, and each proves it inspected something.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HARNESS_PATH = ROOT / "scripts" / "live_validate_analysis.py"
_SPEC = importlib.util.spec_from_file_location(
    "live_validate_analysis", HARNESS_PATH
)
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)

from orbit.runtime.evidence import EvidenceStore  # noqa: E402
from orbit.backend.base import (  # noqa: E402
    ChatResult,
    RecoverableBackendError,
    TokenCount,
)
from orbit.runtime.analysis_runtime import (  # noqa: E402
    ANALYSIS_TOOL_NAME,
    FINISH_TOOL_NAME,
    PLAN_TOOL_NAME,
)

CTX = 8192
# A 64-hex value the stub reports in `/props`, and deliberately NOT the model.
# `/props` really does carry hashes of the chat template, the llama.cpp
# library and the tokenizer; a check that matched one would report the model
# as validated when only the template was.
PROPS_HASH = "c" * 64


class _StubBackend:
    """Enough backend to complete one run, with a switch for failing.

    Deliberately not a mock of the harness's own behaviour: the runtime is
    real, and this only stands in for the model.
    """

    thinking = False

    #: Prose the stub attaches to an ANALYSIS tool call. Real models often
    #: emit both, and a step whose prose is dropped is the defect this
    #: recorder exists to catch -- so at least one step must carry some.
    STEP_PROSE = "checking whether pickle is reachable from the entrypoint"

    def __init__(self, fail=None, plan_questions=None, prose="a finding",
                 step_prose=None, fail_at=None):
        self.fail = fail
        #: Fail only once a phase is reached, rather than on the first call.
        #: `"report"` is the closing report -- a toolless call made after the
        #: loop, and the interrupt an analyst is likeliest to send, the
        #: report being the longest single generation in a run.
        #: `"action"` is an analysis call inside the loop.
        self.fail_at = fail_at
        self.plan_questions = list(plan_questions or [])
        self.prose = prose
        self.step_prose = self.STEP_PROSE if step_prose is None else step_prose
        self.calls = 0

    def health(self):
        return True

    def backend_props(self):
        return {"model_compatibility": {"template_hash": PROPS_HASH}}

    def supports_exact_context_admission(self):
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(tokens=int(40 + chars * 0.25), context_tokens=CTX,
                          rendered_hash="a" * 64, token_hash="b" * 64)

    def chat_stream(self, messages, **kwargs):
        offered = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        if self.fail is not None:
            if self.fail_at is None:
                raise self.fail
            if self.fail_at == "report" and not offered and self.calls >= 2:
                raise self.fail
            if self.fail_at == "action" and ANALYSIS_TOOL_NAME in offered:
                raise self.fail
        self.calls += 1
        calls = []
        content = ""
        if PLAN_TOOL_NAME in offered:
            calls = [self._call(PLAN_TOOL_NAME,
                                {"questions": self.plan_questions})]
        elif FINISH_TOOL_NAME in offered:
            calls = [self._call(FINISH_TOOL_NAME,
                                {"status": "resolved",
                                 "answer_summary": "settled"})]
        elif ANALYSIS_TOOL_NAME in offered:
            calls = [self._call(ANALYSIS_TOOL_NAME,
                                {"code": f"print({self.calls})"})]
            content = self.step_prose
        else:
            content = self.prose
        return ChatResult(
            content=content, model="m", finish_reason="stop",
            tool_calls=calls, prompt_tokens=1, completion_tokens=1,
            cached_tokens=0, prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)

    def _call(self, name, arguments):
        return [{"id": f"c{self.calls}", "type": "function",
                 "function": {"name": name,
                              "arguments": json.dumps(arguments)}}][0]


class _RejectedChildBackend(_StubBackend):
    """Proposes a child question the controller refuses.

    The controller rejects a child that is not grounded in causal evidence;
    that refusal is what `rejected_children` counts, and it is the only way
    to make this field non-zero on a scripted run.
    """

    def chat_stream(self, messages, **kwargs):
        offered = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        if FINISH_TOOL_NAME in offered:
            self.calls += 1
            return ChatResult(
                content="", model="m", finish_reason="stop",
                tool_calls=[self._call(FINISH_TOOL_NAME, {
                    "status": "still_open",
                    "answer_summary": "needs more",
                    # A child that restates its own parent: the controller
                    # refuses it, which is what `rejected_children` counts.
                    "child_question": {
                        "question": self.plan_questions[0]["question"],
                        "missing_fact": "needs a run",
                        "caused_by_evidence_id": "nonexistent",
                    },
                })],
                prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            )
        return super().chat_stream(messages, **kwargs)


class _UnusablePlanBackend(_StubBackend):
    """Answers the first plan call with something unusable, then complies.

    One repair, so `plan_calls` is 2 rather than the 1 every other scenario
    produces.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plans = 0

    def chat_stream(self, messages, **kwargs):
        offered = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        if PLAN_TOOL_NAME in offered:
            self.plans += 1
            self.calls += 1
            if self.plans == 1:
                return ChatResult(
                    content="", model="m", finish_reason="stop",
                    tool_calls=[], prompt_tokens=1, completion_tokens=1,
                    cached_tokens=0, prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )
            self.calls -= 1
        return super().chat_stream(messages, **kwargs)


class _BlockedQuestionBackend(_StubBackend):
    """Reports its question still open, so the run ends with it unresolved."""

    def chat_stream(self, messages, **kwargs):
        offered = [t["function"]["name"] for t in (kwargs.get("tools") or [])]
        if FINISH_TOOL_NAME in offered:
            self.calls += 1
            return ChatResult(
                content="", model="m", finish_reason="stop",
                tool_calls=[self._call(FINISH_TOOL_NAME, {
                    "status": "still_open",
                    "answer_summary": "the evidence did not settle it",
                })],
                prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            )
        return super().chat_stream(messages, **kwargs)


class _StubRun:
    """A finished, uncancelled run holding one report text.

    `_exit_code` is exercised directly here: several of the shapes below
    need an evidence set or a context refusal that no scripted run can
    produce, and the point is completeness across the runtime's messages
    rather than the paths that reach them.
    """

    cancelled = False
    stop_reason = "no open question requires an action"

    def __init__(self, text):
        class _Report:
            pass
        self.final_report = _Report()
        self.final_report.text = text


def module_error(message):
    return RecoverableBackendError(message)


def _run_main(operator_sha="a" * 64, backend=None, artifact_text=None,
              request=None, base_url="http://stub", artifact_bytes=None):
    """Drive the real `main()` and return what it actually did.

    Returns the parsed output JSON with two externally-observed facts added
    under keys that cannot collide with the harness's own: the process exit
    code, and whether the output file exists at all.
    """
    if backend is None:
        backend = _StubBackend()
    with tempfile.TemporaryDirectory(prefix="orbit-harness-test-") as tmp:
        root = pathlib.Path(tmp)
        # The artifact lives alone, so anything appearing beside it was put
        # there by the harness -- the containment check depends on it.
        sample_dir = root / "sample"
        sample_dir.mkdir()
        # An independent witness for the workspace. `mkdtemp` honours
        # TMPDIR, so pointing it here means this function OBSERVES where the
        # workspace was created rather than believing the path the record
        # reports -- a value the code under test produces, and which a
        # falsified path defeats.
        workspaces = root / "workspaces"
        workspaces.mkdir()
        artifact = sample_dir / "sample.py"
        if artifact_bytes is not None:
            artifact.write_bytes(artifact_bytes)
        else:
            artifact.write_text(
                "import os\nimport pickle\nprint(os.name)\n"
                if artifact_text is None else artifact_text
            )
        out = root / "nested" / "record.json"
        argv = ["live_validate_analysis.py", "--base-url", base_url,
                "--artifact", str(artifact), "--output", str(out)]
        if request is not None:
            argv += ["--request", request]
        if operator_sha is not None:
            argv += ["--operator-model-sha", operator_sha]

        previous_tempdir = tempfile.tempdir
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.dict(os.environ, {"TMPDIR": str(workspaces)},
                             clear=False), \
             mock.patch.object(harness, "LlamaServerBackend",
                               lambda **kwargs: backend):
            # `tempfile` caches the directory, so the env patch alone would
            # not take effect.
            tempfile.tempdir = None
            try:
                exit_code = harness.main()
            finally:
                tempfile.tempdir = previous_tempdir

        observed = {
            "_exit_code": exit_code,
            "_output_exists": out.exists(),
            "_artifact_bytes_after": artifact.read_bytes(),
            # What was actually left behind, seen from outside.
            # Everything left in the temp root, not just what matches the
            # workspace prefix. A prefix glob verifies "that name was cleaned
            # up", which a rename or a differently-named directory defeats;
            # this verifies the root is as empty as it started.
            "_leaked_workspaces": sorted(q.name for q in workspaces.iterdir()),
            # Where workspaces were allowed to be created. The record's own
            # `workspace_root` must name a path under here, which is how a
            # falsified value is caught.
            "_workspace_parent": str(workspaces),
            "_files_beside_artifact": sorted(
                q.name for q in artifact.parent.iterdir()
            ),
        }
        if out.exists():
            observed["_record"] = json.loads(out.read_text())
        return observed


class SuccessCaseTests(unittest.TestCase):
    """§3. One complete run, judged only by what it left behind."""

    def setUp(self) -> None:
        self.observed = _run_main()
        self.assertTrue(self.observed["_output_exists"],
                        "main() wrote no output file")
        self.record = self.observed["_record"]

    def test_the_run_succeeds_and_writes_a_parsable_record(self) -> None:
        self.assertEqual(self.observed["_exit_code"], 0)
        self.assertIsNone(self.record["error"])
        self.assertIsInstance(self.record, dict)

    def test_the_controller_actually_engaged(self) -> None:
        """Otherwise every other assertion here holds for an empty run."""
        self.assertGreater(self.record["model_calls"], 0)
        self.assertEqual(self.record["plan_calls"], 1)
        self.assertEqual(self.record["cover_calls"], 1)

    def test_the_report_comes_from_the_authoritative_field(self) -> None:
        self.assertTrue(self.record["final_report_present"])
        self.assertNotEqual(self.record["final_report"], "")

    def test_step_text_is_recorded_separately_from_the_report(self) -> None:
        self.assertIn("last_step_text_len", self.record)
        self.assertIn("final_report", self.record)
        self.assertIsInstance(self.record["last_step_text_len"], int)

    def test_the_reported_workspace_path_is_the_real_one(self) -> None:
        """The record names where the workspace actually was.

        Cleanup is witnessed by counting leftovers, so a falsified path no
        longer breaks that check -- but the path is a diagnostic an operator
        reads, and one pointing somewhere the workspace never lived would
        send them looking in the wrong place.
        """
        reported = pathlib.Path(self.record["workspace_root"])
        self.assertEqual(str(reported.parent),
                         self.observed["_workspace_parent"])
        self.assertTrue(reported.name.startswith("orbit-analysis-session-"))

    def test_the_workspace_is_gone(self) -> None:
        """Observed where the workspace was created, not read from the record.

        The record's `workspace_root` is written by the code under test, so a
        falsified path passes an existence check trivially. This counts what
        is actually left in the temp root this test owns.
        """
        self.assertEqual(self.observed["_leaked_workspaces"], [],
                         "the analysis workspace was left on disk")

    def test_the_artifact_is_not_mutated(self) -> None:
        self.assertEqual(
            self.observed["_artifact_bytes_after"],
            b"import os\nimport pickle\nprint(os.name)\n",
        )


class ActionLastCaseTests(unittest.TestCase):
    """§4. The shape that caused the original misreading.

    A run whose last step executed an action carries no assistant prose on
    that step. Reading the step instead of the report calls such a run a
    failure; inferring presence from spent model calls passes runs that have
    no report at all.
    """

    def setUp(self) -> None:
        # No prose on the action step: that is the shape under test, and it
        # is the normal shape of a native tool call. Asked for explicitly so
        # the emptiness is a choice rather than an accident of the stub.
        backend = _StubBackend(step_prose="", plan_questions=[
            {"question": "is pickle reachable", "missing_fact": "needs a run"},
        ])
        self.observed = _run_main(backend=backend)
        self.record = self.observed["_record"]

    def test_the_last_step_carries_no_prose(self) -> None:
        """The premise. Without it the rest proves nothing."""
        self.assertGreater(len(self.record["steps"]), 0)
        self.assertTrue(self.record["steps"][-1]["action_executed"])
        self.assertEqual(self.record["last_step_text_len"], 0)

    def test_the_full_report_is_still_emitted(self) -> None:
        self.assertTrue(self.record["final_report_present"])
        self.assertNotEqual(self.record["final_report"], "")
        self.assertGreater(len(self.record["final_report"]), 0)


class ReportPresenceIsNotInferredTests(unittest.TestCase):
    """Calls spent are not a report produced.

    Every other case here has a report AND spent calls, so a recorder that
    reported `model_calls > 0` would agree with the truth throughout and the
    suite would never notice. The discriminating shape is a run whose closing
    report fails: the calls were made, the report does not exist, and saying
    it does would tell an operator the analysis concluded when it did not.
    """

    def setUp(self) -> None:
        backend = _StubBackend()
        served = backend.chat_stream

        def fail_the_closing_report(messages, **kwargs):
            # The closing report is a toolless call made after the loop.
            # A *recoverable* failure, which the runtime contains: the run
            # completes and simply has no report, which is the shape that
            # separates "calls were spent" from "a report exists".
            if not kwargs.get("tools") and backend.calls >= 2:
                raise RecoverableBackendError("the report could not be read")
            return served(messages, **kwargs)

        backend.chat_stream = fail_the_closing_report
        self.record = _run_main(backend=backend)["_record"]

    def test_the_premise_holds(self) -> None:
        """Calls really were spent -- otherwise this proves nothing."""
        self.assertGreater(self.record["model_calls"], 0)

    def test_no_report_is_claimed(self) -> None:
        self.assertIs(self.record["final_report_present"], False)
        self.assertEqual(self.record["final_report"], "")
        self.assertEqual(self.record["final_report_evidence_ids"], [])


class StepProseIsRecordedTests(unittest.TestCase):
    """Prose that exists must be counted, not silently reported as zero.

    Every other case here produces steps with no assistant prose, so a
    recorder that hardcoded 0 -- or read a field that does not exist without
    a `getattr` default to hide it -- agreed with the truth throughout. That
    is exactly how the shipped `step.text` bug survived: it reported no prose
    for steps that had hundreds of characters, and nothing disagreed.
    """

    def setUp(self) -> None:
        backend = _StubBackend(plan_questions=[
            {"question": "is pickle reachable", "missing_fact": "needs a run"},
        ])
        self.expected = backend.STEP_PROSE
        self.record = _run_main(backend=backend)["_record"]

    def test_a_step_that_carried_prose_reports_its_length(self) -> None:
        self.assertGreater(len(self.record["steps"]), 0)
        self.assertEqual(
            [s["text_len"] for s in self.record["steps"]],
            [len(self.expected)],
        )

    def test_the_last_step_length_is_the_prose_it_actually_held(self) -> None:
        self.assertEqual(self.record["last_step_text_len"],
                         len(self.expected))
        self.assertGreater(self.record["last_step_text_len"], 0)

    def test_the_step_row_carries_its_evidence_id(self) -> None:
        """A real run produces real ids; asserting only emptiness proves little."""
        ids = [s["evidence_id"] for s in self.record["steps"]]
        self.assertTrue(all(i and i.startswith("ev_") for i in ids), ids)


class FailureCaseTests(unittest.TestCase):
    """§5. A real backend failure, and what the record must say about it."""

    def setUp(self) -> None:
        backend = _StubBackend(fail=RuntimeError("backend exploded"))
        self.observed = _run_main(backend=backend)
        self.record = self.observed["_record"]

    def test_the_failure_is_reported_by_the_exit_code(self) -> None:
        self.assertNotEqual(self.observed["_exit_code"], 0)

    def test_the_cause_is_recorded_rather_than_discarded(self) -> None:
        self.assertIsNotNone(self.record["error"])
        self.assertIn("backend exploded", self.record["error"])

    def test_the_record_is_still_written(self) -> None:
        """A failed run is exactly when the operator needs the record."""
        self.assertTrue(self.observed["_output_exists"])

    def test_report_keys_are_present_and_negative(self) -> None:
        """Absent keys would make a reader's `[...]` raise on the failure path."""
        self.assertIs(self.record["final_report_present"], False)
        self.assertEqual(self.record["final_report"], "")
        self.assertEqual(self.record["last_step_text_len"], 0)

    def test_the_workspace_is_gone_on_the_failure_path_too(self) -> None:
        self.assertEqual(self.observed["_leaked_workspaces"], [])


class CancelledRunTests(unittest.TestCase):
    """A cancelled run must not read as a clean pass.

    This is the case an exception-shaped test cannot reach: the runtime
    CONTAINS `KeyboardInterrupt`, ending the run and reporting `cancelled`
    instead of raising, so nothing arrives at the error handler and `error`
    stays None. An operator who Ctrl-Cs a long run on a real sample, or a CI
    gate reading `$?`, would otherwise be told a half-second abort succeeded.
    """

    def setUp(self) -> None:
        backend = _StubBackend(fail=KeyboardInterrupt())
        self.observed = _run_main(backend=backend)
        self.record = self.observed["_record"]

    def test_the_premise_holds(self) -> None:
        """The runtime contained it: no exception, but a cancelled run."""
        self.assertIs(self.record["cancelled"], True)
        self.assertIsNone(self.record["error"])

    def test_the_exit_code_reports_the_cancellation(self) -> None:
        self.assertNotEqual(self.observed["_exit_code"], 0)

    def test_the_record_is_written_and_says_what_happened(self) -> None:
        self.assertTrue(self.observed["_output_exists"])
        self.assertEqual(self.record["stop_reason"], "cancelled")
        self.assertIs(self.record["final_report_present"], False)

    def test_the_workspace_is_still_cleaned_up(self) -> None:
        self.assertEqual(self.observed["_leaked_workspaces"], [])


class ExitCodeTests(unittest.TestCase):
    """`$?` is the gate, so it must mean the run answered its question.

    Three ways of not answering it all looked like success, and each was
    reached by a path no test drove. Gating on `cancelled` alone fixed only
    the interrupt shape that the test happened to exercise -- the runtime
    does not set `cancelled` for an interrupt during the closing report, and
    that is the likeliest interrupt of all.
    """

    _PLAN = [{"question": "is pickle reachable", "missing_fact": "needs a run"}]

    def _run(self, **kwargs):
        return _run_main(backend=_StubBackend(plan_questions=self._PLAN,
                                              **kwargs))

    def test_a_finished_run_with_a_report_succeeds(self) -> None:
        observed = self._run()
        self.assertEqual(observed["_exit_code"], 0)
        self.assertIs(observed["_record"]["final_report_present"], True)

    def test_a_cancelled_run_fails_even_if_it_carries_a_report(self) -> None:
        """The coupling that makes the `cancelled` term worth keeping.

        A cancelled run never carries a report today, because
        `run_autonomous` guards its closing report with `not cancelled`.
        That is a property of another module, and nothing in this harness
        pins it -- relaxing that guard there makes a Ctrl-C run produce a
        report, satisfy the report test, and exit 0 again. So the exit code
        is asserted directly against a cancelled run holding a report,
        rather than against the runtime's current behaviour.
        """
        class _Report:
            text = "findings"

        class _Cancelled:
            cancelled = True
            final_report = _Report()
            stop_reason = "cancelled"

        self.assertEqual(harness._exit_code(None, _Cancelled()), 1)
        # And the healthy shape still succeeds, so this is not a blanket 1.
        class _Finished:
            cancelled = False
            final_report = _Report()
            stop_reason = "no open question requires an action"

        self.assertEqual(harness._exit_code(None, _Finished()), 0)

    def test_a_cancelled_run_fails_through_the_real_runtime(self) -> None:
        """The same property, driven end to end rather than on a stand-in.

        The unit assertion above fixes `_exit_code`; this one fixes the
        harness around it, so neither can drift into passing a cancelled run
        while the other looks correct.
        """
        observed = self._run(fail=KeyboardInterrupt(), fail_at="action")
        self.assertIs(observed["_record"]["cancelled"], True)
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_an_interrupt_during_the_closing_report_fails(self) -> None:
        """The runtime leaves `cancelled` False here and only drops the report."""
        observed = self._run(fail=KeyboardInterrupt(), fail_at="report")
        self.assertIs(observed["_record"]["cancelled"], False)
        self.assertIs(observed["_record"]["final_report_present"], False)
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_a_recoverable_report_failure_fails(self) -> None:
        observed = self._run(fail=module_error("the report could not be read"),
                             fail_at="report")
        self.assertIs(observed["_record"]["final_report_present"], False)
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_a_report_with_no_usable_text_fails(self) -> None:
        """A report whose entire content says it has none is not an answer.

        When the closing generation returns empty prose the runtime writes a
        placeholder rather than nothing, so `final_report_present` is true
        and the gate passed on a run that concluded nothing.
        """
        observed = _run_main(backend=_StubBackend(
            plan_questions=self._PLAN, prose=""))
        record = observed["_record"]
        self.assertIs(record["final_report_present"], True)
        self.assertEqual(record["final_report"].strip(),
                         harness.NO_USABLE_REPORT_TEXT)
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_an_empty_report_fails_even_with_an_appendix(self) -> None:
        """The realistic shape, and the one an equality check missed.

        The runtime appends its deterministic appendix after the placeholder,
        so a report on any artifact carrying a URI or an address is longer
        than the placeholder itself. Comparing for equality therefore only
        ever matched artifacts with NO indicators -- the rare case -- and let
        the common one through, which is every real sample this harness is
        pointed at.
        """
        observed = _run_main(
            backend=_StubBackend(plan_questions=self._PLAN, prose=""),
            artifact_text="url = 'http://185.234.72.19/panel/gate.php'\n",
        )
        record = observed["_record"]
        # The appendix really is there: the premise of the test.
        self.assertIn("185.234.72.19", record["final_report"])
        self.assertGreater(len(record["final_report"]),
                           len(harness.NO_USABLE_REPORT_TEXT))
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_a_run_that_collected_no_evidence_fails(self) -> None:
        """The second non-answer report, and the one real samples produce.

        A model that answers with prose and never calls a tool leaves the run
        with no evidence at all, and the runtime says exactly that. Gating on
        one placeholder caught the empty-generation case and let this one
        through -- on an artifact too large to cover, which is what a real
        obfuscated dropper is.
        """
        class _ProseOnly(_StubBackend):
            def chat_stream(self, messages, **kwargs):
                self.calls += 1
                return ChatResult(
                    content="I cannot analyse this.", model="m",
                    finish_reason="stop", tool_calls=[], prompt_tokens=1,
                    completion_tokens=1, cached_tokens=0,
                    prompt_tokens_per_second=None,
                    generation_tokens_per_second=None,
                )

        observed = _run_main(
            backend=_ProseOnly(),
            # Large enough to refuse coverage, and carrying an indicator so
            # the appendix is appended after the message -- the shape that
            # defeated an equality test.
            artifact_text=("// padding\n" * 900
                           + "u = 'http://185.234.72.19/gate.php'\n"),
        )
        record = observed["_record"]
        self.assertIs(record["source_covered"], False)
        self.assertIs(record["final_report_present"], True)
        self.assertTrue(record["final_report"].lstrip().startswith(
            harness.NO_EVIDENCE_REPORT))
        self.assertNotEqual(observed["_exit_code"], 0)

    def test_a_report_that_could_not_be_composed_fails(self) -> None:
        """The third shape: the run worked, the report would not fit.

        Asserted on `_exit_code` directly, because reaching it end to end
        needs an evidence set larger than the report context, which no
        scripted run produces.
        """
        class _Uncomposable:
            cancelled = False
            stop_reason = "no open question requires an action"

            class final_report:
                text = (
                    "The report could not be composed: the collected "
                    "evidence no longer fits the context window."
                )

        self.assertEqual(harness._exit_code(None, _Uncomposable()), 1)

    def test_every_non_answer_report_the_runtime_can_emit_fails(self) -> None:
        """One table, because this axis has been wrong twice.

        The first attempt matched one message by equality, which the appendix
        defeated. The second matched one message by prefix, and missed the
        two others the runtime can emit. So the shapes are enumerated from
        the runtime's five `AnalysisReport` construction sites rather than
        from whichever one a test happened to reach, and each is checked bare
        and with an appendix -- the appendix being what made the common case
        the one that slipped through.
        """
        appendix = "\n\n## Verified indicators\n\n- uri: http://x.invalid/a"
        shapes = {
            "no evidence collected": harness.NO_EVIDENCE_REPORT,
            "empty generation": harness.NO_USABLE_REPORT_TEXT,
            "collected evidence will not fit": (
                "The report could not be composed: the collected evidence "
                "no longer fits the context window."
            ),
            "covered source will not fit": (
                "The report could not be composed: the covered source no "
                "longer fits the context window."
            ),
        }
        for label, text in shapes.items():
            for tag, body in (("bare", text), ("with appendix", text + appendix)):
                with self.subTest(shape=label, form=tag):
                    self.assertEqual(
                        harness._exit_code(None, _StubRun(body)), 1,
                        f"{label} ({tag}) reported success",
                    )

    def test_a_genuine_report_is_never_wrongly_failed(self) -> None:
        """The other direction: a false failure blocks a valid validation."""
        genuine = {
            "ordinary finding":
                "The sample is a dropper that contacts a remote host.",
            "quotes the placeholder":
                f"An earlier run logged '{harness.NO_USABLE_REPORT_TEXT}'; "
                "that is not the case here.",
            "mentions composition":
                "Composition of the payload could not be determined, but "
                "the URL is unambiguous.",
            "begins with No":
                "No network activity was observed; the sample is inert.",
        }
        for label, text in genuine.items():
            with self.subTest(report=label):
                self.assertEqual(harness._exit_code(None, _StubRun(text)), 0,
                                 f"{label} was wrongly failed")

    def test_a_real_report_quoting_the_placeholder_still_passes(self) -> None:
        """`in` would be wrong: a genuine report may mention the phrase."""
        prose = (
            "The sample is a dropper. Note that an earlier run recorded "
            f"'{harness.NO_USABLE_REPORT_TEXT}', which is not the case here."
        )
        observed = _run_main(backend=_StubBackend(
            plan_questions=self._PLAN, prose=prose))
        self.assertIn(harness.NO_USABLE_REPORT_TEXT,
                      observed["_record"]["final_report"])
        self.assertEqual(observed["_exit_code"], 0)

    def test_a_backend_outage_mid_run_fails(self) -> None:
        """A report written from partial evidence is not a finished analysis."""
        observed = self._run(fail=module_error("upstream is down"),
                             fail_at="action")
        record = observed["_record"]
        self.assertTrue(record["stop_reason"].startswith("backend error"))
        # The report exists, which is exactly why presence alone is not enough.
        self.assertIs(record["final_report_present"], True)
        self.assertNotEqual(observed["_exit_code"], 0)


class WorkspaceCleanupTests(unittest.TestCase):
    """§6. Cleanup witnessed from outside, across repeated runs."""

    def test_no_analysis_workspace_survives_any_path(self) -> None:
        """Counted in a temp root this test owns, under the real TMPDIR.

        Two earlier versions of this witness were defeated. Globbing a
        hardcoded `/tmp` is blind whenever `TMPDIR` points elsewhere -- and
        it does in this project's own environment notes -- so a run could
        leak 28 workspaces with the suite green. Reading the path out of the
        emitted record is no backstop either: that value is produced by the
        code under test, so falsifying it defeats the check.

        So the workspace root is whatever `tempfile` is actually using, and
        this test owns an empty one, which also removes the order-dependence
        of counting a directory shared with every other suite.
        """
        previous_tempdir = tempfile.tempdir
        with tempfile.TemporaryDirectory(prefix="orbit-cleanup-witness-") as own:
            with mock.patch.dict(os.environ, {"TMPDIR": own}, clear=False):
                # `tempfile` caches the temp dir, so the patch alone is not
                # enough -- clear the cache so `mkdtemp` really lands here.
                tempfile.tempdir = None
                try:
                    self.assertEqual(tempfile.gettempdir(), own)
                    for backend in (
                        _StubBackend(),
                        _StubBackend(fail=RuntimeError("boom")),
                        _StubBackend(plan_questions=[
                            {"question": "q", "missing_fact": "m"}]),
                    ):
                        _run_main(backend=backend)
                    # Everything left behind, not just what matches the
                    # workspace prefix: a differently-named leftover is still
                    # a leak.
                    leaked = sorted(q.name for q in pathlib.Path(own).iterdir())
                finally:
                    # Restored, not blanked. Setting `None` happens to be
                    # right only because that is the default; a runner with
                    # its own `tempfile.tempdir` would have it silently
                    # cleared by this test while `_run_main` preserves it.
                    tempfile.tempdir = previous_tempdir
        self.assertEqual(
            leaked, [],
            f"the harness leaked {len(leaked)} analysis workspace(s)",
        )


class OperatorShaContractTests(unittest.TestCase):
    """§7. Metadata, recorded verbatim, never checked and never a gate."""

    def test_the_value_is_recorded_verbatim(self) -> None:
        for supplied in ("a" * 64, "not-a-hash", ""):
            with self.subTest(sha=supplied):
                record = _run_main(operator_sha=supplied)["_record"]
                self.assertEqual(record["operator_model_sha256"], supplied)

    def test_verification_is_literally_false_in_the_json(self) -> None:
        """Read out of the emitted text, so a truthy-but-not-True value fails."""
        observed = _run_main()
        raw = json.dumps(observed["_record"])
        self.assertIn('"operator_model_sha256_verified": false', raw)
        self.assertIs(
            observed["_record"]["operator_model_sha256_verified"], False
        )

    def test_an_absent_sha_is_still_reported_unverified(self) -> None:
        record = _run_main(operator_sha=None)["_record"]
        self.assertIsNone(record["operator_model_sha256"])
        self.assertIs(record["operator_model_sha256_verified"], False)

    def test_no_sha_value_gates_the_run(self) -> None:
        """However a gate might be phrased, the outcome must not change.

        `PROPS_HASH` is the interesting one: it IS a value the props payload
        carries, so any check comparing against props would treat it as a
        match and every other value as a mismatch.
        """
        baseline = _run_main(operator_sha=None)
        for supplied in ("a" * 64, PROPS_HASH, "not-a-hash", ""):
            with self.subTest(sha=supplied):
                observed = _run_main(operator_sha=supplied)
                self.assertEqual(observed["_exit_code"],
                                 baseline["_exit_code"])
                self.assertEqual(observed["_record"]["stop_reason"],
                                 baseline["_record"]["stop_reason"])
                self.assertIs(
                    observed["_record"]["operator_model_sha256_verified"],
                    False,
                )


class UnhealthyServerTests(unittest.TestCase):
    """A server that is not ready must stop the run, not be analysed anyway."""

    def test_an_unhealthy_server_stops_before_any_work(self) -> None:
        class _Unhealthy(_StubBackend):
            def health(self):
                return False

        backend = _Unhealthy()
        with self.assertRaises(SystemExit) as caught:
            _run_main(backend=backend)
        # It says which server, so the operator knows what to fix.
        self.assertIn("not healthy", str(caught.exception))
        self.assertIn("http://stub", str(caught.exception))
        # And nothing was asked of the model.
        self.assertEqual(backend.calls, 0)


class RecordedFieldsTests(unittest.TestCase):
    """The record's own numbers, asserted against the run that produced them.

    A recorder whose fields are constants is indistinguishable from one that
    works, unless something checks them. Most of this record was unasserted:
    `rejected_free_actions` in particular is documented in the harness as
    "the single number that says whether the old loop came back", and could
    be hardcoded to 0 without any test noticing -- which would silently
    answer the one question a no-COVER validation is run to ask.
    """

    def setUp(self) -> None:
        self.backend = _StubBackend(plan_questions=[
            {"question": "is pickle reachable", "missing_fact": "needs a run"},
        ])
        self.record = _run_main(backend=self.backend)["_record"]

    def test_model_calls_match_what_the_backend_served(self) -> None:
        """The backend is the independent witness; the record is the claim."""
        self.assertEqual(self.record["model_calls"], self.backend.calls)
        self.assertGreater(self.backend.calls, 0)

    def test_counts_track_the_run_rather_than_being_constants(self) -> None:
        """Driven to values other than 1, which a hardcoded read matches.

        Every scenario here otherwise produces exactly one cover call, one
        plan call and one action, so a constant is indistinguishable from a
        read -- the same flaw fixed for the rejection count, not extended to
        its neighbours.
        """
        backend = _StubBackend(plan_questions=[
            {"question": f"question {i}", "missing_fact": "needs a run"}
            for i in range(3)
        ])
        record = _run_main(backend=backend)["_record"]
        self.assertEqual(record["initial_questions"], 3)
        self.assertEqual(record["actions_executed"], 3)
        # `model_calls` is 5 in the one scenario that asserts it, so a
        # constant matches. A three-question run spends more.
        self.assertEqual(record["model_calls"], backend.calls)
        self.assertGreater(record["model_calls"], 5)
        # `plan_calls` is 1 on every scenario, so a constant matches it. An
        # unusable plan earns exactly one repair, which is the only shape
        # that tells a read from a hardcoded 1.
        repaired = _run_main(backend=_UnusablePlanBackend())["_record"]
        self.assertEqual(repaired["plan_calls"], 2)
        self.assertEqual(len(record["steps"]), 3)
        self.assertEqual(len(record["progress"]), 3)
        self.assertEqual(record["resolved_questions"], ["Q1", "Q2", "Q3"])

    def test_an_uncovered_artifact_reports_no_cover_call(self) -> None:
        """`source_covered` and `cover_calls` are read, not assumed true."""
        # Bytes that do not decode as text cannot be covered.
        record = _run_main(artifact_bytes=b"\x00\xff\xfebinary")["_record"]
        self.assertIs(record["source_covered"], False)
        self.assertEqual(record["cover_calls"], 0)

    def test_questions_left_open_are_reported(self) -> None:
        """`open_questions` is `[]` everywhere else, so a constant matches."""
        record = _run_main(backend=_BlockedQuestionBackend(plan_questions=[
            {"question": "is pickle reachable", "missing_fact": "needs a run"},
        ]))["_record"]
        self.assertEqual(record["open_questions"], ["Q1"])
        self.assertEqual(record["resolved_questions"], [])

    def test_the_controller_shape_is_recorded_truthfully(self) -> None:
        self.assertTrue(self.record["source_covered"])
        self.assertEqual(self.record["cover_calls"], 1)
        self.assertEqual(self.record["plan_calls"], 1)
        self.assertEqual(self.record["initial_questions"], 1)
        self.assertEqual(self.record["resolved_questions"], ["Q1"])
        self.assertEqual(self.record["open_questions"], [])
        self.assertEqual(self.record["actions_executed"], 1)
        self.assertEqual(self.record["progress"], ["NEW_CONTENT"])

    def test_the_rejection_count_is_read_not_assumed(self) -> None:
        """Driven to a non-zero value, because 0 proves nothing.

        The runtime fills this from the controller's `rejected_children`, so
        a run that proposes an invalid child question produces a non-zero
        count. Asserting only that a clean run reports 0 cannot tell a
        working recorder from a hardcoded constant.
        """
        backend = _RejectedChildBackend(plan_questions=[
            {"question": "is pickle reachable", "missing_fact": "needs a run"},
        ])
        record = _run_main(backend=backend)["_record"]
        self.assertGreater(
            record["rejected_child_questions"], 0,
            "a rejected child question must be reported, not assumed absent",
        )

    def test_a_clean_run_rejects_nothing(self) -> None:
        self.assertEqual(self.record["rejected_child_questions"], 0)
        self.assertEqual(self.record["actions_executed"], 1)

    def test_the_stop_reason_and_outcome_are_recorded(self) -> None:
        self.assertEqual(self.record["stop_reason"],
                         "no open question requires an action")
        self.assertIs(self.record["cancelled"], False)

    def test_the_report_cites_the_evidence_the_run_produced(self) -> None:
        ids = self.record["final_report_evidence_ids"]
        self.assertTrue(ids, "a run with evidence must cite it")
        self.assertTrue(all(i.startswith("ev_") for i in ids), ids)

    def test_the_inputs_are_echoed_for_the_reader(self) -> None:
        """Driven with non-default values, so a hardcoded echo cannot pass."""
        text = "# a distinctive artifact\nprint('x')\n"
        record = _run_main(
            request="Look for network egress.",
            base_url="http://elsewhere:9999",
            artifact_text=text,
        )["_record"]
        self.assertEqual(record["base_url"], "http://elsewhere:9999")
        self.assertEqual(record["request"], "Look for network egress.")
        self.assertEqual(record["artifact_bytes"], len(text.encode()))
        self.assertEqual(record["artifact_sha256"],
                         hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(record["backend_props"],
                         _StubBackend().backend_props())


class ContainmentTests(unittest.TestCase):
    """Nothing is written beside the artifact.

    The harness is pointed at real malicious samples, so where it writes
    matters as much as what it reads. Checking only that the artifact's own
    bytes are unchanged misses a snapshot dropped into the sample's
    directory: the file survives untouched while the directory does not.
    """

    def test_no_file_appears_beside_the_artifact(self) -> None:
        observed = _run_main()
        self.assertEqual(observed["_files_beside_artifact"], ["sample.py"])

    def test_nothing_is_written_beside_the_artifact_on_failure(self) -> None:
        observed = _run_main(backend=_StubBackend(fail=RuntimeError("boom")))
        self.assertEqual(observed["_files_beside_artifact"], ["sample.py"])


class ComposedReportTextTests(unittest.TestCase):
    """The report the analyst reads must be well formed, not merely matched.

    Extracting the opening clause to a constant moved the separating space
    inside the remaining fragment, where nothing guards it. Deleting it makes
    production emit "The report could not be composed:the collected
    evidence..." -- and the whole analysis suite stays green, because the gate
    is prefix-matched and no test reads the sentence. That is malformed prose
    in the product, invisible to the tests that cover the product.
    """

    def _composed(self, kind):
        """The real text, from the branch that actually builds it.

        An earlier version regex-matched the module source and `eval`ed the
        fragment. That read the LAYOUT rather than the behaviour: reflowing
        the string across different line boundaries -- identical output --
        made it fail. Both branches are reached by refusing admission while
        an appendix exists, which is the condition the runtime itself tests,
        so a reflow is invisible here and a change to the sentence is not.
        """
        from orbit.runtime import analysis_runtime as runtime

        method = {"collected": "report",
                  "covered": "_report_from_coverage"}[kind]
        captured = {}

        def refuse(self, *args, **kwargs):
            raise runtime.ContextAdmissionError("nothing fits")

        with tempfile.TemporaryDirectory(prefix="orbit-composed-") as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "sample.js"
            # An indicator, so the deterministic appendix is non-empty --
            # without one both branches re-raise instead of composing.
            artifact.write_text("var u = 'http://185.234.72.19/gate.php';\n")
            workspace = runtime.AnalysisWorkspace.create()
            try:
                snapshot = workspace.source_root / artifact.name
                snapshot.write_bytes(artifact.read_bytes())
                rt = runtime.AnalysisRuntime(
                    backend=_StubBackend(),
                    source=runtime.AnalysisSource(
                        snapshot_path=snapshot,
                        sha256="0" * 64,
                        size_bytes=snapshot.stat().st_size,
                        original_path=str(artifact),
                    ),
                    evidence_store=EvidenceStore(
                        root=workspace.root / "evidence"),
                    workspace=workspace,
                )
                with mock.patch.object(
                    runtime.AnalysisRuntime, "_admit", refuse
                ):
                    captured["text"] = getattr(rt, method)(
                        question="what does it do?"
                    ).text
            finally:
                workspace.close()
        return captured["text"]

    def test_the_prefix_ends_where_a_sentence_continues(self) -> None:
        """The join, asserted on the constant rather than on a source layout.

        Reaching the collected-evidence branch needs stored evidence plus a
        refused admission, which is more scaffolding than the property is
        worth. The property itself is simple: the prefix ends at the colon,
        so whatever follows must begin with a space, and every use site
        writes `f"{PREFIX} ..."`. Asserting the constant's own shape catches
        a prefix that grows or loses its colon; the covered-source test
        below catches the join itself, in the branch that really builds it.
        """
        from orbit.runtime import analysis_runtime as runtime

        prefix = runtime.REPORT_NOT_COMPOSED_PREFIX
        self.assertTrue(prefix.endswith(":"),
                        f"the prefix must end at the colon: {prefix!r}")
        self.assertFalse(prefix.endswith(" "),
                         "the space belongs to the fragment, not the prefix")

    def test_the_covered_source_report_reads_as_a_sentence(self) -> None:
        from orbit.runtime import analysis_runtime as runtime

        text = self._composed("covered")
        self.assertTrue(text.startswith(runtime.REPORT_NOT_COMPOSED_PREFIX))
        self.assertIn(": the covered source", text)
        rest = text[len(runtime.REPORT_NOT_COMPOSED_PREFIX):]
        self.assertTrue(rest.startswith(" "),
                        f"no space after the prefix: {text[:70]!r}")


class NonAnswerPrefixDriftTests(unittest.TestCase):
    """The prefixes must be the runtime's own strings, not copies of them.

    A copied literal drifts in silence. Rewording the opening clause in the
    runtime once left this suite, the report-visibility suite and the
    evidence-authority guard all green while both composed-failure reports
    began exiting 0 -- the exact false pass the gate exists to prevent.
    Importing removes that failure mode by construction; this test is what
    keeps the import from being replaced by a copy again.
    """

    def test_each_prefix_is_the_runtime_object_not_a_copy(self) -> None:
        from orbit.runtime import analysis_runtime as runtime

        for name in ("NO_USABLE_REPORT_TEXT", "REPORT_NOT_COMPOSED_PREFIX",
                     "NO_EVIDENCE_REPORT"):
            with self.subTest(constant=name):
                value = getattr(runtime, name)
                self.assertIn(
                    value, harness.NON_ANSWER_REPORT_PREFIXES,
                    f"{name} is not among the prefixes the gate matches",
                )

    def test_the_harness_hardcodes_none_of_them(self) -> None:
        """A re-copied literal would satisfy the test above; this catches it."""
        from orbit.runtime import analysis_runtime as runtime

        source = HARNESS_PATH.read_text()
        checked = 0
        for name in ("NO_USABLE_REPORT_TEXT", "REPORT_NOT_COMPOSED_PREFIX",
                     "NO_EVIDENCE_REPORT"):
            value = getattr(runtime, name)
            checked += 1
            for quote in ('"', "'"):
                self.assertNotIn(
                    f"{quote}{value}", source,
                    f"{name} is spelled out in the harness instead of "
                    f"imported (as a {quote}-quoted string)",
                )
        self.assertEqual(checked, 3, "the guard inspected nothing")

    def test_the_runtime_still_uses_them_where_it_reports_no_answer(self) -> None:
        """And the runtime has not gone back to inline strings either."""
        from orbit.runtime import analysis_runtime as runtime

        source = pathlib.Path(runtime.__file__).read_text()
        for name in ("NO_USABLE_REPORT_TEXT", "REPORT_NOT_COMPOSED_PREFIX",
                     "NO_EVIDENCE_REPORT"):
            with self.subTest(constant=name):
                value = getattr(runtime, name)
                # Counting COMPLETE quoted strings is not enough: the
                # realistic regression embeds the prefix at the START of a
                # longer literal, which leaves that count at one. So the
                # opening quote is matched without requiring the closing
                # one, and the constant's own definition is the single
                # occurrence allowed.
                # Either quote style, and the opening quote only: the
                # realistic regression embeds the prefix at the START of a
                # longer literal, which a complete-string count misses.
                occurrences = sum(source.count(f"{q}{value}")
                                  for q in ('"', "'"))
                self.assertEqual(
                    occurrences, 1,
                    f"{name} is spelled out somewhere instead of referenced",
                )


class SourceGuardTests(unittest.TestCase):
    """§9. Only what a run cannot show, and each proves it inspected something.

    A run cannot demonstrate the ABSENCE of a code path that a future edit
    might add. These guards do that, and each asserts it looked at a non-zero
    number of lines -- the previous suite's guard filtered lines by a
    predicate matching none, so its assertion never ran.
    """

    def setUp(self) -> None:
        self.source = HARNESS_PATH.read_text()
        self.lines = self.source.splitlines()

    def test_the_retired_verifying_flag_is_not_reintroduced(self) -> None:
        self.assertNotIn("expect_model_sha", self.source)
        self.assertNotIn("--expect-model-sha", self.source)

    def test_no_comparison_involves_the_operator_sha(self) -> None:
        """Parsed, not pattern-matched on line text.

        A line-based version of this guard skipped anything starting with a
        quote, meaning to skip docstrings -- and skipped every dict-literal
        key line with it, which is exactly where the value is written. An
        explicit `args.operator_model_sha == props.get(...)` inside the record
        passed the whole suite. Reading the syntax tree removes the question
        of how a line happens to be spelled.
        """
        tree = ast.parse(self.source)

        def mentions_sha(node) -> bool:
            return any(
                isinstance(n, ast.Attribute) and n.attr == "operator_model_sha"
                for n in ast.walk(node)
            )

        compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
        self.assertGreater(len(compares), 0,
                           "no comparisons found -- the guard sees nothing")
        offending = [
            ast.unparse(n) for n in compares if mentions_sha(n)
        ]
        self.assertEqual(offending, [],
                         "the operator SHA must never be compared")

        # Nor reached through a method call: `.startswith`, `.endswith`, `in`.
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        reached = [
            ast.unparse(n) for n in calls
            if mentions_sha(n) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"startswith", "endswith", "find", "index"}
        ]
        self.assertEqual(reached, [])

    def test_no_branch_depends_on_the_operator_sha(self) -> None:
        """It is metadata: nothing may take a different path because of it."""
        tree = ast.parse(self.source)
        branches = [n for n in ast.walk(tree)
                    if isinstance(n, (ast.If, ast.IfExp, ast.While))]
        self.assertGreater(len(branches), 0)
        offending = [
            ast.unparse(n.test) for n in branches
            if any(isinstance(x, ast.Attribute)
                   and x.attr == "operator_model_sha"
                   for x in ast.walk(n.test))
        ]
        self.assertEqual(offending, [],
                         "the operator SHA must never gate a branch")

    def test_the_props_blob_is_never_searched(self) -> None:
        inspected = [ln for ln in self.lines if "props" in ln]
        self.assertGreater(len(inspected), 0)
        self.assertNotIn("json.dumps(props)", self.source)

    def test_no_attribute_is_read_through_a_defaulting_getattr(self) -> None:
        """A default turns a renamed field into a plausible zero."""
        offenders = [
            ln.strip() for ln in self.lines
            if "getattr(" in ln and not ln.strip().startswith("#")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

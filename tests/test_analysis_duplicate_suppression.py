"""Autonomous analysis must not spend its action budget re-reading.

A qualified live run spent all nine of its actions printing the same 7706-byte
artifact in nine different formattings, and stopped on the action budget having
never executed the decoder it had already located. The evidence proves it
without reference to what the file meant: nine actions, seven distinct programs,
five distinct outputs, one unchanging source digest.

The rule these tests hold is deliberately narrow. Re-running one program over
one unchanged input against one unchanged workspace is the same experiment, and
an experiment already run cannot establish anything new -- so the runtime
answers it from the evidence it already holds instead of executing it again.
Everything else runs: a different program, a different range, the same program
after the workspace changed, any transformation. Correctness over cleverness,
because a false suppression silently deletes an observation the analyst needed.

Nothing here knows what an artifact means. There is no decoder detection and no
encoding knowledge; the runtime only declines to repeat itself and says what it
already has.
"""

from __future__ import annotations

import unittest

from tests.test_analysis_autonomous import (
    AutonomousTestBase,
    ScriptedBackend,
    emit,
    nondet,
    tool_response,
    write_artifact,
)
from tests.test_analysis_runtime import prose_response

from orbit.runtime.analysis_progress import (
    NEW_CONTENT,
    NO_PROGRESS,
    observation_fingerprint,
)


READ_X = "print(open('/workspace/input').read()[:200], end='')"
READ_Y = "print(open('/workspace/input').read()[200:400], end='')"
# A deterministic transformation: it derives new bytes rather than restating
# input, and it writes them where later steps can address them.
TRANSFORM = (
    "import pathlib\n"
    "raw = open('/workspace/input').read()\n"
    "out = ''.join(chr(ord(c) ^ 0x2a) for c in raw[:64])\n"
    "pathlib.Path('/workspace/work/decoded.txt').write_text(out)\n"
    "print('decoded', len(out), end='')"
)


class FingerprintTests(unittest.TestCase):
    """The identity is three hashes, and every one of them counts."""

    def test_identical_inputs_are_the_same_experiment(self) -> None:
        self.assertEqual(
            observation_fingerprint("code", "src", {"a": "1"}),
            observation_fingerprint("code", "src", {"a": "1"}),
        )

    def test_each_component_changes_the_identity(self) -> None:
        base = observation_fingerprint("code", "src", {"a": "1"})
        self.assertNotEqual(base, observation_fingerprint("other", "src", {"a": "1"}))
        self.assertNotEqual(base, observation_fingerprint("code", "other", {"a": "1"}))
        self.assertNotEqual(base, observation_fingerprint("code", "src", {"a": "2"}))
        self.assertNotEqual(base, observation_fingerprint("code", "src", {}))

    def test_workspace_ordering_does_not_change_the_identity(self) -> None:
        """Two files in a different dict order are the same workspace."""
        self.assertEqual(
            observation_fingerprint("c", "s", {"a": "1", "b": "2"}),
            observation_fingerprint("c", "s", {"b": "2", "a": "1"}),
        )


class SurrogateFilenameTests(unittest.TestCase):
    """A workspace filename the filesystem allows but UTF-8 cannot encode.

    Undecodable bytes in a scratch name reach Python as lone surrogates. The
    fingerprint hashes workspace state, so it must survive them: refusing to
    compute one would convert an unreadable filename into a failed analysis
    step, breaking a recovery path the runtime already handles.
    """

    def test_a_lone_surrogate_in_a_handle_does_not_raise(self) -> None:
        digest = observation_fingerprint("code", "src", {"bad\udcff.bin": "00"})
        self.assertEqual(len(digest), 64)

    def test_it_still_discriminates_across_surrogate_names(self) -> None:
        self.assertNotEqual(
            observation_fingerprint("c", "s", {"a\udcff": "1"}),
            observation_fingerprint("c", "s", {"a\udcfe": "1"}),
        )


class ExactDuplicateSuppressionTests(AutonomousTestBase):
    def test_the_same_read_runs_once_and_is_answered_thereafter(self) -> None:
        """Three identical requests, one execution, no duplicate evidence."""
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 3, prose_response('done'))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 1, "only the first request runs")
        self.assertEqual(run.suppressed_duplicates, 2)
        self.assertEqual(
            [s.suppressed_duplicate_of is not None for s in run.steps][:3],
            [False, True, True],
        )

    def test_a_suppressed_request_creates_no_evidence(self) -> None:
        """No fake record, no second id for one observation."""
        before = len(self.store.records)
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 3, prose_response('done'))
        ).run_autonomous("inspect it", finalize=False)

        # One executed action stores its observation and its raw output. The
        # two suppressed requests store nothing at all.
        created = len(self.store.records) - before
        self.assertEqual(created, 2, "one execution, two records; repeats add none")
        self.assertEqual(run.suppressed_duplicates, 2)

    def test_the_prior_evidence_id_is_named_and_stays_re_attestable(self) -> None:
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 2, prose_response('done'))
        ).run_autonomous("inspect it", finalize=False)

        first = run.steps[0].evidence
        self.assertIsNotNone(first)
        # The id handed back is the one the first execution actually produced.
        self.assertEqual(run.steps[1].suppressed_duplicate_of, first.evidence_id)
        self.assertIsNotNone(
            self.store.reattest_exact(first.evidence_id),
            "suppression must not disturb the evidence it points at",
        )

    def test_the_model_is_told_what_already_answers_it(self) -> None:
        backend = ScriptedBackend(*[tool_response(READ_X)] * 2, prose_response('done'))
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)

        evidence_id = run.steps[1].suppressed_duplicate_of
        tool_messages = [
            m for msgs in backend.seen_messages for m in msgs if m.get("role") == "tool"
        ]
        answered = [m for m in tool_messages if "NO_PROGRESS" in str(m.get("content"))]
        self.assertTrue(answered, "the duplicate must be answered, not silently dropped")
        text = str(answered[-1]["content"])
        self.assertIn(evidence_id, text, "the prior identity must be named")
        self.assertIn(f"evidence:{evidence_id}", text, "and be requestable verbatim")

    def test_a_suppressed_duplicate_is_no_progress_and_not_an_error(self) -> None:
        """Nothing failed, so the error budget must not pay for it."""
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 2, prose_response('done'))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(
            [r.classification for r in run.progress][:2], [NEW_CONTENT, NO_PROGRESS]
        )
        suppressed = [s for s in run.steps if s.suppressed_duplicate_of][0]
        self.assertEqual(run.progress[1].classification, NO_PROGRESS)
        self.assertTrue(run.progress[1].repeated_action)
        self.assertIsNone(suppressed.rejection, "a duplicate is not a refusal")


class ActionBudgetTests(AutonomousTestBase):
    """Duplicates cost a model call; they must not cost an action slot."""

    def test_duplicates_leave_the_budget_for_real_work(self) -> None:
        """The action budget is spent on work, not on repetition.

        Three identical reads and one transformation, against a budget of two
        actions. Under the old accounting the two repeats consumed two of the
        slots, so the transformation never ran. Suppressed, they cost model
        calls and nothing else, and the transformation fits.

        The repeats are separated by nothing here, so only one can be
        consecutive before the existing stall bound applies -- which is why the
        budget, not the stall, is what this asserts.
        """
        backend = ScriptedBackend(
            tool_response(READ_X),
            tool_response(READ_X),      # suppressed: same program, same state
            tool_response(TRANSFORM),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous(
            "inspect it", finalize=False, max_actions=2
        )

        self.assertEqual(run.suppressed_duplicates, 1)
        # Both executions that could establish something did, inside a budget
        # of two -- the repeat did not crowd out the transformation.
        self.assertEqual(run.actions_executed, 2)
        executed = [s for s in run.steps if s.action_executed]
        self.assertTrue(
            executed[-1].artifact_handles,
            "the transformation ran rather than being crowded out",
        )

    def test_the_model_call_ceiling_still_bounds_the_run(self) -> None:
        """Suppression must not turn a bounded loop into an unbounded one."""
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 200)
        ).run_autonomous("inspect it", finalize=False, max_model_calls=6)

        self.assertLessEqual(run.model_calls, 7, "the ceiling still holds")
        self.assertGreater(run.suppressed_duplicates, 0)


class NotSuppressedTests(AutonomousTestBase):
    """Everything that can still change an answer must still run."""

    def test_a_different_range_of_the_same_file_runs(self) -> None:
        run = self.runtime(
            ScriptedBackend(
                tool_response(READ_X), tool_response(READ_Y), prose_response("done")
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 2)
        self.assertEqual(run.suppressed_duplicates, 0)

    def test_the_same_program_runs_again_after_the_workspace_changes(self) -> None:
        """A changed workspace makes it a different experiment."""
        run = self.runtime(
            ScriptedBackend(
                tool_response(READ_X),
                tool_response(write_artifact("derived.txt", "stage two")),
                tool_response(READ_X),
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.suppressed_duplicates, 0, "the state it ran against moved")
        self.assertEqual(run.actions_executed, 3)

    def test_a_transformation_is_never_suppressed_for_resembling_a_read(self) -> None:
        run = self.runtime(
            ScriptedBackend(
                tool_response(READ_X), tool_response(TRANSFORM), prose_response("done")
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.suppressed_duplicates, 0)
        self.assertEqual(run.actions_executed, 2)
        executed = [s for s in run.steps if s.action_executed]
        self.assertTrue(executed[-1].artifact_handles, "it produced a durable artifact")

    def test_two_different_programs_with_identical_output_both_run(self) -> None:
        """Equal stdout is not equal identity, and cannot be known beforehand.

        This is the case the live trace showed twice: different programs whose
        printed result happened to match. Suppressing the second would mean
        predicting its output, so it executes and is judged afterwards.
        """
        run = self.runtime(
            ScriptedBackend(
                tool_response("print('same', end='')"),
                tool_response("x = 1\nprint('same', end='')"),
                prose_response("done"),
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 2, "neither was knowably redundant")
        self.assertEqual(run.suppressed_duplicates, 0)

    def test_malformed_code_still_reaches_the_sandbox_refusal(self) -> None:
        """An unparseable program has no stable identity; it is refused, not suppressed."""
        run = self.runtime(
            ScriptedBackend(
                tool_response("def ("), tool_response("def ("), prose_response("done")
            )
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.suppressed_duplicates, 0)
        self.assertTrue(
            any(s.rejection for s in run.steps),
            "the sandbox's own refusal wording must survive",
        )


class WorkspaceStateIsTotalTests(AutonomousTestBase):
    """A probe must not be suppressed after the thing it probes has moved.

    The fingerprint's workspace component decides whether an experiment would
    be repeated, so it has to cover what a program can actually observe -- not
    only the regular-file contents the sandbox needs for artifact capture. A
    review found directory existence and permission changes invisible to it: a
    probe re-run after a `mkdir` was suppressed and the model was told stale
    bytes were the current answer, with the reference asserting "this exact
    observation already exists". It did not.
    """

    def _probe_around(self, seed, probe, mutate):
        backend = ScriptedBackend(
            *([tool_response(seed)] if seed else []),
            tool_response(probe),
            tool_response(mutate),
            tool_response(probe),
            prose_response("done"),
        )
        run = self.runtime(backend).run_autonomous("inspect it", finalize=False)
        return run, [s.result.stdout for s in run.steps if s.action_executed]

    def test_a_directory_probe_re_runs_after_the_directory_appears(self) -> None:
        run, outputs = self._probe_around(
            None,
            "import os;print(os.path.isdir('/workspace/work/stage2'), end='')",
            "import os;os.makedirs('/workspace/work/stage2', exist_ok=True)\n"
            "print('made', end='')",
        )
        self.assertEqual(run.suppressed_duplicates, 0, "the state it probed moved")
        self.assertEqual(
            [outputs[0], outputs[-1]], ["False", "True"],
            "the second probe must observe the change, not a stale copy",
        )

    def test_a_mode_probe_re_runs_after_a_chmod(self) -> None:
        run, outputs = self._probe_around(
            "open('/workspace/work/f', 'w').write('x')\nprint('seeded', end='')",
            "import os\n"
            "print(oct(os.stat('/workspace/work/f').st_mode & 0o777), end='')",
            "import os;os.chmod('/workspace/work/f', 0o600)\nprint('chmod', end='')",
        )
        self.assertEqual(run.suppressed_duplicates, 0)
        self.assertNotEqual(
            outputs[1], outputs[-1], "the permission change must be observable"
        )

    def test_the_sandbox_scaffolding_directory_does_not_defeat_suppression(
        self,
    ) -> None:
        """`work/tmp` appears on the first run whatever the program did.

        Counting it would make the first two workspace states differ in every
        session, so nothing would ever be suppressed -- the mechanism would be
        silently dead while every test that checks a single repeat still
        passed. This is the test that would have caught that.
        """
        run = self.runtime(
            ScriptedBackend(*[tool_response(READ_X)] * 3, prose_response("done"))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.suppressed_duplicates, 2, "both repeats, not just one")


class PerRunAccountingTests(AutonomousTestBase):
    def test_each_run_reports_only_its_own_suppressions(self) -> None:
        """A second run in one session reports what it suppressed, not the total."""
        runtime = self.runtime(
            ScriptedBackend(
                tool_response(READ_X),
                tool_response(READ_X),
                prose_response("done"),
                tool_response(READ_X),
                tool_response(READ_X),
                prose_response("done"),
            )
        )
        first = runtime.run_autonomous("inspect it", finalize=False)
        second = runtime.run_autonomous("inspect it again", finalize=False)

        self.assertEqual(first.suppressed_duplicates, 1)
        self.assertEqual(
            second.suppressed_duplicates, 2,
            "the second run's own count, not the session running total",
        )
        # The session-level registry is what persists across runs.
        self.assertGreaterEqual(runtime.suppressed_duplicates, 3)


class NonDeterministicOutputTests(AutonomousTestBase):
    """Random or time-varying stdout does not make a repeat into discovery."""

    def test_a_repeat_is_suppressed_even_when_its_output_would_differ(self) -> None:
        run = self.runtime(
            ScriptedBackend(*[tool_response(nondet("scan"))] * 3, prose_response('done'))
        ).run_autonomous("inspect it", finalize=False)

        self.assertEqual(run.actions_executed, 1)
        self.assertEqual(run.suppressed_duplicates, 2)


class ScriptedProgressRegressionTests(AutonomousTestBase):
    """The live failure, reproduced deterministically end to end.

    Read X, request X twice more, then -- having been told the observation
    already exists -- execute a deterministic transformation and inspect what it
    produced. This is the trajectory the real run could not reach.
    """

    def test_the_run_reaches_the_transformation(self) -> None:
        backend = ScriptedBackend(
            tool_response(READ_X),
            tool_response(READ_X),
            # A second consecutive duplicate would end the run on the existing
            # stall bound, which this change deliberately leaves alone: the
            # live trace never had two duplicates in a row. One correction,
            # then the model acts on it.
            tool_response(TRANSFORM),
            tool_response("print(open('/workspace/work/decoded.txt').read(), end='')"),
            prose_response('done'),
        )
        run = self.runtime(backend).run_autonomous("analyse it", finalize=False)

        self.assertEqual(run.suppressed_duplicates, 1, "the repeat was suppressed")
        self.assertEqual(run.actions_executed, 3, "and it cost no action slot")

        # The model saw the guidance before it changed strategy.
        tool_messages = [
            str(m.get("content"))
            for msgs in backend.seen_messages
            for m in msgs
            if m.get("role") == "tool"
        ]
        self.assertTrue(any("NO_PROGRESS" in t for t in tool_messages))

        # The transformation ran, wrote a durable artifact, and its output was
        # stored as evidence that can be re-attested exactly.
        transform_step = run.steps[2]
        self.assertTrue(transform_step.action_executed)
        self.assertTrue(transform_step.artifact_handles)
        self.assertIsNotNone(
            self.store.reattest_exact(transform_step.evidence.evidence_id)
        )

        # And the streak reset: the transformation counts as progress.
        self.assertEqual(
            [r.classification for r in run.progress][:3],
            [NEW_CONTENT, NO_PROGRESS, NEW_CONTENT],
        )

    def test_the_decoded_output_is_readable_afterwards(self) -> None:
        backend = ScriptedBackend(
            tool_response(READ_X),
            tool_response(READ_X),
            tool_response(TRANSFORM),
            tool_response("print(open('/workspace/work/decoded.txt').read(), end='')"),
            prose_response('done'),
        )
        run = self.runtime(backend).run_autonomous("analyse it", finalize=False)

        executed = [s for s in run.steps if s.action_executed]
        last = executed[-1]
        self.assertTrue(last.action_executed, "the derived artifact is addressable")
        self.assertIsNotNone(last.evidence)


if __name__ == "__main__":
    unittest.main()

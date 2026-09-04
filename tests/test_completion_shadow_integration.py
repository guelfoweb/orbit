"""The shadow gate inside a real autonomous run: it must change nothing."""

from __future__ import annotations

import secrets
import unittest
from unittest import mock

from orbit.runtime.completion_shadow import SHADOW_ENV, ShadowObservation

from tests.test_analysis_autonomous import AutonomousTestBase, emit
from tests.test_analysis_runtime import prose_response, tool_response


def _script():
    """Four productive steps then prose, so a run reaches the shadow schedule."""
    return [
        tool_response(emit("one")),
        tool_response(emit("two")),
        tool_response(emit("three")),
        tool_response(emit("four")),
        tool_response(emit("five")),
        prose_response("done"),
    ]


class _CountingBackend:
    """Serves the loop from a script and the verifier from its own answers.

    Kept separate so verifier traffic can never be mistaken for a loop call:
    the loop path is `chat_stream` with tools, the verifier path is `chat`
    with `tools == []`.
    """

    def __init__(self, responses, verifier_answers=None):
        self._responses = list(responses)
        self.calls = 0
        self.tool_modes: list[object] = []
        self._verifier = list(verifier_answers or [])
        self.verifier_calls = 0

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        self.tool_modes.append(tools)
        if tools == []:
            from orbit.backend.base import ChatResult

            self.verifier_calls += 1
            answer = self._verifier.pop(0) if self._verifier else "CONTINUE missing: x"
            return ChatResult(
                content=answer,
                model="m",
                finish_reason="stop",
                tool_calls=[],
                prompt_tokens=50,
                completion_tokens=4,
                cached_tokens=0,
                prompt_tokens_per_second=None,
                generation_tokens_per_second=None,
            )
        if self.calls >= len(self._responses):
            raise AssertionError("loop invoked more times than scripted")
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def count_text_tokens(self, text: str):
        """A deterministic stand-in for the model tokenizer.

        Whole-word counting, not the real vocabulary: these tests are about the
        budget mechanism, not about token fidelity, and a fixture that returned
        None would make every checkpoint skip and hide the behaviour under test.
        """
        from orbit.backend.base import TokenCount

        return TokenCount(tokens=len(text.split()), context_tokens=16384)

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None,
                    on_delta=None, on_progress=None):
        response = self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )
        if on_delta is not None and response.content:
            on_delta(response.content)
        return response


def _trajectory(run):
    """Everything about a run that the shadow is forbidden to influence."""
    return {
        "stop_reason": run.stop_reason,
        "model_calls": run.model_calls,
        "actions": run.actions_executed,
        "steps": len(run.steps),
        "replans": run.replans,
        "classifications": [record.classification for record in run.progress],
        # The content half of the id, not the whole id: the prefix is a
        # uuid4, so two runs never share it whatever the shadow does. The
        # digest is what says the same evidence was produced.
        "evidence_digests": [
            step.evidence.evidence_id.split("_")[-1]
            for step in run.steps
            if step.evidence is not None
        ],
        "report": (run.final_report.text if run.final_report else None),
    }


class ShadowIsObservationalTests(AutonomousTestBase):
    def test_default_off_runs_no_verifier(self) -> None:
        backend = _CountingBackend(_script())
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(SHADOW_ENV, None)
            run = self.runtime(backend).run_autonomous("go", finalize=False)
        self.assertIsNone(run.completion_shadow)
        self.assertNotIn([], backend.tool_modes)

    def test_shadow_on_off_trajectories_are_identical(self) -> None:
        import os

        results = {}
        for label, env in (("off", {}), ("on", {SHADOW_ENV: "1"})):
            with self.subTest(shadow=label):
                base = AutonomousTestBase("run")
                base.setUp()
                self.addCleanup(base._dir.cleanup)
                backend = _CountingBackend(
                    _script(),
                    verifier_answers=["CONTINUE missing: the payload"] * 12,
                )
                with mock.patch.dict("os.environ", env, clear=False):
                    if not env:
                        os.environ.pop(SHADOW_ENV, None)
                    run = base.runtime(backend).run_autonomous("go", finalize=False)
                results[label] = _trajectory(run)
        self.assertEqual(results["off"], results["on"])

    def test_would_stop_does_not_stop_the_run(self) -> None:
        import os

        from orbit.runtime.completion_shadow import ShadowObservation

        backend = _CountingBackend(_script())
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            # Forced rather than scripted: a real COMPLETE must now cite live
            # evidence, and this test is about the loop ignoring the verdict,
            # not about how the verdict was reached.
            with mock.patch(
                "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                side_effect=lambda **kw: ShadowObservation(
                    action=kw["action"],
                    snapshot_digest=secrets.token_hex(32),
                    would_stop=True,
                ),
            ):
                run = self.runtime(backend).run_autonomous("go", finalize=False)
        shadow = run.completion_shadow
        self.assertIsNotNone(shadow)
        self.assertTrue(shadow.would_stop_actions, "the fixture must reach WOULD_STOP")
        # The run ended for its own reason, never because the shadow said stop.
        self.assertNotEqual(run.stop_reason, "would_stop")
        self.assertEqual(run.actions_executed, 5)

    def test_verifier_calls_are_outside_the_loop_budget(self) -> None:
        import os

        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            run = runtime.run_autonomous("go", finalize=False)
        shadow = run.completion_shadow
        self.assertGreater(shadow.calls, 0)
        loop_calls = sum(1 for tools in backend.tool_modes if tools != [])
        # The load-bearing one: the loop's own budget counts loop calls and
        # nothing else, however many verifier calls the shadow made.
        self.assertEqual(run.model_calls, loop_calls)
        # And the LIFETIME counter too. `run.model_calls` is structurally
        # immune -- the shadow sits between the windows it is measured over --
        # so counting a verifier call would leave that assertion green while
        # the runtime's own total, which the terminal renders as session
        # usage, silently absorbed a diagnostic.
        self.assertGreater(backend.verifier_calls, 0,
                           "the verifier must actually have run")
        self.assertEqual(runtime.model_calls, loop_calls)

    def test_verifier_calls_are_tool_free(self) -> None:
        import os

        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            self.runtime(backend).run_autonomous("go", finalize=False)
        self.assertIn([], backend.tool_modes)

    def test_shadow_failure_does_not_end_the_run(self) -> None:
        import os

        backend = _CountingBackend(_script())
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            # Patched at the import site: `analysis_runtime` bound the name at
            # module load, so patching the source module would not take effect
            # and the test would pass without exercising anything.
            with mock.patch(
                "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                side_effect=RuntimeError("verifier exploded"),
            ):
                run = self.runtime(backend).run_autonomous("go", finalize=False)
        self.assertEqual(run.actions_executed, 5)
        self.assertTrue(
            all(not item.would_stop for item in run.completion_shadow.observations)
        )


def _long_script(n: int = 13):
    """Enough productive steps to pass every checkpoint up to the hard bound."""
    return [tool_response(emit(f"s{i}")) for i in range(n)] + [prose_response("done")]


class ActiveStopIsUnreachableTests(AutonomousTestBase):
    """The central safety property, asserted behaviourally rather than by grep.

    A source scan for `would_stop` is evaded by `getattr(obs, "would_" + "stop")`
    or by a helper the scan does not read, and a short fixture is evaded by
    gating the stop above its action count. Forcing every checkpoint to say
    stop, and running to the hard bound, is evaded by neither: any branch that
    acts on the verdict changes the trajectory here.
    """

    def test_forced_would_stop_at_every_checkpoint_changes_nothing(self) -> None:
        import os

        from orbit.runtime.completion_shadow import ShadowObservation

        # The digest is realistic and unpredictable, never a fixed sentinel: a
        # stop conditioned on `snapshot_digest != "forced"` would otherwise
        # truncate real runs while passing this very test.
        def always_stop(**kwargs):
            return ShadowObservation(
                action=kwargs["action"],
                snapshot_digest=secrets.token_hex(32),
                verifier_a="COMPLETE",
                verifier_b="NO_GAP",
                would_stop=True,
            )

        trajectories = {}
        for label, env in (("off", {}), ("on", {SHADOW_ENV: "1"})):
            base = AutonomousTestBase("run")
            base.setUp()
            self.addCleanup(base._dir.cleanup)
            backend = _CountingBackend(_long_script())
            with mock.patch.dict("os.environ", env, clear=False):
                if not env:
                    os.environ.pop(SHADOW_ENV, None)
                with mock.patch(
                    "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                    side_effect=always_stop,
                ):
                    run = base.runtime(backend).run_autonomous("go", finalize=False)
            trajectories[label] = _trajectory(run)
            if label == "on":
                self.assertTrue(
                    run.completion_shadow.would_stop_actions,
                    "the fixture must actually reach WOULD_STOP checkpoints",
                )
        self.assertEqual(trajectories["off"], trajectories["on"])

    def test_forced_would_stop_reaches_the_action_bound(self) -> None:
        """Guards the guard: if the fixture stopped early the test above would
        compare two short runs and prove nothing."""
        import os

        from orbit.runtime.completion_shadow import ShadowObservation

        backend = _CountingBackend(_long_script())
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            with mock.patch(
                "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                side_effect=lambda **kw: ShadowObservation(
                    action=kw["action"],
                    snapshot_digest=secrets.token_hex(32),
                    would_stop=True,
                ),
            ):
                run = self.runtime(backend).run_autonomous("go", finalize=False)
        self.assertGreaterEqual(run.actions_executed, 8)
        self.assertGreaterEqual(len(run.completion_shadow.would_stop_actions), 3)


class ShadowWiringTests(unittest.TestCase):
    def test_loop_never_branches_on_would_stop(self) -> None:
        """A cheap first line only. `ActiveStopIsUnreachableTests` is the real one."""
        import inspect

        from orbit.runtime import analysis_runtime

        source = inspect.getsource(analysis_runtime.AnalysisRuntime.run_autonomous)
        self.assertIn("_observe_completion_shadow", source)
        self.assertNotIn("would_stop", source)

    def test_observation_reads_only_active_evidence(self) -> None:
        """The runtime call site, not just `build_snapshot`, must filter.

        Passing the raw store here would hand a verifier a value the analysis
        has already retracted -- the exact failure rc36 fixed for reports.
        """
        import inspect

        from orbit.runtime import analysis_runtime

        source = inspect.getsource(
            analysis_runtime.AnalysisRuntime._observe_completion_shadow
        )
        self.assertIn("active_records(", source)
        self.assertNotIn(
            "records = list(self.evidence_store.records.values())", source
        )

    def test_verifier_declares_its_own_phase(self) -> None:
        import inspect

        from orbit.runtime import analysis_runtime

        source = inspect.getsource(
            analysis_runtime.AnalysisRuntime._observe_completion_shadow
        )
        self.assertIn("ANALYSIS_COMPLETION_SHADOW_PHASE", source)
        self.assertIn("tools=[]", source)
        self.assertIn('tools_mode="off"', source)
        self.assertNotIn("ANALYSIS_STEP_PHASE", source)


class SnapshotUsesActiveEvidenceOnlyTests(unittest.TestCase):
    """A superseded record must never reach a verifier.

    Replayed against the preserved store from the rc36 finalization failure,
    which contains a genuine artifact rewrite: an earlier `ioc_report.txt`
    naming `gibuyuy37v2v.top` and its correction naming `gibuzuy37v2v.top`.
    Handing both to a verifier would ask it to judge completion against a
    value the analysis had already retracted.
    """

    # Resolved relative to the repo rather than hard-coded to one machine, and
    # skipped cleanly when the preserved archive is not present -- it lives
    # outside the repository by convention, so most checkouts will not have it.
    CHECKPOINT = (
        "RECOVERED-finalization-failure-run-20260825-021057/evidence-store/evidence"
    )

    def setUp(self) -> None:
        import os
        from pathlib import Path

        root = Path(
            os.environ.get(
                "ORBIT_CHECKPOINT_ROOT",
                Path(__file__).resolve().parents[2] / "orbit-checkpoints",
            )
        )
        store = root / self.CHECKPOINT
        if not store.is_dir():
            self.skipTest("preserved evidence store not present")
        self.STORE = str(store)
        from orbit.runtime.evidence import EvidenceStore

        self.store = EvidenceStore(root=Path(self.STORE))
        self.store.load_index()

    def test_superseded_record_is_excluded_from_the_snapshot(self) -> None:
        from orbit.runtime.completion_shadow import build_snapshot
        from orbit.runtime.evidence_authority import active_records

        every = list(self.store.records.values())
        active = active_records(every)
        self.assertLess(len(active), len(every), "fixture must contain a supersession")

        superseded = {
            record.evidence_id for record in every
        } - {record.evidence_id for record in active}

        snapshot = build_snapshot(
            request="extract the C2 and artifacts",
            records=active,
            load_raw=self.store.load_raw,
        )
        rendered = snapshot.render()
        for evidence_id in superseded:
            self.assertNotIn(evidence_id, rendered)

    def test_snapshot_over_all_records_would_leak_the_retracted_version(self) -> None:
        # Pins why the ACTIVE view is load-bearing rather than cosmetic: built
        # over every record, the snapshot carries the superseded evidence id.
        from orbit.runtime.completion_shadow import build_snapshot
        from orbit.runtime.evidence_authority import active_records

        every = list(self.store.records.values())
        active_ids = {record.evidence_id for record in active_records(every)}
        superseded = [r for r in every if r.evidence_id not in active_ids]
        self.assertTrue(superseded)

        naive = build_snapshot(
            request="extract the C2 and artifacts",
            records=every,
            load_raw=self.store.load_raw,
        )
        correct = build_snapshot(
            request="extract the C2 and artifacts",
            records=active_records(every),
            load_raw=self.store.load_raw,
        )
        self.assertNotEqual(naive.digest, correct.digest)
        self.assertIn(superseded[0].evidence_id, naive.render())
        self.assertNotIn(superseded[0].evidence_id, correct.render())

if __name__ == "__main__":
    unittest.main()


class ShadowCancellationTests(AutonomousTestBase):
    """A checkpoint must not be able to swallow the analyst's cancellation.

    The shadow is a diagnostic and its failures are contained -- that is the
    whole point of the handler. But it caught `BaseException`, so a Ctrl-C
    arriving during a checkpoint was recorded as `shadow_error:
    KeyboardInterrupt` and the analysis carried on, resolving questions the
    analyst had asked it to abandon. The three cases are different events and
    must stay distinguishable.
    """

    def _run_with_shadow_raising(self, exc):
        """Drive a run whose shadow observation raises `exc`.

        The failure is injected at `evaluate_completion_shadow`, inside the
        observer's own `try`, which is where a real verifier failure or a
        real interrupt would arrive.
        """
        import os

        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)

        def raising(**_kwargs):
            raise exc

        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False), \
             mock.patch(
                 "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                 side_effect=raising,
             ):
            return runtime.run_autonomous("go", finalize=False)

    def test_an_ordinary_failure_is_still_contained_as_evidence(self) -> None:
        """The behaviour the handler exists for, unchanged."""
        run = self._run_with_shadow_raising(RuntimeError("verifier exploded"))
        self.assertIsNotNone(run.completion_shadow)
        self.assertEqual(
            [o.blocked_by for o in run.completion_shadow.observations],
            ["shadow_error: RuntimeError"],
        )
        # The run was not stopped by a diagnostic failing.
        self.assertFalse(run.cancelled)
        self.assertNotEqual(run.stop_reason, "cancelled")

    def test_a_cancellation_stops_the_run(self) -> None:
        run = self._run_with_shadow_raising(KeyboardInterrupt())
        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, "cancelled")

    def test_a_cancellation_writes_no_shadow_evidence(self) -> None:
        """The analyst's decision is not a diagnostic failure.

        Recording `shadow_error: KeyboardInterrupt` would put their stop into
        the ledger as an observation the verifier made, which it did not.
        """
        run = self._run_with_shadow_raising(KeyboardInterrupt())
        blocked = [o.blocked_by for o in run.completion_shadow.observations]
        self.assertEqual(blocked, [])
        self.assertNotIn("shadow_error: KeyboardInterrupt", blocked)

    def test_a_cancellation_stops_the_analysis_short(self) -> None:
        """The run must END, not carry on past the analyst's stop.

        Asserting `resolved_questions == ()` would be vacuous here -- this
        backend resolves nothing either way -- so the witness is the work
        actually done: a cancelled run stops at the checkpoint, while the
        same run with an ordinary shadow failure continues to completion.
        """
        cancelled = self._run_with_shadow_raising(KeyboardInterrupt())
        ordinary = self._run_with_shadow_raising(RuntimeError("verifier died"))
        self.assertLess(
            cancelled.actions_executed, ordinary.actions_executed,
            "the run continued past the cancellation",
        )
        self.assertEqual(cancelled.resolved_questions, ())

    def test_a_cancellation_leaves_no_report(self) -> None:
        """Consistent with every other cancellation path in the runtime."""
        import os

        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False), \
             mock.patch(
                 "orbit.runtime.analysis_runtime.evaluate_completion_shadow",
                 side_effect=KeyboardInterrupt(),
             ):
            run = runtime.run_autonomous("go", finalize=True)
        self.assertTrue(run.cancelled)
        self.assertIsNone(run.final_report)

    def test_system_exit_is_not_turned_into_shadow_evidence(self) -> None:
        """A `BaseException` that is neither ordinary nor a cancellation.

        It must not be normalised into a diagnostic observation. Propagating
        is the honest outcome: the runtime has no meaning to assign it.
        """
        with self.assertRaises(SystemExit):
            self._run_with_shadow_raising(SystemExit(3))

    def test_generator_exit_is_not_turned_into_shadow_evidence(self) -> None:
        with self.assertRaises(GeneratorExit):
            self._run_with_shadow_raising(GeneratorExit())

    def test_the_three_outcomes_stay_distinct(self) -> None:
        """One table, because conflating any two of them is the defect."""
        ordinary = self._run_with_shadow_raising(ValueError("bad snapshot"))
        self.assertFalse(ordinary.cancelled)
        self.assertEqual(
            [o.blocked_by for o in ordinary.completion_shadow.observations],
            ["shadow_error: ValueError"],
        )

        cancelled = self._run_with_shadow_raising(KeyboardInterrupt())
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.completion_shadow.observations, [])

        with self.assertRaises(SystemExit):
            self._run_with_shadow_raising(SystemExit(1))


class FinalLedgerCancellationTests(AutonomousTestBase):
    """A Ctrl-C during the final ledger write must not escape the run.

    `_write_shadow_final` links the checkpoint ledger to the outcome, and it
    runs after everything else: every step is committed, the closing report
    is written, and `stop_reason` is settled. Its handler caught `Exception`
    -- correct for a diagnostic that must not fail a finished run -- which
    let a `KeyboardInterrupt` through to escape `run_autonomous` entirely.

    That is not merely untidy. The caller holds only a pre-run checkpoint,
    and `repl.py` restores it on an interrupt: the history of every completed
    step is deleted and its evidence left orphaned on disk.
    """

    def _run_with_ledger_raising(self, exc):
        """Drive a complete run whose FINAL ledger write raises `exc`.

        Injected at `_write_shadow_final` itself, so the run reaches the
        point where it is already over -- which is what makes this different
        from an interrupt during the analysis.
        """
        import os

        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)

        def raising(self_, *args, **kwargs):
            raise exc

        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False), \
             mock.patch(
                 "orbit.runtime.analysis_runtime.AnalysisRuntime."
                 "_write_shadow_final",
                 raising,
             ):
            return runtime.run_autonomous("go", finalize=False)

    def test_an_ordinary_failure_still_does_not_end_the_run(self) -> None:
        """The behaviour the handler exists for, unchanged."""
        run = self._run_with_ledger_raising(RuntimeError("ledger unwritable"))
        self.assertFalse(run.cancelled)
        self.assertNotEqual(run.stop_reason, "cancelled")
        # The run's own verdict survives a diagnostic that could not be written.
        self.assertTrue(run.stop_reason)

    def test_a_cancellation_does_not_escape(self) -> None:
        """The defect: this used to raise out of `run_autonomous`."""
        run = self._run_with_ledger_raising(KeyboardInterrupt())
        self.assertTrue(run.cancelled)
        self.assertEqual(run.stop_reason, "cancelled")

    def test_a_cancellation_returns_the_completed_work(self) -> None:
        """What escaping destroyed: the caller gets the run, not an exception.

        The steps and their evidence are the whole point -- an escape sent the
        caller to a checkpoint restore that deleted them.
        """
        ordinary = self._run_with_ledger_raising(RuntimeError("unwritable"))
        run = self._run_with_ledger_raising(KeyboardInterrupt())
        self.assertGreater(len(run.steps), 0)
        self.assertEqual(len(run.progress), len(run.steps))
        self.assertGreater(run.actions_executed, 0)
        # The spend the analyst reads is the run's own, unchanged by an
        # interrupt that arrived after the work was done.
        self.assertGreater(ordinary.model_calls, 0)
        self.assertEqual(run.model_calls, ordinary.model_calls)

    def test_a_cancellation_writes_no_extra_shadow_evidence(self) -> None:
        """The stop is not a diagnostic observation the verifier made."""
        before = self._run_with_ledger_raising(RuntimeError("ledger unwritable"))
        after = self._run_with_ledger_raising(KeyboardInterrupt())
        self.assertEqual(
            len(after.completion_shadow.observations),
            len(before.completion_shadow.observations),
            "the cancellation added a fabricated observation",
        )
        for observation in after.completion_shadow.observations:
            self.assertNotIn("KeyboardInterrupt", observation.blocked_by or "")

    def test_a_cancellation_resolves_nothing_new(self) -> None:
        """No question becomes answered because the ledger write was stopped.

        Driven through a PLANNING fixture on purpose. This module's own
        backend never issues a plan, so `resolved_questions` is `()` on both
        sides and the comparison holds for any implementation -- a mutant
        that marked every question RESOLVED at the ledger survived it. The
        controller fixture makes the two sides differ if anything rewrites
        the ledger's state.
        """
        from tests import test_analysis_controller_runtime as controller_tests

        def run_with(exc):
            # `_Case` is instantiated only to reuse its `_runtime` helper.
            # Its `addCleanup` never fires -- nothing calls `doCleanups` on a
            # bare instance -- so the workspace is released here instead.
            # Without this the test leaked two session directories per run.
            case = controller_tests._Case("run")
            model = controller_tests._Model(
                plan=[controller_tests._question(f"q{i}") for i in range(5)]
            )
            runtime = case._runtime(model)
            self.addCleanup(runtime.close)

            def raising(self_, *args, **kwargs):
                raise exc

            with mock.patch.dict(
                "os.environ", {SHADOW_ENV: "1"}, clear=False
            ), mock.patch(
                "orbit.runtime.analysis_runtime.AnalysisRuntime."
                "_write_shadow_final",
                raising,
            ):
                # A budget that leaves questions BOTH resolved and open: a
                # fixture where everything is already resolved cannot tell a
                # correct ledger from one that marks the rest resolved too.
                return runtime.run_autonomous(
                    "Analyse it.", finalize=False, max_model_calls=6
                )

        ordinary = run_with(RuntimeError("unwritable"))
        cancelled = run_with(KeyboardInterrupt())
        # The cancellation is reported on THIS path too. The other tests
        # assert it through a fixture that issues no plan, where
        # `controller is None` -- so a flag made conditional on controller
        # state would read as True there and ship unnoticed.
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")
        self.assertFalse(ordinary.cancelled)
        # `replans` is asserted HERE, not in the scripted fixture, because
        # only this one produces a non-zero value: comparing 0 == 0 there
        # lets a handler that zeroes the counter pass unnoticed.
        self.assertGreater(ordinary.replans, 0)
        self.assertEqual(cancelled.replans, ordinary.replans)
        # The premise, asserted: some are settled and some are not, so the
        # comparison below can actually distinguish.
        self.assertTrue(ordinary.resolved_questions)
        self.assertTrue(ordinary.open_questions)
        self.assertEqual(cancelled.resolved_questions,
                         ordinary.resolved_questions)
        self.assertEqual(cancelled.open_questions, ordinary.open_questions)

    def test_a_finalized_run_keeps_its_closing_report(self) -> None:
        """The state next to the flags, which every other test here misses.

        Every case in this class runs `finalize=False`, so `final_report` is
        None on both branches and nothing notices a handler that discards it.
        That is the same loss this commit exists to prevent -- the analyst's
        closing report thrown away while writing a diagnostic about the run
        that produced it -- so it is asserted where it can actually be lost.
        """
        import os

        def run_with(exc):
            backend = _CountingBackend(
                _script(),
                verifier_answers=["CONTINUE missing: the payload"] * 12,
            )
            runtime = self.runtime(backend)

            def raising(self_, *args, **kwargs):
                raise exc

            with mock.patch.dict(
                "os.environ", {SHADOW_ENV: "1"}, clear=False
            ), mock.patch(
                "orbit.runtime.analysis_runtime.AnalysisRuntime."
                "_write_shadow_final",
                raising,
            ):
                return runtime.run_autonomous("go", finalize=True)

        ordinary = run_with(RuntimeError("unwritable"))
        cancelled = run_with(KeyboardInterrupt())

        # The premise: this configuration really does produce a report, so
        # the assertion below can distinguish.
        self.assertIsNotNone(ordinary.final_report)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")
        # The report survives the interrupt: it was written before the
        # ledger, and the ledger failing must not retract it.
        self.assertIsNotNone(cancelled.final_report)
        self.assertEqual(cancelled.final_report.text,
                         ordinary.final_report.text)
        # And the run's other counters are the run's own, not the handler's.
        self.assertEqual(cancelled.replans, ordinary.replans)
        self.assertEqual(cancelled.actions_executed,
                         ordinary.actions_executed)

    def test_a_cancellation_with_no_steps_is_still_reported(self) -> None:
        """A budget so tight the run reaches the ledger having done nothing.

        Every other fixture here arrives with steps committed, so a flag
        derived from them -- `cancelled = bool(steps)` -- reads True by
        accident and a regression to that shape would ship silently,
        reporting an interrupt as an ordinary completed run.
        """
        from tests import test_analysis_controller_runtime as controller_tests

        def run_with(exc):
            case = controller_tests._Case("run")
            runtime = case._runtime(
                controller_tests._Model(plan=[controller_tests._question("q")])
            )
            self.addCleanup(runtime.close)

            def raising(self_, *args, **kwargs):
                raise exc

            with mock.patch.dict(
                "os.environ", {SHADOW_ENV: "1"}, clear=False
            ), mock.patch(
                "orbit.runtime.analysis_runtime.AnalysisRuntime."
                "_write_shadow_final",
                raising,
            ):
                return runtime.run_autonomous(
                    "Analyse it.", finalize=False, max_model_calls=2
                )

        cancelled = run_with(KeyboardInterrupt())
        # The premise: the ledger really was reached with nothing done.
        self.assertEqual(len(cancelled.steps), 0)
        self.assertEqual(cancelled.actions_executed, 0)
        # And the interrupt is still reported as one.
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")

    def test_a_cancellation_keeps_what_the_run_suppressed(self) -> None:
        """The duplicate-suppression counters survive the interrupt.

        `suppressed_duplicates` is read into the result AFTER this
        handler runs, so a handler that zeroed it would report a run as
        having found nothing repetitive. Every other fixture in this
        class measures it as 0, which is why the
        driving backend is borrowed from the duplicate-suppression
        module: three identical reads are what makes the counter bite.
        """
        from tests.test_analysis_autonomous import ScriptedBackend, tool_response
        from tests.test_analysis_duplicate_suppression import READ_X
        from tests.test_analysis_runtime import prose_response

        def run_with(exc):
            backend = ScriptedBackend(
                *[tool_response(READ_X)] * 3, prose_response("done")
            )
            runtime = self.runtime(backend)

            def raising(self_, *args, **kwargs):
                raise exc

            with mock.patch.dict(
                "os.environ", {SHADOW_ENV: "1"}, clear=False
            ), mock.patch(
                "orbit.runtime.analysis_runtime.AnalysisRuntime."
                "_write_shadow_final",
                raising,
            ):
                return runtime.run_autonomous("go", finalize=False)

        cancelled = run_with(KeyboardInterrupt())
        # The premise: this fixture really does suppress something.
        self.assertGreater(cancelled.suppressed_duplicates, 0)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")

    def test_a_cancellation_keeps_the_repairs_it_needed(self) -> None:
        """A control repair the run performed survives the interrupt.

        `control_repairs` is reported as a delta against a baseline taken
        before the run, and that baseline is read AFTER this handler --
        so rebinding it here would erase a repair the run really did.
        No fixture in this class produces one, which is why the backend
        is borrowed: it fails to parse the first control response, and
        the runtime repairs it exactly once.
        """
        from tests.test_analysis_controller_runtime import (
            ControlIsolationTests,
            LlamaServerToolCallParseError,
            _Model,
            _question,
        )

        borrowed = ControlIsolationTests("test_a_parse_failure_is_repaired_once")
        borrowed.addCleanup = self.addCleanup
        runtime = borrowed._strict(
            _Model(plan=[_question("a")]),
            failures=[
                LlamaServerToolCallParseError("Failed to parse input at pos 118")
            ],
        )

        def raising(self_, *args, **kwargs):
            raise KeyboardInterrupt()

        with mock.patch.dict(
            "os.environ", {SHADOW_ENV: "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.analysis_runtime.AnalysisRuntime._write_shadow_final",
            raising,
        ):
            cancelled = runtime.run_autonomous("Analyse it.", finalize=False)

        # The premise: this fixture really did repair something.
        self.assertEqual(cancelled.control_repairs, 1)
        # Pinned here rather than in the scripted fixtures, where all three
        # are 0 and would compare 0 == 0. Like `control_repairs`, the first
        # is reported as a delta against a baseline read after this handler.
        self.assertEqual(cancelled.control_attempts, 3)
        self.assertEqual(cancelled.cover_calls, 1)
        self.assertEqual(cancelled.plan_calls, 1)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")

    def test_a_run_already_cancelled_stays_cancelled(self) -> None:
        """The flag is SET here, never toggled.

        A run interrupted during a step already carries `cancelled` True
        and still walks to the ledger, where a second interrupt lands in
        this handler. Every other test here arrives with the flag False,
        so a handler that inverted it instead of setting it would pass
        them all while turning the one case that matters -- a run the
        analyst had already stopped -- back into an ordinary completed
        run.
        """
        backend = _CountingBackend(
            _script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)

        def raising(self_, *args, **kwargs):
            raise KeyboardInterrupt()

        with mock.patch.dict(
            "os.environ", {SHADOW_ENV: "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.analysis_runtime.AnalysisRuntime._write_shadow_final",
            raising,
        ), mock.patch.object(
            type(runtime), "step", side_effect=KeyboardInterrupt
        ):
            cancelled = runtime.run_autonomous("go", finalize=False)

        # The premise: the run was ALREADY cancelled before the ledger.
        self.assertEqual(len(cancelled.steps), 0)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")

    def test_a_cancellation_keeps_the_repair_the_run_made(self) -> None:
        """An action repair the run performed survives the interrupt.

        This is the counter `repairs`, distinct from `control_repairs`:
        one counts a corrected ACTION, the other a re-asked control call.
        It is 0 in every other fixture in this class -- including the one
        pinning `suppressed_duplicates`, which is why zeroing both at once
        was caught by the `suppressed` half alone while `repairs` went
        unpinned. Hence a backend borrowed from the repair suite: the
        first program is broken, the second corrects it.
        """
        from tests.test_analysis_repair import (
            BROKEN,
            FIXED,
            RecordingBackend,
            RepairFlowTests,
            _prose,
            _tool_call,
        )

        borrowed = RepairFlowTests("test_failure_then_repair_then_success")
        borrowed.setUp()
        self.addCleanup(borrowed.doCleanups)
        runtime = borrowed.runtime(
            RecordingBackend(
                _tool_call(BROKEN),
                _tool_call(FIXED, call_id="call_2"),
                _prose("decoded"),
            )
        )

        def raising(self_, *args, **kwargs):
            raise KeyboardInterrupt()

        with mock.patch.dict(
            "os.environ", {SHADOW_ENV: "1"}, clear=False
        ), mock.patch(
            "orbit.runtime.analysis_runtime.AnalysisRuntime._write_shadow_final",
            raising,
        ):
            cancelled = runtime.run_autonomous("analyse", finalize=False)

        # The premise: this fixture really did repair an action.
        self.assertEqual(cancelled.repairs, 1)
        self.assertTrue(cancelled.cancelled)
        self.assertEqual(cancelled.stop_reason, "cancelled")

    def test_system_exit_still_propagates(self) -> None:
        """Not a cancellation and not a diagnostic failure: no meaning to assign."""
        with self.assertRaises(SystemExit):
            self._run_with_ledger_raising(SystemExit(3))

    def test_generator_exit_still_propagates(self) -> None:
        with self.assertRaises(GeneratorExit):
            self._run_with_ledger_raising(GeneratorExit())

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

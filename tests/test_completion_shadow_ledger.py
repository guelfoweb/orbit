"""The shadow ledger: it must record faithfully and decide nothing."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orbit.runtime.completion_shadow import (
    SHADOW_ENV,
    CompletionSnapshot,
    ShadowObservation,
    build_snapshot,
)
from orbit.runtime.completion_shadow_ledger import (
    LEDGER_SCHEMA_VERSION,
    ShadowLedgerWriter,
    ledger_path_for_evidence_root,
    read_ledger,
    verify_snapshot_hashes,
)

from tests.test_analysis_autonomous import AutonomousTestBase
from tests.test_completion_shadow_integration import (
    _CountingBackend,
    _long_script,
    _script,
    _trajectory,
)


def _observation(action: int, **kwargs) -> ShadowObservation:
    snapshot = build_snapshot(
        request="extract the C2",
        records=[type("R", (), {"evidence_id": "ev_a", "metadata": {}})()],
        load_raw=lambda _i: "the decoded url is http://example.invalid",
    )
    defaults = dict(
        action=action,
        snapshot_digest=snapshot.digest,
        snapshot=snapshot,
        verifier_a="CONTINUE",
        calls=1,
        prompt_tokens=120,
        output_tokens=9,
        wall_seconds=1.25,
    )
    defaults.update(kwargs)
    return ShadowObservation(**defaults)


class WriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="orbit-ledger-")
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "run.jsonl"

    def test_one_checkpoint_writes_one_valid_record(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        self.assertTrue(writer.write_checkpoint(_observation(4)))
        result = read_ledger(self.path)
        self.assertEqual(len(result.checkpoints), 1)
        self.assertEqual(result.checkpoints[0]["action"], 4)
        self.assertEqual(result.malformed_lines, ())
        self.assertEqual(verify_snapshot_hashes(result), ())

    def test_snapshot_rehashes_exactly(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(_observation(4))
        result = read_ledger(self.path)
        # The persisted projection must reproduce the digest the verifiers saw,
        # or a stored decision cannot be tied to what it was made from.
        self.assertEqual(verify_snapshot_hashes(result), ())

    def test_tampered_snapshot_is_detected(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(_observation(4))
        raw = self.path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(raw[0])
        payload["snapshot_evidence"][0]["text"] = "something else entirely"
        self.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        self.assertEqual(verify_snapshot_hashes(read_ledger(self.path)), (4,))

    def test_verdicts_and_costs_are_preserved(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(
            _observation(
                6,
                verifier_a="COMPLETE",
                verifier_a_evidence_ids=("ev_a",),
                verifier_b="NO_GAP",
                would_stop=True,
                calls=2,
                prompt_tokens=300,
                output_tokens=14,
                wall_seconds=2.5,
            )
        )
        record = read_ledger(self.path).checkpoints[0]
        self.assertEqual(record["verifier_a"], "COMPLETE")
        self.assertEqual(record["verifier_a_evidence_ids"], ["ev_a"])
        self.assertEqual(record["verifier_b"], "NO_GAP")
        self.assertTrue(record["would_stop"])
        self.assertEqual(record["verifier_calls"], 2)
        self.assertEqual(record["verifier_prompt_tokens"], 300)
        self.assertEqual(record["verifier_output_tokens"], 14)

    def test_continue_and_gap_records(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(
            _observation(4, verifier_a="CONTINUE", verifier_a_detail="missing: the URL")
        )
        writer.write_checkpoint(
            _observation(6, verifier_a="COMPLETE", verifier_b="GAP", blocked_by="verifier_b_gap")
        )
        result = read_ledger(self.path)
        self.assertEqual([c["action"] for c in result.checkpoints], [4, 6])
        self.assertEqual(result.would_stop_actions, ())
        self.assertEqual(result.checkpoints[1]["blocked_by"], "verifier_b_gap")

    def test_incomplete_run_is_detectable(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_run_start(request="r")
        writer.write_checkpoint(_observation(4))
        # No final record: the process died mid-run.
        self.assertFalse(read_ledger(self.path).complete)

    def test_truncated_last_line_keeps_earlier_records(self) -> None:
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(_observation(4))
        writer.write_checkpoint(_observation(6))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema_version": 1, "record": "checkpo')
        result = read_ledger(self.path)
        self.assertEqual(len(result.checkpoints), 2, "paid-for observations survive")
        self.assertEqual(result.malformed_lines, (3,))
        self.assertFalse(result.complete)

    def test_a_finished_run_does_not_vouch_for_the_next_one(self) -> None:
        """Two runs share one session file. A killed second run must not read
        as complete because the first one finished."""
        first = ShadowLedgerWriter(self.path)
        first.write_run_start(request="run one")
        first.write_run_final(stop_reason="done")

        second = ShadowLedgerWriter(self.path)
        second.write_run_start(request="run two")
        second.write_checkpoint(_observation(4))

        result = read_ledger(self.path)
        self.assertFalse(result.complete, "the killed run is not complete")
        self.assertEqual(result.run_start["request"], "run two")
        self.assertEqual(len(result.checkpoints), 1, "only this run's checkpoints")

        second.write_run_final(stop_reason="finished")
        done = read_ledger(self.path)
        self.assertTrue(done.complete)
        self.assertEqual(done.run_final["stop_reason"], "finished")

    def test_a_lost_run_start_does_not_inherit_the_previous_final(self) -> None:
        """Each record opens the file on its own, so a transient failure can
        drop `run_start` alone. Keying the scope off it would then credit the
        previous run's final record to this one."""
        first = ShadowLedgerWriter(self.path, run_id="A")
        first.write_run_start(request="one")
        first.write_run_final(stop_reason="complete", actions_executed=9)

        second = ShadowLedgerWriter(self.path, run_id="B")
        # `write_run_start` lost to a transient error; later writes succeed.
        second.write_checkpoint(_observation(4))

        result = read_ledger(self.path)
        self.assertFalse(result.complete, "run B never finished")
        self.assertIsNone(result.run_final, "run A's final must not vouch for B")
        self.assertEqual([c["action"] for c in result.checkpoints], [4])

    def test_records_without_a_run_id_are_unattributable(self) -> None:
        # A file predating run ids, or otherwise unattributable: nothing may be
        # credited to a specific run rather than guessing.
        self.path.write_text(
            json.dumps(
                {"schema_version": LEDGER_SCHEMA_VERSION, "record": "run_final",
                 "stop_reason": "complete"}
            )
            + "\n",
            encoding="utf-8",
        )
        result = read_ledger(self.path)
        self.assertFalse(result.complete)
        self.assertIsNone(result.run_final)
        self.assertEqual(result.checkpoints, ())

    def test_checkpoints_are_scoped_to_their_own_run(self) -> None:
        first = ShadowLedgerWriter(self.path)
        first.write_run_start(request="one")
        first.write_checkpoint(_observation(4))
        first.write_checkpoint(_observation(6))
        first.write_run_final(stop_reason="done")
        second = ShadowLedgerWriter(self.path)
        second.write_run_start(request="two")
        second.write_checkpoint(_observation(8))
        second.write_run_final(stop_reason="done")
        result = read_ledger(self.path)
        self.assertEqual([c["action"] for c in result.checkpoints], [8])

    def test_unknown_schema_version_is_refused_not_guessed(self) -> None:
        self.path.write_text(
            json.dumps({"schema_version": 999, "record": "checkpoint", "action": 4}) + "\n",
            encoding="utf-8",
        )
        result = read_ledger(self.path)
        self.assertEqual(result.checkpoints, ())
        self.assertEqual(result.unsupported_versions, (999,))

    def test_verbose_answer_is_truncated_where_it_is_parsed(self) -> None:
        """The bound must be enforced at the source, not merely asserted here.

        Written so that removing the truncation in `evaluate_completion_shadow`
        fails this test: the previous version compared against a limit it had
        itself produced, and survived deleting the bound entirely.
        """
        from orbit.runtime.completion_shadow import (
            MAX_PERSISTED_RAW_CHARS,
            CompletionSnapshot,
            evaluate_completion_shadow,
        )

        hostile = ("<think>" + "x" * 200 + "</think>") * 500

        class _Response:
            content = "CONTINUE missing: " + hostile
            prompt_tokens = 10
            completion_tokens = 4

        observation = evaluate_completion_shadow(
            action=4,
            snapshot=CompletionSnapshot("r", (), (), "d" * 64),
            ask=lambda _i, _r: _Response(),
            active_evidence_ids=set(),
            reattest=lambda _i: "raw",
        )
        self.assertLessEqual(len(observation.verifier_a_raw), MAX_PERSISTED_RAW_CHARS)
        self.assertLessEqual(len(observation.verifier_a_detail), MAX_PERSISTED_RAW_CHARS)

        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(observation)
        record = read_ledger(self.path).checkpoints[0]
        self.assertLessEqual(len(record["verifier_a_raw"]), MAX_PERSISTED_RAW_CHARS)

    def test_verifier_b_answer_is_bounded_too(self) -> None:
        from orbit.runtime.completion_shadow import (
            MAX_PERSISTED_RAW_CHARS,
            CompletionSnapshot,
            evaluate_completion_shadow,
        )

        live = "ev_000000000000_0000000000000000"
        answers = [f"COMPLETE evidence: {live}", "GAP missing: " + "y" * 100_000]

        class _Response:
            def __init__(self, text):
                self.content = text
                self.prompt_tokens = 10
                self.completion_tokens = 4

        seen: list[str] = []

        def ask(_instruction, _rendered):
            seen.append(_instruction)
            return _Response(answers[len(seen) - 1])

        observation = evaluate_completion_shadow(
            action=4,
            snapshot=CompletionSnapshot("r", (), (), "d" * 64),
            ask=ask,
            active_evidence_ids={live},
            reattest=lambda _i: "raw",
        )
        self.assertEqual(observation.verifier_b, "GAP")
        self.assertLessEqual(len(observation.verifier_b_raw), MAX_PERSISTED_RAW_CHARS)
        self.assertLessEqual(len(observation.verifier_b_detail), MAX_PERSISTED_RAW_CHARS)

    def test_artifact_handles_are_capped(self) -> None:
        from orbit.runtime.completion_shadow import MAX_SNAPSHOT_ARTIFACTS

        record = type(
            "R",
            (),
            {
                "evidence_id": "ev_a",
                "metadata": {"artifacts": [{"handle": f"/w/{i}"} for i in range(600)]},
            },
        )()
        snapshot = build_snapshot(
            request="r", records=[record], load_raw=lambda _i: "text"
        )
        self.assertEqual(len(snapshot.artifacts), MAX_SNAPSHOT_ARTIFACTS)
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(
            ShadowObservation(
                action=4, snapshot_digest=snapshot.digest, snapshot=snapshot
            )
        )
        stored = read_ledger(self.path).checkpoints[0]
        self.assertEqual(len(stored["snapshot_artifacts"]), MAX_SNAPSHOT_ARTIFACTS)
        self.assertEqual(verify_snapshot_hashes(read_ledger(self.path)), ())

    def test_request_is_bounded_in_the_snapshot(self) -> None:
        from orbit.runtime.completion_shadow import MAX_SNAPSHOT_REQUEST_CHARS

        snapshot = build_snapshot(
            request="q" * (MAX_SNAPSHOT_REQUEST_CHARS * 10),
            records=[],
            load_raw=lambda _i: "",
        )
        self.assertEqual(len(snapshot.request), MAX_SNAPSHOT_REQUEST_CHARS)
        writer = ShadowLedgerWriter(self.path)
        writer.write_checkpoint(
            ShadowObservation(
                action=4, snapshot_digest=snapshot.digest, snapshot=snapshot
            )
        )
        record = read_ledger(self.path).checkpoints[0]
        self.assertEqual(len(record["request"]), MAX_SNAPSHOT_REQUEST_CHARS)
        self.assertEqual(verify_snapshot_hashes(read_ledger(self.path)), ())

    def test_write_failure_is_recorded_not_raised(self) -> None:
        writer = ShadowLedgerWriter(Path("/proc/orbit-cannot-exist/run.jsonl"))
        self.assertFalse(writer.write_checkpoint(_observation(4)))
        self.assertTrue(writer.failures)

    def test_serialization_failure_is_recorded_not_raised(self) -> None:
        writer = ShadowLedgerWriter(self.path)

        class Unserializable:
            pass

        self.assertFalse(writer.write_run_start(bad=Unserializable()))
        self.assertTrue(writer.failures)

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        # Ordinary CLI interrupt semantics matter more than a diagnostic line.
        writer = ShadowLedgerWriter(self.path)
        with mock.patch.object(Path, "open", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                writer.write_checkpoint(_observation(4))

    def test_disabled_writer_writes_nothing(self) -> None:
        writer = ShadowLedgerWriter(self.path, enabled=False)
        self.assertFalse(writer.write_checkpoint(_observation(4)))
        self.assertFalse(self.path.exists())

    def test_records_are_deterministic(self) -> None:
        # Same run id, same input, same bytes. The id itself is deliberately
        # unique per run, so it is held fixed here rather than compared.
        one = Path(self._dir.name) / "a.jsonl"
        two = Path(self._dir.name) / "b.jsonl"
        ShadowLedgerWriter(one, run_id="fixed").write_checkpoint(_observation(4))
        ShadowLedgerWriter(two, run_id="fixed").write_checkpoint(_observation(4))
        self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_each_writer_gets_its_own_run_id(self) -> None:
        self.assertNotEqual(
            ShadowLedgerWriter(self.path).run_id,
            ShadowLedgerWriter(self.path).run_id,
        )


class InLoopLedgerTests(AutonomousTestBase):
    def _ledger(self, runtime) -> Path:
        return ledger_path_for_evidence_root(runtime.evidence_store.root)

    def test_shadow_off_writes_nothing(self) -> None:
        backend = _CountingBackend(_script())
        runtime = self.runtime(backend)
        os.environ.pop(SHADOW_ENV, None)
        runtime.run_autonomous("go", finalize=False)
        self.assertFalse(self._ledger(runtime).exists())

    def test_run_writes_checkpoints_and_a_final_record(self) -> None:
        backend = _CountingBackend(
            _long_script(), verifier_answers=["CONTINUE missing: the payload"] * 12
        )
        runtime = self.runtime(backend)
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            run = runtime.run_autonomous("analyze this", finalize=False)
        result = read_ledger(self._ledger(runtime))
        self.assertIsNotNone(result.run_start)
        self.assertTrue(result.checkpoints)
        self.assertTrue(result.complete)
        self.assertEqual(verify_snapshot_hashes(result), ())
        self.assertEqual(result.run_final["stop_reason"], run.stop_reason)
        self.assertEqual(result.run_final["actions_executed"], run.actions_executed)
        self.assertEqual(result.run_final["model_calls"], run.model_calls)
        self.assertEqual(result.verifier_calls, run.completion_shadow.calls)

    def test_checkpoint_actions_match_the_schedule(self) -> None:
        backend = _CountingBackend(
            _long_script(), verifier_answers=["CONTINUE missing: x"] * 12
        )
        runtime = self.runtime(backend)
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            run = runtime.run_autonomous("go", finalize=False)
        result = read_ledger(self._ledger(runtime))
        self.assertEqual(
            [c["action"] for c in result.checkpoints],
            [o.action for o in run.completion_shadow.observations],
        )

    def test_ledger_failure_cannot_change_the_trajectory(self) -> None:
        trajectories = {}
        for label, broken in (("ok", False), ("broken", True)):
            base = AutonomousTestBase("run")
            base.setUp()
            self.addCleanup(base._dir.cleanup)
            backend = _CountingBackend(
                _long_script(), verifier_answers=["CONTINUE missing: x"] * 12
            )
            runtime = base.runtime(backend)
            with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
                if broken:
                    with mock.patch.object(
                        ShadowLedgerWriter, "_append", side_effect=OSError("disk full")
                    ):
                        run = runtime.run_autonomous("go", finalize=False)
                else:
                    run = runtime.run_autonomous("go", finalize=False)
            trajectories[label] = _trajectory(run)
        self.assertEqual(trajectories["ok"], trajectories["broken"])

    def test_ledger_carries_no_model_facing_content(self) -> None:
        backend = _CountingBackend(
            _long_script(), verifier_answers=["CONTINUE missing: x"] * 12
        )
        runtime = self.runtime(backend)
        with mock.patch.dict("os.environ", {SHADOW_ENV: "1"}, clear=False):
            runtime.run_autonomous("go", finalize=False)
        # The claim this defends is that the ledger is write-only: it never
        # re-enters a prompt. It deliberately does NOT claim reasoning markup
        # is filtered out of a verifier's own answer -- that is bounded to 400
        # characters, and `test_verbose_answer_is_truncated_where_it_is_parsed`
        # is what pins the bound.
        raw = self._ledger(runtime).read_text(encoding="utf-8")
        self.assertTrue(raw)
        for message in runtime.messages:
            content = message.get("content")
            if isinstance(content, str) and content:
                self.assertNotIn("completion-shadow.jsonl", content)
                self.assertNotIn("snapshot_sha256", content)


if __name__ == "__main__":
    unittest.main()

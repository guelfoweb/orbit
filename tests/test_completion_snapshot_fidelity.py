"""Snapshot fidelity: does the verifier see the whole of its input, or less."""

from __future__ import annotations

import unittest
from unittest import mock

from orbit.runtime.completion_shadow import (
    MAX_SNAPSHOT_ARTIFACTS,
    MAX_SNAPSHOT_CHARS_PER_RECORD,
    MAX_SNAPSHOT_RECORDS,
    MAX_SNAPSHOT_REQUEST_CHARS,
    CompletionSnapshot,
    build_snapshot,
    evaluate_completion_shadow,
)


class _Record:
    def __init__(self, evidence_id: str, artifacts=()) -> None:
        self.evidence_id = evidence_id
        self.metadata = {"artifacts": [{"handle": h} for h in artifacts]}


def _snapshot(records, raw, request="r"):
    return build_snapshot(request=request, records=records, load_raw=lambda _i: raw)


class LosslessTests(unittest.TestCase):
    def test_everything_fits_is_lossless(self) -> None:
        s = _snapshot([_Record("ev_a")], "short")
        self.assertTrue(s.fidelity.lossless)
        self.assertEqual(s.fidelity.reasons, ())
        self.assertEqual(s.fidelity.active_records_total, 1)
        self.assertEqual(s.fidelity.active_records_included, 1)

    def test_record_exactly_at_the_cap_is_lossless(self) -> None:
        # The boundary is the interesting case: equal is not truncated.
        s = _snapshot([_Record("ev_a")], "x" * MAX_SNAPSHOT_CHARS_PER_RECORD)
        self.assertTrue(s.fidelity.lossless)
        self.assertEqual(s.fidelity.truncated_records, ())


class LossyTests(unittest.TestCase):
    def test_one_char_over_the_cap_is_lossy(self) -> None:
        s = _snapshot([_Record("ev_a")], "x" * (MAX_SNAPSHOT_CHARS_PER_RECORD + 1))
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("record_content_truncated", s.fidelity.reasons)
        t = s.fidelity.truncated_records[0]
        self.assertEqual(t.authoritative_chars, MAX_SNAPSHOT_CHARS_PER_RECORD + 1)
        self.assertEqual(t.included_chars, MAX_SNAPSHOT_CHARS_PER_RECORD)

    def test_omitted_record_is_lossy_and_counted(self) -> None:
        records = [_Record(f"ev_{i}") for i in range(MAX_SNAPSHOT_RECORDS + 3)]
        s = _snapshot(records, "short")
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("record_count_cap", s.fidelity.reasons)
        self.assertEqual(s.fidelity.omitted_record_count, 3)
        self.assertEqual(s.fidelity.active_records_included, MAX_SNAPSHOT_RECORDS)

    def test_artifact_overflow_is_lossy(self) -> None:
        handles = [f"/w/{i}" for i in range(MAX_SNAPSHOT_ARTIFACTS + 5)]
        s = _snapshot([_Record("ev_a", handles)], "short")
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("artifact_count_cap", s.fidelity.reasons)
        self.assertEqual(s.fidelity.omitted_artifact_count, 5)

    def test_artifact_lost_with_an_evicted_record_is_reported(self) -> None:
        # The cascade: a handle can vanish because its record was evicted,
        # which is a different reason from the artifact cap firing.
        records = [_Record("ev_old", ["/w/gone"])] + [
            _Record(f"ev_{i}") for i in range(MAX_SNAPSHOT_RECORDS)
        ]
        s = _snapshot(records, "short")
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("artifact_omitted_with_record", s.fidelity.reasons)
        self.assertEqual(s.fidelity.artifacts_total, 1)
        self.assertEqual(s.fidelity.artifacts_included, 0)

    def test_a_record_dropped_for_a_missing_id_is_lossy(self) -> None:
        """A record can be skipped for its own reason, not just a cap.

        Without this the snapshot would be one record short and still call
        itself whole, which is the one thing `lossless` must never do.
        """
        s = _snapshot([_Record(""), _Record("ev_a")], "short")
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("record_unusable", s.fidelity.reasons)
        self.assertEqual(s.fidelity.omitted_record_count, 1)
        self.assertEqual(s.fidelity.active_records_total, 2)
        self.assertEqual(s.fidelity.active_records_included, 1)

    def test_request_truncation_is_lossy(self) -> None:
        s = _snapshot([_Record("ev_a")], "short", "q" * (MAX_SNAPSHOT_REQUEST_CHARS + 1))
        self.assertFalse(s.fidelity.lossless)
        self.assertIn("request_truncated", s.fidelity.reasons)
        self.assertTrue(s.fidelity.request_truncated)

    def test_multiple_simultaneous_reasons(self) -> None:
        records = [_Record(f"ev_{i}") for i in range(MAX_SNAPSHOT_RECORDS + 2)]
        s = _snapshot(records, "x" * 5000, "q" * (MAX_SNAPSHOT_REQUEST_CHARS + 1))
        self.assertFalse(s.fidelity.lossless)
        for reason in ("record_count_cap", "record_content_truncated", "request_truncated"):
            self.assertIn(reason, s.fidelity.reasons)


class ModelVisibleBytesUnchangedTests(unittest.TestCase):
    def test_fidelity_does_not_alter_the_snapshot_or_its_hash(self) -> None:
        """Fidelity describes the snapshot; it must not be part of it."""
        one = _snapshot([_Record("ev_a")], "x" * 5000)
        two = _snapshot([_Record("ev_a")], "x" * 5000)
        self.assertEqual(one.digest, two.digest)
        self.assertEqual(one.render(), two.render())
        # The rendered text is what the verifiers read; nothing from fidelity
        # may appear in it.
        for token in ("lossless", "truncated", "authoritative_chars", "fidelity"):
            self.assertNotIn(token, one.render())

    def test_digest_is_stable_against_a_known_value(self) -> None:
        # Pins that adding fidelity did not move the hash for a fixed input.
        s = build_snapshot(request="r", records=[_Record("ev_a")], load_raw=lambda _i: "text")
        import hashlib
        import json

        expected = hashlib.sha256(
            json.dumps(
                {"request": "r", "evidence": [["ev_a", "text"]], "artifacts": []},
                ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(s.digest, expected)


class FidelityGateTests(unittest.TestCase):
    LIVE = "ev_000000000000_0000000000000000"

    def _run(self, snapshot):
        answers = [f"COMPLETE evidence: {self.LIVE}", "NO_GAP"]
        seen: list[str] = []

        class _R:
            def __init__(self, text):
                self.content = text
                self.prompt_tokens = 10
                self.completion_tokens = 2

        def ask(_i, _r):
            seen.append(_i)
            return _R(answers[len(seen) - 1])

        return evaluate_completion_shadow(
            action=4, snapshot=snapshot, ask=ask,
            active_evidence_ids={self.LIVE}, reattest=lambda _i: "raw",
        )

    def test_lossless_snapshot_permits_would_stop(self) -> None:
        obs = self._run(_snapshot([_Record("ev_a")], "short"))
        self.assertTrue(obs.would_stop)
        self.assertIsNone(obs.blocked_by)

    def test_lossy_snapshot_forces_would_stop_false(self) -> None:
        obs = self._run(_snapshot([_Record("ev_a")], "x" * 5000))
        self.assertFalse(obs.would_stop)
        self.assertEqual(obs.blocked_by, "snapshot_lossy")

    def test_absent_fidelity_does_not_fabricate_a_block(self) -> None:
        # A snapshot built without fidelity (an older path) is not treated as
        # lossy; it is simply not gated, which is the pre-existing behaviour.
        obs = self._run(CompletionSnapshot("r", (("ev_a", "t"),), (), "d" * 64))
        self.assertTrue(obs.would_stop)

    def test_gate_runs_after_the_existing_gates(self) -> None:
        # A lossy snapshot whose verifiers disagree is still blocked by the
        # disagreement, not silently relabelled.
        answers = ["CONTINUE missing: x"]

        class _R:
            content = answers[0]
            prompt_tokens = 10
            completion_tokens = 2

        obs = evaluate_completion_shadow(
            action=4, snapshot=_snapshot([_Record("ev_a")], "x" * 5000),
            ask=lambda _i, _r: _R(), active_evidence_ids=set(), reattest=lambda _i: "raw",
        )
        self.assertEqual(obs.blocked_by, "verifier_a_continue")


if __name__ == "__main__":
    unittest.main()

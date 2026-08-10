from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

from orbit.qualification.fixtures import FixtureError, load_fixture_set, load_fixture_text
from orbit.qualification.runner import QualificationRunner
from orbit.qualification.schema import (
    ComparisonMode,
    FixtureObservation,
    LifecycleOutcome,
    RunProvenance,
    StateReuseEvidence,
    Status,
)
from orbit.qualification.validators import compare_fixture_results, validate_observation
from scripts.orbit_qualify_lifecycle import LifecycleExecutor, _port_free


ROOT = Path(__file__).parents[1]


def observation(evidence: StateReuseEvidence | None, *, finish_reason: str = "stop") -> FixtureObservation:
    return FixtureObservation(
        route=None,
        tool_calls=(),
        executed_tools=(),
        final_output="",
        finish_reason=finish_reason,
        model_call_count=1,
        retry_count=0,
        calls=(),
        artifact=None,
        lifecycle=LifecycleOutcome(True, "clean"),
        peak_rss_bytes=None,
        state_reuse=evidence,
    )


def evidence(operation: str, **changes) -> StateReuseEvidence:
    base = StateReuseEvidence(
        operation=operation,
        initialized_before=True,
        initialized_after=True,
        invalidated=True,
        recapture_observed=True,
        capture_count=3,
        restore_count=20,
        fallback_count=0,
        invalidation_count=2,
        cached_tokens_after=0,
        checkpoint_size_after=0,
        partial_state_accepted=False,
        cancellation_observed=True,
        restore_rejected=True,
        fallback_succeeded=True,
        fallback_attempts=1,
        rss_start_bytes=10_000,
        rss_end_bytes=10_100,
        rss_peak_bytes=10_100,
        rss_tolerance_bytes=1_000,
        rss_samples=(10_000, 10_050, 10_100, 10_100),
        process_pid=123,
        process_exit_code=0,
        port_released=True,
        residual_state=(),
    )
    defaults = {
        "reset_invalidation": {"checkpoint_size_after": 75_000_000},
        "restore_failure_fallback": {"initialized_after": False, "restore_count": 0, "fallback_count": 1},
        "repeated_restore_rss": {"cached_tokens_after": 768, "checkpoint_size_after": 75_000_000},
        "teardown_cleanup": {"initialized_after": False},
    }
    return replace(replace(base, **defaults.get(operation, {})), **changes)


class LifecycleFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_set = load_fixture_set(ROOT / "qualification/fixtures/lifecycle-v1.json")
        cls.fixtures = {item.name: item for item in cls.fixture_set.fixtures}

    def test_fixture_contract_is_strict_and_capability_bound(self) -> None:
        self.assertEqual(
            tuple(self.fixtures),
            (
                "reset_invalidation",
                "cancellation",
                "restore_failure_fallback",
                "repeated_restore_rss",
                "teardown_cleanup",
            ),
        )
        self.assertTrue(all(item.capability == "state_reuse" for item in self.fixtures.values()))
        raw = (ROOT / "qualification/fixtures/lifecycle-v1.json").read_text()
        with self.assertRaisesRegex(FixtureError, "unknown_key"):
            load_fixture_text(raw.replace('"operation": "reset_invalidation"', '"operation": "reset_invalidation", "retry": true'))
        with self.assertRaisesRegex(FixtureError, "invalid_lifecycle_operation"):
            load_fixture_text(raw.replace('"operation": "reset_invalidation"', '"operation": "unknown"'))

    def test_reset_requires_invalidation_and_cold_recapture(self) -> None:
        fixture = self.fixtures["reset_invalidation"]
        self.assertIs(validate_observation(fixture, observation(evidence("reset_invalidation"))).status, Status.PASS)
        failed = validate_observation(
            fixture,
            observation(evidence("reset_invalidation", invalidated=False, recapture_observed=False)),
        )
        self.assertIs(failed.status, Status.FAIL)
        self.assertEqual(failed.reason.code, "stale_state_after_reset")
        stale = observation(evidence("reset_invalidation", cached_tokens_after=64))
        self.assertEqual(validate_observation(fixture, stale).reason.code, "stale_state_after_reset")
        empty = observation(evidence("reset_invalidation", checkpoint_size_after=0))
        self.assertEqual(validate_observation(fixture, empty).reason.code, "stale_state_after_reset")
        fallback = observation(evidence("reset_invalidation", fallback_count=1))
        self.assertEqual(validate_observation(fixture, fallback).reason.code, "stale_state_after_reset")
        incomplete = observation(evidence("reset_invalidation", restore_count=None))
        self.assertIs(validate_observation(fixture, incomplete).status, Status.TECHNICAL_STOP)

    def test_cancellation_rejects_partial_or_residual_state(self) -> None:
        fixture = self.fixtures["cancellation"]
        clean = observation(evidence("cancellation", initialized_after=False), finish_reason="cancelled")
        self.assertIs(validate_observation(fixture, clean).status, Status.PASS)
        partial = replace(clean, state_reuse=evidence("cancellation", initialized_after=False, partial_state_accepted=True))
        self.assertEqual(validate_observation(fixture, partial).reason.code, "partial_state_accepted")
        residue = replace(clean, state_reuse=evidence("cancellation", initialized_after=False, residual_state=("checkpoint.tmp",)))
        self.assertEqual(validate_observation(fixture, residue).reason.code, "lifecycle_residue")

    def test_cancellation_worker_error_is_not_reported_as_model_failure(self) -> None:
        executor = LifecycleExecutor(None, None, "http://unused", "orbit-qwen36-native-v1")
        executor._state = mock.Mock(return_value={"initialized": True})
        executor._reuse_call = mock.Mock(side_effect=RuntimeError("boom"))
        executor._wait_in_flight = mock.Mock()
        executor._cancel = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "cancellation worker failed"):
            executor.execute(self.fixtures["cancellation"], Path("/tmp"))

    def test_restore_failure_requires_one_safe_fallback_without_loop(self) -> None:
        fixture = self.fixtures["restore_failure_fallback"]
        self.assertIs(
            validate_observation(
                fixture, observation(evidence("restore_failure_fallback"))
            ).status,
            Status.PASS,
        )
        partial = evidence("restore_failure_fallback", partial_state_accepted=True)
        self.assertEqual(validate_observation(fixture, observation(partial)).reason.code, "partial_restore_accepted")
        loop = evidence("restore_failure_fallback", fallback_attempts=2, fallback_count=2)
        self.assertEqual(validate_observation(fixture, observation(loop)).reason.code, "fallback_loop")
        counted = evidence("restore_failure_fallback", restore_count=1)
        self.assertEqual(validate_observation(fixture, observation(counted)).reason.code, "restore_failure_not_rejected")
        failed = evidence("restore_failure_fallback", fallback_succeeded=False)
        self.assertEqual(validate_observation(fixture, observation(failed)).reason.code, "cold_fallback_failed")

    def test_restore_hook_rejects_skipped_test(self) -> None:
        executor = LifecycleExecutor(None, None, "http://unused", "orbit-qwen36-native-v1")
        skipped = subprocess.CompletedProcess([], 0, "", "Ran 1 test in 0.0s\n\nOK (skipped=1)\n")
        with mock.patch("scripts.orbit_qualify_lifecycle.subprocess.run", return_value=skipped):
            with self.assertRaisesRegex(RuntimeError, "restore-failure hook failed"):
                executor._restore_hook("restore_failure_fallback")

    def test_repeated_restore_uses_explicit_bounded_rss_tolerance(self) -> None:
        fixture = self.fixtures["repeated_restore_rss"]
        self.assertIs(
            validate_observation(fixture, observation(evidence("repeated_restore_rss"))).status,
            Status.PASS,
        )
        growth = evidence(
            "repeated_restore_rss",
            rss_end_bytes=12_000,
            rss_peak_bytes=12_000,
            rss_samples=(10_000, 11_000, 11_500, 12_000),
        )
        self.assertEqual(validate_observation(fixture, observation(growth)).reason.code, "rss_growth_unbounded")
        base, tolerance = 1_000_000_000, 64 * 1024**2
        monotonic = evidence(
            "repeated_restore_rss",
            rss_start_bytes=base,
            rss_end_bytes=base + 32 * 1024**2,
            rss_peak_bytes=base + 32 * 1024**2,
            rss_tolerance_bytes=tolerance,
            rss_samples=(base, base + 8 * 1024**2, base + 16 * 1024**2, base + 32 * 1024**2),
        )
        self.assertEqual(validate_observation(fixture, observation(monotonic)).reason.code, "rss_growth_unbounded")
        staircase = replace(
            monotonic,
            rss_samples=(base, base + 16 * 1024**2, base + 16 * 1024**2, base + 32 * 1024**2),
        )
        self.assertEqual(validate_observation(fixture, observation(staircase)).reason.code, "rss_growth_unbounded")
        inconsistent = replace(monotonic, rss_peak_bytes=base)
        result = validate_observation(fixture, observation(inconsistent))
        self.assertIs(result.status, Status.TECHNICAL_STOP)
        self.assertEqual(result.reason.code, "lifecycle_evidence_invalid")
        non_finite = replace(monotonic, rss_end_bytes=float("nan"))
        self.assertIs(
            validate_observation(fixture, observation(non_finite)).status,
            Status.TECHNICAL_STOP,
        )
        incomplete = replace(monotonic, rss_end_bytes=base + 16 * 1024**2,
                             rss_peak_bytes=base + 16 * 1024**2,
                             rss_samples=(base, base + 16 * 1024**2))
        self.assertIs(
            validate_observation(fixture, observation(incomplete)).status,
            Status.TECHNICAL_STOP,
        )

    def test_teardown_requires_exit_port_release_and_no_residue(self) -> None:
        fixture = self.fixtures["teardown_cleanup"]
        self.assertIs(validate_observation(fixture, observation(evidence("teardown_cleanup"))).status, Status.PASS)
        running = evidence("teardown_cleanup", process_exit_code=None)
        self.assertEqual(validate_observation(fixture, observation(running)).reason.code, "process_not_exited")
        bound = evidence("teardown_cleanup", port_released=False)
        self.assertEqual(validate_observation(fixture, observation(bound)).reason.code, "port_not_released")
        retained = evidence("teardown_cleanup", initialized_after=True, checkpoint_size_after=1)
        self.assertEqual(validate_observation(fixture, observation(retained)).reason.code, "teardown_state_retained")

    def test_missing_or_incomplete_evidence_is_technical_stop(self) -> None:
        fixture = self.fixtures["reset_invalidation"]
        missing = validate_observation(fixture, observation(None))
        self.assertIs(missing.status, Status.TECHNICAL_STOP)
        self.assertEqual(missing.reason.code, "lifecycle_evidence_missing")
        incomplete = validate_observation(
            fixture,
            observation(evidence("reset_invalidation", invalidated=None)),
        )
        self.assertIs(incomplete.status, Status.TECHNICAL_STOP)
        self.assertEqual(incomplete.reason.code, "lifecycle_evidence_incomplete")
        invalid = validate_observation(
            fixture,
            observation(evidence("reset_invalidation", capture_count=True)),
        )
        self.assertIs(invalid.status, Status.TECHNICAL_STOP)
        self.assertEqual(invalid.reason.code, "lifecycle_evidence_invalid")

    def test_teardown_rejects_process_residue_and_malformed_pid(self) -> None:
        fixture = self.fixtures["teardown_cleanup"]
        residue = evidence("teardown_cleanup", residual_state=("process:456",))
        self.assertEqual(validate_observation(fixture, observation(residue)).reason.code, "lifecycle_residue")
        invalid = evidence("teardown_cleanup", process_pid=0)
        result = validate_observation(fixture, observation(invalid))
        self.assertIs(result.status, Status.TECHNICAL_STOP)
        self.assertEqual(result.reason.code, "lifecycle_evidence_invalid")

    def test_unsupported_state_reuse_is_not_applicable(self) -> None:
        fixture_set = load_fixture_set(ROOT / "qualification/fixtures/lifecycle-v1.json")
        profile = {
            "compatibility_profile": "orbit-gemma4-native-v1",
            "verified": True,
            "capabilities": {"route_prefix_reuse": False},
        }
        provenance = RunProvenance(
            1, fixture_set.content_hash, "git", "orbit-gemma4-native-v1", "model",
            "template", "hash", "backend", "revision", {}, {}, {},
        )
        with tempfile.TemporaryDirectory() as directory:
            run = QualificationRunner(
                fixture_set, profile, provenance, object(), Path(directory)
            ).run(("reset_invalidation",))
        self.assertIs(run.fixtures[0].status, Status.NOT_APPLICABLE)
        self.assertEqual(run.fixtures[0].reason.code, "capability_not_supported")

    def test_lifecycle_parity_compares_outcomes_not_rss_noise(self) -> None:
        fixture = self.fixtures["repeated_restore_rss"]
        baseline = validate_observation(
            fixture, observation(evidence("repeated_restore_rss"))
        )
        candidate = validate_observation(
            fixture,
            replace(
                observation(evidence(
                    "repeated_restore_rss",
                    rss_start_bytes=20_000,
                    rss_end_bytes=20_500,
                    rss_peak_bytes=21_000,
                    rss_samples=(20_000, 21_000, 20_500, 20_500),
                )),
                model_call_count=2,
            ),
        )
        parity = compare_fixture_results(
            fixture, baseline, candidate, comparison_mode=ComparisonMode.OPTIMIZATION
        )
        self.assertTrue(parity.equivalent)
        self.assertTrue(parity.performance_comparison_valid)

    def test_port_release_ignores_time_wait_but_rejects_active_listener(self) -> None:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            self.assertFalse(_port_free(port))
        self.assertTrue(_port_free(port))


if __name__ == "__main__":
    unittest.main()

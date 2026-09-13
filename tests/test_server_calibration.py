"""The bounded thread sweep: what it measures, and what it refuses to.

These run against a fake client rather than a real model. What is being pinned
is the calibrator's policy -- which candidates it tries, what it rejects, what
it does when a decode fails, what it leaves behind -- and none of that needs 20
GiB of weights to be true. The real rates are measured separately, on hardware,
and recorded in workdir/diag/autocalibration/.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_server import server_calibration as module  # noqa: E402
from orbit.native_server.server_calibration import (  # noqa: E402
    CALIBRATION_DECODE_TOKENS,
    REFERENCE_DECODE_TOKENS,
    REFERENCE_PREFILL_TOKENS,
    CandidateMeasurement,
    _turn_score,
    calibrate_threads,
    calibration_tokens,
    restore_threads,
)
from orbit.native_server.server_profile import HostTopology  # noqa: E402

DELL = HostTopology(cpu_model="hybrid", physical_cores=16, logical_cpus=16,
                    total_ram_mib=31 * 1024, swap_total_mib=2048)


class _Lib:
    """A fake llama.cpp just real enough to be timed and mis-set."""

    def __init__(self, *, rates=None, fail_at=None) -> None:
        # threads -> (prefill seconds per token, decode seconds per token)
        self.rates = rates or {}
        self.fail_at = fail_at
        self.threads_calls: list[tuple] = []
        self.clears = 0
        self.decodes = 0
        self.current = (0, 0)

    def llama_get_memory(self, ctx):
        return 1

    def llama_memory_clear(self, mem, flag):
        self.clears += 1

    def llama_set_n_threads(self, ctx, n, nb):
        self.current = (n, nb)
        self.threads_calls.append((n, nb))

    def llama_batch_get_one(self, ptr, n):
        return ("batch", n)

    def llama_decode(self, ctx, batch):
        self.decodes += 1
        if self.fail_at is not None and self.current[0] == self.fail_at:
            return 1
        per_token = self.rates.get(self.current[0], (0.0, 0.0))
        # Sleep is how a fake reports a rate. Kept microscopic: the ordering of
        # the candidates is what matters, not the absolute numbers.
        import time

        n = batch[1]
        time.sleep(per_token[0] * n if n > 1 else per_token[1])
        return 0

    def llama_synchronize(self, ctx):
        return None


class _Session:
    ctx_tgt = 12345


class _Config:
    ubatch_size = 128


class _Client:
    def __init__(self, lib) -> None:
        self.lib = type("W", (), {"lib": lib})()
        self._session = _Session()
        self.config = _Config()

    def tokenize(self, text):
        return [1, 2, 3, 4, 5, 6, 7, 8]


def _fast_decode_slow_prefill():
    """The measured shape: more threads help prefill and hurt decode."""
    return {
        6: (0.000_020, 0.000_060),
        8: (0.000_012, 0.000_110),
        12: (0.000_010, 0.000_130),
        16: (0.000_010, 0.000_200),
    }


class ScoreTests(unittest.TestCase):
    """The turn model, which is what actually picks the winner."""

    def test_the_reference_turn_is_evaluated_tokens_not_prompt_tokens(self) -> None:
        """Guards the constant whose value flips the answer on real hardware.

        A tools-on CHAT turn is 978 prompt tokens of which 768 are restored.
        Scoring against 978 picks a thread count whose decode is half as fast;
        scoring against the 210 actually evaluated does not. If someone
        "corrects" this to the prompt size, the Dell regresses ~2x.
        """
        self.assertLess(REFERENCE_PREFILL_TOKENS, 400)
        self.assertGreater(REFERENCE_DECODE_TOKENS, 0)

    def test_a_zero_rate_can_never_win(self) -> None:
        self.assertEqual(_turn_score(0.0, 10.0), 0.0)
        self.assertEqual(_turn_score(10.0, 0.0), 0.0)

    def test_decode_is_weighted_enough_to_beat_raw_prefill(self) -> None:
        # The measured Dell rows: 6 threads vs 16 threads.
        six = _turn_score(50.8, 16.4)
        sixteen = _turn_score(64.6, 5.4)
        self.assertGreater(six, sixteen)


class SweepTests(unittest.TestCase):
    def test_the_measured_shape_selects_the_decode_friendly_candidate(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        result = calibrate_threads(_Client(lib), topology=DELL)
        self.assertIsNotNone(result)
        values, table = result
        self.assertEqual(values["threads"], 6)
        self.assertEqual(values["threads_batch"], 6)
        self.assertEqual(len(table), 4)

    def test_every_candidate_evaluates_the_same_token_count(self) -> None:
        """T: no warm/cold mismatch between candidates."""
        lib = _Lib(rates=_fast_decode_slow_prefill())
        _, table = calibrate_threads(_Client(lib), topology=DELL)
        prefill = {row["prefill_tokens"] for row in table}
        decode = {row["decode_tokens"] for row in table}
        self.assertEqual(len(prefill), 1)
        self.assertEqual(decode, {CALIBRATION_DECODE_TOKENS})

    def test_the_kv_is_cleared_before_and_after_every_candidate(self) -> None:
        """T13. Nothing a benchmark did may be visible to a real request."""
        lib = _Lib(rates=_fast_decode_slow_prefill())
        calibrate_threads(_Client(lib), topology=DELL)
        self.assertEqual(lib.clears, 2 * 4)

    def test_a_decode_failure_is_recorded_not_raised(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill(), fail_at=12)
        result = calibrate_threads(_Client(lib), topology=DELL)
        self.assertIsNotNone(result)
        _, table = result
        failed = [row for row in table if row["threads"] == 12]
        self.assertTrue(failed[0]["rejected"].startswith("error:"))

    def test_every_candidate_failing_yields_no_result(self) -> None:
        """T6. The caller falls back; it is never handed a made-up winner."""
        lib = _Lib(rates=_fast_decode_slow_prefill(), fail_at=None)
        lib.llama_decode = lambda ctx, batch: 1
        self.assertIsNone(calibrate_threads(_Client(lib), topology=DELL))

    def test_a_swapping_candidate_cannot_win(self) -> None:
        """T9. However fast it looked."""
        lib = _Lib(rates={6: (0.001, 0.001), 16: (0.000_001, 0.000_001)})
        client = _Client(lib)
        real = module._swap_used_mib
        calls = {"n": 0}

        def swapping():
            # Grows only while the fastest candidate runs.
            calls["n"] += 1
            return 0 if lib.current[0] != 16 else calls["n"] * 1024

        module._swap_used_mib = swapping
        try:
            result = calibrate_threads(client, topology=DELL, candidates=[6, 16])
        finally:
            module._swap_used_mib = real
        self.assertIsNotNone(result)
        values, table = result
        self.assertEqual(values["threads"], 6)
        rejected = [r for r in table if r["threads"] == 16][0]
        self.assertIn("swap_growth", rejected["rejected"])
        self.assertEqual(rejected["score"], 0.0)

    def test_the_budget_stops_the_sweep(self) -> None:
        lib = _Lib(rates={t: (0.002, 0.002) for t in (6, 8, 12, 16)})
        _, table = calibrate_threads(
            _Client(lib), topology=DELL, budget_seconds=0.0001
        )
        self.assertLess(len(table), 4)

    def test_no_candidates_yields_no_result(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        self.assertIsNone(
            calibrate_threads(_Client(lib), topology=DELL, candidates=[])
        )

    def test_it_declines_fields_it_cannot_measure(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        self.assertIsNone(
            calibrate_threads(
                _Client(lib), topology=DELL, fields=("batch", "ubatch")
            )
        )
        self.assertEqual(lib.decodes, 0)

    def test_only_the_requested_fields_are_returned(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        values, _ = calibrate_threads(
            _Client(lib), topology=DELL, fields=("threads_batch",)
        )
        self.assertEqual(set(values), {"threads_batch"})


class TokenTests(unittest.TestCase):
    def test_the_prompt_is_deterministic_and_the_requested_length(self) -> None:
        client = _Client(_Lib())
        first = calibration_tokens(client, 100)
        second = calibration_tokens(client, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)

    def test_the_prompt_is_synthetic_not_a_route_prompt(self) -> None:
        """It must not accidentally measure the tokens the anchor holds."""
        seen: list[str] = []
        client = _Client(_Lib())
        client.tokenize = lambda text: (seen.append(text), [1, 2, 3])[1]
        calibration_tokens(client, 10)
        joined = " ".join(seen).lower()
        for marker in ("<|im_start|>", "system", "tool", "you are"):
            self.assertNotIn(marker, joined)


class RestoreTests(unittest.TestCase):
    """T14. The context must not be left on the last candidate tried."""

    def test_the_resolved_counts_are_reapplied(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        client = _Client(lib)
        calibrate_threads(client, topology=DELL)
        self.assertEqual(lib.current, (16, 16))   # last candidate, not the winner
        restore_threads(client, 6, 6)
        self.assertEqual(lib.current, (6, 6))

    def test_restoring_without_a_context_is_a_no_op(self) -> None:
        lib = _Lib()
        client = _Client(lib)
        client._session.ctx_tgt = 0
        restore_threads(client, 6, 6)
        self.assertEqual(lib.threads_calls, [])


if __name__ == "__main__":
    unittest.main()

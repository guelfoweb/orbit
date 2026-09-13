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
    TIE_MARGIN,
    WARMUP_REJECTION,
    CandidateMeasurement,
    _turn_score,
    calibrate_threads,
    calibration_tokens,
    restore_threads,
)
from orbit.native_server.server_profile import HostTopology  # noqa: E402


def _scored(table):
    """The rows that compete: everything but the warm-up (failed or not)."""
    return [
        row for row in table
        if not str(row["rejected"] or "").startswith(WARMUP_REJECTION)
    ]

DELL = HostTopology(cpu_model="hybrid", physical_cores=16, logical_cpus=16,
                    total_ram_mib=31 * 1024, swap_total_mib=2048)


class _Lib:
    """A fake llama.cpp just real enough to be timed and mis-set."""

    def __init__(self, *, rates=None, fail_at=None, cold_decodes=0, cold_factor=1.0) -> None:
        # threads -> (prefill seconds per token, decode seconds per token)
        self.rates = rates or {}
        self.fail_at = fail_at
        self.threads_calls: list[tuple] = []
        self.clears = 0
        self.decodes = 0
        self.current = (0, 0)
        # The page-in a real model pays once: the first `cold_decodes` calls
        # run `cold_factor` times slower, whichever thread count is active.
        self.cold_decodes = cold_decodes
        self.cold_factor = cold_factor

    def llama_n_threads(self, ctx):
        return self.current[0]

    def llama_n_threads_batch(self, ctx):
        return self.current[1]

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
        seconds = per_token[0] * n if n > 1 else per_token[1]
        if self.decodes <= self.cold_decodes:
            seconds *= self.cold_factor
        time.sleep(seconds)
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
        self.assertEqual(len(_scored(table)), 4)

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
        # Four candidates plus the warm-up, each cleared before and after.
        self.assertEqual(lib.clears, 2 * 5)

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
        result = calibrate_threads(
            _Client(lib), topology=DELL, budget_seconds=0.0001
        )
        # The warm-up alone exhausts a budget this small. The first timed
        # candidate still runs -- one real measurement is left behind rather
        # than none -- and everything after it is abandoned.
        self.assertIsNotNone(result)
        values, table = result
        self.assertEqual(len(_scored(table)), 1)
        self.assertEqual(values["threads"], 6)
        # warm-up + one candidate: 4 prefill steps + the decode run, twice.
        self.assertEqual(lib.decodes, 2 * (512 // 128 + CALIBRATION_DECODE_TOKENS))

    def test_the_warm_up_counts_against_the_budget(self) -> None:
        """The ceiling is a promise about wall time, not about useful work.

        Warm-up and one candidate each take ~0.1 s here; the budget is 0.15 s.
        Counted, the second candidate sees ~0.2 s elapsed and is abandoned;
        uncounted, it would see ~0.1 s and run.
        """
        per_token = 0.1 / (512 + CALIBRATION_DECODE_TOKENS)
        lib = _Lib(rates={t: (per_token, per_token) for t in (6, 8, 12, 16)})
        seen = []
        real = module.measure_candidate

        def spy(client, threads, *, tokens):
            seen.append(threads)
            return real(client, threads, tokens=tokens)

        module.measure_candidate = spy
        try:
            calibrate_threads(_Client(lib), topology=DELL, budget_seconds=0.15)
        finally:
            module.measure_candidate = real
        self.assertEqual(seen, [6, 6])

    def test_the_warm_up_row_reaches_the_event_sink_too(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        events = []
        _, table = calibrate_threads(_Client(lib), topology=DELL, on_event=events.append)
        self.assertEqual(len(events), len(table))
        self.assertEqual(events[0].rejected, WARMUP_REJECTION)

    def test_a_swapping_warm_up_keeps_its_reason_beside_the_marker(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        real = module._swap_used_mib
        calls = {"n": 0}

        def swapping():
            calls["n"] += 1
            return 1024 if calls["n"] == 2 else 0   # after the warm-up only

        module._swap_used_mib = swapping
        try:
            values, table = calibrate_threads(_Client(lib), topology=DELL)
        finally:
            module._swap_used_mib = real
        self.assertTrue(table[0]["rejected"].startswith(WARMUP_REJECTION + ":swap_growth"))
        self.assertEqual(values["threads"], 6)

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


class WarmUpTests(unittest.TestCase):
    """The first candidate must not lose for being first.

    Measured on the Dell after a fresh boot: candidate 1 (6 threads) ran with
    17,957 major page faults at 25 tok/s prefill, candidate 2 (8 threads) with
    1,903 at 47 tok/s, and 8 was cached as the winner. Warm, the same 6
    measures 41 tok/s and wins. These pin the untimed pass that pays the
    page-in before anything is scored.
    """

    def _cold_lib(self):
        # Every decode in the first candidate's worth of calls is 3x slower:
        # 512 prefill tokens in ubatch-128 steps, then the decode run.
        first_candidate_decodes = 512 // 128 + CALIBRATION_DECODE_TOKENS
        return _Lib(
            rates=_fast_decode_slow_prefill(),
            cold_decodes=first_candidate_decodes,
            cold_factor=3.0,
        )

    def test_a_cold_first_candidate_cannot_flip_the_winner(self) -> None:
        lib = self._cold_lib()
        values, table = calibrate_threads(_Client(lib), topology=DELL)
        self.assertEqual(values["threads"], 6)

    def test_the_warm_up_is_recorded_marked_and_never_wins(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        _, table = calibrate_threads(_Client(lib), topology=DELL)
        warm = [row for row in table if row["rejected"] == WARMUP_REJECTION]
        self.assertEqual(len(warm), 1)
        self.assertEqual(warm[0]["score"], 0.0)
        self.assertEqual(table[0]["rejected"], WARMUP_REJECTION)
        # It measured the same work as a real candidate.
        self.assertEqual(warm[0]["prefill_tokens"], table[1]["prefill_tokens"])
        self.assertEqual(warm[0]["decode_tokens"], CALIBRATION_DECODE_TOKENS)

    def test_the_warm_up_uses_the_first_candidate_and_walks_the_same_path(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        calibrate_threads(_Client(lib), topology=DELL)
        self.assertEqual(lib.threads_calls[0], (6, 6))
        self.assertEqual([c[0] for c in lib.threads_calls], [6, 6, 8, 12, 16])

    def test_without_the_warm_up_the_cold_candidate_would_have_lost(self) -> None:
        """The control: the fake really does reproduce the defect."""
        lib = self._cold_lib()
        real = module.measure_candidate
        calls = {"n": 0}

        def skip_warm_up(client, threads, *, tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                # Stand in for a warm-up that does not touch the weights.
                return CandidateMeasurement(
                    threads=threads, prefill_tokens=len(tokens),
                    decode_tokens=CALIBRATION_DECODE_TOKENS, prefill_seconds=1,
                    decode_seconds=1, prefill_tps=1, decode_tps=1, rss_mib=0,
                    swap_used_mib=0, swap_growth_mib=0, score=0.0,
                )
            return real(client, threads, tokens=tokens)

        module.measure_candidate = skip_warm_up
        try:
            values, _ = calibrate_threads(_Client(lib), topology=DELL)
        finally:
            module.measure_candidate = real
        self.assertNotEqual(values["threads"], 6)

    def test_a_failing_warm_up_does_not_abort_the_sweep(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill())
        real_decode = lib.llama_decode
        state = {"n": 0}

        def flaky(ctx, batch):
            state["n"] += 1
            if state["n"] == 1:
                return 1  # the warm-up's first decode fails
            return real_decode(ctx, batch)

        lib.llama_decode = flaky
        events = []
        result = calibrate_threads(_Client(lib), topology=DELL, on_event=events.append)
        self.assertIsNotNone(result)
        values, table = result
        self.assertTrue(table[0]["rejected"].startswith(WARMUP_REJECTION))
        self.assertEqual(values["threads"], 6)
        # The failed row reached the sink like every other one.
        self.assertEqual(len(events), len(table))

    def test_a_failing_timed_candidate_reaches_the_event_sink_too(self) -> None:
        lib = _Lib(rates=_fast_decode_slow_prefill(), fail_at=12)
        events = []
        _, table = calibrate_threads(_Client(lib), topology=DELL, on_event=events.append)
        self.assertEqual(len(events), len(table))
        self.assertTrue(any(str(e.rejected).startswith("error:") for e in events))


class TieBreakTests(unittest.TestCase):
    """More threads must earn their place by more than the noise."""

    def _rows(self, scores):
        return [
            CandidateMeasurement(
                threads=t, prefill_tokens=512, decode_tokens=24, prefill_seconds=1,
                decode_seconds=1, prefill_tps=1, decode_tps=1, rss_mib=0,
                swap_used_mib=0, swap_growth_mib=0, score=s,
            )
            for t, s in scores.items()
        ]

    def _winner(self, scores):
        rows = self._rows(scores)
        real = module.measure_candidate
        module.measure_candidate = lambda client, threads, *, tokens: next(
            r for r in rows if r.threads == threads
        )
        try:
            values, _ = calibrate_threads(
                _Client(_Lib()), topology=DELL, candidates=list(scores)
            )
        finally:
            module.measure_candidate = real
        return values["threads"]

    def test_within_the_margin_the_fewer_threads_win(self) -> None:
        # The Dell, warm: 6 -> 0.1092, 8 -> 0.1082, 12 -> 0.1047.
        self.assertEqual(self._winner({6: 0.1092, 8: 0.1082, 12: 0.1047}), 6)
        # And when the larger count edges ahead by less than the margin.
        self.assertEqual(self._winner({6: 0.100, 8: 0.105, 12: 0.108}), 6)

    def test_beyond_the_margin_the_better_candidate_wins(self) -> None:
        self.assertEqual(self._winner({6: 0.100, 8: 0.125}), 8)
        self.assertEqual(self._winner({6: 0.050, 12: 0.100, 16: 0.060}), 12)

    def test_the_margin_is_a_fraction_of_the_top_score(self) -> None:
        top = 0.2
        just_inside = top * (1.0 - TIE_MARGIN) + 1e-9
        just_outside = top * (1.0 - TIE_MARGIN) - 1e-6
        self.assertEqual(self._winner({6: just_inside, 16: top}), 6)
        self.assertEqual(self._winner({6: just_outside, 16: top}), 16)

    def test_the_boundary_is_inclusive_across_float_rounding(self) -> None:
        """A score written as exactly 90 % of top is inside, whatever the ulps say."""
        for top in (0.001, 0.1, 0.1092, 0.2, 0.3, 0.7):
            self.assertEqual(self._winner({6: top * 0.9, 16: top}), 6, top)

    def test_the_margin_is_bounded(self) -> None:
        """Wide enough for ~5% repeat noise, narrow enough that a real gain wins."""
        self.assertGreaterEqual(TIE_MARGIN, 0.05)
        self.assertLessEqual(TIE_MARGIN, 0.15)


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
        self.assertEqual((lib.llama_n_threads(1), lib.llama_n_threads_batch(1)), (16, 16))
        restore_threads(client, 6, 6)
        self.assertEqual(lib.current, (6, 6))
        self.assertEqual((lib.llama_n_threads(1), lib.llama_n_threads_batch(1)), (6, 6))

    def test_restoring_without_a_context_is_a_no_op(self) -> None:
        lib = _Lib()
        client = _Client(lib)
        client._session.ctx_tgt = 0
        restore_threads(client, 6, 6)
        self.assertEqual(lib.threads_calls, [])


if __name__ == "__main__":
    unittest.main()

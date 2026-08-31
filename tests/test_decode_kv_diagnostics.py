"""Physical KV state around `llama_decode`, and ANALYSIS call correlation.

DIAG-INSTRUMENT-1. A reproducible autonomous analysis died on
`llama_decode failed during generation: 1` -- "could not find a KV slot for the
batch" -- and the diagnostics could not say why: the six ANALYSIS_STEP calls
emitted no records at all, and nothing reported the physical sequence frontier.

Two defects, tested here together because they are one missing story:

* the REPL handed ANALYSIS the raw backend while CHAT used the instrumented
  one, so `phase` / `model_call_id` correlation existed for chat and not for
  analysis;
* no decode site reported `seq_pos_min` / `seq_pos_max`, the submitted batch
  size, or the return code, so a KV-slot refusal left no evidence.

These tests pin the diagnostics, not a fix for the underlying failure. Nothing
here asserts that a decode succeeds -- only that when one happens, its physical
state is recorded, and that a disabled run records nothing and queries nothing.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import kv_diag as native_diag  # noqa: E402
from orbit.runtime.kv_diag import instrument_backend  # noqa: E402


class _CountingLib:
    """A libllama stand-in that counts every native KV query it receives."""

    def __init__(self, *, seq_min: int = 0, seq_max: int = 100) -> None:
        self.seq_min = seq_min
        self.seq_max = seq_max
        self.calls: list[str] = []

    def llama_get_memory(self, ctx):
        self.calls.append("get_memory")
        return "memory-handle"

    def llama_memory_seq_pos_min(self, memory, seq_id):
        self.calls.append("seq_pos_min")
        return self.seq_min

    def llama_memory_seq_pos_max(self, memory, seq_id):
        self.calls.append("seq_pos_max")
        return self.seq_max


class _DiagFile:
    """Enable diagnostics into a fresh file for the duration of a block."""

    def __enter__(self):
        handle = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        handle.close()
        self.path = handle.name
        self._env = mock.patch.dict(
            os.environ, {"ORBIT_KV_DIAG": "1", "ORBIT_KV_DIAG_FILE": self.path}
        )
        self._env.start()
        return self

    def __exit__(self, *exc):
        self._env.stop()
        return False

    def records(self) -> list[dict]:
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class DecodeKvStateTests(unittest.TestCase):
    """What the emitter records, and from which context."""

    def test_a_successful_generation_decode_records_physical_state(self) -> None:
        lib = _CountingLib(seq_min=0, seq_max=100)
        with _DiagFile() as diag:
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="ctx-tgt", lib=lib, ctx_capacity=8192,
                batch_tokens=1, decode_rc=0, iteration=7,
            )
            record = diag.records()[-1]

        self.assertEqual(record["event"], "kv_diag_decode_kv_state")
        self.assertEqual(record["stage"], "generation")
        self.assertEqual(record["decode_rc"], 0)
        self.assertEqual(record["iteration"], 7)
        self.assertEqual(record["batch_tokens"], 1)
        self.assertEqual(record["ctx_capacity"], 8192)
        self.assertEqual(record["seq_pos_min"], 0)
        self.assertEqual(record["seq_pos_max"], 100)

    def test_a_refused_batch_records_the_state_that_refused_it(self) -> None:
        """rc=1 must arrive WITH the physical state, not as a bare code.

        llama.cpp restores the memory on a KV-slot refusal, so the frontier read
        here is the one the batch was rejected against -- which is the whole
        point of recording it.
        """
        lib = _CountingLib(seq_min=0, seq_max=8191)
        with _DiagFile() as diag:
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="ctx-tgt", lib=lib, ctx_capacity=8192,
                batch_tokens=1, decode_rc=1, iteration=271,
            )
            record = diag.records()[-1]

        self.assertEqual(record["decode_rc"], 1)
        self.assertEqual(record["seq_pos_max"], 8191)
        self.assertEqual(record["physical_frontier"], 8192)
        self.assertEqual(
            record["remaining_physical_positions"], 0,
            "a full context must be visible as zero remaining positions",
        )

    def test_the_frontier_is_seq_pos_max_plus_one(self) -> None:
        """Positions are 0-based and inclusive, so n tokens end at n-1."""
        lib = _CountingLib(seq_min=0, seq_max=4095)
        with _DiagFile() as diag:
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="c", lib=lib, ctx_capacity=8192,
                batch_tokens=1, decode_rc=0,
            )
            record = diag.records()[-1]

        self.assertEqual(record["physical_frontier"], 4096)
        self.assertEqual(record["remaining_physical_positions"], 8192 - 4096)

    def test_the_state_is_read_from_the_context_it_was_given(self) -> None:
        """The query must target the passed context, not some other handle."""
        seen: list[object] = []

        class Recording(_CountingLib):
            def llama_get_memory(self, ctx):
                seen.append(ctx)
                return "memory-handle"

        with _DiagFile():
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="the-target-context", lib=Recording(),
                ctx_capacity=8192, batch_tokens=1, decode_rc=0,
            )

        # Two lookups, one per bound (min and max); both must target the same
        # context the caller passed.
        self.assertEqual(seen, ["the-target-context", "the-target-context"])

    def test_prefill_and_generation_records_are_distinguishable(self) -> None:
        lib = _CountingLib()
        with _DiagFile() as diag:
            native_diag.emit_decode_kv_state(
                stage="prefill", ctx="c", lib=lib, ctx_capacity=8192,
                batch_tokens=256, decode_rc=0, range_start=512, range_end=768,
            )
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="c", lib=lib, ctx_capacity=8192,
                batch_tokens=1, decode_rc=0, iteration=3,
            )
            prefill, generation = diag.records()[-2:]

        self.assertEqual(prefill["stage"], "prefill")
        self.assertEqual(prefill["batch_tokens"], 256)
        self.assertEqual((prefill["range_start"], prefill["range_end"]), (512, 768))
        self.assertEqual(generation["stage"], "generation")
        self.assertEqual(generation["batch_tokens"], 1)
        self.assertEqual(generation["iteration"], 3)

    def test_a_missing_native_symbol_degrades_to_null_not_an_exception(self) -> None:
        """Diagnostics must never be able to break a run."""
        class Bare:
            def llama_get_memory(self, ctx):
                return "memory-handle"

        with _DiagFile() as diag:
            native_diag.emit_decode_kv_state(
                stage="generation", ctx="c", lib=Bare(), ctx_capacity=8192,
                batch_tokens=1, decode_rc=0,
            )
            record = diag.records()[-1]

        self.assertIsNone(record["seq_pos_max"])
        self.assertIsNone(record["physical_frontier"])
        self.assertIsNone(record["remaining_physical_positions"])


class DiagnosticsOffTests(unittest.TestCase):
    """Disabled diagnostics must cost nothing at all."""

    def test_disabled_diagnostics_issue_no_native_kv_query(self) -> None:
        lib = _CountingLib()
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": ""}):
            self.assertFalse(native_diag.enabled())
            # The production call sites are guarded by `enabled()`; this asserts
            # the guard is what protects the native query, by showing the query
            # only ever happens through the emitter.
            if native_diag.enabled():
                native_diag.emit_decode_kv_state(
                    stage="generation", ctx="c", lib=lib, ctx_capacity=8192,
                    batch_tokens=1, decode_rc=0,
                )
        self.assertEqual(lib.calls, [], "no native KV query may run when disabled")

    def test_every_decode_call_site_is_guarded_by_the_env_check(self) -> None:
        """The OFF contract is structural: no guard, no zero-overhead claim.

        Asserted over the AST rather than by running a decode, because the
        property is "this native query is unreachable when disabled" -- which a
        passing run cannot demonstrate and a missing guard would silently break.
        """
        import ast

        source = (ROOT / "src/orbit/native_llama/client.py").read_text(encoding="utf-8")
        guarded = unguarded = 0
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.If):
                continue
            if "emit_decode_kv_state" not in ast.unparse(node):
                continue
            if "kv_diag_enabled" in ast.unparse(node.test):
                guarded += 1
            else:
                unguarded += 1

        self.assertEqual(unguarded, 0, "every decode diagnostic must be env-gated")
        self.assertEqual(guarded, 2, "generation and prefill sites are both guarded")

    def test_instrument_backend_is_a_no_op_when_disabled(self) -> None:
        class Raw:
            thinking = False

        raw = Raw()
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": ""}):
            self.assertIs(instrument_backend(raw), raw)


class AnalysisCorrelationTests(unittest.TestCase):
    """ANALYSIS must call the instrumented backend, as CHAT already does."""

    def test_the_analysis_backend_is_instrumented_when_diagnostics_are_on(self) -> None:
        from orbit.terminal.repl import Repl

        class Raw:
            thinking = False

        repl = object.__new__(Repl)
        object.__setattr__(repl, "backend", Raw())
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": "1"}):
            wrapped = repl._analysis_backend()

        self.assertEqual(type(wrapped).__name__, "_DiagnosticBackend")
        self.assertIsInstance(getattr(wrapped, "_backend"), Raw)

    def test_the_analysis_backend_is_untouched_when_diagnostics_are_off(self) -> None:
        from orbit.terminal.repl import Repl

        class Raw:
            thinking = False

        raw = Raw()
        repl = object.__new__(Repl)
        object.__setattr__(repl, "backend", raw)
        with mock.patch.dict(os.environ, {"ORBIT_KV_DIAG": ""}):
            self.assertIs(repl._analysis_backend(), raw)

    def test_both_analysis_entry_points_use_the_instrumented_backend(self) -> None:
        """Pinned at the source level: `/analysis` and the confined path both.

        A behavioural test would need a live artifact and workspace; what must
        not regress is that neither call site goes back to `self.backend`.
        """
        source = (ROOT / "src/orbit/terminal/repl.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("backend=self._analysis_backend(),"), 2)
        self.assertNotIn("                backend=self.backend,\n", source)


if __name__ == "__main__":
    unittest.main()

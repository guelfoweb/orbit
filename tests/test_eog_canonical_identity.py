"""EOG-terminated completions must leave a canonical, reusable pair.

An MTP step decodes `id_last + draft` into the target as one validate batch, and
on the boundary-committed path `prompt_tgt` is extended with all of it up front.
When a stop token appears inside the accept loop the old code jumped straight to
`done`, skipping the `n_past` resync and the `seq_rm` trims that `full_accept`
performs. That left `prompt_tgt` longer than `n_past`, `n_past` stale, and the
physical frontier above both -- so BOTH publication-gate clauses failed and the
completion published an EMPTY identity.

That was fail-closed, not corrupting: the next turn simply went cold. But EOG
inside the accept loop is the NORMAL way a turn ends, so resident reuse was
being lost on the common case rather than the rare one.

These execute the real cleanup arithmetic extracted from the shim rather than
asserting on source text, so a change to the shipped formula changes the
results here.
"""
from __future__ import annotations

import ctypes
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp"


def _shim_text() -> str:
    return SHIM.read_text()


class _EogCleanup:
    """Compile the REAL EOG cleanup arithmetic lifted out of the shim.

    Only the opaque native calls are replaced by observable side effects; the
    frontier formula, the resize condition, the pending re-anchor argument and
    the trim bounds are taken verbatim from the shipped source.
    """

    _lib = None

    @classmethod
    def _extract(cls) -> str:
        text = _shim_text()
        start = text.index("                // Canonicalize before leaving.")
        end = text.index("goto done;", start)
        body = text[start:end]
        # Strip comments so prose cannot satisfy a pattern.
        body = re.sub(r"//[^\n]*", "", body)
        for needed in (
            "frontier_logical_base + 1 + i",
            "prompt_tgt.resize(committed_size)",
            "common_speculative_accept(session->spec, 0, (uint16_t) i)",
            "llama_memory_seq_rm(mem_tgt, 0, n_past, -1)",
            "llama_memory_seq_rm(mem_dft, 0, n_past, -1)",
            "session->last_target_untrusted = true",
        ):
            if needed not in body:
                raise AssertionError(
                    f"EOG cleanup no longer contains {needed!r}; update the "
                    f"extractor rather than letting it bind to a stale shape"
                )
        # Bind the native calls to recordable stubs, preserving the arguments.
        body = body.replace(
            "common_speculative_accept(session->spec, 0, (uint16_t) i)",
            "out->pending_accept_arg = (int32_t) i",
        )
        body = body.replace(
            "llama_memory_seq_rm(mem_tgt, 0, n_past, -1)",
            "stub_seq_rm(out, 0, n_past, tgt_rm_ok)",
        )
        body = body.replace(
            "llama_memory_seq_rm(mem_dft, 0, n_past, -1)",
            "stub_seq_rm(out, 1, n_past, dft_rm_ok)",
        )
        body = re.sub(r"trace_\w+\([^;]*\);", "", body)
        body = body.replace("session->debug_seq_rm_count += 2;", "out->seq_rm_calls += 2;")
        body = body.replace(
            "session->last_target_untrusted = true;", "out->target_untrusted = true;"
        )
        body = body.replace("prompt_tgt.resize(committed_size);", "prompt_size = committed_size;")
        body = body.replace("prompt_tgt.size()", "prompt_size")
        return """
        #include <cstdint>
        #include <cstddef>
        struct Out {
            int32_t prompt_size;
            int32_t n_past;
            int32_t pending_accept_arg;
            int32_t tgt_rm_from;
            int32_t dft_rm_from;
            int32_t seq_rm_calls;
            bool target_untrusted;
        };
        static bool stub_seq_rm(Out * out, int which, int32_t from, bool result) {
            if (which == 0) { out->tgt_rm_from = from; } else { out->dft_rm_from = from; }
            return result;
        }
        extern "C" void eog_cleanup(Out * out, size_t frontier_logical_base,
                                    size_t i, size_t initial_prompt_size,
                                    bool tgt_rm_ok, bool dft_rm_ok) {
            size_t prompt_size = initial_prompt_size;
            int32_t n_past = 0;
            out->pending_accept_arg = -1;
            out->tgt_rm_from = -1;
            out->dft_rm_from = -1;
            out->seq_rm_calls = 0;
            out->target_untrusted = false;
        %s
            out->prompt_size = (int32_t) prompt_size;
            out->n_past = n_past;
        }
        """ % body

    @classmethod
    def lib(cls):
        if cls._lib is None:
            tmp = Path(tempfile.mkdtemp(prefix="orbit_eog_"))
            src = tmp / "e.cpp"
            src.write_text(cls._extract())
            so = tmp / "e.so"
            rc = subprocess.run(
                ["g++", "-std=c++17", "-fPIC", "-shared", "-O0", "-o", str(so), str(src)],
                capture_output=True, text=True, check=False,
            )
            if rc.returncode != 0:
                raise AssertionError(f"EOG harness build failed: {rc.stderr[:900]}")
            cls._lib = ctypes.CDLL(str(so))
        return cls._lib


class _Out(ctypes.Structure):
    _fields_ = [
        ("prompt_size", ctypes.c_int32),
        ("n_past", ctypes.c_int32),
        ("pending_accept_arg", ctypes.c_int32),
        ("tgt_rm_from", ctypes.c_int32),
        ("dft_rm_from", ctypes.c_int32),
        ("seq_rm_calls", ctypes.c_int32),
        ("target_untrusted", ctypes.c_bool),
    ]


def _run(base, i, initial_size, tgt_ok=True, dft_ok=True) -> _Out:
    out = _Out()
    _EogCleanup.lib().eog_cleanup(
        ctypes.byref(out), ctypes.c_size_t(base), ctypes.c_size_t(i),
        ctypes.c_size_t(initial_size), ctypes.c_bool(tgt_ok), ctypes.c_bool(dft_ok),
    )
    return out


class EogCanonicalStateTests(unittest.TestCase):
    """The committed frontier, and both halves trimmed to it."""

    def test_boundary_path_trims_stop_and_speculative_tail(self) -> None:
        """3: EOG with an accepted speculative tail after the stop.

        base=30, draft of 4 pre-committed, stop at i=2. The two tokens after the
        stop must not survive: identity is base + predecessor + accepted = 33.
        """
        out = _run(base=30, i=2, initial_size=35)
        self.assertEqual(out.prompt_size, 33)
        self.assertEqual(out.n_past, 33)

    def test_non_boundary_path_resize_is_a_no_op(self) -> None:
        """1: a completion that already has the right size is left alone.

        On the non-boundary path prompt_tgt is grown one token at a time, so it
        is already base+1+i when the stop is seen. The cleanup must not shorten
        it further -- doing so would drop a legitimately committed token.
        """
        out = _run(base=30, i=2, initial_size=33)
        self.assertEqual(out.prompt_size, 33)
        self.assertEqual(out.n_past, 33)

    def test_eog_with_no_trailing_draft_tokens(self) -> None:
        """2: the stop is the last token of the batch; nothing to discard."""
        out = _run(base=30, i=3, initial_size=34)
        self.assertEqual(out.prompt_size, 34)
        self.assertEqual(out.n_past, 34)

    def test_stop_at_the_first_generated_position(self) -> None:
        """6: i=0 keeps only the predecessor, never a negative frontier."""
        out = _run(base=30, i=0, initial_size=35)
        self.assertEqual(out.prompt_size, 31)
        self.assertEqual(out.n_past, 31)
        self.assertGreater(out.n_past, 0)

    def test_both_halves_are_trimmed_to_the_same_frontier(self) -> None:
        """4/5: target and draft must be trimmed together, at n_past.

        Trimming one and not the other leaves a mismatched pair that still looks
        frontier-plausible to a later claim.
        """
        out = _run(base=30, i=2, initial_size=35)
        self.assertEqual(out.tgt_rm_from, out.n_past)
        self.assertEqual(out.dft_rm_from, out.n_past)
        self.assertEqual(out.seq_rm_calls, 2)

    def test_pending_is_reanchored_to_the_trimmed_frontier(self) -> None:
        """11: pending_pos must name n_past-1 after cleanup.

        accept() ran earlier with the full accepted count, so pending_pos was
        above the stop. Re-accepting with `i` moves it to base+i == n_past-1.
        """
        for base, i in ((30, 2), (30, 0), (100, 5)):
            with self.subTest(base=base, i=i):
                out = _run(base=base, i=i, initial_size=base + 1 + 6)
                self.assertEqual(out.pending_accept_arg, i)
                self.assertEqual(base + out.pending_accept_arg, out.n_past - 1)

    def test_identity_length_matches_the_physical_frontier(self) -> None:
        """10: len(identity) == frontier+1, the publication gate's invariant."""
        for base, i, size in ((30, 2, 35), (30, 0, 35), (7, 1, 12)):
            with self.subTest(base=base, i=i):
                out = _run(base=base, i=i, initial_size=size)
                physical_frontier = out.n_past - 1
                self.assertEqual(out.prompt_size, physical_frontier + 1)

    def test_refused_target_trim_poisons_the_pair(self) -> None:
        """7/12: a refused rollback must fail closed into a cold rebuild."""
        out = _run(base=30, i=2, initial_size=35, tgt_ok=False)
        self.assertTrue(
            out.target_untrusted,
            "a refused trim leaves tokens above the frontier; publishing would "
            "let a later claim rest on a poisoned frontier",
        )

    def test_refused_draft_trim_poisons_the_pair(self) -> None:
        out = _run(base=30, i=2, initial_size=35, dft_ok=False)
        self.assertTrue(out.target_untrusted)

    def test_successful_trim_leaves_the_pair_trusted(self) -> None:
        out = _run(base=30, i=2, initial_size=35)
        self.assertFalse(out.target_untrusted)


class EogSourceContractTests(unittest.TestCase):
    """Structural properties the extracted harness cannot express."""

    def setUp(self) -> None:
        self.text = _shim_text()

    def test_the_loop_eog_exit_canonicalizes_before_leaving(self) -> None:
        """9: the stop path must not reach `done` without cleaning up.

        Scoped by brace depth: the cleanup must sit between the `is_eog` test
        and its `goto done`, not merely somewhere in the file.
        """
        at = self.text.index("            if (llama_vocab_is_eog(vocab_tgt, id_last)) {", self.text.index("for (size_t i = 0; i < ids.size()"))
        block = self.text[at:self.text.index("goto done;", at)]
        code = re.sub(r"//[^\n]*", "", block)
        for required in ("resize", "n_past =", "seq_rm", "common_speculative_accept"):
            self.assertIn(
                required, code,
                f"the EOG exit reaches `goto done` without {required}: the pair "
                f"would be left incoherent and identity dropped",
            )

    def test_full_accept_commits_the_whole_step(self) -> None:
        """A non-EOG full accept must NOT be truncated.

        The EOG cleanup trims to the stop; `full_accept` must keep everything,
        syncing n_past to the grown prompt_tgt. A resize to the step's base here
        would silently drop every token the step accepted while still looking
        internally consistent to the publication gate.
        """
        # There are several `if (full_accept)` blocks; the one that owns the
        # frontier is the one declaring need_replay_after_failed_rm.
        at = self.text.index(
            "        if (full_accept) {\n            bool need_replay_after_failed_rm"
        )
        block = self.text[at:at + 700]
        code = re.sub(r"//[^\n]*", "", block)
        self.assertIn("n_past = (int32_t) prompt_tgt.size();", code)
        self.assertNotIn(
            "prompt_tgt.resize(", code,
            "full_accept must commit the whole step, never truncate it",
        )

    def test_publication_gate_requires_both_clauses(self) -> None:
        """The gate must check frontier AND length, executed not asserted.

        Either clause alone is insufficient: a stale n_past can satisfy the
        length check while the physical frontier has moved, and vice versa.
        Publishing on a half-check is what makes an identity that misdescribes
        KV, which is a false cache hit rather than a slow path.
        """
        at = self.text.index("session->last_resident_tokens = prompt_tgt;")
        cond_at = self.text.rindex("if (", 0, at)
        cond = re.sub(r"//[^\n]*", "", self.text[cond_at:at])
        for clause in ("llama_memory_seq_pos_max(mem_tgt, 0) == n_past - 1",
                       "prompt_tgt.size() == (size_t) n_past"):
            self.assertIn(
                clause, cond,
                "the publication gate must keep both clauses; dropping one "
                "publishes an identity that can misdescribe physical KV",
            )
        self.assertIn("&&", cond, "the two clauses must both hold, not either")
        self.assertIn(
            "session->last_resident_tokens.clear();", self.text[at:at + 400],
            "a failed gate must publish EMPTY so the next turn rebuilds cold",
        )

    def test_the_pair_verdict_the_cleanup_relies_on_is_intact(self) -> None:
        """The EOG rollback fails closed only through the canonical verdict.

        `last_target_untrusted = true` on a refused trim is load-bearing only
        because the exit verdict turns a long frontier into `untrusted`. Gutting
        that verdict -- or dropping any of its four conjuncts -- makes a refused
        rollback publish anyway. Sibling suites cover this, but a reader of the
        EOG fix should not have to know that to trust it, so the dependency is
        pinned where the fix lives.
        """
        at = self.text.index("session->persistent_pair_untrusted =\n            !(")
        expr = re.sub(r"//[^\n]*", "", self.text[at:self.text.index(";", at)])
        for term in ("target_ok", "draft_ok", "pending_aligned", "identity_ok"):
            self.assertIn(
                term, expr,
                f"the canonical pair verdict lost {term}: a refused EOG "
                f"rollback could then be reported as a trusted pair",
            )
        self.assertIn("!(", expr, "the verdict must negate the conjunction")
        self.assertNotIn(
            "= false", expr,
            "a hardcoded verdict would trust a pair that was never proven",
        )

    def test_the_first_sample_eog_exit_needs_no_cleanup(self) -> None:
        """The early EOG exits precede any draft commit, so they stay bare.

        Adding a trim there would be wrong, not merely redundant: nothing has
        been committed yet and prompt_tgt is already canonical.
        """
        first = self.text.index("if (llama_vocab_is_eog(vocab_tgt, id_last)) {")
        block = self.text[first:self.text.index("goto done;", first)]
        self.assertNotIn("resize", block)
        self.assertNotIn("seq_rm", block)


if __name__ == "__main__":
    unittest.main()

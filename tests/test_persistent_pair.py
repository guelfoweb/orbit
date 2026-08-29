"""Persistent self-MTP pair: target KV + draft KV + pending_h, across completions.

The pair is only reusable when all three halves agree at the same frontier AND
the speculative implementation still belongs to the same target context. Frontier
agreement alone is insufficient: the D3b-R2 review proved a draft can be
frontier-correct and content-wrong, which is why `pending_pos` -- the position of
the predecessor row that seeds the next append -- is part of the predicate.

These are behavioural state-machine tests over the real predicate, compiled from
the shim, plus contract checks that pin the reset split.
"""
from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp"


def _shim_text() -> str:
    return SHIM.read_text()


class _Predicate:
    """Compile the REAL predicate extracted from the shim, against stubs.

    An earlier version of this file compiled a hand-written mirror of the
    predicate. Mutating the shim could not affect it, so five mutation classes
    survived: the tests were proving a copy correct, not the code. The body is
    now lifted verbatim out of the shim, so any change to it changes these
    results.
    """

    _lib = None

    @classmethod
    def _extract(cls) -> str:
        text = _shim_text()
        start = text.index("static bool persistent_pair_is_reusable(")
        end = text.index("\n}", start) + 2
        return text[start:end]

    @classmethod
    def lib(cls):
        if cls._lib is not None:
            return cls._lib
        body = cls._extract()
        harness = """
        #include <cstdint>
        #include <cstddef>
        typedef void * llama_memory_t;
        struct llama_context;
        struct common_speculative;

        // Stubbed environment, driven by the test.
        static int32_t g_tgt_max, g_dft_max, g_pend_pos;
        static uint64_t g_pend_gen;
        static bool g_pend_ok;

        static int32_t llama_memory_seq_pos_max(llama_memory_t m, int) {
            return m == (llama_memory_t) 1 ? g_tgt_max : g_dft_max;
        }
        static bool common_speculative_pending_state(
                const common_speculative *, int,
                int32_t * pos, uint64_t * fp, uint64_t * gen) {
            if (!g_pend_ok) { return false; }
            if (pos) { *pos = g_pend_pos; }
            if (fp)  { *fp  = 1; }
            if (gen) { *gen = g_pend_gen; }
            return true;
        }

        struct orbit_mtp_session {
            bool persistent_pair_untrusted;
            common_speculative * spec;
            llama_context * spec_pinned_ctx_tgt;
        };

        __BODY__

        extern "C" bool drive(
                bool untrusted, bool has_spec, bool identity_ok,
                int32_t tgt_max, int32_t dft_max,
                bool pend_ok, int32_t pend_pos, uint64_t pend_gen) {
            g_tgt_max = tgt_max; g_dft_max = dft_max;
            g_pend_ok = pend_ok; g_pend_pos = pend_pos; g_pend_gen = pend_gen;
            orbit_mtp_session s;
            s.persistent_pair_untrusted = untrusted;
            s.spec = has_spec ? (common_speculative *) 0x1 : nullptr;
            llama_context * ctx = (llama_context *) 0x2;
            s.spec_pinned_ctx_tgt = identity_ok ? ctx : (llama_context *) 0x3;
            return persistent_pair_is_reusable(
                &s, ctx, (llama_memory_t) 1, (llama_memory_t) 2);
        }
        """.replace("__BODY__", body)

        tmp = Path(tempfile.mkdtemp(prefix="orbit_pair_"))
        (tmp / "p.cpp").write_text(harness)
        rc = subprocess.run(
            ["g++", "-std=c++17", "-fPIC", "-shared", "-O0",
             "-o", str(tmp / "p.so"), str(tmp / "p.cpp")],
            capture_output=True, text=True, check=False,
        )
        if rc.returncode != 0:
            raise AssertionError(f"predicate build failed: {rc.stderr[:800]}")
        lib = ctypes.CDLL(str(tmp / "p.so"))
        lib.drive.restype = ctypes.c_bool
        lib.drive.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
                              ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
                              ctypes.c_int32, ctypes.c_uint64]
        cls._lib = lib
        return lib


class PairReusabilityTests(unittest.TestCase):
    """The predicate that decides soft vs hard reset."""

    def _ok(self, untrusted=False, has_spec=True, identity_ok=True,
            tgt=28, dft=28, pend_ok=True, pend_pos=28, pend_gen=7):
        return _Predicate.lib().drive(
            untrusted, has_spec, identity_ok, tgt, dft, pend_ok, pend_pos, pend_gen)

    def test_canonical_pair_is_reusable(self) -> None:
        """All three halves agree at F=28, pending names F, identity holds."""
        self.assertTrue(self._ok())

    def test_untrusted_pair_is_refused(self) -> None:
        self.assertFalse(self._ok(untrusted=True))

    def test_missing_spec_is_refused(self) -> None:
        self.assertFalse(self._ok(has_spec=False))

    def test_ctx_identity_mismatch_is_refused(self) -> None:
        """A preserved impl keeps the ctx_tgt it was built with."""
        self.assertFalse(self._ok(identity_ok=False))

    def test_frontier_disagreement_is_refused(self) -> None:
        self.assertFalse(self._ok(tgt=28, dft=27))
        self.assertFalse(self._ok(tgt=27, dft=28))

    def test_empty_pair_is_refused(self) -> None:
        """A cold pair (-1/-1) has nothing to reuse."""
        self.assertFalse(self._ok(tgt=-1, dft=-1, pend_pos=-1, pend_gen=0))

    def test_pending_unavailable_is_refused(self) -> None:
        self.assertFalse(self._ok(pend_ok=False))

    def test_pending_pos_must_name_the_frontier(self) -> None:
        """This is what frontier equality alone cannot prove."""
        for wrong in (27, 29, 0, -1):
            with self.subTest(pend_pos=wrong):
                self.assertFalse(self._ok(pend_pos=wrong))
        self.assertTrue(self._ok(pend_pos=28))

    def test_never_written_pending_is_refused(self) -> None:
        """gen==0 means the constructor's zero row, never a canonical carryover."""
        self.assertFalse(self._ok(pend_gen=0))

    def test_frontier_aligned_but_content_wrong_is_refused(self) -> None:
        """The exact D3b-R2 defect class: frontiers agree, predecessor does not."""
        self.assertFalse(self._ok(tgt=28, dft=28, pend_pos=14))


class ResetSplitContractTests(unittest.TestCase):
    """Soft preserves the pair; hard rebuilds it."""

    def setUp(self) -> None:
        self.text = _shim_text()

    def test_soft_reset_touches_only_request_bookkeeping(self) -> None:
        at = self.text.find("static void soft_reset_request_state(")
        self.assertGreater(at, -1)
        body = self.text[at:self.text.find("\n}", at)]
        for forbidden in ("llama_memory_clear", "common_speculative_free",
                          "common_speculative_init", "spec_epoch"):
            self.assertNotIn(forbidden, body,
                             f"soft reset must not perform {forbidden}")
        # Comments are stripped: an assertion that merely finds a name would be
        # satisfied by prose mentioning it, which is how the claim-clearing bug
        # below stayed invisible.
        code = re.sub(r"//[^\n]*", "", body)
        for expected in ("request_boundary_ckpt", "request_boundary_prompt_tgt"):
            self.assertIn(expected, code)

    def test_soft_reset_must_not_clear_the_resident_claim(self) -> None:
        """Clearing the claim here would make resident reuse unreachable.

        `soft_reset_request_state` runs at the TOP of the very completion the
        claim was set for, and the claim is read further down. Clearing it here
        zeroes it before that read, so `resident_ok` is false on exactly the
        turns where the pair IS reusable -- the feature can never activate, and
        the failure is silent because it degrades to a correct full replay.

        Nothing leaks by leaving it: the claim is consumed one-shot where it is
        read, and the hard-reset path clears it separately.
        """
        at = self.text.find("static void soft_reset_request_state(")
        body = self.text[at:self.text.find("\n}", at)]
        code = re.sub(r"//[^\n]*", "", body)
        self.assertNotIn(
            "pending_resident_prefix_len", code,
            "the soft reset must not touch the resident claim: it would be "
            "zeroed before the completion that owns it can read it",
        )

    def test_the_claim_is_consumed_one_shot_where_it_is_read(self) -> None:
        """Since the soft reset no longer clears it, the read site must.

        Read-then-zero on adjacent lines is what keeps a claim from surviving
        into a later completion that never earned it.
        """
        read_at = self.text.find("const int32_t resident_claim = session->pending_resident_prefix_len;")
        self.assertGreater(read_at, -1, "the claim must be read into a local")
        window = self.text[read_at:read_at + 220]
        self.assertIn(
            "session->pending_resident_prefix_len = 0;", window,
            "the claim must be zeroed immediately where it is consumed",
        )

    def test_hard_reset_still_rebuilds(self) -> None:
        at = self.text.find("static bool reset_speculative_request_state(")
        self.assertGreater(at, -1)
        body = self.text[at:at + 1400]
        self.assertIn("common_speculative_free", body)
        self.assertIn("common_speculative_init", body)
        self.assertIn("spec_epoch++", body)

    def test_no_draft_clear_outside_the_resident_guard(self) -> None:
        """A draft clear on a resident pass destroys the half being preserved.

        Both clears must sit inside the same `!resident_pass` guard. A clear
        placed before or outside it would empty the draft while the target keeps
        its prefix, leaving pending_h naming a predecessor the draft no longer
        holds -- a mismatched pair that still looks frontier-plausible.
        """
        lines = self.text.splitlines()
        sites = [i for i, l in enumerate(lines) if "llama_memory_clear(mem_dft, true);" in l]
        self.assertTrue(sites, "expected draft clear sites")
        for i in sites:
            window = "\n".join(lines[max(0, i - 12):i])
            self.assertTrue(
                "if (!resident_pass) {" in window or "orbit_mtp_session_reset" in window
                or "reset_speculative_request_state" in window
                or "llama_get_memory(session->ctx_dft)" in window,
                f"draft clear at line {i+1} is not guarded against a resident pass",
            )

    def test_hard_reset_path_poisons_the_pair(self) -> None:
        """The hard branch must mark untrusted; a rebuilt pair has proven nothing."""
        at = self.text.find('trace_request_reset_mode("hard");')
        self.assertGreater(at, -1, "the hard branch must be labelled")
        block = self.text[at:at + 500]
        self.assertIn("reset_speculative_request_state", block)
        self.assertIn("session->persistent_pair_untrusted = true;", block)
        self.assertNotIn("persistent_pair_untrusted = false", block)

    def test_soft_branch_does_not_grant_trust(self) -> None:
        """Soft reset preserves existing trust; it must not manufacture it."""
        at = self.text.find('trace_request_reset_mode("soft");')
        self.assertGreater(at, -1)
        start = self.text.rfind("if (soft) {", 0, at)
        block = self.text[start:at + 120]
        self.assertNotIn("persistent_pair_untrusted = false", block)

    def test_hard_reset_marks_the_pair_untrusted(self) -> None:
        """A rebuilt pair has proven nothing yet."""
        self.assertIn("session->persistent_pair_untrusted = true;", self.text)

    def test_identity_is_pinned_at_every_construction(self) -> None:
        pins = self.text.count(
            "session->spec_pinned_ctx_tgt = session->spec_params.draft.ctx_tgt;")
        inits = self.text.count("common_speculative_init(session->spec_params, 1);")
        self.assertEqual(pins, inits,
                         "every constructed impl must pin the target it belongs to")

    def test_resident_admission_requires_the_whole_pair(self) -> None:
        at = self.text.find("const bool resident_ok =")
        self.assertGreater(at, -1)
        pred = self.text[at:self.text.find(";", at)]
        for term in ("pair_trusted_from_previous", "identity_ok",
                     "llama_memory_seq_pos_max(mem_dft, 0) == resident_claim - 1",
                     "entry_pend_pos == resident_claim - 1", "entry_pend_gen > 0"):
            self.assertIn(term, pred, f"resident admission must require {term}")

    def test_defect_a_strict_proper_prefix_preserved(self) -> None:
        self.assertIn("resident_claim < (int32_t) prompt_tgt.size()", self.text)
        self.assertNotIn("resident_claim <= (int32_t) prompt_tgt.size()", self.text)

    def test_boundary_precedence_preserved(self) -> None:
        self.assertIn("&& !resident_pass", self.text)

    def test_can_partial_rollback_not_wired_into_the_live_path(self) -> None:
        """Its probe is destructive; it must stay out of the decision path."""
        live = [l for l in self.text.splitlines()
                if "can_partial_rollback(" in l and "static bool" not in l]
        self.assertEqual(live, [], f"can_partial_rollback must stay unwired: {live}")


class StalePendingTests(unittest.TestCase):
    """A preserved carryover must never seed a from-zero replay.

    `soft_reset_request_state` keeps `pending_h`, which is correct for a suffix
    append at F+1. But the soft/hard decision is made independently of whether
    the completion actually takes the resident path: a soft reset followed by a
    rejected resident claim (claim 0, n_rs_seq 0, claim >= prompt size) replays
    from position 0. `process()` then seeds draft slot 0 from the carryover where
    the zero row is required -- and because `process()` overwrites the carryover
    before the exit check runs, the resulting draft would still be certified
    trusted. The vendored code states the invariant: "-1 == no predecessor:
    correct only for a batch starting at position 0".
    """

    def setUp(self) -> None:
        self.text = _shim_text()

    def test_carryover_is_discarded_before_a_from_zero_replay(self) -> None:
        at = self.text.find("if (!resident_pass && session->spec) {")
        self.assertGreater(at, -1, "a non-resident pass must discard the carryover")
        block = self.text[at:at + 600]
        self.assertIn("common_speculative_reset_pending(session->spec, 0)", block)

    def test_failure_to_discard_poisons_the_pair(self) -> None:
        """If the carryover cannot be proven safe, refuse to reuse the impl."""
        at = self.text.find("if (!resident_pass && session->spec) {")
        block = self.text[at:at + 600]
        self.assertIn("session->persistent_pair_untrusted = true;", block)

    def test_discard_is_observable(self) -> None:
        self.assertIn('trace_pending_discarded("replay_from_zero")', self.text)

    def test_session_reset_poisons_the_pair(self) -> None:
        """orbit_mtp_session_reset destroys the draft half and the impl.

        A trust bit surviving that operation would assert a pair that no longer
        exists. Physical checks catch it downstream, but a flag outliving what it
        describes is a latent false-trust path.
        """
        at = self.text.find("extern \"C\" bool orbit_mtp_session_reset(")
        self.assertGreater(at, -1)
        body = self.text[at:self.text.find("\n}", at)]
        self.assertIn("session->persistent_pair_untrusted = true;", body)

        # Presence is not reachability. A mutant that prefixes the assignment
        # with `if (false)` leaves the string intact while disabling it, so
        # assert the statement is not guarded by any conditional: walk back from
        # the assignment to the previous statement boundary and require nothing
        # conditional in between.
        assign = body.rindex("session->persistent_pair_untrusted = true;")
        prev_end = max(body.rfind(";", 0, assign), body.rfind("}", 0, assign))
        between = body[prev_end + 1:assign]
        stripped = "\n".join(
            line for line in between.splitlines()
            if not line.strip().startswith("//")
        ).strip()
        self.assertEqual(
            stripped, "",
            f"the poisoning assignment must be unconditional, found: {stripped!r}",
        )

    def test_reset_pending_restores_the_constructor_state(self) -> None:
        """The vendored reset must match what a fresh implementation has."""
        spec = (ROOT / "src/orbit/native_llama/vendor/source/llama.cpp"
                / "common/speculative.cpp").read_text()
        at = spec.find("bool reset_pending(llama_seq_id seq_id) override {")
        self.assertGreater(at, -1)
        body = spec[at:spec.find("\n    }", at)]
        for expected in ("pending_pos[seq_id] = -1;", "pending_fp[seq_id] = 0ull;",
                         "pending_gen[seq_id] = 0ull;", "0.0f"):
            self.assertIn(expected, body)


class PairTrustLifecycleTests(unittest.TestCase):
    """Armed before mutation, cleared only at a proven-canonical exit."""

    def setUp(self) -> None:
        self.text = _shim_text()

    def test_pair_is_armed_before_any_mutation(self) -> None:
        arm = self.text.find("session->persistent_pair_untrusted = true;\n\n    bool need_replay")
        self.assertGreater(arm, -1, "the pair must be poisoned before mutating")
        for mutation in ("llama_memory_clear(mem_tgt, true);", "llama_decode(ctx_tgt,"):
            self.assertGreater(self.text.find(mutation, arm), arm)

    def test_trust_requires_all_three_halves_and_identity(self) -> None:
        at = self.text.find("session->persistent_pair_untrusted =\n            !(")
        self.assertGreater(at, -1)
        expr = self.text[at:self.text.find(";", at)]
        for term in ("target_ok", "draft_ok", "pending_aligned", "identity_ok"):
            self.assertIn(term, expr)

    def _trust_predicate(self):
        """Compile the REAL trust expression out of the shim.

        Earlier versions of the two tests below evaluated Python dict literals
        with no reference to the shim at all, so they passed even if the entire
        canonical-exit trust block were deleted. The expression is now lifted
        from the source, so its semantics -- not just the presence of its
        terms -- are under test.
        """
        at = self.text.find("session->persistent_pair_untrusted =\n            !(")
        if at < 0:
            self.fail("canonical-exit trust assignment not found")
        expr = self.text[at:self.text.find(";", at)].split("=", 1)[1].strip()
        src = """
        #include <cstdint>
        extern "C" bool untrusted(bool target_ok, bool draft_ok,
                                  bool pending_aligned, bool identity_ok) {
            return %s;
        }
        """ % expr
        lib = _PredicateHarness.build("pairtrust", src) if "_PredicateHarness" in globals() \
            else self._build_local(src)
        lib.untrusted.restype = ctypes.c_bool
        return lib

    @staticmethod
    def _build_local(src: str):
        tmp = Path(tempfile.mkdtemp(prefix="orbit_trust_"))
        cpp = tmp / "t.cpp"
        cpp.write_text(src)
        so = tmp / "t.so"
        rc = subprocess.run(
            ["g++", "-std=c++17", "-fPIC", "-shared", "-O0", "-o", str(so), str(cpp)],
            capture_output=True, text=True, check=False,
        )
        if rc.returncode != 0:
            raise AssertionError(f"trust harness build failed: {rc.stderr[:600]}")
        return ctypes.CDLL(str(so))

    def test_recovery_is_possible(self) -> None:
        """The flag must not latch: a fully canonical exit clears it."""
        lib = self._trust_predicate()
        self.assertFalse(
            lib.untrusted(True, True, True, True),
            "an exit with all four halves canonical must restore trust, or the "
            "pair could never be reused after the first completion",
        )

    def test_any_single_failure_poisons_the_pair(self) -> None:
        """Each of the four terms must independently deny trust."""
        lib = self._trust_predicate()
        for i, missing in enumerate(
            ("target_ok", "draft_ok", "pending_aligned", "identity_ok")
        ):
            args = [True, True, True, True]
            args[i] = False
            with self.subTest(missing=missing):
                self.assertTrue(
                    lib.untrusted(*args),
                    f"{missing} does not poison the pair -- a corrupt half "
                    f"would be reported as canonical",
                )


if __name__ == "__main__":
    unittest.main()

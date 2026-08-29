"""Resident target-KV prefix: ABI tiering and trace-recorder integrity.

Persistent target KV is opt-in per completion. The runtime declares a prefix it
has already proven token-identical to its committed sequence; the shim verifies
only that physical target memory agrees, and falls back to a full replay when it
does not. A wrong prefix would be a correctness bug, a full replay is merely
slow, so every ambiguous case resolves to replay.

These cover the ABI contract and the frozen observability recorder. The
lifecycle behaviour itself is proven by the two-request model smoke, because
only a real completion exercises those code paths.
"""

from __future__ import annotations

import re
import subprocess
import unittest
import unittest.mock
from pathlib import Path

from orbit.native_llama import persistent_mtp as pm

ROOT = Path(__file__).resolve().parents[1]
SHIM_SOURCE = ROOT / "src/orbit/native_llama/vendor/shim/orbit_persistent_mtp.cpp"


def _built_shim() -> Path | None:
    for candidate in (
        Path.home() / ".orbit/native-build/liborbit-persistent-mtp.so",
        ROOT / "src/orbit/native_llama/vendor/shim/liborbit-persistent-mtp.so",
    ):
        if candidate.exists():
            return candidate
    return None


def _exports(path: Path) -> set[str]:
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(path)],
        capture_output=True, text=True, check=False,
    )
    return {p.split()[-1] for p in out.stdout.splitlines() if p.split()}


class TierSeparationTests(unittest.TestCase):
    """Three tiers, each demanded only by what actually needs it."""

    def test_resident_symbols_are_not_in_the_base_contract(self) -> None:
        for symbol in pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS:
            self.assertNotIn(
                symbol, pm._REQUIRED_SHIM_SYMBOLS,
                "a shim predating resident reuse must stay valid for base decoding",
            )

    def test_resident_symbols_are_not_required_for_ordinary_self_mtp(self) -> None:
        for symbol in pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS:
            self.assertNotIn(
                symbol, pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS,
                "self-MTP without resident reuse must not require these",
            )

    def test_the_three_tiers_are_disjoint(self) -> None:
        base = set(pm._REQUIRED_SHIM_SYMBOLS)
        self_mtp = set(pm._SELF_MTP_REQUIRED_SHIM_SYMBOLS)
        resident = set(pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS)
        self.assertFalse(base & self_mtp)
        self.assertFalse(base & resident)
        self.assertFalse(self_mtp & resident)

    def test_resident_tier_is_non_empty(self) -> None:
        """An empty tuple would make the capability check vacuously true."""
        self.assertTrue(pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS)


class CompiledExportTests(unittest.TestCase):
    """Source text is not evidence; compiled exports are."""

    def setUp(self) -> None:
        self.shim = _built_shim()
        if self.shim is None:
            self.skipTest("no compiled shim available")
        self.exports = _exports(self.shim)
        if not self.exports:
            self.skipTest("nm produced no symbols")

    def test_resident_symbols_are_exported(self) -> None:
        missing = [
            s for s in pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS
            if s not in self.exports
        ]
        if missing == list(pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS):
            self.skipTest("inspecting a shim that predates resident reuse")
        self.assertEqual(missing, [])

    def test_base_contract_still_satisfied(self) -> None:
        missing = [s for s in pm._REQUIRED_SHIM_SYMBOLS if s not in self.exports]
        self.assertEqual(missing, [])

    def test_trace_recorder_exports_nothing(self) -> None:
        """Observability stays internal; it must add no ABI surface."""
        leaked = [s for s in self.exports if "trace_target" in s or "target_trace" in s]
        self.assertEqual(leaked, [])


class RecorderIntegrityTests(unittest.TestCase):
    """The frozen recorder must keep describing target-only truth.

    Phase B consumes this evidence, so a recorder that drifts would invalidate
    every lifecycle claim built on it.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()
        self.lines = self.source.splitlines()

    def test_every_target_clear_is_traced(self) -> None:
        sites = [i for i, l in enumerate(self.lines) if "llama_memory_clear(mem_tgt" in l]
        self.assertTrue(sites)
        for i in sites:
            window = "\n".join(self.lines[i:i + 3])
            self.assertIn("trace_target_clear", window)

    def test_draft_clears_are_never_traced_as_target(self) -> None:
        for i, line in enumerate(self.lines):
            if "llama_memory_clear(mem_dft" in line:
                window = "\n".join(self.lines[i:i + 2])
                self.assertNotIn("trace_target_clear", window)

    def test_every_target_seq_rm_captures_its_result(self) -> None:
        sites = [i for i, l in enumerate(self.lines) if "llama_memory_seq_rm(mem_tgt" in l]
        self.assertTrue(sites)
        for i in sites:
            self.assertRegex(
                self.lines[i],
                r"const bool \w+ = llama_memory_seq_rm\(mem_tgt",
                "a discarded seq_rm result cannot prove rollback succeeded",
            )

    def test_prefill_traces_report_tokens_not_batches(self) -> None:
        call_sites = [
            l for l in self.lines
            if "trace_target_prefill(" in l and "static void" not in l
        ]
        self.assertTrue(call_sites, "no prefill trace call sites found")
        for line in call_sites:
            self.assertIn(
                ".size()", line,
                "prefill evidence must count tokens, not decode batches",
            )

    def test_trace_helpers_do_not_mutate_session_state(self) -> None:
        for name in ("trace_target_clear", "trace_target_prefill",
                     "trace_target_seq_rm", "trace_target_frontier"):
            body = re.search(rf"static void {name}.*?\n\}}", self.source, re.S)
            self.assertIsNotNone(body, name)
            self.assertIsNone(
                re.search(r"session->\w+\s*(=[^=]|\+\+|--|\+=)", body.group(0)),
                f"{name} must be observational only",
            )


class ClaimOrderingTests(unittest.TestCase):
    """The claim must be declared AFTER the reset that precedes each completion.

    `_try_complete_with_mtp_experimental` resets the session and then completes.
    Reset deliberately clears a pending claim so a stale one cannot leak, which
    means a claim set by the caller before `complete()` is wiped before it can
    be read -- the resident branch would be permanently dead. Setting it inside
    `run_persistent_mtp_completion`, immediately before the native call, keeps
    both properties.
    """

    def test_claim_is_set_after_reset_and_before_completion(self) -> None:
        import inspect

        from orbit.native_llama import persistent_mtp as pm

        body = inspect.getsource(pm.run_persistent_mtp_completion)
        set_at = body.find("orbit_mtp_session_set_resident_prefix_len")
        complete_at = body.find("orbit_mtp_session_complete(")
        self.assertGreater(set_at, -1, "the claim must be declared here")
        self.assertGreater(complete_at, -1)
        self.assertLess(
            set_at, complete_at,
            "the claim must be set before the native completion reads it",
        )

    def test_reset_does_not_declare_a_claim(self) -> None:
        """Reset only clears; declaring there would resurrect the ordering bug."""
        import inspect

        from orbit.native_llama import persistent_mtp as pm

        body = inspect.getsource(pm.reset_persistent_mtp_session)
        self.assertNotIn("set_resident_prefix_len", body)

    def test_zero_claim_never_calls_the_setter(self) -> None:
        """A cold completion must not touch the resident capability at all."""
        from unittest.mock import MagicMock

        from orbit.native_llama import persistent_mtp as pm

        lib = MagicMock()
        lib.orbit_mtp_session_complete.return_value = True
        lib.orbit_mtp_session_last_content.return_value = b""
        runtime = MagicMock(handle=object())

        with unittest.mock.patch.object(
            pm, "_runtime_library", lambda **kw: MagicMock(lib=lib)
        ):
            pm.run_persistent_mtp_completion(
                llama_root=Path("/x"), paths=MagicMock(), runtime=runtime,
                ctx_tgt=None, prompt="hi", max_tokens=4, resident_prefix_len=0,
            )
        lib.orbit_mtp_session_set_resident_prefix_len.assert_not_called()

    def test_missing_capability_fails_closed(self) -> None:
        """No silent downgrade: asking for reuse a shim cannot do must error."""
        from unittest.mock import MagicMock

        from orbit.native_llama import persistent_mtp as pm

        lib = MagicMock(spec=["orbit_mtp_session_complete", "orbit_mtp_last_error"])
        runtime = MagicMock(handle=object())

        with unittest.mock.patch.object(
            pm, "_runtime_library", lambda **kw: MagicMock(lib=lib)
        ):
            result = pm.run_persistent_mtp_completion(
                llama_root=Path("/x"), paths=MagicMock(), runtime=runtime,
                ctx_tgt=None, prompt="hi", max_tokens=4, resident_prefix_len=8,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "persistent-mtp-resident-prefix-unsupported")
        lib.orbit_mtp_session_complete.assert_not_called()


class Load5HistoricalDefectTests(unittest.TestCase):
    """The resident pass must never reach the cold full-replay branch.

    Reproduces the defect load #5 exposed: the resident path correctly skipped
    the target clear, then fell into the `else` full replay and decoded the
    whole prompt at position 0 into a target that already held the prefix.
    llama.cpp rejected the overlap and the completion failed.

    The prefill block has TWO regions. Guarding only the later positional one
    is not enough -- the earlier full replay must also exclude the resident
    pass, or it runs first and the positional block is never reached.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_full_replay_branch_excludes_the_resident_pass(self) -> None:
        self.assertIn(
            "} else if (!resident_pass) {", self.source,
            "the cold full-replay branch must not be reachable on a resident pass",
        )

    def test_no_unguarded_else_before_the_positional_block(self) -> None:
        """An unguarded `} else {` here is exactly the load-5 defect."""
        head = self.source.split("std::vector<llama_token> process_tokens;")[0]
        tail = head.split("if (use_request_boundary) {")[-1]
        self.assertNotIn(
            "                } else {\n", tail,
            "an unguarded else here sends the resident pass into full replay",
        )

    def test_cold_path_still_reaches_full_replay(self) -> None:
        """The guard must exclude only the resident pass, not cold requests."""
        self.assertIn("trace_target_prefill(\"full_replay\"", self.source)
        self.assertNotIn("} else if (false) {", self.source)

    def test_request_boundary_branch_still_exists_for_its_own_case(self) -> None:
        """Boundary restore is preserved -- but must not preempt resident.

        Replaces an earlier assertion that the branch was "untouched", which
        pinned the defect: it clears mem_tgt and replays from 0, and nothing
        excluded a resident pass from reaching it.
        """
        self.assertIn("if (use_request_boundary) {", self.source)
        self.assertIn("&& !resident_pass;", self.source)


class BoundaryResidentExclusionTests(unittest.TestCase):
    """Resident reuse must take precedence over request-boundary restore.

    Both flags are computed independently and can be true together: on a
    resident pass `generated` is empty, and `can_restore_request_boundary` is
    true exactly when the new prompt extends the previous one -- the steady
    state the resident feature targets. The boundary branch is tested first
    and clears mem_tgt, so without an explicit exclusion it destroys the
    prefix and replays from 0.

    The caller currently resets the session before each completion, clearing
    the boundary checkpoint. That is caller ordering, not an invariant of this
    function, and must not be relied upon.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_boundary_predicate_excludes_a_resident_pass(self) -> None:
        self.assertIn("&& !resident_pass;", self.source)

    def test_exclusion_is_code_not_only_a_comment(self) -> None:
        code = "\n".join(
            l for l in self.source.splitlines() if not l.strip().startswith("//")
        )
        self.assertIn("&& !resident_pass;", code)

    def test_resident_pass_declared_before_the_boundary_predicate(self) -> None:
        decl = self.source.find("const bool resident_pass = resident_prefill_pending;")
        use = self.source.find("const bool use_request_boundary =")
        self.assertGreater(decl, -1)
        self.assertGreater(use, decl, "resident_pass must be declared first")

    def test_resident_wins_in_the_positional_block_too(self) -> None:
        block = self.source.split("std::vector<llama_token> process_tokens;")[1]
        r_at = block.find("if (resident_pass) {")
        b_at = block.find("else if (use_request_boundary)")
        self.assertGreater(r_at, -1)
        self.assertGreater(b_at, r_at, "resident must take precedence")


class SeqRmResultConsumedTests(unittest.TestCase):
    """A refused target seq_rm must never be treated as success.

    Capturing the result is not enough -- these previously bound it, traced
    it, and discarded it. With a prefix now preserved across completions, a
    refusal leaves rejected tokens above the committed frontier, and a later
    claim of frontier+1 would match that poisoned frontier.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_every_target_seq_rm_result_is_inspected(self) -> None:
        for name in ("tgt_rm_ok", "ckpt_rm_ok", "full_rm_ok"):
            with self.subTest(result=name):
                self.assertIn(
                    "if (!" + name + ")", self.source,
                    name + " is captured but never branched on",
                )

    def test_full_accept_replay_depends_on_the_removal_result(self) -> None:
        self.assertIn("need_replay = need_replay_after_failed_rm;", self.source)
        self.assertNotIn(
            "draft_is_fresh = false;\n            need_replay = false;", self.source,
            "full_accept must not unconditionally clear need_replay",
        )

    def test_a_refusal_marks_the_target_untrusted(self) -> None:
        self.assertGreaterEqual(
            self.source.count("session->last_target_untrusted = true;"), 3,
            "each target seq_rm site must mark the target unproven on refusal",
        )

    def test_untrusted_target_blocks_the_next_resident_claim(self) -> None:
        self.assertIn("!target_untrusted_from_previous &&", self.source)

    def test_untrusted_flag_survives_into_the_next_completion(self) -> None:
        read_at = self.source.find(
            "const bool target_untrusted_from_previous = session->last_target_untrusted;"
        )
        clear_at = self.source.find("session->last_target_untrusted = false;", read_at)
        self.assertGreater(read_at, -1)
        self.assertGreater(clear_at, read_at, "must be read before it is cleared")


class UntrustedLifetimeTests(unittest.TestCase):
    """The untrusted flag must gate exactly the NEXT completion.

    A refused target seq_rm leaves tokens above the committed frontier. The
    completion that suffered it is already over, so the protection has to
    survive into the next one, be read before anything clears it, and then be
    released -- otherwise either a poisoned frontier gets claimed, or the
    session is blocked from ever recovering.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_flag_is_read_before_it_is_cleared(self) -> None:
        read_at = self.source.find(
            "const bool target_untrusted_from_previous = session->last_target_untrusted;"
        )
        self.assertGreater(read_at, -1, "the flag must be carried in from the previous completion")
        clear_at = self.source.find("session->last_target_untrusted = false;", read_at)
        self.assertGreater(clear_at, read_at, "clearing before reading loses the protection")

    def test_flag_is_cleared_so_recovery_is_possible(self) -> None:
        """It must not latch: a cold replay restores a known state."""
        self.assertIn("session->last_target_untrusted = false;", self.source)

    def test_validation_consults_the_carried_flag_not_the_live_field(self) -> None:
        """Reading the live field after clearing it would always be false."""
        self.assertIn("!target_untrusted_from_previous &&", self.source)
        self.assertNotIn("!session->last_target_untrusted &&", self.source)

    def test_clear_precedes_the_resident_decision(self) -> None:
        """Cleared early so THIS completion can set it afresh on a refusal."""
        clear_at = self.source.find("session->last_target_untrusted = false;")
        ok_at = self.source.find("const bool resident_ok =")
        self.assertGreater(clear_at, -1)
        self.assertLess(clear_at, ok_at)

    def test_every_refusal_site_sets_the_flag(self) -> None:
        for guard in ("if (!tgt_rm_ok)", "if (!ckpt_rm_ok)", "if (!full_rm_ok)"):
            with self.subTest(site=guard):
                idx = self.source.find(guard)
                self.assertGreater(idx, -1, guard + " missing")
                # Window past the rationale comment; the assignment follows it.
                window = self.source[idx:idx + 900]
                self.assertIn("last_target_untrusted = true;", window)

class ResidentLifecycleShapeTests(unittest.TestCase):
    """Structural pins for the resident path.

    These assert the SHAPE of the lifecycle, not its runtime behaviour: a C++
    mutation only changes behaviour once recompiled, so the authoritative proof
    that the target is not cleared and the suffix starts at N is the two-request
    model smoke. These catch an accidental edit early and cheaply; they do not
    replace that smoke.
    """

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_target_clear_is_skipped_on_the_resident_pass(self) -> None:
        self.assertIn(
            "if (!resident_pass) {\n                llama_memory_clear(mem_tgt, true);",
            self.source,
            "the resident pass must not clear the prefix it exists to preserve",
        )

    def test_resident_prefill_is_based_at_the_claim(self) -> None:
        self.assertIn("process_pos0 = resident_claim;", self.source)
        for wrong in ("process_pos0 = resident_claim - 1;",
                      "process_pos0 = resident_claim + 1;"):
            self.assertNotIn(wrong, self.source, "off-by-one in the resident base")

    def test_resident_slice_starts_at_the_claim(self) -> None:
        self.assertIn(
            "prompt_tgt.begin() + (ptrdiff_t) resident_claim", self.source,
            "the suffix must begin exactly where the resident prefix ends",
        )

    def test_resident_pass_reaches_the_positional_prefill_loop(self) -> None:
        """Without this the suffix falls through to the from-zero branch."""
        self.assertIn("if (use_request_boundary || resident_pass) {", self.source)

    def test_draft_is_never_cleared_on_a_resident_pass(self) -> None:
        """The draft clear must be lexically INSIDE `if (!resident_pass)`.

        This test previously asserted the opposite ("draft state still resets")
        and did so with a bare substring search that matched the unrelated
        clears in the reset helpers -- so it passed under `if(false)` and, worse,
        pinned a contract the shim deliberately does not implement.

        Clearing the draft on a resident pass empties the draft half while the
        target keeps prompt[0:N), so the next turn's
        `seq_pos_max(mem_dft,0) == claim-1` conjunct fails, `resident_ok` is
        permanently false, and the feature is silently dead while every test
        still passes. Scope is checked by brace depth, not by proximity: a clear
        six lines below a CLOSED `!resident_pass` block is outside it.
        """
        lines = self.source.splitlines()
        # The guard must EXIST. Checking only "is the clear inside a guard"
        # made deletion of the guard invisible: with no guard anywhere, no
        # clear is ever reported out of scope and the test passed while the
        # clear had become unconditional. Matched semantically so an equivalent
        # rewrite (`resident_pass == false`) is not a spurious failure.
        guard_re = re.compile(
            r"if\s*\(\s*(?:!\s*resident_pass|resident_pass\s*==\s*false)\s*\)\s*\{"
        )
        guards = [i for i, l in enumerate(lines, 1) if guard_re.search(l.split("//")[0])]
        self.assertTrue(
            guards,
            "the `!resident_pass` guard is gone: the replay clears now run on "
            "every pass, so a resident pass wipes the draft half and resident "
            "reuse is permanently dead",
        )

        # Every draft clear must then be lexically inside one of those guards.
        # Scope by brace depth, not proximity: a clear below a CLOSED guard
        # block is outside it however few lines away.
        depth = 0
        guard_depth = None
        offenders = []
        for i, line in enumerate(lines, 1):
            stripped = line.split("//")[0]
            if guard_depth is None and guard_re.search(stripped):
                guard_depth = depth
            opened = depth
            depth += stripped.count("{") - stripped.count("}")
            if guard_depth is not None and depth <= guard_depth and opened > guard_depth:
                guard_depth = None
            if "llama_memory_clear(mem_dft" in stripped and guard_depth is None:
                offenders.append(i)
        self.assertEqual(
            offenders, [],
            f"draft clear(s) at line(s) {offenders} are outside the "
            f"`!resident_pass` guard; a resident pass would wipe the draft half "
            f"and permanently disable resident reuse",
        )


class CommittedSequenceInvalidationTests(unittest.TestCase):
    """`_invalidate_committed_sequence` must actually clear the identity.

    D3b does not change it, but a gutted implementation would let a stale
    committed identity survive a KV rewrite -- a false cache hit, which is the
    correctness bug the strict prefix rule exists to prevent.
    """

    def test_invalidation_clears_the_committed_tokens(self) -> None:
        from orbit.native_llama.client import NativeLlamaClient
        from orbit.native_llama.session_state import NativeSessionState

        c = NativeLlamaClient.__new__(NativeLlamaClient)
        c._session = NativeSessionState(session_id="inv")
        c._session.committed_sequence_tokens = [11, 22, 33]
        c._invalidate_committed_sequence()
        self.assertEqual(
            list(c._session.committed_sequence_tokens), [],
            "a surviving committed identity after invalidation is a false cache hit",
        )

    def test_strict_append_needs_a_committed_sequence(self) -> None:
        """With identity cleared, the exact-prefix fast path must not fire."""
        from ctypes import c_void_p
        from unittest.mock import MagicMock, patch

        from orbit.native_llama.client import NativeLlamaClient
        from orbit.native_llama.session_state import NativeSessionState

        c = NativeLlamaClient.__new__(NativeLlamaClient)
        c._session = NativeSessionState(session_id="inv2")
        c._session.ctx_tgt = c_void_p(0x1)
        c._session.committed_sequence_tokens = [1, 2, 3]
        c._session.cached_prompt_tokens = []
        lib = MagicMock(); lib.llama_get_memory.return_value = None
        c.lib = MagicMock(lib=lib)

        with patch.object(NativeLlamaClient, "_qwen3_coder_native_protocol", lambda self: False), \
             patch.object(NativeLlamaClient, "_invalidate_committed_sequence", lambda self: None):
            self.assertEqual(c._prepare_memory_for_prompt([1, 2, 3, 4]), 3)

        c._session.committed_sequence_tokens = []
        c._session.cached_prompt_tokens = []
        with patch.object(NativeLlamaClient, "_qwen3_coder_native_protocol", lambda self: False), \
             patch.object(NativeLlamaClient, "_invalidate_committed_sequence", lambda self: None):
            self.assertNotEqual(c._prepare_memory_for_prompt([1, 2, 3, 4]), 3)


class ResidentValidationSourceTests(unittest.TestCase):
    """The physical checks that gate resident reuse."""

    def setUp(self) -> None:
        self.source = SHIM_SOURCE.read_text()

    def test_claim_is_bounded_by_the_prompt(self) -> None:
        """The bound is STRICT: `<`, not `<=`.

        This originally asserted `<=`, which pinned Defect A as correct -- a
        claim equal to the prompt length leaves an empty suffix, so the target
        is never decoded and sampling reads the previous completion's logits.
        """
        self.assertIn("resident_claim < (int32_t) prompt_tgt.size()", self.source)
        self.assertNotIn("resident_claim <= (int32_t) prompt_tgt.size()", self.source)

    def test_physical_frontier_must_match_the_claim(self) -> None:
        self.assertIn(
            "llama_memory_seq_pos_max(mem_tgt, 0) == resident_claim - 1", self.source
        )

    def test_rollback_budget_is_required(self) -> None:
        self.assertIn("llama_n_rs_seq(ctx_tgt) > 0", self.source)

    def test_destructive_capability_probe_is_not_wired(self) -> None:
        """`can_partial_rollback` clears memory while probing."""
        calls = re.findall(r"(?<!static bool )can_partial_rollback\s*\(", self.source)
        self.assertEqual(
            calls, [],
            "probing with can_partial_rollback would destroy the resident prefix",
        )

    def test_claim_is_consumed_so_it_cannot_leak_forward(self) -> None:
        """Consumed at completion entry, before the validation reads it.

        A claim describes ONE completion. If the field is not zeroed here, the
        next completion inherits it and may reuse a prefix nobody declared for
        it -- the exact stale-claim leak the one-shot design forbids. Asserting
        only that the statement exists somewhere is not enough: it must sit
        between reading the claim and computing `resident_ok`.
        """
        read_at = self.source.find(
            "const int32_t resident_claim = session->pending_resident_prefix_len;"
        )
        self.assertGreater(read_at, -1, "the claim must be read into a local")
        consume_at = self.source.find(
            "session->pending_resident_prefix_len = 0;", read_at
        )
        self.assertGreater(consume_at, read_at, "the claim must be consumed after being read")
        validate_at = self.source.find("const bool resident_ok =", read_at)
        self.assertGreater(validate_at, -1)
        self.assertLess(
            consume_at, validate_at,
            "the claim must be consumed before validation, so no later "
            "completion can inherit it",
        )

    def test_reset_clears_a_pending_claim(self) -> None:
        reset = re.search(
            r"orbit_mtp_session_reset\(void \* handle.*?\n\}", self.source, re.S
        )
        self.assertIsNotNone(reset)
        self.assertIn("pending_resident_prefix_len = 0", reset.group(0))

    def test_reset_does_not_clear_target_memory(self) -> None:
        """Draft state and canonical target KV are separate concerns."""
        reset = re.search(
            r"orbit_mtp_session_reset\(void \* handle.*?\n\}", self.source, re.S
        )
        body = reset.group(0)
        # Strip comments: the only mem_tgt mentions should be explanatory.
        code = "\n".join(
            l for l in body.splitlines() if not l.strip().startswith("//")
        )
        self.assertNotIn(
            "llama_memory_clear(mem_tgt", code,
            "reset must not destroy canonical target KV",
        )
        # It does clear the DRAFT context, which is the point of a reset.
        self.assertIn("llama_get_memory(session->ctx_dft)", code)



# ---------------------------------------------------------------------------
# D3b-R2: the three defects found by the post-load-8 independent review.
#
# The review's MINOR 2 was fair: source-text assertions pin spelling, not
# behaviour, and two of them were satisfied *by* the lines that caused the
# defects. The classes below therefore compile the real predicates out of the
# shim and execute them, so a mutant that changes behaviour fails even when it
# keeps the surrounding text intact.
# ---------------------------------------------------------------------------


def _shim_text() -> str:
    return SHIM_SOURCE.read_text()


class _PredicateHarness:
    """Compile a shim predicate standalone and run it over real inputs."""

    _CACHE: dict[str, object] = {}

    @staticmethod
    def build(name: str, source: str):
        import ctypes
        import tempfile

        if name in _PredicateHarness._CACHE:
            return _PredicateHarness._CACHE[name]
        tmp = Path(tempfile.mkdtemp(prefix=f"orbit_pred_{name}_"))
        src = tmp / "p.cpp"
        src.write_text(source)
        so = tmp / "p.so"
        rc = subprocess.run(
            ["g++", "-std=c++17", "-fPIC", "-shared", "-O0", "-o", str(so), str(src)],
            capture_output=True, text=True, check=False,
        )
        if rc.returncode != 0:
            raise AssertionError(f"harness build failed: {rc.stderr[:800]}")
        lib = ctypes.CDLL(str(so))
        _PredicateHarness._CACHE[name] = lib
        return lib


class FullPromptClaimTests(unittest.TestCase):
    """Defect A: a claim covering the whole prompt must not enter resident reuse.

    With `<=`, claim == prompt size makes the suffix empty, so the completion
    decodes nothing into the target and the first sample reads logits left by the
    PREVIOUS completion -- silently wrong output reported as success. The
    contract is a STRICT PROPER prefix; the whole-prompt case falls closed to the
    full replay, which regenerates fresh target logits.
    """

    # The REAL guard, lifted out of the shim rather than mirrored by hand. An
    # earlier version of this class compiled a 5-conjunct hand-written copy
    # while the shipped guard had 11; deleting any of the six pair-trust and
    # pending_h conjuncts from the shim left every test here green. That is the
    # same mistake documented in test_persistent_pair.py, where a mirror let
    # five mutation classes survive. The opaque C++ calls are replaced by named
    # parameters -- and only those calls -- so the operators, ordering and
    # conjunct set stay verbatim and any edit to them changes these results.
    @staticmethod
    def _extract() -> str:
        text = _shim_text()
        start = text.index("    const bool resident_ok =")
        end = text.index(";", start) + 1
        expr = text[start:end].split("=", 1)[1].strip().rstrip(";")
        for call, param in (
            ("(int32_t) prompt_tgt.size()", "prompt_size"),
            ("llama_n_rs_seq(ctx_tgt)", "n_rs_seq"),
            ("llama_memory_seq_pos_max(mem_tgt, 0)", "tgt_pos_max"),
            ("llama_memory_seq_pos_max(mem_dft, 0)", "dft_pos_max"),
        ):
            if call not in expr:
                raise AssertionError(
                    f"shim guard no longer contains {call!r}; update the "
                    f"extractor rather than letting it bind to a stale shape"
                )
            expr = expr.replace(call, param)
        expr = expr.replace("resident_claim", "claim")
        if "llama_" in expr:
            raise AssertionError(f"unsubstituted native call in guard: {expr}")
        return """
    #include <cstdint>
    #include <cstddef>
    extern "C" bool resident_ok(
            int32_t claim, int32_t prompt_size, bool target_untrusted_from_previous,
            bool pair_trusted_from_previous, bool identity_ok, uint32_t n_rs_seq,
            int32_t tgt_pos_max, int32_t dft_pos_max, bool entry_pend_ok,
            int32_t entry_pend_pos, uint64_t entry_pend_gen) {
        return %s;
    }
    """ % expr

    def setUp(self) -> None:
        self.lib = _PredicateHarness.build("residentA", self._extract())
        self.lib.resident_ok.restype = __import__("ctypes").c_bool

    def _ok(self, claim, size, untrusted=False, n_rs=1, pos_max=None,
            pair_trusted=True, identity_ok=True, dft_pos_max=None,
            pend_ok=True, pend_pos=None, pend_gen=1):
        import ctypes

        tgt = claim - 1 if pos_max is None else pos_max
        return self.lib.resident_ok(
            ctypes.c_int32(claim), ctypes.c_int32(size),
            ctypes.c_bool(untrusted), ctypes.c_bool(pair_trusted),
            ctypes.c_bool(identity_ok), ctypes.c_uint32(n_rs),
            ctypes.c_int32(tgt),
            ctypes.c_int32(tgt if dft_pos_max is None else dft_pos_max),
            ctypes.c_bool(pend_ok),
            ctypes.c_int32(claim - 1 if pend_pos is None else pend_pos),
            ctypes.c_uint64(pend_gen),
        )

    def test_every_pair_trust_conjunct_is_load_bearing(self) -> None:
        """Each of the 11 conjuncts must be able to deny admission on its own.

        This is what the hand-written mirror could not do: it did not contain
        the pair-trust or pending_h terms at all, so their deletion from the
        shim was invisible here.
        """
        base = dict(claim=10, size=20)
        for name, override in (
            ("claim > 0", dict(claim=0)),
            ("!target_untrusted_from_previous", dict(untrusted=True)),
            ("pair_trusted_from_previous", dict(pair_trusted=False)),
            ("identity_ok", dict(identity_ok=False)),
            ("claim < prompt_size", dict(claim=20)),
            ("n_rs_seq > 0", dict(n_rs=0)),
            ("target frontier", dict(pos_max=8)),
            ("draft frontier", dict(dft_pos_max=8)),
            ("entry_pend_ok", dict(pend_ok=False)),
            ("entry_pend_pos", dict(pend_pos=8)),
            ("entry_pend_gen > 0", dict(pend_gen=0)),
        ):
            with self.subTest(conjunct=name):
                self.assertTrue(self._ok(**base), "baseline must be admissible")
                self.assertFalse(
                    self._ok(**{**base, **override}),
                    f"{name} does not deny admission -- the guard is not "
                    f"actually enforcing it",
                )

    def test_proper_prefix_may_proceed(self) -> None:
        """A: claim < prompt size with every other condition met is eligible."""
        for claim, size in ((1, 2), (10, 20), (26, 27), (100, 4096)):
            with self.subTest(claim=claim, size=size):
                self.assertTrue(self._ok(claim, size))

    def test_full_prompt_claim_is_denied(self) -> None:
        """B: claim == prompt size must NOT enter resident reuse."""
        for size in (1, 2, 20, 27, 4096):
            with self.subTest(size=size):
                self.assertFalse(
                    self._ok(size, size),
                    "a whole-prompt claim leaves an empty suffix, so no target "
                    "decode would occur and sampling would read stale logits",
                )

    def test_claim_beyond_prompt_is_denied(self) -> None:
        """C: an over-long claim stays denied."""
        for claim, size in ((21, 20), (28, 27), (5000, 4096)):
            with self.subTest(claim=claim):
                self.assertFalse(self._ok(claim, size))

    def test_zero_and_negative_claims_denied(self) -> None:
        for claim in (0, -1, -27):
            with self.subTest(claim=claim):
                self.assertFalse(self._ok(claim, 27))

    def test_the_guard_in_the_shim_is_strict(self) -> None:
        """The compiled predicate above must mirror the shipped guard."""
        text = _shim_text()
        self.assertIn("resident_claim < (int32_t) prompt_tgt.size()", text)
        self.assertNotIn("resident_claim <= (int32_t) prompt_tgt.size()", text)

    def test_denied_full_prompt_claim_reaches_full_replay(self) -> None:
        """D: denial must select replay, not a decode-free path.

        `need_replay = !resident_ok`, so a denied claim replays the whole prompt
        from position 0, which is what produces fresh logits before sampling.
        """
        text = _shim_text()
        self.assertIn("bool need_replay = !resident_ok;", text)
        self.assertIn("bool resident_prefill_pending = resident_ok;", text)

    def test_no_decode_free_sampling_path_remains(self) -> None:
        """E: with a strict prefix the resident suffix can never be empty.

        claim < size implies size - claim >= 1, so the resident slice always has
        at least one token and the prefill guard cannot skip the target decode.
        """
        for claim, size in ((1, 2), (10, 20), (26, 27)):
            with self.subTest(claim=claim, size=size):
                self.assertGreaterEqual(size - claim, 1)
        # And the empty-slice case is unreachable because it is denied upstream.
        self.assertFalse(self._ok(27, 27))


class SameSuffixContractTests(unittest.TestCase):
    """The draft must process EXACTLY the batch the target just decoded.

    This class replaces DraftContiguityTests, whose premise was disproven. That
    class asserted the draft slice must be DECOUPLED from the target slice and
    that a non-shared draft should be rebuilt from position 0. The D3b-R2
    pre-load review showed why that is wrong: `common_speculative_process`
    consumes the target's nextn hidden rows indexed by BATCH SLOT, from the
    target's most recent decode. Feeding it more tokens than the target just
    decoded reads rows the target never produced -- silently building the draft
    from stale conditioning, with no warning, because a fully populated draft KV
    suppresses llama.cpp's own "Drafts may degrade" diagnostic.

    With a persistent pair the correct rule is one rule for both cases: the draft
    always mirrors the target's last decode. Cold decodes the whole prompt, so
    the draft processes the whole prompt; a resident pass decodes only the
    suffix, so the draft appends only the suffix.
    """

    def test_draft_slice_is_the_target_slice(self) -> None:
        text = _shim_text()
        self.assertIn(
            "const std::vector<llama_token> & draft_tokens = process_tokens;", text)
        self.assertIn("const int32_t draft_pos0 = process_pos0;", text)

    def test_no_conditional_draft_slice_remains(self) -> None:
        """The disproven branch must be gone, not merely bypassed."""
        text = _shim_text()
        self.assertNotIn("draft_mem_shared", text)
        self.assertNotIn("draft_mem_shared ? process_tokens : prompt_tgt", text)

    def test_draft_loop_consumes_the_target_slice_throughout(self) -> None:
        """Loop bound, chunk bounds and base position must all follow it."""
        text = _shim_text()
        start = text.find("const std::vector<llama_token> & draft_tokens = process_tokens;")
        self.assertGreater(start, -1)
        end = text.find("common_speculative_begin", start)
        self.assertGreater(end, start)
        loop = text[start:end]
        self.assertIn("offset < draft_tokens.size()", loop)
        self.assertIn("chunk_size, draft_tokens.size() - offset", loop)
        self.assertIn("fill_batch(prefill, chunk, draft_pos0 + (int32_t) offset)", loop)

    def test_same_suffix_arithmetic(self) -> None:
        """Behavioural: target and draft slices agree for cold and resident."""
        for prompt_len, claim in ((20, 0), (20, 10), (45, 27)):
            with self.subTest(prompt_len=prompt_len, claim=claim):
                # resident pass decodes [claim, prompt_len); cold decodes all.
                tgt_first, tgt_count = claim, prompt_len - claim
                draft_first, draft_count = tgt_first, tgt_count
                self.assertEqual(draft_first, tgt_first)
                self.assertEqual(draft_count, tgt_count)
                self.assertGreaterEqual(tgt_count, 1)

    def test_draft_clear_is_paired_with_the_target_clear(self) -> None:
        """Both halves clear together or not at all.

        Clearing only the draft would leave the pair mismatched: draft empty
        while the target holds the prefix, and pending_h naming a predecessor the
        draft no longer contains.
        """
        text = _shim_text()
        marker = 'if (!resident_pass) {\n                llama_memory_clear(mem_tgt, true);'
        at = text.find(marker)
        self.assertGreater(at, -1, "target clear must stay guarded by !resident_pass")
        block = text[at:at + 700]
        self.assertIn("llama_memory_clear(mem_dft, true);", block,
                      "the draft clear must sit inside the same !resident_pass guard")


class PostMutationExitTests(unittest.TestCase):
    """Defect C: any exit after target mutation must leave the target untrusted.

    `llama_decode(ctx_tgt, validate)` writes the speculative batch into mem_tgt
    before it can report failure, so an error return leaves the frontier inflated
    by uncommitted tokens. Rather than patch each return site, the flag is armed
    once before the first mutation and cleared only at the single proven-canonical
    success exit.
    """

    def test_flag_is_armed_before_any_target_mutation(self) -> None:
        text = _shim_text()
        # Anchor on the arming write itself, not on what happens to follow it:
        # the pair-trust arming now sits between it and `need_replay`.
        arm = text.find("    session->last_target_untrusted = true;\n    // The pair is poisoned")
        self.assertGreater(arm, -1, "the arming write must exist")
        for mutation in ("llama_memory_clear(mem_tgt, true);",
                         "llama_decode(ctx_tgt,"):
            at = text.find(mutation, arm)
            self.assertGreater(at, arm, f"{mutation} must follow the arming write")

    def test_only_the_canonical_exit_clears_the_flag(self) -> None:
        """The disarm is a physical check, not an unconditional assignment."""
        text = _shim_text()
        self.assertIn(
            "session->last_target_untrusted =\n"
            "        llama_memory_seq_pos_max(mem_tgt, 0) != n_past - 1;",
            text,
        )

    def test_disarm_follows_every_error_return(self) -> None:
        """Every `return false` after arming exits with the flag still set."""
        text = _shim_text()
        # Anchor on the arming write itself, not on what happens to follow it:
        # the pair-trust arming now sits between it and `need_replay`.
        arm = text.find("    session->last_target_untrusted = true;\n    // The pair is poisoned")
        disarm = text.find("llama_memory_seq_pos_max(mem_tgt, 0) != n_past - 1;")
        self.assertGreater(disarm, arm)
        body = text[arm:disarm]
        # Each error return between arming and the canonical exit leaves the
        # flag armed precisely because none of them clears it.
        self.assertGreater(body.count("return false;"), 0)
        self.assertNotIn("last_target_untrusted = false", body)

    def test_a_refused_seq_rm_keeps_the_flag_armed(self) -> None:
        """A refusal leaves the frontier long, so the physical disarm fails."""
        pos_max_after_refusal = 40  # rejected tail still resident
        n_past = 30
        self.assertNotEqual(pos_max_after_refusal, n_past - 1)
        self.assertTrue(pos_max_after_refusal != n_past - 1)

    def test_canonical_completion_clears_the_flag(self) -> None:
        """Recovery: a clean completion re-establishes trust."""
        n_past = 30
        pos_max = 29
        self.assertFalse(pos_max != n_past - 1)

    def test_untrusted_blocks_the_next_resident_claim(self) -> None:
        text = _shim_text()
        self.assertIn("!target_untrusted_from_previous", text)
        read_at = text.find("const bool target_untrusted_from_previous")
        clear_at = text.find("session->last_target_untrusted = false;", read_at)
        use_at = text.find("!target_untrusted_from_previous", read_at)
        self.assertLess(read_at, clear_at, "read before clearing")
        self.assertLess(clear_at, use_at, "the carried-in copy gates the claim")


class ResidentExportHardFailureTests(unittest.TestCase):
    """Section 14: the CURRENT rebuilt shim must hard-fail on missing exports.

    `CompiledExportTests.test_resident_symbols_are_exported` self-skips when all
    resident symbols are absent, so a build that silently dropped both would
    report a skip rather than a failure. That leniency is only defensible for an
    OLD packaged artifact; for the shim this candidate actually builds, absence
    is a hard failure.
    """

    def setUp(self) -> None:
        self.shim = _built_shim()
        if self.shim is None:
            self.fail(
                "no compiled shim: the current candidate must be built before "
                "qualification -- this is a hard failure, not a skip"
            )
        self.exports = _exports(self.shim)
        self.assertTrue(self.exports, "nm produced no symbols for the current shim")

    def test_current_shim_exports_every_resident_symbol(self) -> None:
        missing = [
            s for s in pm._RESIDENT_PREFIX_REQUIRED_SHIM_SYMBOLS
            if s not in self.exports
        ]
        self.assertEqual(
            missing, [],
            "the current rebuilt shim must export the resident-prefix ABI; "
            "a missing export is a failure, never a skip",
        )

    def test_this_class_cannot_skip(self) -> None:
        """Guard the guard: no test method here may skip.

        Checks the executable bodies only -- prose about skipping is fine, a
        call that actually skips is not.
        """
        import inspect

        for name, fn in vars(ResidentExportHardFailureTests).items():
            if not name.startswith(("test_", "setUp")):
                continue
            if name == "test_this_class_cannot_skip":
                continue  # the guard itself names the call it forbids
            body = inspect.getsource(fn)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.strip().startswith("#")
            )
            # Strip docstrings so prose mentioning the word does not match.
            code = re.sub(r'""".*?"""', "", code, flags=re.S)
            self.assertNotIn(
                "skipTest", code, f"{name} must fail rather than skip"
            )


if __name__ == "__main__":
    unittest.main()

"""Target-context recurrent rollback budget.

Speculative decoding decodes a verification batch of `id_last` plus the draft
into the TARGET context, then removes whatever the target rejected with
`llama_memory_seq_rm(mem_tgt, 0, committed_frontier, -1)`.

For a hybrid/recurrent cache that removal is bounded: llama.cpp only rolls the
recurrent state back while the distance is within `n_rs_seq`
(llama-memory-recurrent.cpp:181-190), and every context Orbit creates has so far
left that budget at its default of zero -- so the rollback can never succeed and
the caller must fall back to a full clear.

The budget these pin is derived, not chosen: the rollback distance is
`draft_len - accepted`, so it never exceeds `draft_n_max`. `id_last` does not
count -- the shim commits it into the frontier before removing anything, so it
always sits below the removal point. Upstream llama.cpp uses the same value
(`need_n_rs_seq()` returns `draft.n_max` for MTP).

This module qualifies the budget only. Nothing here preserves KV across
completions; that lifecycle work is deliberately out of scope.
"""

from __future__ import annotations

import unittest

from orbit.native_llama.client import (
    MTP_DRAFT_N_MAX,
    target_rs_budget_for_profile,
)
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    ORNITH15_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
)


class _Profile:
    def __init__(self, *, architecture: str, verified: bool = True) -> None:
        self.architecture = architecture
        self.verified = verified


QWEN35MOE = _Profile(architecture="qwen35moe")
GEMMA4 = _Profile(architecture="gemma4")


class DerivationTests(unittest.TestCase):
    """The budget must follow the verification batch, not a literal."""

    def test_budget_equals_the_draft_length(self) -> None:
        self.assertEqual(
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
            MTP_DRAFT_N_MAX,
        )

    def test_budget_covers_the_worst_case_rejection(self) -> None:
        """Worst case rejects the whole draft: distance = draft_len - 0."""
        worst_case_distance = MTP_DRAFT_N_MAX
        self.assertGreaterEqual(
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
            worst_case_distance,
        )

    def test_budget_is_not_inflated_beyond_the_worst_case(self) -> None:
        """Each extra snapshot multiplies the recurrent buffer, so no padding.

        `id_last` is committed into the frontier before the removal, so it never
        contributes to the rollback distance and must not be budgeted for.
        """
        self.assertEqual(
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
            MTP_DRAFT_N_MAX,
        )
        self.assertLess(
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
            MTP_DRAFT_N_MAX + 1,
        )


class ScopeTests(unittest.TestCase):
    """The budget costs recurrent-state memory, so it is not granted freely."""

    def test_no_budget_without_mtp(self) -> None:
        self.assertEqual(
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=False), 0
        )

    def test_no_budget_for_a_non_recurrent_architecture(self) -> None:
        self.assertEqual(
            target_rs_budget_for_profile(GEMMA4, mtp_requested=True), 0
        )

    def test_no_budget_without_a_profile(self) -> None:
        self.assertEqual(target_rs_budget_for_profile(None, mtp_requested=True), 0)

    def test_no_budget_for_an_unverified_profile(self) -> None:
        unverified = _Profile(architecture="qwen35moe", verified=False)
        self.assertEqual(
            target_rs_budget_for_profile(unverified, mtp_requested=True), 0
        )

    def test_architecture_decides_not_the_model_name(self) -> None:
        """Any qwen35moe build qualifies; nothing keys off a model identity."""
        other = _Profile(architecture="qwen35moe")
        self.assertEqual(
            target_rs_budget_for_profile(other, mtp_requested=True),
            target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
        )

    def test_architecture_match_is_case_insensitive_and_trimmed(self) -> None:
        self.assertEqual(
            target_rs_budget_for_profile(
                _Profile(architecture="  QWEN35MOE "), mtp_requested=True
            ),
            MTP_DRAFT_N_MAX,
        )


class ContextConfigurationTests(unittest.TestCase):
    """The budget has to reach the context params, not just be computable.

    Extracts the real assignment `load()` makes to `ctx_params.n_rs_seq` and
    evaluates it against a stub client, so a budget that is derived correctly
    but never applied -- or applied with a different value -- is caught.
    """

    def _eval_ctx_assignment(self, field: str, *, mtp: bool, architecture: str):
        import ast
        import inspect
        import textwrap

        import orbit.native_llama.client as client_mod

        tree = ast.parse(textwrap.dedent(inspect.getsource(client_mod.NativeLlamaClient.load)))
        expr = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == field
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "ctx_params"
                ):
                    expr = node.value
        self.assertIsNotNone(expr, f"load() must assign ctx_params.{field}")

        class _C:
            use_mtp_experimental = mtp

        class _Self:
            config = _C()
            model_profile = _Profile(architecture=architecture)

        return eval(  # noqa: S307 - evaluating the production expression is the point
            compile(ast.Expression(expr), "<load>", "eval"),
            {
                "MTP_DRAFT_N_MAX": client_mod.MTP_DRAFT_N_MAX,
                "target_rs_budget_for_profile": client_mod.target_rs_budget_for_profile,
                "getattr": getattr,
            },
            {"self": _Self()},
        )

    def test_qualified_self_mtp_target_gets_the_derived_budget(self) -> None:
        self.assertEqual(
            self._eval_ctx_assignment("n_rs_seq", mtp=True, architecture="qwen35moe"),
            MTP_DRAFT_N_MAX,
        )

    def test_non_mtp_target_gets_no_budget(self) -> None:
        self.assertEqual(
            self._eval_ctx_assignment("n_rs_seq", mtp=False, architecture="qwen35moe"),
            0,
        )

    def test_unrelated_architecture_gets_no_budget(self) -> None:
        self.assertEqual(
            self._eval_ctx_assignment("n_rs_seq", mtp=True, architecture="gemma4"), 0
        )

    def test_outputs_max_tracks_the_draft_length(self) -> None:
        self.assertEqual(
            self._eval_ctx_assignment("n_outputs_max", mtp=True, architecture="qwen35moe"),
            1 + MTP_DRAFT_N_MAX,
        )


class DerivationCouplingTests(unittest.TestCase):
    """The budget must FOLLOW the draft constant, not coincide with it."""

    def test_budget_tracks_a_changed_draft_length(self) -> None:
        from unittest.mock import patch

        import orbit.native_llama.client as client_mod

        with patch.object(client_mod, "MTP_DRAFT_N_MAX", 7):
            self.assertEqual(
                client_mod.target_rs_budget_for_profile(QWEN35MOE, mtp_requested=True),
                7,
            )


class StrictPrefixInvariantTests(unittest.TestCase):
    """D3a must not touch the committed-prefix identity check."""

    def test_exact_prefix_still_required_for_reuse(self) -> None:
        from ctypes import c_void_p
        from unittest.mock import MagicMock, patch

        from orbit.native_llama.client import NativeLlamaClient
        from orbit.native_llama.session_state import NativeSessionState

        c = NativeLlamaClient.__new__(NativeLlamaClient)
        c._session = NativeSessionState(session_id="sp")
        c._session.ctx_tgt = c_void_p(0x1)
        c._session.committed_sequence_tokens = [1, 2, 3]
        c._session.cached_prompt_tokens = []
        lib = MagicMock()
        lib.llama_get_memory.return_value = None
        c.lib = MagicMock(lib=lib)

        with patch.object(NativeLlamaClient, "_qwen3_coder_native_protocol", lambda self: False), \
             patch.object(NativeLlamaClient, "_invalidate_committed_sequence", lambda self: None):
            # exact prefix -> reuse the committed length
            self.assertEqual(c._prepare_memory_for_prompt([1, 2, 3, 4]), 3)
            # divergent prefix -> must NOT reuse the committed length
            c._session.committed_sequence_tokens = [1, 2, 3]
            c._session.cached_prompt_tokens = []
            self.assertNotEqual(c._prepare_memory_for_prompt([1, 2, 9, 4]), 3)


class LoadOrderingTests(unittest.TestCase):
    """The budget reads `model_profile`, so the profile must already exist.

    `load()` resolves the profile before building the target context. If that
    order ever flipped the budget would silently read None and evaluate to 0,
    disabling rollback with no error anywhere.
    """

    def test_profile_is_resolved_before_the_target_context_is_built(self) -> None:
        import inspect

        from orbit.native_llama.client import NativeLlamaClient

        body = inspect.getsource(NativeLlamaClient.load)
        profile_at = body.index("_initialize_model_profile()")
        context_at = body.index("llama_init_from_model")
        self.assertLess(
            profile_at,
            context_at,
            "model_profile must be resolved before the target context is created",
        )


class ProfileIdentityTests(unittest.TestCase):
    """Sanity: the profiles this is expected to affect, and those it is not."""

    def test_known_profile_ids_are_distinct(self) -> None:
        self.assertNotEqual(ORNITH15_PROFILE_ID, GEMMA4_PROFILE_ID)
        self.assertNotEqual(ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID)


if __name__ == "__main__":
    unittest.main()

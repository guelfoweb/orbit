"""Production reachability: cold turn bootstraps identity, next turn reuses it.

Persistent self-MTP state is only worth anything if the real `/chat` path can
reach it. That requires two things the runtime previously did not do: publish a
committed identity after an MTP turn, and derive a resident claim from it on the
next turn. Before this plumbing existed the MTP entry point invalidated identity
unconditionally, so `len(committed)` was always 0 and no claim could ever be
derived -- the backend's resident path was unreachable in production.

These tests drive the real methods on a real client object with the native layer
stubbed, so they prove the runtime decision, not a description of it.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.mtp_completion import MtpCompletionResult


def _client(committed=(), prompt_tokens=(), coder=False):
    """A client with just enough wired for the identity decisions."""
    c = object.__new__(NativeLlamaClient)
    c._session = MagicMock()
    c._session.committed_sequence_tokens = list(committed)
    c.tokenize = lambda text: list(prompt_tokens)
    c._qwen3_coder_native_protocol = lambda: coder
    return c


def _result(success=True, pair_canonical=True, generated=(7, 8),
            resident=(1, 2, 3, 7)):
    """A completion result.

    `resident` is what the backend measured as physically present in target KV,
    and is deliberately NOT `prompt + generated`: a sampled token enters
    `generated` before it is decoded into the sequence, so the two differ by the
    final token. Identity is published from `resident` alone.
    """
    return MtpCompletionResult(
        enabled=True, success=success, error=None,
        pair_canonical=pair_canonical, generated_tokens=tuple(generated),
        resident_tokens=tuple(resident),
    )


class ResidentClaimDerivationTests(unittest.TestCase):
    """Python decides eligibility; strict equality only."""

    def test_no_committed_identity_yields_no_claim(self) -> None:
        c = _client(committed=(), prompt_tokens=(1, 2, 3))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", []), 0)

    def test_exact_proper_prefix_yields_the_committed_length(self) -> None:
        c = _client(prompt_tokens=(1, 2, 3, 4, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 3)

    def test_prefix_mismatch_yields_no_claim(self) -> None:
        c = _client(prompt_tokens=(1, 2, 9, 4, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_claim_equal_to_prompt_length_is_denied(self) -> None:
        """Defect A: a whole-prompt claim leaves no suffix, so sampling would
        read logits from the previous completion."""
        c = _client(prompt_tokens=(1, 2, 3))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_claim_longer_than_the_prompt_is_denied(self) -> None:
        c = _client(prompt_tokens=(1, 2))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_no_longest_common_prefix_relaxation(self) -> None:
        """A shared head is not identity; anything short of exact yields 0."""
        c = _client(prompt_tokens=(1, 2, 3, 99, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3, 4]), 0)

    def test_coder_protocol_opts_out(self) -> None:
        c = _client(prompt_tokens=(1, 2, 3, 4), coder=True)
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2]), 0)

    def test_tokenizer_failure_falls_back_to_cold(self) -> None:
        c = _client(prompt_tokens=(1, 2, 3, 4))
        def boom(_text):
            raise RuntimeError("tokenizer unavailable")
        c.tokenize = boom
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2]), 0)


class CommittedPublicationTests(unittest.TestCase):
    """Publication follows the backend's physical verdict, nothing else."""

    def _publish(self, client, result):
        client._invalidate_committed_sequence = MagicMock()
        client._session.committed_sequence_tokens = [9, 9, 9]
        client._publish_mtp_committed_identity(result, "p")
        return client

    @staticmethod
    def _published(client):
        """What identity the helper actually left on the session."""
        return client._session.committed_sequence_tokens

    def test_canonical_success_publishes_identity(self) -> None:
        """A: cold canonical success must publish, or reuse cannot bootstrap."""
        c = self._publish(_client(prompt_tokens=(1, 2, 3)), _result())
        self.assertEqual(self._published(c), [1, 2, 3, 7])
        c._invalidate_committed_sequence.assert_not_called()

    def test_failed_completion_drops_identity(self) -> None:
        c = self._publish(_client(prompt_tokens=(1, 2, 3)),
                          _result(success=False))
        c._invalidate_committed_sequence.assert_called_once()
        self.assertEqual(self._published(c), [9, 9, 9],
                         "nothing may be published on a non-canonical exit")

    def test_resident_reuse_with_poisoned_pair_does_not_publish(self) -> None:
        """G: resident_reuse_active is NOT pair_canonical."""
        r = MtpCompletionResult(
            enabled=True, success=True, error=None,
            resident_reuse_active=True, pair_canonical=False,
            generated_tokens=(7, 8), resident_tokens=(1, 2, 3, 7),
        )
        c = self._publish(_client(prompt_tokens=(1, 2, 3)), r)
        c._invalidate_committed_sequence.assert_called_once()
        self.assertEqual(self._published(c), [9, 9, 9],
                         "nothing may be published on a non-canonical exit")

    def test_cold_completion_ending_canonical_publishes(self) -> None:
        """H: no resident reuse, but the pair ended canonical -> publish."""
        r = MtpCompletionResult(
            enabled=True, success=True, error=None,
            resident_reuse_active=False, pair_canonical=True,
            generated_tokens=(7, 8), resident_tokens=(1, 2, 3, 7),
        )
        c = self._publish(_client(prompt_tokens=(1, 2, 3)), r)
        self.assertEqual(self._published(c), [1, 2, 3, 7])

    def test_missing_resident_ids_drops_identity(self) -> None:
        """Without a measured resident sequence there is nothing sound to claim.

        Retokenizing the text is not equivalent, and prompt + generated
        overstates residency by one token, so the only safe action is to drop.
        """
        c = self._publish(_client(prompt_tokens=(1, 2, 3)),
                          _result(resident=()))
        c._invalidate_committed_sequence.assert_called_once()
        self.assertEqual(self._published(c), [9, 9, 9],
                         "nothing may be published on a non-canonical exit")

    def test_identity_is_the_measured_resident_sequence(self) -> None:
        """Identity must equal what the backend measured, verbatim."""
        c = _client(prompt_tokens=(1, 2, 3))
        c._invalidate_committed_sequence = MagicMock()
        c._publish_mtp_committed_identity(
            _result(generated=(7, 8), resident=(1, 2, 3, 7)), "p"
        )
        self.assertEqual(c._session.committed_sequence_tokens, [1, 2, 3, 7])

    def test_identity_is_never_prompt_plus_generated(self) -> None:
        """The off-by-one that would corrupt the next turn.

        prompt=(1,2,3) and generated=(7,8) would reconstruct [1,2,3,7,8] -- one
        token longer than the [1,2,3,7] actually resident. Publishing that makes
        the next turn skip prefill for a position that was never decoded, so
        every following token lands one position early and output is silently
        wrong. The published identity must track the measurement, not the
        reconstruction.
        """
        c = _client(prompt_tokens=(1, 2, 3))
        c._invalidate_committed_sequence = MagicMock()
        c._publish_mtp_committed_identity(
            _result(generated=(7, 8), resident=(1, 2, 3, 7)), "p"
        )
        published = c._session.committed_sequence_tokens
        self.assertNotEqual(
            published, [1, 2, 3, 7, 8],
            "identity must not be rebuilt from prompt + generated_tokens",
        )
        self.assertEqual(len(published), 4)


class ClaimWiringTests(unittest.TestCase):
    """The derived claim must actually reach the native call.

    Deriving N correctly is worthless if the value is not passed, and a mutation
    that hardcodes 0 at the call site is invisible to tests that only exercise
    the derivation helper. These assert the wiring itself, by AST rather than by
    substring, so an unreachable or constant argument is caught.
    """

    def setUp(self) -> None:
        import ast, inspect
        self.ast = ast
        src = inspect.getsource(NativeLlamaClient._try_complete_with_mtp_experimental)
        self.fn = ast.parse(src.lstrip()).body[0]

    def _completion_call(self):
        for node in self.ast.walk(self.fn):
            if (isinstance(node, self.ast.Call)
                    and getattr(node.func, "id", None) == "run_persistent_mtp_completion"):
                return node
        self.fail("run_persistent_mtp_completion is not called")

    def test_the_completion_receives_a_resident_claim(self) -> None:
        call = self._completion_call()
        kw = {k.arg: k.value for k in call.keywords}
        self.assertIn("resident_prefix_len", kw,
                      "the derived claim must be passed to the native completion")

    def test_the_claim_is_the_derived_value_not_a_constant(self) -> None:
        """A hardcoded 0 would silently disable resident reuse forever."""
        call = self._completion_call()
        kw = {k.arg: k.value for k in call.keywords}
        node = kw["resident_prefix_len"]
        self.assertNotIsInstance(
            node, self.ast.Constant,
            "resident_prefix_len must be the derived variable, not a literal",
        )
        self.assertEqual(getattr(node, "id", None), "resident_prefix_len")

    def test_the_claim_is_derived_before_the_reset(self) -> None:
        """Derivation reads committed identity; the reset clears the claim, so
        the value must be computed first and applied afterwards."""
        derive = reset = complete = None
        for node in self.ast.walk(self.fn):
            if isinstance(node, self.ast.Call):
                attr = getattr(node.func, "attr", None)
                name = getattr(node.func, "id", None)
                if attr == "_resident_prefix_len_for_mtp" and derive is None:
                    derive = node.lineno
                if name == "reset_persistent_mtp_session" and reset is None:
                    reset = node.lineno
                if name == "run_persistent_mtp_completion" and complete is None:
                    complete = node.lineno
        self.assertIsNotNone(derive); self.assertIsNotNone(reset)
        self.assertIsNotNone(complete)
        self.assertLess(derive, reset, "derive the claim before the reset")
        self.assertLess(reset, complete, "apply the claim after the reset")

    def test_identity_is_snapshotted_before_the_reset(self) -> None:
        snapshot = None
        for node in self.ast.walk(self.fn):
            if (isinstance(node, self.ast.Name)
                    and node.id == "committed_at_entry"
                    and isinstance(node.ctx, self.ast.Store)):
                snapshot = node.lineno
                break
        self.assertIsNotNone(snapshot, "identity must be snapshotted at entry")


class BootstrapLifecycleTests(unittest.TestCase):
    """The decisive production-reachability property, end to end."""

    def test_cold_turn_then_resident_turn(self) -> None:
        """Turn 1 bootstraps identity; turn 2 derives a claim from it.

        The identity carried between turns is the backend's MEASURED resident
        sequence. Turn 1 prompts (1,2,3) and samples (4,5), but only (4) has
        been decoded into KV when the completion ends, so residency is
        [1,2,3,4] -- not the [1,2,3,4,5] a prompt+generated reconstruction would
        produce. Turn 2's claim must describe the former, or it would name a
        position that was never decoded.
        """
        # TURN 1: no identity, so no claim; a cold canonical success publishes.
        c = _client(committed=(), prompt_tokens=(1, 2, 3))
        self.assertEqual(c._resident_prefix_len_for_mtp("p1", []), 0)

        c._session.committed_sequence_tokens = []
        c._invalidate_committed_sequence = (
            lambda: setattr(c._session, "committed_sequence_tokens", [])
        )
        c._publish_mtp_committed_identity(
            _result(generated=(4, 5), resident=(1, 2, 3, 4)), "p1"
        )
        committed = c._session.committed_sequence_tokens
        self.assertEqual(committed, [1, 2, 3, 4])
        self.assertNotEqual(
            committed, [1, 2, 3, 4, 5],
            "identity must be the measured residency, not prompt + generated",
        )

        # TURN 2: the next prompt begins with exactly that sequence.
        turn2_tokens = committed + [6, 7]
        c2 = _client(committed=committed, prompt_tokens=tuple(turn2_tokens))
        claim = c2._resident_prefix_len_for_mtp("p2", committed)
        self.assertEqual(claim, len(committed))
        self.assertTrue(0 < claim < len(turn2_tokens))

    def test_a_poisoned_turn_forces_the_next_turn_cold(self) -> None:
        c = _client(committed=(), prompt_tokens=(1, 2, 3))
        c._session.committed_sequence_tokens = [1, 2, 3]
        c._invalidate_committed_sequence = (
            lambda: setattr(c._session, "committed_sequence_tokens", [])
        )
        c._publish_mtp_committed_identity(
            _result(pair_canonical=False, generated=(4, 5),
                    resident=(1, 2, 3, 4)), "p1")
        self.assertEqual(c._session.committed_sequence_tokens, [],
                         "a poisoned pair must leave no identity behind")
        c2 = _client(committed=[], prompt_tokens=(1, 2, 3, 4))
        self.assertEqual(c2._resident_prefix_len_for_mtp("p2", []), 0)


if __name__ == "__main__":
    unittest.main()

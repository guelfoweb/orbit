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
    """A client with just enough wired for the identity decisions.

    A REAL session, not a mock: committed identity is owned by the session's
    `CommittedIdentity`, so a mocked session would bypass the owner entirely
    and the tests would assert against a stand-in rather than the real state.
    """
    from orbit.native_llama.committed_identity import CommittedIdentity
    from orbit.native_llama.session_state import NativeSessionState

    c = object.__new__(NativeLlamaClient)
    c._session = NativeSessionState(session_id="test")
    c.tokenize = lambda text: list(prompt_tokens)
    c._qwen3_coder_native_protocol = lambda: coder
    c._session.bind_committed_identity(
        CommittedIdentity(
            tokenize=lambda text: c.tokenize(text),
            coder_protocol=lambda: c._qwen3_coder_native_protocol(),
        )
    )
    c._session.committed_sequence_tokens = list(committed)
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
        c = _client(committed=(1, 2, 3), prompt_tokens=(1, 2, 3, 4, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 3)

    def test_prefix_mismatch_yields_no_claim(self) -> None:
        c = _client(committed=(1, 2, 3), prompt_tokens=(1, 2, 9, 4, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_claim_equal_to_prompt_length_is_denied(self) -> None:
        """Defect A: a whole-prompt claim leaves no suffix, so sampling would
        read logits from the previous completion."""
        c = _client(committed=(1, 2, 3), prompt_tokens=(1, 2, 3))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_claim_longer_than_the_prompt_is_denied(self) -> None:
        c = _client(committed=(1, 2, 3), prompt_tokens=(1, 2))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0)

    def test_no_longest_common_prefix_relaxation(self) -> None:
        """A shared head is not identity; anything short of exact yields 0."""
        c = _client(committed=(1, 2, 3, 4), prompt_tokens=(1, 2, 3, 99, 5))
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2, 3, 4]), 0)

    def test_coder_protocol_opts_out(self) -> None:
        c = _client(committed=(1, 2), prompt_tokens=(1, 2, 3, 4), coder=True)
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2]), 0)

    def test_tokenizer_failure_falls_back_to_cold(self) -> None:
        c = _client(committed=(1, 2), prompt_tokens=(1, 2, 3, 4))
        def boom(_text):
            raise RuntimeError("tokenizer unavailable")
        c.tokenize = boom
        self.assertEqual(c._resident_prefix_len_for_mtp("p", [1, 2]), 0)


class CommittedPublicationTests(unittest.TestCase):
    """Publication follows the backend's physical verdict, nothing else."""

    def _publish(self, client, result):
        """Drive the real publisher over real state.

        `_invalidate_committed_sequence` is deliberately NOT mocked: identity is
        owned by the session's `CommittedIdentity`, which clears itself, so a
        mock there would observe a stand-in instead of the authoritative state.
        A dropped identity is therefore an EMPTY sequence, not a call count.
        """
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

    def test_failed_completion_drops_identity(self) -> None:
        c = self._publish(_client(prompt_tokens=(1, 2, 3)),
                          _result(success=False))
        self.assertEqual(self._published(c), [],
                         "a non-canonical exit must leave no identity behind")

    def test_resident_reuse_with_poisoned_pair_does_not_publish(self) -> None:
        """G: resident_reuse_active is NOT pair_canonical."""
        r = MtpCompletionResult(
            enabled=True, success=True, error=None,
            resident_reuse_active=True, pair_canonical=False,
            generated_tokens=(7, 8), resident_tokens=(1, 2, 3, 7),
        )
        c = self._publish(_client(prompt_tokens=(1, 2, 3)), r)
        self.assertEqual(self._published(c), [],
                         "a non-canonical exit must leave no identity behind")

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
        self.assertEqual(self._published(c), [],
                         "a non-canonical exit must leave no identity behind")

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


class OwnershipContractTests(unittest.TestCase):
    """The extraction's own guarantees: one owner, and no lost state.

    These cover behaviour that exists ONLY because identity moved out of
    `client.py`, so nothing older can protect them.
    """

    def test_binding_a_wired_owner_carries_tokens_across(self) -> None:
        """Binding must not silently drop an already-recorded identity.

        A session records identity before the client finishes wiring its
        tokenizer; swapping in the wired owner must move those tokens, not
        discard them. Dropping them would look like a harmless reset and
        silently cost every subsequent reuse.
        """
        from orbit.native_llama.committed_identity import CommittedIdentity
        from orbit.native_llama.session_state import NativeSessionState

        session = NativeSessionState(session_id="carry")
        session.committed_sequence_tokens = [1, 2, 3]
        wired = CommittedIdentity(tokenize=lambda t: [], coder_protocol=lambda: False)
        session.bind_committed_identity(wired)

        self.assertEqual(wired.tokens, [1, 2, 3], "bind dropped the identity")
        self.assertEqual(session.committed_sequence_tokens, [1, 2, 3])
        self.assertIs(session.committed_identity, wired, "one owner only")

    def test_commit_records_prompt_and_generated_exactly(self) -> None:
        """`commit` must record the full resident sequence, in order.

        Understating it -- dropping the generated tail -- is a false cache hit:
        the next turn would claim a prefix shorter than KV and then diverge.
        """
        c = _client()
        c._commit_sequence([1, 2, 3], [4, 5])
        self.assertEqual(c._session.committed_sequence_tokens, [1, 2, 3, 4, 5])

    def test_a_duck_typed_session_keeps_its_pre_extraction_behaviour(self) -> None:
        """A session that only carries the token attribute must still work.

        Identity used to live in `session.committed_sequence_tokens`, so any
        object exposing it was a valid session. Several call sites rely on that.
        Requiring a `NativeSessionState` would be a behaviour change.
        """
        import types

        c = object.__new__(NativeLlamaClient)
        c._session = types.SimpleNamespace(committed_sequence_tokens=[1, 2])
        c.tokenize = lambda text: [1, 2, 3]
        c._qwen3_coder_native_protocol = lambda: False

        c._commit_sequence([7], [8])
        self.assertEqual(c._session.committed_sequence_tokens, [7, 8])
        c._invalidate_committed_sequence()
        self.assertEqual(c._session.committed_sequence_tokens, [])

    def test_a_duck_typed_session_still_derives_a_resident_claim(self) -> None:
        """The claim must use the CLIENT's tokenizer, on any session shape.

        This is the operation that actually regressed once: an adapter that
        stubbed the tokenizer returned 0 here where the pre-extraction code
        returned the committed length, silently refusing every reuse. The
        failure was invisible to the suite because the duck-typed test only
        exercised commit and invalidate -- the two operations that did NOT
        regress. Derivation is the one that did.
        """
        import types

        def client(committed, prompt, coder=False):
            c = object.__new__(NativeLlamaClient)
            c._session = types.SimpleNamespace(
                committed_sequence_tokens=list(committed), session_id="duck"
            )
            c.tokenize = lambda text: list(prompt)
            c._qwen3_coder_native_protocol = lambda: coder
            return c

        exact = client([1, 2, 3], [1, 2, 3, 4, 5])
        self.assertEqual(
            exact._resident_prefix_len_for_mtp("p", [1, 2, 3]), 3,
            "an exact proper prefix must yield the committed length; 0 here "
            "means the client's tokenizer is not reaching the derivation",
        )
        self.assertEqual(
            client([1, 2, 9], [1, 2, 3, 4, 5])._resident_prefix_len_for_mtp("p", [1, 2, 9]), 0,
            "a divergent prefix must refuse reuse",
        )
        self.assertEqual(
            client([1, 2, 3], [1, 2, 3])._resident_prefix_len_for_mtp("p", [1, 2, 3]), 0,
            "a whole-prompt claim leaves no suffix to decode",
        )
        self.assertEqual(
            client([1, 2], [1, 2, 3, 4], coder=True)._resident_prefix_len_for_mtp("p", [1, 2]), 0,
            "the coder opt-out must still reach the derivation",
        )

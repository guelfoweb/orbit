"""The contract of `_initialize_self_mtp_session`'s return value.

RUNTIME-AUDIT-2. The boolean is a fall-through guard -- "self-MTP handled this
attempt; do not try external-draft" -- and NOT a success signal. Four of its
five `True` sites build no runtime.

The distinction was untested rather than untrue. Flipping the SUCCESS return
`True -> False` survived the whole suite, while flipping any of the four
terminal-failure returns was caught. That asymmetry is the gap these tests
close: every `True` site is pinned here with the runtime it did or did not
build, so a future reader cannot reintroduce the "True means built" reading by
returning False from the one site where it happens to be true.

No production behaviour is asserted to change; these tests pass unmodified
against the audited source.
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from orbit.native_llama import client as client_mod
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.mtp_session_lifecycle import MtpSessionLifecycle
from orbit.native_llama.session_state import NativeSessionState


class _Eligibility:
    """Patch BOTH stages of `_self_mtp_eligible`, as a context manager.

    Eligibility is decided in two stages: the verified identity is consulted
    for whether any artifact of this profile declares `self_mtp` at all, and
    only then is the ~46 s digest of the artifact worth paying. Both are
    patched so these tests are isolated from the real
    `VERIFIED_NATIVE_MODEL_IDENTITIES` registry -- otherwise the outcome of
    every case below would depend on live registry contents rather than on the
    branch each test is pinning.
    """

    def __init__(self, *, supported=None, error=None):
        identity = types.SimpleNamespace(
            artifact_capabilities={"self_mtp": ("sha",)}
        )
        digest = (
            mock.patch.object(
                client_mod, "verified_artifact_supports", side_effect=error
            )
            if error is not None
            else mock.patch.object(
                client_mod, "verified_artifact_supports", return_value=supported
            )
        )
        self._patches = [
            mock.patch.object(
                client_mod, "verified_native_model_identity", return_value=identity
            ),
            digest,
        ]

    def __enter__(self):
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False


class _Runtime:
    """A constructed self-MTP runtime, deterministic and native-free."""

    self_mtp = True

    def __init__(self) -> None:
        self.ctx_dft = "ctx-dft"
        self.spec = "spec"


def _client(*, ctx_tgt="ctx-tgt", mtp_supported=True, draft="/models/draft.gguf"):
    """A client whose EXTERNAL-DRAFT path is fully eligible.

    Deliberate: if external-draft were unavailable anyway, "it was not
    attempted" would prove nothing about the return value. Here the only thing
    that can stop it is the boolean.
    """
    c = object.__new__(NativeLlamaClient)
    session = NativeSessionState(session_id="contract")
    session.ctx_tgt = ctx_tgt
    c._session = session
    c._mtp_lifecycle = MtpSessionLifecycle(lambda: session, free_runtime=lambda r: None)
    c.paths = types.SimpleNamespace(
        llama_root=None, model="artifact.gguf", mtp_available=True,
        draft_mtp_model=draft, fallback_reason=None,
    )
    c.config = types.SimpleNamespace(
        use_mtp_experimental=True, context_tokens=8, batch_size=1,
        ubatch_size=1, threads=1, threads_batch=1,
    )
    c.model_profile = types.SimpleNamespace(
        profile_id=ORNITH15_PROFILE_ID, verified=True, mtp_supported=mtp_supported
    )
    c._model = "model-handle"
    c._free_persistent_mtp_session = lambda: None
    return c


class TerminalOutcomesReturnTrueTests(unittest.TestCase):
    """Every failure that OWNS the attempt returns True and builds nothing."""

    def test_a_capability_error_is_terminal_and_built_nothing(self) -> None:
        c = _client()
        with _Eligibility(error=OSError("io")):
            handled = c._initialize_self_mtp_session()

        self.assertTrue(handled, "a capability error must be terminal")
        self.assertIsNone(
            c._mtp_lifecycle.runtime,
            "True here cannot mean 'built': nothing was constructed",
        )
        self.assertFalse(c._session.mtp_enabled)
        self.assertTrue(c._session.mtp_failed)

    def test_a_missing_target_context_is_terminal_and_built_nothing(self) -> None:
        c = _client(ctx_tgt=None)
        with _Eligibility(supported=True):
            handled = c._initialize_self_mtp_session()

        self.assertTrue(handled)
        self.assertIsNone(c._mtp_lifecycle.runtime)
        self.assertEqual(c._session.mtp_failure_reason, "target-context-missing")

    def test_a_constructor_failure_is_terminal_and_built_nothing(self) -> None:
        c = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", side_effect=RuntimeError("boom")
        ):
            handled = c._initialize_self_mtp_session()

        self.assertTrue(handled)
        self.assertIsNone(c._mtp_lifecycle.runtime)
        self.assertTrue(c._session.mtp_failed)

    def test_a_missing_self_mtp_abi_is_terminal_and_built_nothing(self) -> None:
        """A shim without the self-MTP symbols surfaces as a constructor error.

        Same terminal shape as any other construction failure, pinned
        separately because it is the case a rebuilt or older shim produces.
        """
        c = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session",
            side_effect=AttributeError("orbit_mtp_session_create"),
        ):
            handled = c._initialize_self_mtp_session()

        self.assertTrue(handled)
        self.assertIsNone(c._mtp_lifecycle.runtime)
        self.assertIn("orbit_mtp_session_create", c._session.mtp_failure_reason)


class SuccessAlsoReturnsTrueTests(unittest.TestCase):
    """The one True that DID build. Untested before this module existed."""

    def test_a_constructed_session_is_terminal(self) -> None:
        c = _client()
        runtime = _Runtime()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", return_value=runtime
        ):
            handled = c._initialize_self_mtp_session()

        self.assertTrue(handled, "a built session must also be terminal")
        self.assertIs(c._mtp_lifecycle.runtime, runtime)
        self.assertTrue(c._session.mtp_enabled)
        self.assertFalse(c._session.mtp_failed)

    def test_success_and_terminal_failure_are_indistinguishable_by_return(self) -> None:
        """The audit's core finding, asserted rather than described.

        Both return True; only the lifecycle state separates them. A caller
        reading this boolean as "built" would be wrong exactly half the time.
        """
        built = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", return_value=_Runtime()
        ):
            built_returned = built._initialize_self_mtp_session()

        failed = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", side_effect=RuntimeError("boom")
        ):
            failed_returned = failed._initialize_self_mtp_session()

        self.assertEqual(built_returned, failed_returned)
        self.assertNotEqual(
            built._session.mtp_enabled, failed._session.mtp_enabled,
            "the states must differ even though the return values do not",
        )


class NotHandledReturnsFalseTests(unittest.TestCase):
    """The only False: not qualified, and no state touched."""

    def test_an_ineligible_artifact_is_not_handled(self) -> None:
        c = _client()
        with _Eligibility(supported=False):
            handled = c._initialize_self_mtp_session()

        self.assertFalse(handled)
        self.assertIsNone(c._mtp_lifecycle.runtime)
        self.assertFalse(c._session.mtp_failed)
        self.assertIsNone(
            c._session.mtp_failure_reason,
            "an unhandled attempt must leave the verdict to the next path",
        )


class CallerHonoursTheGuardTests(unittest.TestCase):
    """What the single production caller actually does with each value.

    External-draft is eligible in every case here, so its absence or presence
    is caused by the boolean alone.
    """

    def test_a_terminal_failure_suppresses_external_draft(self) -> None:
        c = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", side_effect=RuntimeError("boom")
        ), mock.patch.object(
            client_mod, "create_persistent_mtp_session"
        ) as external:
            c._initialize_persistent_mtp_session()

        external.assert_not_called()
        self.assertTrue(
            c._session.mtp_failed,
            "the failure is reported rather than papered over by a fallback",
        )

    def test_a_built_session_suppresses_external_draft(self) -> None:
        c = _client()
        with _Eligibility(supported=True), mock.patch.object(
            client_mod, "create_self_mtp_session", return_value=_Runtime()
        ), mock.patch.object(
            client_mod, "create_persistent_mtp_session"
        ) as external:
            c._initialize_persistent_mtp_session()

        external.assert_not_called()
        self.assertTrue(c._session.mtp_enabled)

    def test_an_unhandled_attempt_lets_external_draft_run(self) -> None:
        c = _client()
        with _Eligibility(supported=False), mock.patch.object(
            client_mod, "create_persistent_mtp_session", return_value=_Runtime()
        ) as external:
            c._initialize_persistent_mtp_session()

        external.assert_called_once()
        self.assertTrue(
            c._session.mtp_enabled,
            "the fall-through path is genuinely reachable, which is what makes "
            "suppressing it in the terminal cases a decision rather than a no-op",
        )


if __name__ == "__main__":
    unittest.main()

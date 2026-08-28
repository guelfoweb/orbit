"""Whether MTP decoding may RUN is runtime state, not artifact metadata.

The historical defect this pins: a self-MTP session was constructed correctly
-- right constructor, one model, `mtp_enabled=True` -- and then every
completion silently decoded normally, because the gate asked
`paths.mtp_available` (and `profile.mtp_supported`), both of which describe the
EXTERNAL-DRAFT registry world and are False for a single-GGUF artifact.

The result was the worst possible shape: the client reported MTP enabled while
drafted/accepted stayed at zero. Measured on the real 21 GiB artifact.

The distinction these tests hold in place:

  metadata  -> HOW a session may be built
  runtime   -> WHETHER speculative decoding may execute
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.native_llama import client as client_mod  # noqa: E402
from orbit.native_llama.client import NativeLlamaClient  # noqa: E402
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID  # noqa: E402
from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime  # noqa: E402


def _runtime(self_mtp: bool) -> PersistentMtpSessionRuntime:
    return PersistentMtpSessionRuntime(
        handle="native-handle", ctx_dft="ctx-dft", spec="spec",
        library=SimpleNamespace(lib=mock.Mock()), self_mtp=self_mtp,
    )


def _client(
    *,
    mtp_requested: bool = True,
    mtp_available: bool = False,
    profile_mtp_supported: bool = False,
    runtime: PersistentMtpSessionRuntime | None = None,
    mtp_enabled: bool = True,
    ctx_tgt: object = "ctx-tgt",
    profile_id: str = ORNITH15_PROFILE_ID,
) -> NativeLlamaClient:
    c = NativeLlamaClient.__new__(NativeLlamaClient)
    c.config = SimpleNamespace(
        use_mtp_experimental=mtp_requested, thinking=False,
        context_tokens=8192, batch_size=256, ubatch_size=128,
        threads=6, threads_batch=6,
    )
    c.paths = SimpleNamespace(
        model=Path("/m.gguf"), llama_root=Path("/llama"),
        mtp_available=mtp_available,
        draft_mtp_model=Path("/draft.gguf") if mtp_available else None,
        fallback_reason=None if mtp_available else "mtp-not-declared",
    )
    c.model_profile = SimpleNamespace(
        profile_id=profile_id, verified=True, mtp_supported=profile_mtp_supported
    )
    c._model = "model-handle"
    c._persistent_mtp_runtime = runtime
    c._session = SimpleNamespace(
        ctx_tgt=ctx_tgt, ctx_dft="ctx-dft" if runtime else None,
        spec="spec" if runtime else None,
        mtp_enabled=mtp_enabled and runtime is not None,
        mtp_failed=False, mtp_failure_reason=None,
    )
    c.mtp_fallback_reason = None
    c.last_mtp_completion = None
    c.cancel_event = threading.Event()
    c._invalidate_committed_sequence = lambda: None
    c._thinking_enabled = lambda t: False
    return c


def _reached_decode(c: NativeLlamaClient) -> bool:
    """Did the production gate hand control to the speculative decode path?

    `run_persistent_mtp_completion` is the first call past every gate, so its
    invocation is the honest signal that MTP decoding actually ran.
    """
    # Raise from the decode call itself: reaching it is the signal, and the
    # exception stops execution before downstream result handling needs a
    # fully-shaped MtpCompletionResult. Building one here would couple this
    # test to fields it is not about.
    sentinel = RuntimeError("reached-speculative-decode")
    with mock.patch.object(
        client_mod, "run_persistent_mtp_completion", side_effect=sentinel
    ) as decode, mock.patch.object(
        client_mod, "reset_persistent_mtp_session",
        side_effect=lambda **kw: kw["runtime"],
    ):
        try:
            c._try_complete_with_mtp_experimental("hi", max_tokens=4)
        except RuntimeError as exc:
            if exc is not sentinel:
                raise
    return decode.called


class HistoricalDefectTests(unittest.TestCase):
    """Fails against the pre-fix candidate; passes only once the gate is right."""

    def test_self_mtp_session_reaches_the_speculative_decode_path(self) -> None:
        c = _client(
            mtp_available=False,          # exactly the real self-MTP shape
            profile_mtp_supported=False,  # Ornith's profile flag
            runtime=_runtime(self_mtp=True),
        )
        self.assertTrue(
            _reached_decode(c),
            "self-MTP session constructed but completion silently decoded normally",
        )

    def test_self_mtp_never_reports_enabled_while_bypassing_mtp(self) -> None:
        """The precise failure shape: enabled=True with zero drafting."""
        c = _client(mtp_available=False, runtime=_runtime(self_mtp=True))
        reached = _reached_decode(c)
        result = c.last_mtp_completion
        if not reached:
            self.assertFalse(
                getattr(result, "enabled", False) and getattr(result, "success", False),
                "claimed MTP success without entering the MTP path",
            )


class ExternalDraftUnchangedTests(unittest.TestCase):
    def test_external_draft_session_still_reaches_decode(self) -> None:
        c = _client(
            mtp_available=True, profile_mtp_supported=True,
            profile_id=QWEN3_CODER_PROFILE_ID, runtime=_runtime(self_mtp=False),
        )
        self.assertTrue(_reached_decode(c))


class NegativeGateTests(unittest.TestCase):
    def test_no_mtp_request_does_not_decode_speculatively(self) -> None:
        c = _client(mtp_requested=False, runtime=_runtime(self_mtp=True))
        self.assertFalse(_reached_decode(c))

    def test_no_constructed_session_does_not_decode_speculatively(self) -> None:
        c = _client(runtime=None)
        self.assertFalse(_reached_decode(c))

    def test_stale_self_mtp_flag_without_a_handle_is_refused(self) -> None:
        """`self_mtp=True` must never be sufficient on its own."""
        c = _client(runtime=None, mtp_enabled=True)
        c._session.mtp_enabled = True   # stale: no runtime behind it
        self.assertFalse(_reached_decode(c))

    def test_stale_mtp_available_without_a_session_is_refused(self) -> None:
        """Metadata saying a draft exists must not imply a live session."""
        c = _client(mtp_available=True, profile_mtp_supported=True, runtime=None)
        c._session.mtp_enabled = True
        self.assertFalse(_reached_decode(c))

    def test_missing_target_context_is_refused(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True), ctx_tgt=None)
        self.assertFalse(_reached_decode(c))

    def test_thinking_mode_is_refused(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True))
        c._thinking_enabled = lambda t: True
        self.assertFalse(_reached_decode(c))

    def test_cancelled_request_is_refused(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True))
        c.cancel_event.set()
        self.assertFalse(_reached_decode(c))


class MetadataGatesStillApplyWithoutSelfMtpTests(unittest.TestCase):
    """The bypass is narrow: it exists only for a real self-MTP session.

    Without these, `self_mtp_session = True` unconditionally -- or derived from
    `mtp_enabled` rather than the runtime -- would pass every other test while
    disabling the profile and draft-availability gates for every model.
    """

    def test_unsupported_profile_still_blocks_when_not_self_mtp(self) -> None:
        c = _client(
            mtp_available=True, profile_mtp_supported=False,
            profile_id=QWEN3_CODER_PROFILE_ID, runtime=_runtime(self_mtp=False),
        )
        self.assertFalse(_reached_decode(c))
        self.assertEqual(c.mtp_fallback_reason, "model_profile_mtp_unsupported")

    def test_missing_draft_artifact_still_blocks_when_not_self_mtp(self) -> None:
        c = _client(
            mtp_available=False, profile_mtp_supported=True,
            profile_id=QWEN3_CODER_PROFILE_ID, runtime=_runtime(self_mtp=False),
        )
        self.assertFalse(_reached_decode(c))
        self.assertEqual(c.mtp_fallback_reason, "mtp-not-declared")

    def test_the_bypass_requires_the_runtime_not_the_enabled_flag(self) -> None:
        """`mtp_enabled` is set for BOTH architectures; only the runtime says which.

        With no runtime at all the decode is unreachable either way, so the
        distinguishing observation is the *reason*: deriving the bypass from
        `mtp_enabled` would skip the metadata gates and leave it unrecorded.
        """
        c = _client(
            mtp_available=False, profile_mtp_supported=False,
            profile_id=QWEN3_CODER_PROFILE_ID, runtime=None,
        )
        c._session.mtp_enabled = True
        self.assertFalse(_reached_decode(c))
        self.assertEqual(c.mtp_fallback_reason, "model_profile_mtp_unsupported")

    def test_absent_runtime_records_missing_draft_artifact(self) -> None:
        c = _client(
            mtp_available=False, profile_mtp_supported=True,
            profile_id=QWEN3_CODER_PROFILE_ID, runtime=None,
        )
        c._session.mtp_enabled = True
        self.assertFalse(_reached_decode(c))
        self.assertEqual(c.mtp_fallback_reason, "mtp-not-declared")


class TeardownClearsReadinessTests(unittest.TestCase):
    def test_free_clears_runtime_and_enabled(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True))
        with mock.patch.object(client_mod, "free_persistent_mtp_session"):
            c._free_persistent_mtp_session()
        self.assertIsNone(c._persistent_mtp_runtime)
        self.assertFalse(c._session.mtp_enabled)
        self.assertFalse(_reached_decode(c))

    def test_repeated_free_is_safe_and_stays_not_ready(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True))
        with mock.patch.object(client_mod, "free_persistent_mtp_session") as freed:
            c._free_persistent_mtp_session()
            c._free_persistent_mtp_session()
        self.assertEqual(freed.call_count, 1)
        self.assertFalse(_reached_decode(c))


if __name__ == "__main__":
    unittest.main()

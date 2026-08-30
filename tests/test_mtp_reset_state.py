"""A session reset must not leave the previous session's MTP status behind.

`reset_session_state` clears the conversation: the KV memory, the cached prompt
tokens, the committed sequence, every prefix and rolling anchor, and -- through
`publish(runtime, clear_failure_reason=True)` -- the session-level
`mtp_failure_reason`. Two client-level fields describing the *last request's*
MTP outcome were left standing, and both reach `/props`.

They fail in opposite directions, which is why each needs its own test:

* `mtp_fallback_reason` is read by `orbit_smoke_harness.mtp_state_from_props`
  as a fallback for `mtp_failure_reason`. A stale one turns a freshly reset,
  healthy backend from `ready` into **`failed`** -- under-reporting.
* `last_mtp_completion` carries `success` plus twenty-odd metrics. A stale one
  reports `on` for a session that has completed nothing yet, and publishes the
  previous session's acceptance ratios and token counts as if they were this
  one's -- over-reporting.

These drive the real `reset_session_state` and assert through the same
`mtp_state_from_props` calculation the harness uses, rather than reading the
private fields alone.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient
from orbit.native_llama.mtp_completion import MtpCompletionResult
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime
from scripts.orbit_smoke_harness import mtp_state_from_props


def _paths() -> NativeLlamaPaths:
    return NativeLlamaPaths(
        llama_root=Path("/llama"),
        build_bin=Path("/llama/build/bin"),
        library=Path("/llama/build/bin/libllama.so"),
        model=Path("/models/target.gguf"),
        draft_mtp_model=Path("/models/draft.gguf"),
        mtp_available=True,
        fallback_reason=None,
        model_id="gemma4-12b-it-q4km",
    )


def _props(client: NativeLlamaClient) -> dict[str, object]:
    """The MTP-status fields of `/props`, built exactly as `app.py` builds them."""
    snapshot = client.session_snapshot()
    return {
        "mtp_experimental_enabled": client.config.use_mtp_experimental,
        "mtp_initialized": snapshot.mtp_initialized,
        "mtp_failure_reason": snapshot.mtp_failure_reason,
        "mtp_fallback_reason": client.mtp_fallback_reason,
        "mtp_last_completion_success": client.last_mtp_completion.success,
    }


class MtpResetStateTests(unittest.TestCase):
    """Every case drives the real reset, then reads the real status shape."""

    def _healthy_client(self, mocked_lib_cls, mocked_reset) -> NativeLlamaClient:
        client = NativeLlamaClient(_paths(), NativeClientConfig(use_mtp_experimental=True))
        fake_lib = mock.Mock()
        fake_lib.llama_get_memory.return_value = object()
        mocked_lib_cls.return_value.lib = fake_lib
        client._session.ctx_tgt = object()
        client._persistent_mtp_runtime = PersistentMtpSessionRuntime(
            handle=object(), ctx_dft=object(), spec=object()
        )
        mocked_reset.return_value = PersistentMtpSessionRuntime(
            handle=object(), ctx_dft=object(), spec=object()
        )
        # `mtp_initialized` is derived from these three, and the harness needs
        # it true for the status to be anything but "requested".
        client._session.ctx_dft = object()
        client._session.spec = object()
        client._session.mtp_enabled = True
        return client

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_case_a_a_stale_fallback_reason_does_not_survive_reset(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """CASE A: fallback stale, completion neutral.

        The harness reads `mtp_fallback_reason` when `mtp_failure_reason` is
        empty, so a leftover reason reports a healthy backend as `failed`.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.mtp_fallback_reason = "thinking-mode"

        self.assertEqual(mtp_state_from_props(_props(client))["status"], "failed")

        client.reset_session_state()

        self.assertIsNone(client.mtp_fallback_reason)
        self.assertEqual(
            mtp_state_from_props(_props(client))["status"], "ready",
            "a reset session must not report the previous request's failure",
        )

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_case_b_a_stale_successful_completion_does_not_survive_reset(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """CASE B: completion stale, fallback neutral.

        The opposite direction: a leftover success reports `on` for a session
        that has not completed anything since the reset.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None, output_tokens=42, elapsed_ms=12.5
        )

        self.assertEqual(mtp_state_from_props(_props(client))["status"], "on")

        client.reset_session_state()

        self.assertFalse(
            client.last_mtp_completion.success,
            "a reset session has completed nothing; it must not claim a success",
        )
        self.assertEqual(mtp_state_from_props(_props(client))["status"], "ready")

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_case_c_both_stale_fields_are_cleared_by_reset(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """CASE C: both stale. Neither clear may depend on the other."""
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.mtp_fallback_reason = "thinking-mode"
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None, output_tokens=42
        )

        client.reset_session_state()

        self.assertIsNone(client.mtp_fallback_reason)
        self.assertFalse(client.last_mtp_completion.success)
        self.assertEqual(mtp_state_from_props(_props(client))["status"], "ready")

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_the_stale_completion_metrics_do_not_survive_reset(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """`/props` publishes the completion's metrics, not just its success.

        A surviving result would attribute the previous session's acceptance
        ratios and token counts to the new one.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None,
            output_tokens=42, acceptance_ratio=0.87, elapsed_ms=99.0,
        )

        client.reset_session_state()

        self.assertEqual(client.last_mtp_completion.output_tokens, 0)
        self.assertIsNone(client.last_mtp_completion.acceptance_ratio)
        self.assertIsNone(client.last_mtp_completion.error)

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_the_cleared_completion_keeps_the_configured_enabled_flag(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """`enabled` describes configuration, not the last request.

        `_mtp_last_completion_payload` returns `None` when it is false, so
        resetting it to `False` would hide the payload for a client that has
        MTP enabled.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None
        )

        client.reset_session_state()

        self.assertTrue(
            client.last_mtp_completion.enabled,
            "reset must restore the configured value, not a blanket False",
        )

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_reset_does_not_disturb_the_session_level_mtp_fields(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """Only the two client-level fields change; the lifecycle keeps its own.

        `publish(clear_failure_reason=True)` already owns the session fields,
        and this fix must not reach into them.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.mtp_fallback_reason = "thinking-mode"

        client.reset_session_state()
        snapshot = client.session_snapshot()

        self.assertTrue(snapshot.mtp_enabled)
        self.assertTrue(snapshot.mtp_initialized)
        self.assertIsNone(snapshot.mtp_failure_reason)
        self.assertIsNotNone(
            client._persistent_mtp_runtime,
            "the reset republishes the runtime; it must not be discarded",
        )

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_a_failed_runtime_reset_still_clears_the_stale_status(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """The discard path returns early -- the clears must precede it.

        A reset whose native re-init fails still destroyed the conversation, so
        the previous request's status is just as stale there.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.mtp_fallback_reason = "thinking-mode"
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None
        )
        mocked_reset.side_effect = RuntimeError("reinit failed")

        client.reset_session_state()

        self.assertIsNone(client.mtp_fallback_reason)
        self.assertFalse(client.last_mtp_completion.success)

    @mock.patch("orbit.native_llama.client.reset_persistent_mtp_session")
    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_the_early_return_paths_leave_session_state_to_its_owner(
        self, mocked_lib_cls, mocked_reset
    ) -> None:
        """Session fields stay the lifecycle's, on the paths `publish` never reaches.

        On the happy path `publish(runtime, clear_failure_reason=True)` runs
        last and would repair session state damaged earlier, hiding an
        over-reaching clear. The discard path returns before it, so this is
        where such damage would actually survive.
        """
        client = self._healthy_client(mocked_lib_cls, mocked_reset)
        client.mtp_fallback_reason = "thinking-mode"
        mocked_reset.side_effect = RuntimeError("reinit failed")

        client.reset_session_state()
        snapshot = client.session_snapshot()

        self.assertIsNone(client.mtp_fallback_reason)
        self.assertEqual(
            snapshot.mtp_failure_reason, "reinit failed",
            "the discard reason belongs to the lifecycle; this fix must not "
            "overwrite or clear it",
        )
        self.assertFalse(snapshot.mtp_enabled)

    @mock.patch("orbit.native_llama.client.LlamaLibrary")
    def test_reset_without_a_persistent_runtime_still_clears_the_status(
        self, mocked_lib_cls
    ) -> None:
        """The no-runtime path returns early too, and is equally stale."""
        client = NativeLlamaClient(_paths(), NativeClientConfig(use_mtp_experimental=True))
        fake_lib = mock.Mock()
        fake_lib.llama_get_memory.return_value = object()
        mocked_lib_cls.return_value.lib = fake_lib
        client._session.ctx_tgt = object()
        client._persistent_mtp_runtime = None
        client.mtp_fallback_reason = "thinking-mode"
        client.last_mtp_completion = MtpCompletionResult(
            enabled=True, success=True, error=None
        )

        client.reset_session_state()

        self.assertIsNone(client.mtp_fallback_reason)
        self.assertFalse(client.last_mtp_completion.success)


if __name__ == "__main__":
    unittest.main()

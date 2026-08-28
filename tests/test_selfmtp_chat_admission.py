"""Production `/chat` admission for a constructed self-MTP runtime.

The defect these cover: `complete_chat` derived `allow_mtp` from
`profile.mtp_supported`, which describes the EXTERNAL-DRAFT architecture and is
False for Ornith. A correctly constructed single-GGUF self-MTP session was
therefore never admitted through the production chat path, and every request
decoded normally with drafted/accepted/draft-calls all zero -- while `/props`
still reported `mtp_enabled: true`.

Admission must instead ask a runtime question: does a usable MTP session exist?
That is architecture-neutral, so it admits external-draft and self-MTP alike
without conflating them, and it is the SAME predicate the completion gate uses
(client.py:2309), so admission and execution cannot disagree.

These drive the real `complete_chat`, capturing the value it hands to
`complete_prompt`, rather than asserting against a helper.
"""

from __future__ import annotations

import unittest
from ctypes import c_void_p
from unittest.mock import patch

from orbit.native_llama import client as client_mod
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID, QWEN3_CODER_PROFILE_ID
from orbit.native_llama.persistent_mtp import PersistentMtpSessionRuntime
from orbit.native_llama.session_state import NativeSessionState


class _Profile:
    """Stands in for the resolved registry profile."""

    def __init__(self, *, mtp_supported: bool, profile_id: str = ORNITH15_PROFILE_ID):
        self.mtp_supported = mtp_supported
        self.profile_id = profile_id
        self.verified = True
        self.uses_native_chat_bridge = False
        self.thinking_supported = True
        self.gemma_prefix_reuse_supported = False
        self.route_prefix_reuse_supported = False
        self.multimodal_supported = False
        self.family = "qwen35moe"


def _runtime(*, self_mtp: bool) -> PersistentMtpSessionRuntime:
    return PersistentMtpSessionRuntime(
        handle=c_void_p(0x1000),
        ctx_dft=c_void_p(0x2000),
        spec=c_void_p(0x3000),
        library=object(),
        self_mtp=self_mtp,
    )


def _client(
    *,
    runtime: PersistentMtpSessionRuntime | None,
    mtp_enabled: bool,
    mtp_supported: bool,
    profile_id: str = ORNITH15_PROFILE_ID,
) -> NativeLlamaClient:
    c = NativeLlamaClient.__new__(NativeLlamaClient)
    c.model_profile = _Profile(mtp_supported=mtp_supported, profile_id=profile_id)
    c._persistent_mtp_runtime = runtime
    c._session = NativeSessionState(session_id="d1")
    c._session.mtp_enabled = mtp_enabled
    c._session.ctx_tgt = c_void_p(0x4000)
    c._final_prefix_status = type("S", (), {"last_used": False})()
    c._media_marker = None
    return c


def admitted(c: NativeLlamaClient, *, tools=None, thinking=False) -> bool:
    """Run the REAL complete_chat and report what it admitted.

    `complete_prompt` is intercepted so nothing native is touched; the captured
    `allow_mtp_experimental` is the production admission decision.
    """
    seen: dict[str, object] = {}

    def fake_complete_prompt(self, prompt, **kw):
        seen["allow_mtp"] = kw.get("allow_mtp_experimental")
        raise _Stop()

    with patch.object(NativeLlamaClient, "complete_prompt", fake_complete_prompt), \
         patch.object(NativeLlamaClient, "_thinking_enabled", lambda self, v: bool(thinking)), \
         patch.object(NativeLlamaClient, "_ensure_prompt_cache_mode", lambda self, m: None), \
         patch.object(NativeLlamaClient, "apply_chat_template", lambda self, *a, **k: "PROMPT"), \
         patch.object(NativeLlamaClient, "_route_anchor_segments_for_prompt", lambda self, *a, **k: None), \
         patch.object(NativeLlamaClient, "_final_prefix_segments_for_prompt", lambda self, *a, **k: None), \
         patch.object(NativeLlamaClient, "_final_prefix_experiment_eligible", lambda self, v: False), \
         patch.object(NativeLlamaClient, "_ornith_rolling_route_eligible", lambda self, **k: False), \
         patch.object(NativeLlamaClient, "_ornith_rolling_analysis_eligible", lambda self, **k: False), \
         patch.object(client_mod, "prepare_multimodal_messages", lambda m, media_marker=None: None):
        try:
            c.complete_chat([{"role": "user", "content": "hi"}], tools=tools)
        except _Stop:
            pass
    return bool(seen.get("allow_mtp"))


class _Stop(Exception):
    pass


class HistoricalDefectTests(unittest.TestCase):
    """The exact Mission-C/diagnosis configuration."""

    def test_constructed_self_mtp_is_admitted_despite_profile_metadata(self) -> None:
        # CURRENT exact artifact, self-MTP runtime constructed, and the registry
        # profile says mtp_supported=False because it describes external draft.
        c = _client(runtime=_runtime(self_mtp=True), mtp_enabled=True, mtp_supported=False)
        self.assertTrue(
            admitted(c),
            "a constructed self-MTP runtime must be admitted through production chat",
        )


class ArchitectureNeutralityTests(unittest.TestCase):
    def test_external_draft_runtime_still_admitted(self) -> None:
        c = _client(runtime=_runtime(self_mtp=False), mtp_enabled=True, mtp_supported=True)
        self.assertTrue(admitted(c))

    def test_external_draft_admitted_even_if_profile_metadata_absent(self) -> None:
        """Admission is a runtime question; it must not re-consult metadata."""
        c = _client(runtime=_runtime(self_mtp=False), mtp_enabled=True, mtp_supported=False)
        self.assertTrue(admitted(c))

    def test_admission_does_not_depend_on_which_constructor_ran(self) -> None:
        a = _client(runtime=_runtime(self_mtp=True), mtp_enabled=True, mtp_supported=False)
        b = _client(runtime=_runtime(self_mtp=False), mtp_enabled=True, mtp_supported=False)
        self.assertEqual(admitted(a), admitted(b))


class NegativeAdmissionTests(unittest.TestCase):
    def test_ordinary_client_without_mtp_denied(self) -> None:
        c = _client(runtime=None, mtp_enabled=False, mtp_supported=False)
        self.assertFalse(admitted(c))

    def test_failed_construction_denied(self) -> None:
        # Both constructors leave the runtime None when construction raises.
        c = _client(runtime=None, mtp_enabled=False, mtp_supported=True)
        self.assertFalse(admitted(c))

    def test_stale_self_mtp_flag_without_runtime_denied(self) -> None:
        c = _client(runtime=None, mtp_enabled=True, mtp_supported=False)
        self.assertFalse(admitted(c))

    def test_stale_runtime_after_reset_failure_denied(self) -> None:
        """Reset failure clears mtp_enabled but leaves the runtime referenced.

        Admitting on the handle alone would admit a request the completion gate
        (client.py:2309) then rejects.
        """
        c = _client(runtime=_runtime(self_mtp=True), mtp_enabled=False, mtp_supported=False)
        self.assertFalse(admitted(c))

    def test_profile_metadata_true_without_session_denied(self) -> None:
        c = _client(runtime=None, mtp_enabled=False, mtp_supported=True)
        self.assertFalse(admitted(c))


class UnchangedGateTests(unittest.TestCase):
    """Admission must not widen the pre-existing non-MTP conditions."""

    def test_tools_still_block_admission(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True), mtp_enabled=True, mtp_supported=False)
        self.assertFalse(admitted(c, tools=[{"type": "function"}]))

    def test_explicit_false_request_still_denied(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True), mtp_enabled=True, mtp_supported=False)
        seen: dict[str, object] = {}

        def fake_complete_prompt(self, prompt, **kw):
            seen["allow_mtp"] = kw.get("allow_mtp_experimental")
            raise _Stop()

        with patch.object(NativeLlamaClient, "complete_prompt", fake_complete_prompt), \
             patch.object(NativeLlamaClient, "_thinking_enabled", lambda self, v: False), \
             patch.object(NativeLlamaClient, "_ensure_prompt_cache_mode", lambda self, m: None), \
             patch.object(NativeLlamaClient, "apply_chat_template", lambda self, *a, **k: "P"), \
             patch.object(NativeLlamaClient, "_route_anchor_segments_for_prompt", lambda self, *a, **k: None), \
             patch.object(NativeLlamaClient, "_final_prefix_segments_for_prompt", lambda self, *a, **k: None), \
             patch.object(NativeLlamaClient, "_final_prefix_experiment_eligible", lambda self, v: False), \
             patch.object(NativeLlamaClient, "_ornith_rolling_route_eligible", lambda self, **k: False), \
             patch.object(NativeLlamaClient, "_ornith_rolling_analysis_eligible", lambda self, **k: False), \
             patch.object(client_mod, "prepare_multimodal_messages", lambda m, media_marker=None: None):
            try:
                c.complete_chat(
                    [{"role": "user", "content": "hi"}], allow_mtp_experimental=False
                )
            except _Stop:
                pass
        self.assertFalse(bool(seen.get("allow_mtp")))


class NoPerRequestHashingTests(unittest.TestCase):
    """Admission must consume qualified runtime state, never re-hash 21 GiB."""

    def test_admission_never_calls_artifact_digest(self) -> None:
        c = _client(runtime=_runtime(self_mtp=True), mtp_enabled=True, mtp_supported=False)
        import orbit.native_llama.artifact_capabilities as ac

        calls: list[str] = []
        real = ac.artifact_sha256
        try:
            ac.artifact_sha256 = lambda p: (calls.append(str(p)), real(p))[1]
            for _ in range(5):
                admitted(c)
        finally:
            ac.artifact_sha256 = real
        self.assertEqual(calls, [], "admission must not hash the artifact per request")


if __name__ == "__main__":
    unittest.main()

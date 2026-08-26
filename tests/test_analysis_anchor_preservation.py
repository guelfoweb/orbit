"""The ANALYSIS rolling checkpoint must survive a tools-free detour.

Production already does this -- `_ensure_prompt_cache_mode` treats
`tools:thinking=off` <-> `chat:thinking=off` as a qualified transition and
preserves both Ornith lineages -- but nothing pinned it for the *analysis*
slot. The route slot has `test_full_turn_cycle_keeps_the_checkpoint_alive`;
this is the missing counterpart.

It is load-bearing for any tools-free call made between two analysis steps:
lose it and the next step falls to a cold prefill instead of restoring.
"""

from __future__ import annotations

import unittest

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID


class _Profile:
    verified = True
    profile_id = ORNITH15_PROFILE_ID


class _Session:
    prompt_cache_mode: str | None = None


def _client(current_mode: str) -> tuple[NativeLlamaClient, dict]:
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.model_profile = _Profile()
    client._session = _Session()
    client._session.prompt_cache_mode = current_mode
    captured: dict = {}
    client.reset_session_state = lambda **kwargs: captured.update(kwargs)
    return client, captured


class AnalysisAnchorSurvivesToolFreeDetourTests(unittest.TestCase):
    def test_step_to_tool_free_call_preserves_the_checkpoint(self) -> None:
        client, captured = _client("tools:thinking=off")
        client._ensure_prompt_cache_mode("chat:thinking=off")
        self.assertTrue(
            captured.get("preserve_ornith_rolling_route_checkpoint"),
            "a tools-free call between analysis steps must not discard the anchor",
        )

    def test_tool_free_call_back_to_step_preserves_the_checkpoint(self) -> None:
        client, captured = _client("chat:thinking=off")
        client._ensure_prompt_cache_mode("tools:thinking=off")
        self.assertTrue(captured.get("preserve_ornith_rolling_route_checkpoint"))

    def test_thinking_transition_is_still_destructive(self) -> None:
        # The preservation is narrow on purpose. A verifier that turned
        # thinking on would silently cost the next step a full prefill.
        client, captured = _client("tools:thinking=off")
        client._ensure_prompt_cache_mode("chat:thinking=on")
        self.assertFalse(captured.get("preserve_ornith_rolling_route_checkpoint"))

    def test_multimodal_transition_is_still_destructive(self) -> None:
        client, captured = _client("tools:thinking=off")
        client._ensure_prompt_cache_mode("multimodal:thinking=off")
        self.assertFalse(captured.get("preserve_ornith_rolling_route_checkpoint"))

    def test_preservation_requires_the_verified_ornith_profile(self) -> None:
        client, captured = _client("tools:thinking=off")
        client.model_profile.verified = False
        client._ensure_prompt_cache_mode("chat:thinking=off")
        self.assertFalse(captured.get("preserve_ornith_rolling_route_checkpoint"))


if __name__ == "__main__":
    unittest.main()

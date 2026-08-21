"""The rolling route checkpoint and the internal prompt-cache mode switch.

A conversational turn alternates between two internal cache modes: the route
call runs with tools attached (`tools:thinking=off`), the final call without
(`chat:thinking=off`). Every switch calls `reset_session_state()`, which is
where conversation-derived KV state goes to die -- correctly, for a user reset,
and catastrophically for an optimisation that only pays off across turns.

So the checkpoint has to survive exactly one thing: that internal switch, for
the one profile qualified for it. It must still die on a real reset, a new
session, a profile change, or any identity drift. These tests pin both halves,
because a preservation rule that leaks into `/reset` would keep one
conversation's tokens resident into the next.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import (
    GEMMA4_PROFILE_ID,
    ORNITH15_PROFILE_ID,
    QWEN36_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
)
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
)

TOOLS_MODE = "tools:thinking=off"
CHAT_MODE = "chat:thinking=off"
MULTIMODAL_MODE = "multimodal:thinking=off"
THINKING_MODE = "tools:thinking=on"

ROUTE_A = [10, 11, 12, 13]


def identity(**overrides) -> RollingRouteIdentity:
    base = dict(
        strategy_id=ROLLING_ROUTE_STRATEGY_ID,
        session_id="default",
        profile_id=ORNITH15_PROFILE_ID,
        model_id="/models/ornith.gguf",
        template_id="tpl",
        tool_schema_hash="tools",
        capability_summary_hash="caps",
        runtime_policy_hash="policy",
        native_version="libllama.so",
        tools_mode="on",
        reset_generation=0,
    )
    base.update(overrides)
    return RollingRouteIdentity(**base)


def checkpoint_a() -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=identity(),
        tokens=list(ROUTE_A),
        checkpoint_data=b"checkpoint-a",
        created_at_monotonic=1.0,
    )


class _Profile:
    def __init__(self, profile_id: str, verified: bool = True) -> None:
        self.profile_id = profile_id
        self.verified = verified


class _Session:
    def __init__(self, mode: str | None) -> None:
        self.prompt_cache_mode = mode


def mode_client(profile_id: str = ORNITH15_PROFILE_ID, *, mode: str | None = TOOLS_MODE,
                verified: bool = True):
    """A client whose only live behaviour is the mode-transition decision."""
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.model_profile = _Profile(profile_id, verified)
    client._session = _Session(mode)
    client._rolling_route_anchor_state = checkpoint_a()
    client._reset_generation = 0
    recorded: list[dict] = []

    def fake_reset(**kwargs):
        recorded.append(dict(kwargs))
        # Mirror the production reset's effect on rolling state so the test
        # observes what a real reset would actually do.
        if not kwargs.get("preserve_ornith_rolling_route_checkpoint"):
            client._reset_generation += 1
            client._rolling_route_anchor_state = RollingRouteAnchorState(
                invalidation_reason="session_reset"
            )

    client.reset_session_state = fake_reset  # type: ignore[method-assign]
    return client, recorded


class QualifiedTransitionPreservesTest(unittest.TestCase):
    """The defect this mission exists to fix."""

    def test_tools_to_chat_preserves_the_rolling_checkpoint(self) -> None:
        client, recorded = mode_client(mode=TOOLS_MODE)
        client._ensure_prompt_cache_mode(CHAT_MODE)

        self.assertEqual(len(recorded), 1, "the mode switch still resets the session")
        self.assertTrue(
            client._rolling_route_anchor_state.valid,
            "the final call's mode switch must not destroy the route checkpoint",
        )
        self.assertEqual(client._rolling_route_anchor_state.tokens, ROUTE_A)

    def test_chat_to_tools_preserves_the_rolling_checkpoint(self) -> None:
        client, _ = mode_client(mode=CHAT_MODE)
        client._ensure_prompt_cache_mode(TOOLS_MODE)
        self.assertTrue(
            client._rolling_route_anchor_state.valid,
            "returning to the route mode must find the checkpoint intact",
        )

    def test_full_turn_cycle_keeps_the_checkpoint_alive(self) -> None:
        # route (tools) -> final (chat) -> next route (tools)
        client, _ = mode_client(mode=TOOLS_MODE)
        client._ensure_prompt_cache_mode(CHAT_MODE)
        client._ensure_prompt_cache_mode(TOOLS_MODE)
        self.assertTrue(client._rolling_route_anchor_state.valid)
        self.assertEqual(client._rolling_route_anchor_state.tokens, ROUTE_A)
        self.assertEqual(
            client._reset_generation, 0, "an internal switch is not a lifecycle reset"
        )


class UnqualifiedTransitionsStillDestroyTest(unittest.TestCase):
    def test_thinking_mode_change_is_not_qualified(self) -> None:
        client, _ = mode_client(mode=TOOLS_MODE)
        client._ensure_prompt_cache_mode(THINKING_MODE)
        self.assertFalse(
            client._rolling_route_anchor_state.valid,
            "only the exact tools<->chat pair is qualified",
        )

    def test_multimodal_transition_is_not_qualified(self) -> None:
        client, _ = mode_client(mode=TOOLS_MODE)
        client._ensure_prompt_cache_mode(MULTIMODAL_MODE)
        self.assertFalse(client._rolling_route_anchor_state.valid)

    def test_other_profiles_do_not_get_ornith_preservation(self) -> None:
        for profile_id in (GEMMA4_PROFILE_ID, QWEN36_PROFILE_ID, QWEN3_CODER_PROFILE_ID):
            with self.subTest(profile=profile_id):
                client, recorded = mode_client(profile_id, mode=TOOLS_MODE)
                client._ensure_prompt_cache_mode(CHAT_MODE)
                self.assertFalse(
                    recorded[0].get("preserve_ornith_rolling_route_checkpoint", False),
                    "rolling preservation is Ornith-only",
                )

    def test_unverified_ornith_does_not_get_preservation(self) -> None:
        client, recorded = mode_client(mode=TOOLS_MODE, verified=False)
        client._ensure_prompt_cache_mode(CHAT_MODE)
        self.assertFalse(recorded[0].get("preserve_ornith_rolling_route_checkpoint", False))


class DestructiveResetStillDestroysTest(unittest.TestCase):
    """The preservation flag must not leak into real resets."""

    def test_reset_session_state_without_the_flag_invalidates(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_route_anchor_state = checkpoint_a()
        client._reset_generation = 0
        client._invalidate_rolling_route_anchor("session_reset")
        self.assertFalse(client._rolling_route_anchor_state.valid)

    def test_reset_default_does_not_preserve_rolling_state(self) -> None:
        # A caller that says nothing must get the destructive behaviour.
        import inspect

        signature = inspect.signature(NativeLlamaClient.reset_session_state)
        param = signature.parameters.get("preserve_ornith_rolling_route_checkpoint")
        self.assertIsNotNone(param, "the preserve option must exist")
        self.assertIs(param.default, False, "preservation must be opt-in, never the default")

    def test_real_reset_body_honours_the_preserve_flag(self) -> None:
        """Drive the real reset body, not a stub, in both directions.

        The transition tests stub `reset_session_state`, so only this can see
        whether the flag is actually consulted inside it.
        """
        import inspect
        import textwrap

        source = inspect.getsource(NativeLlamaClient.reset_session_state)
        start = source.index("        if not preserve_ornith_rolling_route_checkpoint:")
        end = source.index("        self._invalidate_final_prefix(", start)
        block = textwrap.dedent(source[start:end])

        for preserve, expect_valid in ((True, True), (False, False)):
            with self.subTest(preserve=preserve):
                client = NativeLlamaClient.__new__(NativeLlamaClient)
                client._rolling_route_anchor_state = checkpoint_a()
                client._reset_generation = 0
                exec(
                    block,
                    {
                        "self": client,
                        "preserve_ornith_rolling_route_checkpoint": preserve,
                    },
                )
                self.assertEqual(
                    client._rolling_route_anchor_state.valid,
                    expect_valid,
                    "the reset body must consult the preserve flag",
                )
                self.assertEqual(client._reset_generation, 0 if preserve else 1)

    def test_server_reset_endpoint_never_preserves_rolling_state(self) -> None:
        # /reset must clear conversation-derived KV: unlike the fixed
        # Qwen3-Coder prefix, this checkpoint holds the user's own tokens.
        import inspect

        from orbit.native_server.app import OrbitNativeServer

        source = inspect.getsource(OrbitNativeServer.reset_session)
        self.assertNotIn(
            "preserve_ornith_rolling_route_checkpoint",
            source,
            "/reset must not preserve conversation-derived rolling state",
        )


if __name__ == "__main__":
    unittest.main()

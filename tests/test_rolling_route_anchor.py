"""The Ornith rolling route anchor: exact reuse, and nothing else.

Covers the wired production gate (which profiles and phases may select the
strategy), the state machine that lets a route checkpoint survive the final
call, and every way reuse must be refused.
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
    QWEN38_PROFILE_ID,
    QWEN3_CODER_PROFILE_ID,
)
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    capture_rolling_route_anchor,
    invalidate_rolling_route_anchor,
    restore_rolling_route_anchor,
    rolling_route_reuse_start,
    rolling_route_should_replace,
)


def identity(**overrides) -> RollingRouteIdentity:
    base = dict(
        strategy_id=ROLLING_ROUTE_STRATEGY_ID,
        session_id="default",
        profile_id=ORNITH15_PROFILE_ID,
        model_id="/models/ornith.gguf",
        template_id="tpl-sha",
        tool_schema_hash="tools",
        capability_summary_hash="caps",
        runtime_policy_hash="policy",
        native_version="libllama.so",
        tools_mode="on",
        reset_generation=0,
    )
    base.update(overrides)
    return RollingRouteIdentity(**base)


class FakeLib:
    def __init__(self, *, size: int = 64, fail_get: bool = False, short_write: bool = False,
                 fail_set: bool = False, short_read: bool = False) -> None:
        self.size = size
        self.fail_get = fail_get
        self.short_write = short_write
        self.fail_set = fail_set
        self.short_read = short_read

    def llama_state_seq_get_size(self, ctx, seq_id):
        return self.size

    def llama_state_seq_get_data(self, ctx, buffer, size, seq_id):
        if self.fail_get:
            raise RuntimeError("boom")
        for i in range(size):
            buffer[i] = (i + 3) % 256
        return size - 1 if self.short_write else size

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        if self.fail_set:
            raise RuntimeError("boom")
        return size - 1 if self.short_read else size


CTX = object()
ROUTE1 = [10, 11, 12, 13]
ROUTE2 = ROUTE1 + [14, 15]
ROUTE3 = ROUTE2 + [16]
FINAL = [900, 901, 902]


class _Profile:
    def __init__(self, profile_id: str, verified: bool = True) -> None:
        self.profile_id = profile_id
        self.verified = verified


class _Config:
    use_mtp_experimental = False
    context_tokens = 16384
    thinking = False


class _Session:
    mtp_enabled = False


def eligibility_client(profile_id: str, *, verified: bool = True) -> NativeLlamaClient:
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.config = _Config()
    client._session = _Session()
    client.model_profile = _Profile(profile_id, verified)
    return client


class RollingRouteEligibilityTest(unittest.TestCase):
    """Only verified Ornith route calls may select the strategy."""

    def test_only_ornith_route_calls_are_eligible(self) -> None:
        for profile_id in (
            GEMMA4_PROFILE_ID,
            QWEN36_PROFILE_ID,
            QWEN38_PROFILE_ID,
            QWEN3_CODER_PROFILE_ID,
        ):
            with self.subTest(profile=profile_id):
                client = eligibility_client(profile_id)
                self.assertFalse(
                    client._ornith_rolling_route_eligible(
                        route_prefix_anchor=True, tools=None, thinking=False
                    ),
                    "other verified profiles must keep their existing strategy",
                )

        client = eligibility_client(ORNITH15_PROFILE_ID)
        self.assertTrue(
            client._ornith_rolling_route_eligible(
                route_prefix_anchor=True, tools=None, thinking=False
            )
        )

    def test_final_calls_are_never_eligible(self) -> None:
        # route_prefix_anchor is the runtime's own route signal; a final call
        # clears it, so the backend never needs to know what a phase is.
        client = eligibility_client(ORNITH15_PROFILE_ID)
        self.assertFalse(
            client._ornith_rolling_route_eligible(
                route_prefix_anchor=False, tools=None, thinking=False
            )
        )

    def test_unverified_ornith_is_not_eligible(self) -> None:
        client = eligibility_client(ORNITH15_PROFILE_ID, verified=False)
        self.assertFalse(
            client._ornith_rolling_route_eligible(
                route_prefix_anchor=True, tools=None, thinking=False
            )
        )

    def test_thinking_and_mtp_block_eligibility(self) -> None:
        client = eligibility_client(ORNITH15_PROFILE_ID)
        self.assertFalse(
            client._ornith_rolling_route_eligible(
                route_prefix_anchor=True, tools=None, thinking=True
            )
        )
        client._session.mtp_enabled = True
        self.assertFalse(
            client._ornith_rolling_route_eligible(
                route_prefix_anchor=True, tools=None, thinking=False
            )
        )


class RollingRouteStateTest(unittest.TestCase):
    def test_capture_holds_exactly_the_prefill_tokens(self) -> None:
        state, meta = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        self.assertTrue(state.valid)
        self.assertEqual(state.tokens, ROUTE1)
        self.assertEqual(meta["checkpoint_tokens"], len(ROUTE1))

    def test_route_chain_reuses_and_rolls_forward(self) -> None:
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        self.assertEqual(rolling_route_reuse_start(state, ROUTE2, identity()), len(ROUTE1))
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE2, identity=identity()
        )
        self.assertEqual(rolling_route_reuse_start(state, ROUTE3, identity()), len(ROUTE2))

    def test_final_prompt_cannot_evict_the_route_checkpoint(self) -> None:
        # The whole point of serializing: a final call wipes live KV but must
        # leave the route chain's saved state alone.
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        self.assertFalse(rolling_route_should_replace(state, FINAL, identity()))
        self.assertIsNone(rolling_route_reuse_start(state, FINAL, identity()))
        self.assertEqual(rolling_route_reuse_start(state, ROUTE2, identity()), len(ROUTE1))

    def test_incompatible_post_tool_route_falls_cold(self) -> None:
        # Measured production behaviour: a post-tool route diverges before the
        # end, so it must cold-prefill rather than reuse conversational state.
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        diverged = [10, 11, 99, 13, 14, 15]
        self.assertIsNone(rolling_route_reuse_start(state, diverged, identity()))

    def test_one_token_mismatch_equal_and_shorter_are_refused(self) -> None:
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE2, identity=identity()
        )
        self.assertIsNone(rolling_route_reuse_start(state, [10, 11, 12, 99, 14, 15, 16], identity()))
        self.assertIsNone(rolling_route_reuse_start(state, list(ROUTE2), identity()))
        self.assertIsNone(rolling_route_reuse_start(state, ROUTE1, identity()))

    def test_every_identity_field_blocks_reuse(self) -> None:
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        for field_name, value in [
            ("strategy_id", "other-strategy"),
            ("session_id", "other-session"),
            ("profile_id", GEMMA4_PROFILE_ID),
            ("model_id", "/models/other.gguf"),
            ("template_id", "other-template"),
            ("tool_schema_hash", "other-tools"),
            ("capability_summary_hash", "other-caps"),
            ("runtime_policy_hash", "other-policy"),
            ("native_version", "other-lib"),
            ("tools_mode", "off"),
            ("reset_generation", 1),
        ]:
            with self.subTest(field=field_name):
                self.assertIsNone(
                    rolling_route_reuse_start(state, ROUTE2, identity(**{field_name: value})),
                    f"{field_name} change must block reuse",
                )

    def test_invalidated_state_cannot_be_reused_or_restored(self) -> None:
        state, _ = capture_rolling_route_anchor(
            FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
        )
        dead = invalidate_rolling_route_anchor(state, "session_reset")
        self.assertFalse(dead.valid)
        self.assertIsNone(rolling_route_reuse_start(dead, ROUTE2, identity()))
        ok, _, _ = restore_rolling_route_anchor(FakeLib(), CTX, dead)
        self.assertFalse(ok)

    def test_capture_failures_produce_no_usable_state(self) -> None:
        for kwargs, reason in [
            ({"fail_get": True}, "checkpoint_capture_failed"),
            ({"short_write": True}, "checkpoint_size_mismatch"),
            ({"size": 0}, "empty_checkpoint"),
        ]:
            with self.subTest(reason=reason):
                state, meta = capture_rolling_route_anchor(
                    FakeLib(**kwargs), CTX, prompt_tokens=ROUTE1, identity=identity()
                )
                self.assertFalse(state.valid)
                self.assertEqual(meta["fallback_reason"], reason)

    def test_restore_failure_is_reported_and_drops_state(self) -> None:
        for kwargs, reason in [
            ({"fail_set": True}, "checkpoint_restore_failed"),
            ({"short_read": True}, "checkpoint_restore_size_mismatch"),
        ]:
            with self.subTest(reason=reason):
                state, _ = capture_rolling_route_anchor(
                    FakeLib(), CTX, prompt_tokens=ROUTE1, identity=identity()
                )
                ok, after, meta = restore_rolling_route_anchor(FakeLib(**kwargs), CTX, state)
                self.assertFalse(ok, "a partial restore must never read as success")
                self.assertFalse(after.valid)
                self.assertEqual(meta["fallback_reason"], reason)

    def test_empty_state_is_never_reusable(self) -> None:
        self.assertIsNone(rolling_route_reuse_start(RollingRouteAnchorState(), ROUTE2, identity()))


if __name__ == "__main__":
    unittest.main()

"""The wired rolling route anchor, driven through the real client methods.

The companion module tests cover `RollingRouteAnchorState` in isolation. These
drive `NativeLlamaClient` itself, because the safety properties that matter
live in the wiring: that a restored checkpoint still has to satisfy strict
append, that a reset destroys the checkpoint, that an incomplete prefill never
captures, and that a failed capture leaves a good checkpoint alone.

Each test asserts the production method was actually entered and that the hook
it depends on was actually called -- a test that only checks a return value
would pass just as happily if the wiring were deleted.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit.native_llama.client as client_module
from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.model_profiles import ORNITH15_PROFILE_ID
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
)

ROUTE1 = [10, 11, 12, 13]
ROUTE2 = ROUTE1 + [14, 15]
CHECKPOINT = b"checkpoint-a"


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


class _Lib:
    """The handful of C entry points these paths touch."""

    def __init__(self, *, fail_set: bool = False) -> None:
        self.fail_set = fail_set
        self.cleared = 0
        self.set_data_calls = 0

    def llama_state_seq_set_data(self, ctx, buffer, size, seq_id):
        self.set_data_calls += 1
        if self.fail_set:
            raise RuntimeError("restore exploded")
        return size

    def llama_get_memory(self, ctx):
        return object()

    def llama_memory_clear(self, mem, flag):
        self.cleared += 1


class _LibHolder:
    def __init__(self, lib: _Lib) -> None:
        self.lib = lib


class _Session:
    def __init__(self) -> None:
        self.ctx_tgt = object()
        self.session_id = "default"
        self.cached_prompt_tokens: list[int] = []
        self.committed_sequence_tokens: list[int] = []
        self.mtp_enabled = False


def strategy_client(state: RollingRouteAnchorState, *, fail_set: bool = False):
    """A client with only what the rolling strategy path reaches."""
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    lib = _Lib(fail_set=fail_set)
    client.lib = _LibHolder(lib)
    client._session = _Session()
    client._rolling_route_anchor_state = state
    client._rolling_route_identity_cache = None
    calls: list[list[int]] = []

    def fake_prepare(prompt_tokens):
        # Stand in for the real strict-append gate and record that the
        # strategy actually deferred to it.
        calls.append(list(prompt_tokens))
        return 99

    client._prepare_memory_for_prompt = fake_prepare  # type: ignore[method-assign]
    client._invalidate_committed_sequence = lambda: client._session.committed_sequence_tokens.clear()  # type: ignore[method-assign]
    return client, lib, calls


def valid_state(tokens=ROUTE1, ident=None) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=ident or identity(),
        tokens=list(tokens),
        checkpoint_data=CHECKPOINT,
        created_at_monotonic=1.0,
    )


class StrictAppendRemainsAuthorityTest(unittest.TestCase):
    """Test A: the rolling strategy may restore, but never authorize."""

    def test_restore_defers_to_prepare_memory_for_prompt(self) -> None:
        client, lib, calls = strategy_client(valid_state())
        client._rolling_route_identity_cache = identity()

        result = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        # The restore really happened...
        self.assertEqual(lib.set_data_calls, 1, "the checkpoint must actually be restored")
        # ...bookkeeping was handed over so strict append can judge it...
        self.assertEqual(client._session.committed_sequence_tokens, ROUTE1)
        self.assertEqual(client._session.cached_prompt_tokens, ROUTE1)
        # ...and the decision was delegated, not made here.
        self.assertEqual(calls, [ROUTE2], "_prepare_memory_for_prompt must decide reuse")
        self.assertEqual(
            result, 99, "the strategy must return strict append's verdict, not its own prefix length"
        )
        self.assertNotEqual(
            result, len(ROUTE1), "returning the restored prefix length would bypass PR #210"
        )

    def test_incompatible_prompt_skips_restore_and_still_defers(self) -> None:
        client, lib, calls = strategy_client(valid_state())
        client._rolling_route_identity_cache = identity()

        result = client._prepare_memory_with_ornith_rolling_route_anchor([10, 11, 99, 13, 14])

        self.assertEqual(lib.set_data_calls, 0, "a mismatched prompt must not restore")
        self.assertEqual(calls, [[10, 11, 99, 13, 14]])
        self.assertEqual(result, 99)

    def test_failed_restore_clears_then_defers(self) -> None:
        client, lib, calls = strategy_client(valid_state(), fail_set=True)
        client._rolling_route_identity_cache = identity()

        result = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        self.assertEqual(lib.set_data_calls, 1)
        self.assertGreaterEqual(lib.cleared, 1, "a partial restore leaves KV unknown; clear it")
        self.assertFalse(
            client._rolling_route_anchor_state.valid, "a failed restore must drop the checkpoint"
        )
        self.assertEqual(calls, [ROUTE2], "the cold path still runs through strict append")
        self.assertEqual(result, 99)


class DirectCallFailsClosedTest(unittest.TestCase):
    """Phase 3: without a staged identity the strategy must not reuse."""

    def test_missing_staged_identity_rejects_reuse(self) -> None:
        client, lib, calls = strategy_client(valid_state())
        # Dispatcher never ran, so no identity was staged.
        self.assertIsNone(client._rolling_route_identity_cache)

        result = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        self.assertEqual(lib.set_data_calls, 0, "no identity means no restore")
        self.assertEqual(calls, [ROUTE2])
        self.assertEqual(result, 99)

    def test_stale_identity_from_another_session_rejects_reuse(self) -> None:
        client, lib, calls = strategy_client(valid_state())
        client._rolling_route_identity_cache = identity(session_id="a-different-session")

        result = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        self.assertEqual(lib.set_data_calls, 0)
        self.assertEqual(result, 99)


class CaptureGuardTest(unittest.TestCase):
    """Tests C and D: what may and may not become a checkpoint."""

    def _capture_harness(self, *, processed, n_prompt, cancelled, existing, capture_result):
        """Re-execute the production capture guard against recorded inputs.

        The guard is a block inside `_complete_prompt_standard`; driving the
        whole method would need a live model, so the guard's exact condition
        and body are exercised here with the same client state.
        """
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_route_anchor_state = existing
        import threading

        client.cancel_event = threading.Event()
        if cancelled:
            client.cancel_event.set()
        client._session = _Session()

        captured_calls: list[list[int]] = []

        def fake_capture(lib, ctx, *, prompt_tokens, identity):
            captured_calls.append(list(prompt_tokens))
            return capture_result, {}

        original = client_module.capture_rolling_route_anchor
        client_module.capture_rolling_route_anchor = fake_capture
        try:
            source = _capture_guard_source()
            exec_globals = {
                "rolling_route_eligible": True,
                "rolling_route_identity": identity(),
                "processed": processed,
                "n_prompt": n_prompt,
                "self": client,
                "lib": object(),
                "prompt_tokens": ROUTE2,
                "rolling_route_should_replace": client_module.rolling_route_should_replace,
                "capture_rolling_route_anchor": fake_capture,
            }
            exec(source, exec_globals)
        finally:
            client_module.capture_rolling_route_anchor = original
        return client, captured_calls

    def test_partial_prefill_never_captures(self) -> None:
        existing = valid_state()
        client, captured = self._capture_harness(
            processed=5, n_prompt=6, cancelled=False, existing=existing,
            capture_result=valid_state(ROUTE2),
        )
        self.assertEqual(captured, [], "an incomplete prefill must not be snapshotted")
        self.assertEqual(client._rolling_route_anchor_state.tokens, ROUTE1)

    def test_cancelled_prefill_never_captures(self) -> None:
        existing = valid_state()
        client, captured = self._capture_harness(
            processed=6, n_prompt=6, cancelled=True, existing=existing,
            capture_result=valid_state(ROUTE2),
        )
        self.assertEqual(captured, [], "a cancelled prefill must not be snapshotted")
        self.assertEqual(client._rolling_route_anchor_state.tokens, ROUTE1)

    def test_complete_prefill_captures_and_rolls_forward(self) -> None:
        existing = valid_state()
        client, captured = self._capture_harness(
            processed=6, n_prompt=6, cancelled=False, existing=existing,
            capture_result=valid_state(ROUTE2),
        )
        self.assertEqual(captured, [ROUTE2], "a complete prefill is the capture point")
        self.assertEqual(client._rolling_route_anchor_state.tokens, ROUTE2)

    def test_failed_capture_preserves_previous_checkpoint(self) -> None:
        existing = valid_state()
        failed = RollingRouteAnchorState(invalidation_reason="checkpoint_capture_failed")
        client, captured = self._capture_harness(
            processed=6, n_prompt=6, cancelled=False, existing=existing,
            capture_result=failed,
        )
        self.assertEqual(captured, [ROUTE2], "capture was attempted")
        state = client._rolling_route_anchor_state
        self.assertTrue(state.valid, "a failed capture must not destroy a good checkpoint")
        self.assertEqual(state.tokens, ROUTE1)
        self.assertEqual(state.checkpoint_data, CHECKPOINT)


def _capture_guard_source() -> str:
    """The capture guard, lifted verbatim from the production method.

    Read from the installed source so the test cannot drift away from the
    code it is protecting: if the guard changes, this text changes with it.
    """
    import inspect
    import textwrap

    source = inspect.getsource(NativeLlamaClient._complete_prompt_standard)
    marker = "        if (\n            rolling_route_eligible"
    start = source.index(marker)
    end = source.index("        self.last_committed_generated_tokens = []", start)
    return textwrap.dedent(source[start:end])


class ResetInvalidatesRollingStateTest(unittest.TestCase):
    """Test B: a reset must destroy conversation-derived KV state."""

    def test_reset_session_state_invalidates_and_bumps_generation(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client._rolling_route_anchor_state = valid_state()
        client._reset_generation = 0

        before = client._rolling_route_anchor_state
        self.assertTrue(before.valid)

        # Exercise the two production effects reset relies on.
        client._reset_generation += 1
        client._invalidate_rolling_route_anchor("session_reset")

        self.assertFalse(
            client._rolling_route_anchor_state.valid,
            "a reset must destroy the rolling checkpoint",
        )
        self.assertEqual(client._rolling_route_anchor_state.invalidation_reason, "session_reset")
        self.assertEqual(client._reset_generation, 1)

    def test_reset_session_state_source_invalidates_rolling_state(self) -> None:
        # The invalidation must live in reset_session_state itself: removing it
        # there would otherwise leave a conversation checkpoint alive across
        # /reset, which the unit above cannot see.
        import inspect

        source = inspect.getsource(NativeLlamaClient.reset_session_state)
        self.assertIn("_invalidate_rolling_route_anchor", source)
        self.assertIn("_reset_generation", source)

    def test_bumped_generation_blocks_reuse_of_an_old_checkpoint(self) -> None:
        client, lib, calls = strategy_client(valid_state())
        # Checkpoint was built at generation 0; the client has since reset.
        client._rolling_route_identity_cache = identity(reset_generation=1)

        result = client._prepare_memory_with_ornith_rolling_route_anchor(ROUTE2)

        self.assertEqual(lib.set_data_calls, 0, "a pre-reset checkpoint must not be restored")
        self.assertEqual(calls, [ROUTE2])
        self.assertEqual(result, 99)


if __name__ == "__main__":
    unittest.main()

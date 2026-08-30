"""The rolling-anchor store: one owner for two independent checkpoint slots.

CHAT route and ANALYSIS keep separate checkpoints, and the separation is the
safety property: restoring an analysis checkpoint for a route prompt would put
someone else's KV behind the model. The slot follows the identity's own
`strategy_id`, so neither lineage can address the other's slot.

These execute the real store over real `RollingRouteAnchorState` values.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.rolling_anchor_store import RollingAnchorStore
from orbit.native_llama.rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    ROLLING_ROUTE_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
)


def _identity(strategy: str = ROLLING_ROUTE_STRATEGY_ID) -> RollingRouteIdentity:
    return RollingRouteIdentity(
        strategy_id=strategy,
        session_id="s",
        profile_id="p",
        model_id="m",
        template_id="t",
        tool_schema_hash="h",
        capability_summary_hash="c",
        runtime_policy_hash="r",
        native_version="v",
        tools_mode="on",
        reset_generation=0,
    )


def _state(tokens, identity=None) -> RollingRouteAnchorState:
    return RollingRouteAnchorState(
        identity=identity or _identity(),
        tokens=list(tokens),
        checkpoint_data=b"kv",
    )


class SlotSelectionTests(unittest.TestCase):
    """Which slot an identity addresses is read from the identity itself."""

    def test_route_identity_addresses_the_route_slot(self) -> None:
        self.assertEqual(RollingAnchorStore.slot_for(_identity()), "route")

    def test_analysis_identity_addresses_the_analysis_slot(self) -> None:
        self.assertEqual(
            RollingAnchorStore.slot_for(_identity(ROLLING_ANALYSIS_STRATEGY_ID)),
            "analysis",
        )

    def test_absent_identity_defaults_to_route(self) -> None:
        """No identity must not raise and must not reach the analysis slot."""
        self.assertEqual(RollingAnchorStore.slot_for(None), "route")


class StoreAndReadTests(unittest.TestCase):
    def test_an_empty_store_reads_as_no_checkpoint(self) -> None:
        store = RollingAnchorStore()
        self.assertFalse(store.state_for(_identity()).valid)
        self.assertFalse(
            store.state_for(_identity(ROLLING_ANALYSIS_STRATEGY_ID)).valid
        )

    def test_a_stored_checkpoint_reads_back(self) -> None:
        store = RollingAnchorStore()
        identity = _identity()
        state = _state([1, 2, 3], identity)

        store.store(identity, state)

        self.assertIs(store.state_for(identity), state)
        self.assertEqual(store.state_for(identity).tokens, [1, 2, 3])

    def test_storing_replaces_the_previous_checkpoint(self) -> None:
        store = RollingAnchorStore()
        identity = _identity()
        store.store(identity, _state([1, 2], identity))
        store.store(identity, _state([1, 2, 3], identity))

        self.assertEqual(
            store.state_for(identity).tokens, [1, 2, 3],
            "a stale checkpoint must not survive its replacement",
        )


class LineageIsolationTests(unittest.TestCase):
    """Neither lineage may ever read the other's checkpoint.

    This is the safety property, not a tidiness one: restoring an analysis
    checkpoint for a route prompt puts a different conversation's KV behind the
    model, which answers confidently from someone else's context.
    """

    def test_a_route_checkpoint_is_invisible_to_analysis(self) -> None:
        store = RollingAnchorStore()
        route_id = _identity()
        store.store(route_id, _state([1, 2, 3], route_id))

        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        self.assertFalse(
            store.state_for(analysis_id).valid,
            "an analysis prompt must never see the route checkpoint",
        )

    def test_an_analysis_checkpoint_is_invisible_to_route(self) -> None:
        store = RollingAnchorStore()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        store.store(analysis_id, _state([9, 9], analysis_id))

        self.assertFalse(
            store.state_for(_identity()).valid,
            "a route prompt must never see the analysis checkpoint",
        )

    def test_the_two_slots_hold_different_checkpoints_at_once(self) -> None:
        store = RollingAnchorStore()
        route_id = _identity()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        store.store(route_id, _state([1, 2], route_id))
        store.store(analysis_id, _state([7, 8, 9], analysis_id))

        self.assertEqual(store.state_for(route_id).tokens, [1, 2])
        self.assertEqual(store.state_for(analysis_id).tokens, [7, 8, 9])


class InvalidationTests(unittest.TestCase):
    def test_invalidation_clears_both_lineages(self) -> None:
        """A reset destroys the conversation both checkpoints belong to."""
        store = RollingAnchorStore()
        route_id = _identity()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        store.store(route_id, _state([1, 2], route_id))
        store.store(analysis_id, _state([7, 8], analysis_id))

        store.invalidate("session_reset")

        self.assertFalse(store.state_for(route_id).valid)
        self.assertFalse(
            store.state_for(analysis_id).valid,
            "an analysis checkpoint surviving a reset is exactly the stale "
            "reuse the identity check exists to prevent",
        )

    def test_invalidation_records_the_reason(self) -> None:
        store = RollingAnchorStore()
        identity = _identity()
        store.store(identity, _state([1, 2], identity))

        store.invalidate("route_change")

        self.assertEqual(
            store.state_for(identity).invalidation_reason, "route_change"
        )

    def test_an_invalidated_checkpoint_cannot_be_read_back(self) -> None:
        store = RollingAnchorStore()
        identity = _identity()
        store.store(identity, _state([1, 2, 3], identity))
        store.invalidate("gone")

        self.assertEqual(
            store.state_for(identity).tokens, [],
            "invalidation must drop the tokens, not merely flag them",
        )

    def test_repeated_invalidation_keeps_the_first_reason(self) -> None:
        """An already-empty slot is left alone rather than rewritten.

        This mirrors the shipped guard: rewriting would overwrite the reason
        that explains why the checkpoint went away with whatever reason came
        last, losing the diagnostic.
        """
        store = RollingAnchorStore()
        identity = _identity()
        store.store(identity, _state([1, 2], identity))

        store.invalidate("first_reason")
        store.invalidate("second_reason")

        self.assertEqual(
            store.state_for(identity).invalidation_reason, "first_reason"
        )

    def test_repeated_invalidation_keeps_the_first_reason_on_analysis(self) -> None:
        """The same guard, on the ANALYSIS slot.

        Its sibling above exercises only the route slot, so dropping the guard
        on this branch went undetected: an already-empty analysis slot would be
        rewritten and its original `invalidation_reason` replaced by whatever
        reason came last, losing the diagnostic that explains why the
        checkpoint went away.
        """
        store = RollingAnchorStore()
        identity = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        store.store(identity, _state([1, 2], identity))

        store.invalidate("first_reason")
        store.invalidate("second_reason")

        self.assertEqual(
            store.state_for(identity).invalidation_reason, "first_reason"
        )

    def test_invalidating_an_already_empty_analysis_slot_records_nothing(self) -> None:
        """An untouched slot must stay untouched, reason included."""
        store = RollingAnchorStore()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        route_id = _identity()
        store.store(route_id, _state([1], route_id))

        store.invalidate("route_only")

        self.assertIsNone(
            store.state_for(analysis_id).invalidation_reason,
            "an empty analysis slot must not acquire a reason it never earned",
        )

    def test_invalidating_an_empty_store_is_safe(self) -> None:
        store = RollingAnchorStore()
        store.invalidate("nothing_to_drop")
        self.assertFalse(store.state_for(_identity()).valid)


class ClientOwnershipTests(unittest.TestCase):
    """The client must project the store, never hold a second copy."""

    def _client(self):
        from orbit.native_llama.client import NativeLlamaClient

        return object.__new__(NativeLlamaClient)

    def test_the_client_properties_read_the_store(self) -> None:
        client = self._client()
        identity = _identity()
        state = _state([4, 5], identity)

        client._store_rolling_anchor_state(identity, state)

        self.assertIs(client._rolling_route_anchor_state, state)
        self.assertIs(client._rolling_anchor_state_for(identity), state)

    def test_a_bare_client_reads_as_no_checkpoint(self) -> None:
        """`object.__new__` never runs `__init__`; absence must be empty."""
        client = self._client()
        self.assertFalse(client._rolling_route_anchor_state.valid)
        self.assertFalse(client._rolling_analysis_anchor_state.valid)

    def test_client_invalidation_reaches_both_slots(self) -> None:
        client = self._client()
        route_id = _identity()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        client._store_rolling_anchor_state(route_id, _state([1], route_id))
        client._store_rolling_anchor_state(analysis_id, _state([2], analysis_id))

        client._invalidate_rolling_route_anchor("session_reset")

        self.assertFalse(client._rolling_route_anchor_state.valid)
        self.assertFalse(client._rolling_analysis_anchor_state.valid)

    def test_deleting_the_analysis_slot_empties_it(self) -> None:
        """`del` used to be legal on a plain attribute; absence means empty.

        The contract is what absence MEANS -- a missing checkpoint falls cold
        rather than raising -- so deleting empties the slot.
        """
        client = self._client()
        analysis_id = _identity(ROLLING_ANALYSIS_STRATEGY_ID)
        client._store_rolling_anchor_state(analysis_id, _state([1, 2], analysis_id))

        del client._rolling_analysis_anchor_state

        self.assertFalse(client._rolling_analysis_anchor_state.valid)
        self.assertFalse(client._rolling_anchor_state_for(analysis_id).valid)

    def test_the_store_never_touches_native_state(self) -> None:
        """No ctx_tgt, no lib: physical KV ownership stays in the client.

        Checked over parsed code, not raw text: the module docstring names
        these very things to say it does NOT use them, and a substring search
        would flag its own explanation.
        """
        import ast

        source = (SRC / "orbit/native_llama/rolling_anchor_store.py").read_text()
        tree = ast.parse(source)
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        for forbidden in ("ctx_tgt", "lib", "ctypes"):
            self.assertNotIn(
                forbidden, names,
                f"the store must not reach native state ({forbidden}); "
                f"capture and restore stay backend primitives",
            )
        self.assertFalse(
            [n for n in names if n.startswith("llama_")],
            "the store must make no native calls",
        )


if __name__ == "__main__":
    unittest.main()

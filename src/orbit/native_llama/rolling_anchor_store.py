"""Ownership of the rolling route-anchor checkpoint slots.

A rolling anchor is a KV checkpoint the runtime may restore instead of
prefilling a prompt from cold. There are two independent lineages -- CHAT route
and ANALYSIS -- and keeping them apart is the point: a route prompt must never
meet an analysis checkpoint. The slot follows the identity's own `strategy_id`,
so the separation is data-driven rather than a branch on model or call-site
names, and the backend never learns the distinction.

This owns the two slots and the operations over them: which slot an identity
addresses, what is stored there, storing a new state, and invalidating both.

Deliberately NOT here:

* **Eligibility.** Whether a call takes the rolling strategy at all is policy
  and stays in the client.
* **Identity construction.** Building a `RollingRouteIdentity` needs the model
  path, profile, config, metadata identity and reset generation -- client-wide
  facts. This module consumes an identity; it never assembles one.
* **Physical KV.** No `ctx_tgt`, no `lib`, no native calls. Capturing and
  restoring a checkpoint are backend primitives the client executes; this side
  only records the resulting state.
* **Committed identity.** That belongs to `CommittedIdentity`. The restore path
  publishes through its existing contract; nothing here touches it.

Reuse *authorization* also stays outside: `rolling_route_reuse_start` and the
strict exact-prefix rule in `_prepare_memory_for_prompt` remain the single
authority on whether reuse is allowed.
"""
from __future__ import annotations

from .rolling_route_anchor import (
    ROLLING_ANALYSIS_STRATEGY_ID,
    RollingRouteAnchorState,
    RollingRouteIdentity,
    invalidate_rolling_route_anchor,
)


class RollingAnchorStore:
    """The single owner of the two rolling-anchor checkpoint slots."""

    __slots__ = ("_route", "_analysis")

    def __init__(self) -> None:
        self._route = RollingRouteAnchorState()
        self._analysis = RollingRouteAnchorState()

    @staticmethod
    def slot_for(identity: RollingRouteIdentity | None) -> str:
        """Which checkpoint an identity addresses, decided by its own strategy.

        The backend still knows nothing about CHAT or ANALYSIS: it reads the
        strategy the caller already put in the identity it supplied.
        """
        if identity is not None and identity.strategy_id == ROLLING_ANALYSIS_STRATEGY_ID:
            return "analysis"
        return "route"

    @property
    def route_state(self) -> RollingRouteAnchorState:
        return self._route

    @property
    def analysis_state(self) -> RollingRouteAnchorState:
        return self._analysis

    def state_for(self, identity: RollingRouteIdentity | None) -> RollingRouteAnchorState:
        """What is stored in the slot this identity addresses.

        An absent slot reads as empty rather than raising: a checkpoint that is
        not there must fall cold, never fail the call.
        """
        if self.slot_for(identity) == "analysis":
            return self._analysis or RollingRouteAnchorState()
        return self._route

    def store(
        self, identity: RollingRouteIdentity | None, state: RollingRouteAnchorState
    ) -> None:
        """Record a checkpoint in the slot this identity addresses."""
        if self.slot_for(identity) == "analysis":
            self._analysis = state
        else:
            self._route = state

    def invalidate(self, reason: str) -> None:
        """Invalidate BOTH lineages.

        A reset destroys the conversation whose tokens these are, and an
        analysis checkpoint surviving it would be exactly the stale-state reuse
        the identity check exists to prevent.

        An already-empty slot is left untouched rather than rewritten with a
        fresh invalidated state: that mirrors the shipped behaviour, and it
        keeps the recorded `invalidation_reason` of the first invalidation
        instead of overwriting it with whatever reason came last.
        """
        if self._route.valid or self._route.identity is not None:
            self._route = invalidate_rolling_route_anchor(self._route, reason)
        analysis = self._analysis
        if analysis is not None and (analysis.valid or analysis.identity is not None):
            self._analysis = invalidate_rolling_route_anchor(analysis, reason)

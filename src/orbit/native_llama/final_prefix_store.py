"""Ownership of the final-prefix anchor state and its transitions.

The final-prefix experiment caches the stable prefix of a final-answer prompt as
a KV checkpoint, so a follow-up turn can restore it instead of prefilling. Two
pieces of state describe that: the `PrefixAnchorState` checkpoint itself, and a
`FinalPrefixExperimentStatus` counting captures, restores and fallbacks.

The status transitions were written out by hand at eight sites -- "ready" twice,
"not ready" three times, "unused" four -- each a handful of field assignments.
That duplication is how the counters drift away from the checkpoint they
describe.

This owns both, and the transitions over them. It owns no policy and touches
nothing native:

* **Eligibility and planning** stay in the client: whether the experiment
  applies, which segments form the prefix, and the identity kwargs are decided
  from config, profile and paths.
* **Native capture and restore** stay in the client. `capture_prefix_anchor`
  and `restore_prefix_anchor` need `ctx_tgt` and `lib`, and the client calls
  them and hands the RESULT here. This module never sees a context or a
  library, so physical KV ownership does not move.
* **Committed identity** belongs to `CommittedIdentity`, and rolling anchors to
  `RollingAnchorStore`. Neither is touched, and neither is merged with this:
  they look similar and are not the same thing.

Ordering matters and is preserved exactly: a failed capture records a fallback
and leaves no valid checkpoint, so a stale anchor can never become newly
reusable because the counters were updated in a different order.
"""
from __future__ import annotations

from dataclasses import dataclass

from .prefix_anchor import PrefixAnchorState


@dataclass
class FinalPrefixExperimentStatus:
    """What the final-prefix experiment has done so far, for reporting."""

    initialized: bool = False
    prefix_tokens: int = 0
    capture_count: int = 0
    restore_count: int = 0
    fallback_count: int = 0
    failure_reason: str | None = None
    last_used: bool = False


class FinalPrefixStore:
    """The single owner of the final-prefix checkpoint and its status."""

    __slots__ = ("_anchor", "_status")

    def __init__(self) -> None:
        self._anchor = PrefixAnchorState()
        self._status = FinalPrefixExperimentStatus()

    @property
    def anchor(self) -> PrefixAnchorState:
        return self._anchor

    @anchor.setter
    def anchor(self, state: PrefixAnchorState) -> None:
        self._anchor = state

    @property
    def status(self) -> FinalPrefixExperimentStatus:
        return self._status

    def mark_ready(self, prefix_tokens: int, *, captured: bool = False,
                   restored: bool = False) -> None:
        """Record that the prefix is resident and usable.

        `captured` and `restored` increment the counter for whichever produced
        it -- they are separate totals, so a restore must not inflate captures.
        """
        self._status.initialized = True
        self._status.prefix_tokens = prefix_tokens
        if captured:
            self._status.capture_count += 1
        if restored:
            self._status.restore_count += 1
        self._status.failure_reason = None
        self._status.last_used = True

    def mark_not_ready(self) -> None:
        """Record that no prefix is resident, without counting a fallback.

        Used on the paths that immediately follow with `record_fallback`, so
        the fallback total is incremented exactly once.
        """
        self._status.initialized = False
        self._status.prefix_tokens = 0

    def record_fallback(self, reason: str) -> None:
        """Record that the experiment gave way to the normal path."""
        self._status.fallback_count += 1
        self._status.failure_reason = reason
        self._status.last_used = False

    def mark_unused(self) -> None:
        """Record that this turn did not use the prefix, changing nothing else.

        Deliberately narrow: the checkpoint may still be perfectly valid, and
        the counters describe history, not the current turn.
        """
        self._status.last_used = False

    def invalidate(self, reason: str) -> None:
        """Drop the checkpoint and mark it unusable.

        The anchor is replaced with an empty state rather than flagged, so no
        stale checkpoint can survive to be restored later.
        """
        self._anchor = PrefixAnchorState()
        self._status.initialized = False
        self._status.prefix_tokens = 0
        self._status.failure_reason = reason
        self._status.last_used = False

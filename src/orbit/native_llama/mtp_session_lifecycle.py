"""Ownership of the persistent-MTP session handle and its derived state.

An MTP session is one native runtime (`PersistentMtpSessionRuntime`) plus four
session fields that describe it: `ctx_dft`, `spec`, `mtp_enabled`, and the
failure pair `mtp_failed` / `mtp_failure_reason`. Those fields are not
independent -- they are a projection of whether a runtime exists and how its
last construction went -- yet the transitions were written out by hand at six
publish sites and five failure sites, which is how they drift apart.

This collaborator owns the handle and those transitions. It owns no policy:
which path to try, whether an artifact qualifies, whether MTP was requested, and
the SOFT/HARD reset decision all stay where they are. It is told what happened
and records it consistently.

Deliberately kept apart:

* **self-MTP and external-draft remain distinct.** They differ in who owns the
  model, so they are separate entry points here, exactly as they are today. No
  branch on model names, and no shared "construct MTP" abstraction that would
  imply the two are interchangeable.
* **Native ownership is unchanged.** Self-MTP borrows the caller's model and
  target context and owns only the draft context and speculative state; the
  runtime object carries that fact and teardown goes through the same
  `free_persistent_mtp_session` in the same order.
* **Committed identity is not touched.** That belongs to `CommittedIdentity`
  (REF-1) and is reached only through its own contract.
"""
from __future__ import annotations

from typing import Callable


class MtpSessionLifecycle:
    """The single owner of the MTP runtime handle and its derived state.

    `session_of` returns the `NativeSessionState` whose fields project this
    runtime; `free_runtime` tears a runtime down. Nothing else is required, so
    the client is not passed in and cannot be reached from here.
    """

    __slots__ = ("_session_of", "_free_runtime", "_runtime")

    def __init__(self, session_of: Callable[[], object], *,
                 free_runtime: Callable[[object], None]) -> None:
        # A getter, not the session itself: a client built via `object.__new__`
        # may not have one yet, and state writes must land on whichever session
        # it eventually assigns -- as they did when these were plain attribute
        # writes in the client.
        self._session_of = session_of
        self._free_runtime = free_runtime
        self._runtime = None

    @property
    def _session(self):
        return self._session_of()

    @property
    def runtime(self):
        """The live runtime, or None when no MTP session is constructed."""
        return self._runtime

    def clear_state(self) -> None:
        """Reset the derived fields to "no session", without freeing anything.

        Used at the start of initialization, where the runtime has already been
        freed separately and only the projection needs resetting.
        """
        self._session.ctx_dft = None
        self._session.spec = None
        self._session.mtp_enabled = False
        self._session.mtp_failed = False
        self._session.mtp_failure_reason = None

    def publish(self, runtime, *, clear_failure_reason: bool = False) -> None:
        """Record a constructed runtime as the live MTP session.

        `clear_failure_reason` mirrors the existing call sites: initialization
        and the in-completion republish leave a previously recorded reason
        alone, while `reset_session_state` and the post-cancel rebuild clear it.
        Preserved rather than unified -- collapsing them would be a behaviour
        change, however tidy it looks.
        """
        self._runtime = runtime
        self._session.ctx_dft = runtime.ctx_dft
        self._session.spec = runtime.spec
        self._session.mtp_enabled = True
        self._session.mtp_failed = False
        if clear_failure_reason:
            self._session.mtp_failure_reason = None

    def attach(self, runtime) -> None:
        """Point the session at a runtime WITHOUT declaring it ready.

        Distinct from `publish`: this records the handle and its two derived
        context fields but leaves `mtp_enabled` / `mtp_failed` exactly as they
        were. The completion path uses it where the runtime may be the one
        already in use, and where readiness is decided later by the outcome --
        using `publish` there would flip a failed session to enabled before it
        had succeeded.
        """
        self._runtime = runtime
        self._session.ctx_dft = runtime.ctx_dft
        self._session.spec = runtime.spec

    def mark_ready(self) -> None:
        """Declare the current session healthy after a successful completion.

        The counterpart to `record_failure`: no runtime changes hands, only the
        verdict. Kept separate from `publish` because there is nothing to
        publish -- the runtime is already attached.
        """
        self._session.mtp_failed = False
        self._session.mtp_failure_reason = None
        self._session.mtp_enabled = True

    def record_failure(self, reason: str, *, disable: bool = False) -> None:
        """Record that MTP could not be constructed or continued.

        `disable` distinguishes the two shapes already in the code: an
        initialization failure marks the session failed but leaves
        `mtp_enabled` as it was, while a failure that invalidates a live
        session also disables it.
        """
        self._session.mtp_failed = True
        self._session.mtp_failure_reason = reason
        if disable:
            self._session.mtp_enabled = False

    def record_unavailable(self, reason: str) -> None:
        """Record that MTP does not apply, without marking it broken.

        Distinct from `record_failure`: nothing went wrong. The artifact or
        profile simply does not offer this architecture, so `mtp_failed` stays
        False and only the reason is recorded.

        The distinction is preserved because the pre-extraction code drew it,
        not because anything currently depends on it: `mtp_failed` has no
        production reader today and is absent from `NativeSessionSnapshot`.
        That makes it a latent invariant rather than a live one -- but
        "unsupported" and "broken" are genuinely different states, the field is
        public on the session dataclass, and collapsing them during a
        behaviour-preserving extraction would be an unforced change.
        """
        self._session.mtp_failure_reason = reason

    def discard(self, reason: str) -> None:
        """Drop a session whose runtime is no longer usable.

        The runtime is dropped WITHOUT freeing it: this is the path where the
        native side already tore itself down (a failed reset), so freeing again
        would be a double free.
        """
        self._runtime = None
        self._session.ctx_dft = None
        self._session.spec = None
        self._session.mtp_enabled = False
        self._session.mtp_failed = True
        self._session.mtp_failure_reason = reason

    def free(self) -> None:
        """Tear down the live session, if any. Idempotent.

        The derived fields are cleared in a `finally`, so a raising teardown
        still leaves the session describing "no MTP" rather than pointing at
        freed handles. `mtp_failure_reason` is deliberately NOT cleared here:
        the existing teardown leaves it in place.
        """
        runtime = self._runtime
        if runtime is None:
            return
        try:
            self._free_runtime(runtime)
        finally:
            self._runtime = None
            self._session.ctx_dft = None
            self._session.spec = None
            self._session.mtp_enabled = False

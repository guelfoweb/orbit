from __future__ import annotations

from ctypes import c_void_p
from dataclasses import dataclass, field

from .events import NativeTimings


DEFAULT_NATIVE_SESSION_ID = "default"


def _unavailable_tokenizer(_text: str) -> list[int]:
    """Stand-in tokenizer for a session with no client bound.

    Raising is the correct behaviour, not a defect: every caller of the
    tokenizing path already treats a tokenizer failure as "no reuse", so an
    unbound session refuses reuse rather than guessing.
    """
    raise RuntimeError("no tokenizer bound to this session")


@dataclass(frozen=True)
class NativeSessionSnapshot:
    session_id: str
    cached_tokens: int
    in_flight: bool
    cancel_requested: bool
    backend_mode: str
    last_metrics: NativeTimings | None
    mtp_enabled: bool
    mtp_initialized: bool
    mtp_failure_reason: str | None


@dataclass
class NativeSessionState:
    session_id: str = DEFAULT_NATIVE_SESSION_ID
    ctx_tgt: c_void_p | None = None
    sampler: c_void_p | None = None
    cached_prompt_tokens: list[int] = field(default_factory=list)
    # Exact tokens currently resident in the backend KV sequence: the prompt
    # that was prefilled plus every generated token that was successfully
    # decoded into it. Empty whenever that identity cannot be proven.
    #
    # Owned entirely by `CommittedIdentity` (see the property below). The
    # session holds no copy of its own: two states needing synchronisation is
    # exactly the failure this ownership move exists to prevent.
    #
    # TODO(REF-follow-up): being a dataclass field, this owner participates in
    # the generated `__eq__`/`__repr__` and is shared by `dataclasses.replace`
    # and `copy.copy`, so two sessions with identical tokens now compare
    # unequal and a replaced session would share one identity. Nothing does any
    # of that today -- there is no `replace`, `asdict`, `copy` or session-to-
    # session `==` in src/ or tests/ -- so fixing it (compare=False, repr=False,
    # plus __deepcopy__) is deferred rather than smuggled into a
    # behaviour-preserving extraction.
    _committed_identity_owner: object | None = None
    prompt_cache_mode: str | None = None
    in_flight: bool = False
    cancel_requested: bool = False
    continuation_ready: bool = False
    last_metrics: NativeTimings | None = None
    # Reserved for future persistent MTP session state.
    ctx_dft: c_void_p | None = None
    spec: c_void_p | None = None
    mtp_enabled: bool = False
    mtp_failed: bool = False
    mtp_failure_reason: str | None = None

    def bind_committed_identity(self, owner) -> None:
        """Adopt a fully-wired owner, carrying any tokens already recorded.

        The session always has an owner (see `committed_identity`); this
        replaces it with one that has the client's tokenizer and profile
        available, which the bare default cannot supply.
        """
        current = self.committed_identity
        if current is not owner and current.tokens:
            owner.adopt(current.tokens)
        self._committed_identity_owner = owner

    @property
    def committed_identity(self):
        """The sole owner of committed-token identity for this session.

        Created on first access so a session is never without one -- several
        call sites build a client via `__new__` and assign `_session` directly,
        and identity must behave identically there.
        """
        owner = self._committed_identity_owner
        if owner is None:
            from .committed_identity import CommittedIdentity

            owner = CommittedIdentity(
                tokenize=_unavailable_tokenizer,
                coder_protocol=lambda: False,
            )
            self._committed_identity_owner = owner
        return owner

    @property
    def committed_sequence_tokens(self) -> list[int]:
        return self.committed_identity.tokens

    @committed_sequence_tokens.setter
    def committed_sequence_tokens(self, value) -> None:
        self.committed_identity.adopt(value)

    def snapshot(self, *, backend_mode: str = "no-mtp") -> NativeSessionSnapshot:
        return NativeSessionSnapshot(
            session_id=self.session_id,
            cached_tokens=len(self.cached_prompt_tokens),
            in_flight=self.in_flight,
            cancel_requested=self.cancel_requested,
            backend_mode=backend_mode,
            last_metrics=self.last_metrics,
            mtp_enabled=self.mtp_enabled,
            mtp_initialized=self.ctx_dft is not None and self.spec is not None and self.mtp_enabled,
            mtp_failure_reason=self.mtp_failure_reason,
        )

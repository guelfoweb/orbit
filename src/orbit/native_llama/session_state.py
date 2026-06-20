from __future__ import annotations

from ctypes import c_void_p
from dataclasses import dataclass, field

from .events import NativeTimings


DEFAULT_NATIVE_SESSION_ID = "default"


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
    chat_visible_frontier_tokens: list[int] = field(default_factory=list)
    committed_frontier_tokens: list[int] = field(default_factory=list)
    raw_emitted_token_ids: list[int] = field(default_factory=list)
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

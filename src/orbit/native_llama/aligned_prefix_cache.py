"""One bounded, exact-token native prefill checkpoint; no response caching."""
from __future__ import annotations

from ctypes import c_ubyte
from dataclasses import dataclass, field
import os
from typing import Any, Mapping

PREFIX_TOKENS = 1024
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("ORBIT_ALIGNED_PREFIX_REUSE", "1").strip().lower() in {"1", "true", "yes", "on"}


def qualified_config(config: Any) -> bool:
    # Initial native qualification covers this exact CPU call layout. In
    # particular, 1024 lies on the unchanged 64-token cold-decode boundary.
    values = ("context_tokens", "threads", "threads_batch", "batch_size", "ubatch_size", "progress_step")
    return (tuple(getattr(config, key, None) for key in values) == (8192, 6, 6, 256, 128, 64)
            and getattr(config, "use_extra_bufts", None) is False
            and getattr(config, "gpu_layers", None) == 0
            and getattr(config, "low_memory", None) is False
            and all(getattr(config, name, None) is False for name in (
                "use_mtp_experimental", "mtp_probe_enabled", "mtp_dry_run_enabled",
                "mtp_accept_probe_enabled", "mtp_decode_probe_enabled", "moe_expert_usage_enabled",
            )))


@dataclass(frozen=True)
class Snapshot:
    identity: str
    tokens: tuple[int, ...] = field(repr=False)
    data: bytes = field(repr=False, compare=False)


class AlignedPrefixCache:
    def __init__(self) -> None:
        self.snapshot: Snapshot | None = None
        self.captures = 0
        self.restores = 0
        self.failure_reason: str | None = None

    def clear(self, reason: str) -> None:
        self.snapshot = None
        self.failure_reason = reason

    def matches(self, prompt: list[int], identity: str) -> bool:
        saved = self.snapshot
        return bool(saved is not None and saved.identity == identity
                    and 0 < len(saved.data) <= MAX_CHECKPOINT_BYTES
                    and len(prompt) > PREFIX_TOKENS
                    and tuple(prompt[:PREFIX_TOKENS]) == saved.tokens)

    def capture(self, lib: Any, ctx: Any, prefix: list[int], identity: str) -> bool:
        if len(prefix) != PREFIX_TOKENS or not ctx:
            return False
        try:
            size = int(lib.llama_state_seq_get_size(ctx, 0))
            if not 0 < size <= MAX_CHECKPOINT_BYTES:
                self.failure_reason = "checkpoint_size_limit"
                return False
            # Evict before allocation: at most one resident snapshot, plus the
            # temporary ctypes copy. Failed capture falls back to cold work.
            self.clear("replacing_checkpoint")
            buffer = (c_ubyte * size)()
            written = int(lib.llama_state_seq_get_data(ctx, buffer, size, 0))
            if written != size:
                self.failure_reason = "capture_size_mismatch"
                return False
            self.snapshot = Snapshot(identity, tuple(prefix), bytes(buffer))
        except Exception:
            self.clear("capture_failed")
            return False
        self.captures += 1
        self.failure_reason = None
        return True

    def restore(self, lib: Any, ctx: Any, prompt: list[int], identity: str) -> bool:
        # The caller clears physical KV and committed identity before a restore,
        # and must clear again after failure: a failed write can be partial.
        if not ctx or not self.matches(prompt, identity):
            return False
        saved = self.snapshot
        assert saved is not None
        try:
            buffer = (c_ubyte * len(saved.data)).from_buffer_copy(saved.data)
            consumed = int(lib.llama_state_seq_set_data(ctx, buffer, len(saved.data), 0))
            if consumed != len(saved.data):
                self.clear("restore_size_mismatch")
                return False
        except Exception:
            self.clear("restore_failed")
            return False
        self.restores += 1
        self.failure_reason = None
        return True

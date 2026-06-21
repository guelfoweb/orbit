from __future__ import annotations

from ctypes import CDLL, CFUNCTYPE, c_bool, c_char_p, c_int32, c_long, c_uint32, c_void_p
from dataclasses import dataclass
from pathlib import Path
import ctypes
import os
import subprocess

from .build_support import compile_cpp_helper
from .mtp_completion import MtpCompletionResult
from .native_artifacts import packaged_shim_path, require_legacy_llama_root
from .native_names import persistent_mtp_shim_filename, platform_runtime_libs
from .paths import NativeLlamaPaths

_REQUIRED_SHIM_SYMBOLS = (
    "orbit_mtp_session_complete",
    "orbit_mtp_session_set_followup_suffix_tokens",
)

MtpTokenCallback = CFUNCTYPE(None, c_char_p, c_void_p)
MtpProgressCallback = CFUNCTYPE(None, c_int32, c_int32, c_int32, c_void_p)


def _noop_token_callback(_text: bytes | None, _user_data) -> None:
    return None


def _noop_progress_callback(_phase: int, _current: int, _total: int, _user_data) -> None:
    return None


@dataclass(frozen=True)
class PersistentMtpSessionRuntime:
    handle: c_void_p
    ctx_dft: c_void_p
    spec: c_void_p
    rss_before_kb: int | None = None
    rss_after_init_kb: int | None = None
    rss_peak_kb: int | None = None


class PersistentMtpLibrary:
    def __init__(self, build_bin: Path, shim_path: Path) -> None:
        self.build_bin = build_bin
        self.shim_path = shim_path
        self._handles: list[CDLL] = []
        self.lib = self._load_library()
        self._configure_api()

    def _load_library(self) -> CDLL:
        flags = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        for dep in platform_runtime_libs():
            path = self.build_bin / dep
            if path.exists():
                self._handles.append(ctypes.CDLL(str(path), mode=flags))
        return ctypes.CDLL(str(self.shim_path), mode=flags)

    def _configure_api(self) -> None:
        lib = self.lib
        lib.orbit_mtp_last_error.argtypes = []
        lib.orbit_mtp_last_error.restype = c_char_p
        lib.orbit_mtp_session_create.argtypes = [c_char_p, c_void_p, c_uint32, c_uint32, c_uint32, c_int32, c_int32]
        lib.orbit_mtp_session_create.restype = c_void_p
        lib.orbit_mtp_session_reset.argtypes = [c_void_p, c_void_p]
        lib.orbit_mtp_session_reset.restype = c_bool
        lib.orbit_mtp_session_free.argtypes = [c_void_p]
        lib.orbit_mtp_session_free.restype = None
        lib.orbit_mtp_session_ctx_dft.argtypes = [c_void_p]
        lib.orbit_mtp_session_ctx_dft.restype = c_void_p
        lib.orbit_mtp_session_spec.argtypes = [c_void_p]
        lib.orbit_mtp_session_spec.restype = c_void_p
        lib.orbit_mtp_session_rss_before_kb.argtypes = [c_void_p]
        lib.orbit_mtp_session_rss_before_kb.restype = c_long
        lib.orbit_mtp_session_rss_after_init_kb.argtypes = [c_void_p]
        lib.orbit_mtp_session_rss_after_init_kb.restype = c_long
        lib.orbit_mtp_session_rss_peak_kb.argtypes = [c_void_p]
        lib.orbit_mtp_session_rss_peak_kb.restype = c_long
        lib.orbit_mtp_session_complete.argtypes = [
            c_void_p,
            c_void_p,
            c_char_p,
            c_int32,
            MtpTokenCallback,
            MtpProgressCallback,
            c_void_p,
        ]
        lib.orbit_mtp_session_complete.restype = c_bool
        lib.orbit_mtp_session_last_content.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_content.restype = c_char_p
        lib.orbit_mtp_session_last_output_tokens.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_output_tokens.restype = c_int32
        lib.orbit_mtp_session_last_draft_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_draft_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_accepted_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_accepted_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_rejected_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_rejected_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_reused_draft_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_reused_draft_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_reused_accepted_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_reused_accepted_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_reused_rejected_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_reused_rejected_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_acceptance_ratio.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_acceptance_ratio.restype = ctypes.c_double
        lib.orbit_mtp_session_last_fresh_acceptance_ratio.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_fresh_acceptance_ratio.restype = ctypes.c_double
        lib.orbit_mtp_session_last_consumed_acceptance_ratio.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_consumed_acceptance_ratio.restype = ctypes.c_double
        lib.orbit_mtp_session_last_target_decode_calls.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_target_decode_calls.restype = c_int32
        lib.orbit_mtp_session_last_draft_decode_calls.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_draft_decode_calls.restype = c_int32
        lib.orbit_mtp_session_last_elapsed_ms.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_elapsed_ms.restype = ctypes.c_double
        lib.orbit_mtp_session_last_tokens_per_second.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_tokens_per_second.restype = ctypes.c_double
        lib.orbit_mtp_session_last_full_accept_steps.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_full_accept_steps.restype = c_int32
        lib.orbit_mtp_session_last_replay_steps.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_replay_steps.restype = c_int32
        lib.orbit_mtp_session_last_partial_accept_steps.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_partial_accept_steps.restype = c_int32
        lib.orbit_mtp_session_last_partial_no_replay_steps.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_partial_no_replay_steps.restype = c_int32
        lib.orbit_mtp_session_last_replay_fallback_steps.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_replay_fallback_steps.restype = c_int32
        lib.orbit_mtp_session_last_seq_rm_supported.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_seq_rm_supported.restype = c_bool
        lib.orbit_mtp_session_last_rollback_tokens_total.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_rollback_tokens_total.restype = c_int32
        lib.orbit_mtp_session_last_checkpoint_count.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_checkpoint_count.restype = c_int32
        lib.orbit_mtp_session_last_restore_count.argtypes = [c_void_p]
        lib.orbit_mtp_session_last_restore_count.restype = c_int32


def build_persistent_mtp_shim(
    *,
    llama_root: Path | None,
    build_dir: Path | None = None,
    build_bin: Path | None = None,
    runner=subprocess.run,
) -> Path:
    packaged = packaged_shim_path(persistent_mtp_shim_filename())
    if packaged is not None and _shim_exports_required_symbols(packaged):
        return packaged
    artifact_name = persistent_mtp_shim_filename()
    llama_root = require_legacy_llama_root(llama_root, artifact_name=artifact_name)
    build_root = build_dir or (Path.home() / ".orbit" / "native-build")
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_persistent_mtp.cpp"
    output = build_root / artifact_name
    return compile_cpp_helper(
        artifact_label="persistent mtp shim",
        source=source,
        output=output,
        llama_root=llama_root,
        build_bin=build_bin,
        runner=runner,
        shared=True,
    )


def _shim_exports_required_symbols(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    flags = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    try:
        lib = ctypes.CDLL(str(path), mode=flags)
    except OSError:
        return False
    return all(hasattr(lib, symbol) for symbol in _REQUIRED_SHIM_SYMBOLS)


def create_persistent_mtp_session(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    ctx_tgt: c_void_p,
    context_tokens: int,
    batch_size: int,
    ubatch_size: int,
    threads: int,
    threads_batch: int,
    build_dir: Path | None = None,
    runner=subprocess.run,
    library_factory=PersistentMtpLibrary,
) -> PersistentMtpSessionRuntime:
    if not paths.mtp_available or paths.draft_mtp_model is None:
        raise RuntimeError(paths.fallback_reason or "draft-mtp-unavailable")
    shim = build_persistent_mtp_shim(llama_root=llama_root, build_dir=build_dir, runner=runner)
    library = library_factory(paths.build_bin, shim)
    handle = library.lib.orbit_mtp_session_create(
        str(paths.draft_mtp_model).encode(),
        ctx_tgt,
        context_tokens,
        batch_size,
        ubatch_size,
        threads,
        threads_batch,
    )
    if not handle:
        raise RuntimeError(_decode_error(library.lib.orbit_mtp_last_error()))
    return PersistentMtpSessionRuntime(
        handle=c_void_p(handle),
        ctx_dft=library.lib.orbit_mtp_session_ctx_dft(handle),
        spec=library.lib.orbit_mtp_session_spec(handle),
        rss_before_kb=_long_or_none(library.lib.orbit_mtp_session_rss_before_kb(handle)),
        rss_after_init_kb=_long_or_none(library.lib.orbit_mtp_session_rss_after_init_kb(handle)),
        rss_peak_kb=_long_or_none(library.lib.orbit_mtp_session_rss_peak_kb(handle)),
    )


def reset_persistent_mtp_session(
    *,
    paths: NativeLlamaPaths,
    runtime: PersistentMtpSessionRuntime,
    ctx_tgt: c_void_p,
    build_dir: Path | None = None,
    library_factory=PersistentMtpLibrary,
    llama_root: Path,
) -> PersistentMtpSessionRuntime:
    shim = build_persistent_mtp_shim(llama_root=llama_root, build_dir=build_dir)
    library = library_factory(paths.build_bin, shim)
    ok = library.lib.orbit_mtp_session_reset(runtime.handle, ctx_tgt)
    if not ok:
        raise RuntimeError(_decode_error(library.lib.orbit_mtp_last_error()))
    return PersistentMtpSessionRuntime(
        handle=runtime.handle,
        ctx_dft=library.lib.orbit_mtp_session_ctx_dft(runtime.handle),
        spec=library.lib.orbit_mtp_session_spec(runtime.handle),
        rss_before_kb=runtime.rss_before_kb,
        rss_after_init_kb=runtime.rss_after_init_kb,
        rss_peak_kb=runtime.rss_peak_kb,
    )


def free_persistent_mtp_session(
    *,
    paths: NativeLlamaPaths,
    runtime: PersistentMtpSessionRuntime,
    build_dir: Path | None = None,
    library_factory=PersistentMtpLibrary,
    llama_root: Path,
) -> None:
    shim = build_persistent_mtp_shim(llama_root=llama_root, build_dir=build_dir)
    library = library_factory(paths.build_bin, shim)
    library.lib.orbit_mtp_session_free(runtime.handle)


def run_persistent_mtp_completion(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    runtime: PersistentMtpSessionRuntime | None,
    ctx_tgt: c_void_p,
    prompt: str,
    max_tokens: int,
    on_token=None,
    on_progress=None,
    build_dir: Path | None = None,
    library_factory=PersistentMtpLibrary,
) -> MtpCompletionResult:
    if runtime is None:
        return MtpCompletionResult(enabled=True, success=False, error="persistent-mtp-uninitialized")
    shim = build_persistent_mtp_shim(llama_root=llama_root, build_dir=build_dir)
    library = library_factory(paths.build_bin, shim)
    if on_token is not None:
        def _token_cb(text: bytes | None, _user_data) -> None:
            if text:
                on_token(text.decode(errors="replace"))
        token_cb = MtpTokenCallback(_token_cb)
    else:
        token_cb = MtpTokenCallback(_noop_token_callback)
    if on_progress is not None:
        def _progress_cb(phase: int, current: int, total: int, _user_data) -> None:
            on_progress(phase, current, total)
        progress_cb = MtpProgressCallback(_progress_cb)
    else:
        progress_cb = MtpProgressCallback(_noop_progress_callback)
    ok = library.lib.orbit_mtp_session_complete(
        runtime.handle,
        ctx_tgt,
        prompt.encode(),
        max(1, min(max_tokens, 32)),
        token_cb,
        progress_cb,
        None,
    )
    if not ok:
        return MtpCompletionResult(
            enabled=True,
            success=False,
            error=_decode_error(library.lib.orbit_mtp_last_error()),
        )
    return MtpCompletionResult(
        enabled=True,
        success=True,
        error=None,
        content=_decode_text(library.lib.orbit_mtp_session_last_content(runtime.handle)),
        output_tokens=int(library.lib.orbit_mtp_session_last_output_tokens(runtime.handle)),
        draft_tokens_total=int(library.lib.orbit_mtp_session_last_draft_tokens_total(runtime.handle)),
        accepted_tokens_total=int(library.lib.orbit_mtp_session_last_accepted_tokens_total(runtime.handle)),
        rejected_tokens_total=int(library.lib.orbit_mtp_session_last_rejected_tokens_total(runtime.handle)),
        reused_draft_tokens_total=int(library.lib.orbit_mtp_session_last_reused_draft_tokens_total(runtime.handle)),
        reused_accepted_tokens_total=int(library.lib.orbit_mtp_session_last_reused_accepted_tokens_total(runtime.handle)),
        reused_rejected_tokens_total=int(library.lib.orbit_mtp_session_last_reused_rejected_tokens_total(runtime.handle)),
        acceptance_ratio=float(library.lib.orbit_mtp_session_last_acceptance_ratio(runtime.handle)),
        fresh_acceptance_ratio=float(library.lib.orbit_mtp_session_last_fresh_acceptance_ratio(runtime.handle)),
        consumed_acceptance_ratio=float(library.lib.orbit_mtp_session_last_consumed_acceptance_ratio(runtime.handle)),
        target_decode_calls=int(library.lib.orbit_mtp_session_last_target_decode_calls(runtime.handle)),
        draft_decode_calls=int(library.lib.orbit_mtp_session_last_draft_decode_calls(runtime.handle)),
        elapsed_ms=float(library.lib.orbit_mtp_session_last_elapsed_ms(runtime.handle)),
        tokens_per_second=float(library.lib.orbit_mtp_session_last_tokens_per_second(runtime.handle)),
        full_accept_steps=int(library.lib.orbit_mtp_session_last_full_accept_steps(runtime.handle)),
        replay_steps=int(library.lib.orbit_mtp_session_last_replay_steps(runtime.handle)),
        partial_accept_steps=int(library.lib.orbit_mtp_session_last_partial_accept_steps(runtime.handle)),
        partial_no_replay_steps=int(library.lib.orbit_mtp_session_last_partial_no_replay_steps(runtime.handle)),
        replay_fallback_steps=int(library.lib.orbit_mtp_session_last_replay_fallback_steps(runtime.handle)),
        seq_rm_supported=bool(library.lib.orbit_mtp_session_last_seq_rm_supported(runtime.handle)),
        rollback_tokens_total=int(library.lib.orbit_mtp_session_last_rollback_tokens_total(runtime.handle)),
        checkpoint_count=int(library.lib.orbit_mtp_session_last_checkpoint_count(runtime.handle)),
        restore_count=int(library.lib.orbit_mtp_session_last_restore_count(runtime.handle)),
    )


def _decode_error(value: bytes | None) -> str:
    if not value:
        return "persistent mtp operation failed"
    return value.decode(errors="replace")


def _decode_text(value: bytes | None) -> str:
    if not value:
        return ""
    return value.decode(errors="replace")


def _long_or_none(value: int | None) -> int | None:
    if not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from .build_support import compile_cpp_helper
from .native_artifacts import packaged_shim_path, require_legacy_llama_root
from .paths import NativeLlamaPaths


@dataclass(frozen=True)
class MtpCompletionResult:
    enabled: bool
    success: bool
    error: str | None
    content: str = ""
    output_tokens: int = 0
    draft_tokens_total: int = 0
    accepted_tokens_total: int = 0
    rejected_tokens_total: int = 0
    acceptance_ratio: float | None = None
    fresh_acceptance_ratio: float | None = None
    consumed_acceptance_ratio: float | None = None
    reused_draft_tokens_total: int = 0
    reused_accepted_tokens_total: int = 0
    reused_rejected_tokens_total: int = 0
    target_decode_calls: int = 0
    draft_decode_calls: int = 0
    elapsed_ms: float | None = None
    tokens_per_second: float | None = None
    full_accept_steps: int = 0
    replay_steps: int = 0
    partial_accept_steps: int = 0
    partial_no_replay_steps: int = 0
    replay_fallback_steps: int = 0
    seq_rm_supported: bool = False
    rollback_tokens_total: int = 0
    checkpoint_count: int = 0
    restore_count: int = 0
    validate_steps: int = 0
    rows_requested_total: int = 0
    rows_consumed_estimated_total: int = 0
    rows_wasted_estimated_total: int = 0
    rows_wasted_estimated_ratio: float | None = None
    accepted_draft_hist_0: int = 0
    accepted_draft_hist_1: int = 0
    accepted_draft_hist_2: int = 0
    accepted_draft_hist_3: int = 0
    accepted_draft_hist_ge4: int = 0
    trace_json: str | None = None
    timing_json: str | None = None
    validate_trace_json: str | None = None
    target_decode_trace_json: str | None = None
    output_token_hashes_json: str | None = None
    first_sample_trace_json: str | None = None


def run_mtp_completion(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    prompt: str,
    max_tokens: int,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> MtpCompletionResult:
    if not paths.mtp_available or paths.draft_mtp_model is None:
        return MtpCompletionResult(enabled=True, success=False, error=paths.fallback_reason or "draft-mtp-unavailable")

    helper = build_mtp_completion_helper(llama_root=llama_root, build_dir=build_dir, runner=runner)
    completed = runner(
        [str(helper), str(paths.model), str(paths.draft_mtp_model), prompt, str(max_tokens)],
        capture_output=True,
        text=True,
        env={"LD_LIBRARY_PATH": str(paths.build_bin)},
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return MtpCompletionResult(enabled=True, success=False, error=f"mtp completion failed with exit code {completed.returncode}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return MtpCompletionResult(enabled=True, success=False, error=f"invalid mtp completion output: {exc}")
    if completed.returncode != 0 or not payload.get("ok"):
        return MtpCompletionResult(
            enabled=True,
            success=False,
            error=str(payload.get("error") or f"mtp completion failed with exit code {completed.returncode}"),
            content=str(payload.get("content") or ""),
            output_tokens=_int_or_default(payload.get("output_tokens")),
            draft_tokens_total=_int_or_default(payload.get("draft_tokens_total")),
            accepted_tokens_total=_int_or_default(payload.get("accepted_tokens_total")),
            rejected_tokens_total=_int_or_default(payload.get("rejected_tokens_total")),
            acceptance_ratio=_float_or_none(payload.get("acceptance_ratio")),
            target_decode_calls=_int_or_default(payload.get("target_decode_calls")),
            draft_decode_calls=_int_or_default(payload.get("draft_decode_calls")),
            elapsed_ms=_float_or_none(payload.get("elapsed_ms")),
            tokens_per_second=_float_or_none(payload.get("tokens_per_second")),
        )
    return MtpCompletionResult(
        enabled=True,
        success=True,
        error=None,
        content=str(payload.get("content") or ""),
        output_tokens=_int_or_default(payload.get("output_tokens")),
        draft_tokens_total=_int_or_default(payload.get("draft_tokens_total")),
        accepted_tokens_total=_int_or_default(payload.get("accepted_tokens_total")),
        rejected_tokens_total=_int_or_default(payload.get("rejected_tokens_total")),
        acceptance_ratio=_float_or_none(payload.get("acceptance_ratio")),
        target_decode_calls=_int_or_default(payload.get("target_decode_calls")),
        draft_decode_calls=_int_or_default(payload.get("draft_decode_calls")),
        elapsed_ms=_float_or_none(payload.get("elapsed_ms")),
        tokens_per_second=_float_or_none(payload.get("tokens_per_second")),
    )


def build_mtp_completion_helper(
    *,
    llama_root: Path | None,
    build_dir: Path | None = None,
    build_bin: Path | None = None,
    runner=subprocess.run,
) -> Path:
    packaged = packaged_shim_path("orbit-mtp-completion")
    if packaged is not None:
        return packaged
    llama_root = require_legacy_llama_root(llama_root, artifact_name="orbit-mtp-completion")
    build_root = build_dir or (Path.home() / ".orbit" / "native-build")
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_mtp_completion.cpp"
    output = build_root / "orbit-mtp-completion"
    return compile_cpp_helper(
        artifact_label="mtp completion helper",
        source=source,
        output=output,
        llama_root=llama_root,
        build_bin=build_bin,
        runner=runner,
    )


def _int_or_default(value) -> int:
    return value if isinstance(value, int) else 0


def _float_or_none(value) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None

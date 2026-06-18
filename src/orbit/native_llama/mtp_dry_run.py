from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from .native_artifacts import packaged_shim_path, require_legacy_llama_root
from .paths import NativeLlamaPaths


@dataclass(frozen=True)
class MtpDryRunResult:
    enabled: bool
    success: bool
    error: str | None
    draft_tokens: int = 0
    rss_before_kb: int | None = None
    rss_after_kb: int | None = None
    rss_peak_kb: int | None = None
    target_load_s: float | None = None
    draft_load_s: float | None = None
    target_ctx_s: float | None = None
    draft_ctx_s: float | None = None
    speculative_init_s: float | None = None
    prompt_decode_s: float | None = None
    draft_s: float | None = None


def run_mtp_dry_run(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> MtpDryRunResult:
    if not paths.mtp_available or paths.draft_mtp_model is None:
        return MtpDryRunResult(enabled=True, success=False, error=paths.fallback_reason or "draft-mtp-unavailable")

    helper = build_mtp_dry_run_helper(llama_root=llama_root, build_dir=build_dir, runner=runner)
    completed = runner(
        [str(helper), str(paths.model), str(paths.draft_mtp_model)],
        capture_output=True,
        text=True,
        env={"LD_LIBRARY_PATH": str(paths.build_bin)},
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return MtpDryRunResult(enabled=True, success=False, error=f"mtp dry run failed with exit code {completed.returncode}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return MtpDryRunResult(enabled=True, success=False, error=f"invalid mtp dry run output: {exc}")
    if completed.returncode != 0 or not payload.get("ok"):
        return MtpDryRunResult(
            enabled=True,
            success=False,
            error=str(payload.get("error") or f"mtp dry run failed with exit code {completed.returncode}"),
            draft_tokens=_int_or_default(payload.get("draft_tokens")),
            rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
            rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
            rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
            target_load_s=_float_or_none(payload.get("target_load_s")),
            draft_load_s=_float_or_none(payload.get("draft_load_s")),
            target_ctx_s=_float_or_none(payload.get("target_ctx_s")),
            draft_ctx_s=_float_or_none(payload.get("draft_ctx_s")),
            speculative_init_s=_float_or_none(payload.get("speculative_init_s")),
            prompt_decode_s=_float_or_none(payload.get("prompt_decode_s")),
            draft_s=_float_or_none(payload.get("draft_s")),
        )
    return MtpDryRunResult(
        enabled=True,
        success=True,
        error=None,
        draft_tokens=_int_or_default(payload.get("draft_tokens")),
        rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
        rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
        rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
        target_load_s=_float_or_none(payload.get("target_load_s")),
        draft_load_s=_float_or_none(payload.get("draft_load_s")),
        target_ctx_s=_float_or_none(payload.get("target_ctx_s")),
        draft_ctx_s=_float_or_none(payload.get("draft_ctx_s")),
        speculative_init_s=_float_or_none(payload.get("speculative_init_s")),
        prompt_decode_s=_float_or_none(payload.get("prompt_decode_s")),
        draft_s=_float_or_none(payload.get("draft_s")),
    )


def build_mtp_dry_run_helper(
    *,
    llama_root: Path | None,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> Path:
    packaged = packaged_shim_path("orbit-mtp-dry-run")
    if packaged is not None:
        return packaged
    llama_root = require_legacy_llama_root(llama_root, artifact_name="orbit-mtp-dry-run")
    build_root = build_dir or (Path.home() / ".orbit" / "native-build")
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_mtp_dry_run.cpp"
    output = build_root / "orbit-mtp-dry-run"
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output

    bin_dir = llama_root / "build/bin"
    command = [
        "g++",
        "-std=c++17",
        str(source),
        f"-I{llama_root / 'include'}",
        f"-I{llama_root / 'common'}",
        f"-I{llama_root}",
        f"-I{llama_root / 'ggml/include'}",
        f"-I{llama_root / 'src'}",
        f"-Wl,-rpath,{bin_dir}",
        str(bin_dir / "libllama-common.so"),
        str(bin_dir / "libllama.so"),
        str(bin_dir / "libggml.so"),
        str(bin_dir / "libggml-base.so"),
        str(bin_dir / "libggml-cpu.so"),
        "-o",
        str(output),
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"failed to build mtp dry run helper: {detail or completed.returncode}")
    return output


def _int_or_none(value) -> int | None:
    return value if isinstance(value, int) else None


def _int_or_default(value) -> int:
    return value if isinstance(value, int) else 0


def _float_or_none(value) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None

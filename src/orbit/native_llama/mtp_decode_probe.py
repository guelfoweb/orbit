from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from .native_artifacts import packaged_shim_path, require_legacy_llama_root
from .paths import NativeLlamaPaths


@dataclass(frozen=True)
class MtpDecodePromptResult:
    name: str
    output_tokens: int
    draft_tokens_total: int
    accepted_tokens_total: int
    rejected_tokens_total: int
    acceptance_ratio: float | None
    target_decode_calls: int
    draft_decode_calls: int
    elapsed_ms: float | None
    tokens_per_second: float | None
    error: str | None = None


@dataclass(frozen=True)
class MtpDecodeProbeResult:
    enabled: bool
    success: bool
    error: str | None
    prompts: tuple[MtpDecodePromptResult, ...] = ()
    rss_before_kb: int | None = None
    rss_after_kb: int | None = None
    rss_peak_kb: int | None = None


def run_mtp_decode_probe(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> MtpDecodeProbeResult:
    if not paths.mtp_available or paths.draft_mtp_model is None:
        return MtpDecodeProbeResult(enabled=True, success=False, error=paths.fallback_reason or "draft-mtp-unavailable")

    helper = build_mtp_decode_probe_helper(llama_root=llama_root, build_dir=build_dir, runner=runner)
    completed = runner(
        [str(helper), str(paths.model), str(paths.draft_mtp_model)],
        capture_output=True,
        text=True,
        env={"LD_LIBRARY_PATH": str(paths.build_bin)},
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return MtpDecodeProbeResult(enabled=True, success=False, error=f"mtp decode probe failed with exit code {completed.returncode}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return MtpDecodeProbeResult(enabled=True, success=False, error=f"invalid mtp decode probe output: {exc}")
    prompts = tuple(_prompt_result(item) for item in payload.get("prompts", []) if isinstance(item, dict))
    if completed.returncode != 0 or not payload.get("ok"):
        return MtpDecodeProbeResult(
            enabled=True,
            success=False,
            error=str(payload.get("error") or f"mtp decode probe failed with exit code {completed.returncode}"),
            prompts=prompts,
            rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
            rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
            rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
        )
    return MtpDecodeProbeResult(
        enabled=True,
        success=True,
        error=None,
        prompts=prompts,
        rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
        rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
        rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
    )


def build_mtp_decode_probe_helper(
    *,
    llama_root: Path | None,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> Path:
    packaged = packaged_shim_path("orbit-mtp-decode-probe")
    if packaged is not None:
        return packaged
    llama_root = require_legacy_llama_root(llama_root, artifact_name="orbit-mtp-decode-probe")
    build_root = build_dir or (Path.home() / ".orbit" / "native-build")
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_mtp_decode_probe.cpp"
    output = build_root / "orbit-mtp-decode-probe"
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
        raise RuntimeError(f"failed to build mtp decode probe helper: {detail or completed.returncode}")
    return output


def _prompt_result(item: dict) -> MtpDecodePromptResult:
    return MtpDecodePromptResult(
        name=str(item.get("name") or "prompt"),
        output_tokens=_int_or_default(item.get("output_tokens")),
        draft_tokens_total=_int_or_default(item.get("draft_tokens_total")),
        accepted_tokens_total=_int_or_default(item.get("accepted_tokens_total")),
        rejected_tokens_total=_int_or_default(item.get("rejected_tokens_total")),
        acceptance_ratio=_float_or_none(item.get("acceptance_ratio")),
        target_decode_calls=_int_or_default(item.get("target_decode_calls")),
        draft_decode_calls=_int_or_default(item.get("draft_decode_calls")),
        elapsed_ms=_float_or_none(item.get("elapsed_ms")),
        tokens_per_second=_float_or_none(item.get("tokens_per_second")),
        error=str(item.get("error")) if item.get("error") is not None else None,
    )


def _int_or_none(value) -> int | None:
    return value if isinstance(value, int) else None


def _int_or_default(value) -> int:
    return value if isinstance(value, int) else 0


def _float_or_none(value) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None

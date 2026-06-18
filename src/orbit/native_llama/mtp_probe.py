from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

from .paths import NativeLlamaPaths


@dataclass(frozen=True)
class MtpProbeResult:
    enabled: bool
    initialized: bool
    error: str | None
    rss_before_kb: int | None = None
    rss_after_kb: int | None = None
    rss_peak_kb: int | None = None


def run_mtp_probe(
    *,
    llama_root: Path,
    paths: NativeLlamaPaths,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> MtpProbeResult:
    if not paths.mtp_available or paths.draft_mtp_model is None:
        return MtpProbeResult(enabled=True, initialized=False, error=paths.fallback_reason or "draft-mtp-unavailable")

    helper = build_mtp_probe_helper(llama_root=llama_root, build_dir=build_dir, runner=runner)
    completed = runner(
        [str(helper), str(paths.model), str(paths.draft_mtp_model)],
        capture_output=True,
        text=True,
        env={"LD_LIBRARY_PATH": str(paths.build_bin)},
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return MtpProbeResult(enabled=True, initialized=False, error=f"mtp probe failed with exit code {completed.returncode}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return MtpProbeResult(enabled=True, initialized=False, error=f"invalid mtp probe output: {exc}")
    if completed.returncode != 0 or not payload.get("ok"):
        return MtpProbeResult(
            enabled=True,
            initialized=False,
            error=str(payload.get("error") or f"mtp probe failed with exit code {completed.returncode}"),
            rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
            rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
            rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
        )
    return MtpProbeResult(
        enabled=True,
        initialized=True,
        error=None,
        rss_before_kb=_int_or_none(payload.get("rss_before_kb")),
        rss_after_kb=_int_or_none(payload.get("rss_after_kb")),
        rss_peak_kb=_int_or_none(payload.get("rss_peak_kb")),
    )


def build_mtp_probe_helper(
    *,
    llama_root: Path,
    build_dir: Path | None = None,
    runner=subprocess.run,
) -> Path:
    build_root = build_dir or (Path.home() / ".orbit" / "native-build")
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "vendor" / "shim" / "orbit_mtp_probe.cpp"
    output = build_root / "orbit-mtp-probe"
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
        raise RuntimeError(f"failed to build mtp probe helper: {detail or completed.returncode}")
    return output


def _int_or_none(value) -> int | None:
    return value if isinstance(value, int) else None

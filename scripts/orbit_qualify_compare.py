#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orbit.backend.llama_server import LlamaServerBackend  # noqa: E402
from orbit.qualification.fixtures import load_fixture_set  # noqa: E402
from orbit.qualification.reporting import format_comparison_summary, write_comparison  # noqa: E402
from orbit.qualification.runner import (  # noqa: E402
    QualificationRunner,
    RuntimeFixtureExecutor,
    build_optimization_comparison,
)
from orbit.qualification.schema import ComparisonExecution  # noqa: E402
from scripts.orbit_qualify import build_provenance  # noqa: E402


CONTROLLED_ENV = frozenset({
    "ORBIT_KV_PREFIX_PREWARM",
    "ORBIT_QWEN_ROUTE_PREFIX_REUSE",
    "ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE",
})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two process-isolated Orbit qualification modes.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fixtures", default=str(ROOT / "qualification/fixtures/optimizations-v1.json"))
    parser.add_argument("--fixture", action="append", dest="fixture_names")
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    baseline_env = _parse_overrides(args.baseline_env)
    candidate_env = _parse_overrides(args.candidate_env)
    fixture_set = load_fixture_set(args.fixtures)
    selected = tuple(args.fixture_names) if args.fixture_names else None
    with tempfile.TemporaryDirectory(prefix="orbit-qualification-comparison-") as directory:
        root = Path(directory)
        fixture_root = root / "fixtures"
        baseline = _run_side(
            "baseline", baseline_env, args, fixture_set, selected, fixture_root, root / "baseline.log"
        )
        _reset_fixture_root(fixture_root, root)
        candidate = _run_side(
            "candidate", candidate_env, args, fixture_set, selected, fixture_root, root / "candidate.log"
        )
    comparison = build_optimization_comparison(fixture_set, baseline, candidate)
    write_comparison(comparison, args.output)
    print(format_comparison_summary(comparison))
    return 0 if comparison.performance_comparison_valid else 1


def _run_side(
    label,
    overrides,
    args,
    fixture_set,
    selected,
    fixture_root: Path,
    log_path: Path,
) -> ComparisonExecution:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    for name in CONTROLLED_ENV:
        env.pop(name, None)
    env.update(overrides)
    env["PYTHONPATH"] = str(SRC)
    command = _server_command(args, port)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    try:
        backend = LlamaServerBackend(base_url=base_url, timeout=args.request_timeout)
        _wait_ready(backend, process, args.startup_timeout, log_path)
        startup_wall = time.perf_counter() - started
        props = backend.backend_props()
        profile = props.get("model_compatibility")
        if not isinstance(profile, dict):
            raise RuntimeError("server did not expose model compatibility metadata")
        fixture_root.mkdir(parents=True, exist_ok=False)
        run = QualificationRunner(
            fixture_set=fixture_set,
            profile=profile,
            provenance=build_provenance(
                args.profile, fixture_set.content_hash, profile, props, process.pid
            ),
            executor=RuntimeFixtureExecutor(backend, process_pid=process.pid),
            workdir=fixture_root,
        ).run(selected)
        return ComparisonExecution(
            label=label,
            server_pid=process.pid,
            startup_wall_seconds=startup_wall,
            configuration=_configuration(args, overrides),
            result=run,
        )
    finally:
        _stop(process)


def _server_command(args, port: int) -> list[str]:
    return [
        sys.executable, "-m", "orbit.terminal.cli", "server",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", str(args.model.resolve()),
        "--ctx", str(args.ctx), "--threads", str(args.threads),
        "--threads-batch", str(args.threads_batch), "--batch", str(args.batch),
        "--ubatch", str(args.ubatch), "--think", "off",
    ]


def _wait_ready(backend, process, timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            raise RuntimeError(f"server exited during startup: {detail}")
        if backend.health():
            return
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout:.0f}s")


def _stop(process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _parse_overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or name not in CONTROLLED_ENV or name in result:
            raise ValueError(f"invalid or duplicate controlled environment override: {name}")
        result[name] = setting
    return result


def _configuration(args, overrides: dict[str, str]) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "model_filename": args.model.name,
        "ctx": args.ctx,
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "thinking": "off",
        "cpu_affinity": affinity,
        "startup_timeout_seconds": args.startup_timeout,
        "request_timeout_seconds": args.request_timeout,
        "environment": dict(sorted(overrides.items())),
        "startup_wall_scope": "process launch through healthy server readiness; model load and prewarm included",
    }


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _reset_fixture_root(path: Path, root: Path) -> None:
    if path.parent != root:
        raise RuntimeError("comparison fixture root escaped its temporary directory")
    if path.exists():
        shutil.rmtree(path)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerBackend  # noqa: E402
from orbit.qualification.fixtures import load_fixture_set  # noqa: E402
from orbit.qualification.reporting import format_summary, write_result  # noqa: E402
from orbit.qualification.runner import QualificationRunner, RuntimeFixtureExecutor  # noqa: E402
from orbit.qualification.schema import RunProvenance, Status  # noqa: E402
from orbit.terminal.runtime_status import collect_host_info  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify an independently managed Orbit server.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--fixtures", default=str(ROOT / "qualification/fixtures/core-v1.json"))
    parser.add_argument("--fixture", action="append", dest="fixture_names")
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    backend = LlamaServerBackend(base_url=args.base_url, timeout=args.request_timeout)
    if not backend.health():
        raise RuntimeError("Orbit server is not healthy")
    props = backend.backend_props()
    profile = props.get("model_compatibility")
    if not isinstance(profile, dict):
        raise RuntimeError("server did not expose model compatibility metadata")
    fixture_set = load_fixture_set(args.fixtures)
    with tempfile.TemporaryDirectory(prefix="orbit-qualification-") as workdir:
        run = QualificationRunner(
            fixture_set=fixture_set,
            profile=profile,
            provenance=_provenance(args.profile, fixture_set.content_hash, profile, props, args.server_pid),
            executor=RuntimeFixtureExecutor(backend, process_pid=args.server_pid),
            workdir=Path(workdir),
        ).run(tuple(args.fixture_names) if args.fixture_names else None)
    write_result(run, args.output)
    print(format_summary(run))
    return 0 if run.overall_status is Status.PASS else 1


def _provenance(
    expected_profile: str,
    fixture_hash: str,
    profile: dict[str, Any],
    props: dict[str, Any],
    server_pid: int | None,
) -> RunProvenance:
    capabilities = props.get("native_backend_capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    backend = capabilities.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    runtime_keys = ("ctx_size", "threads", "threads_batch", "batch_size", "ubatch_size", "parallel_slots")
    return RunProvenance(
        qualification_schema_version=1, fixture_set_hash=fixture_hash,
        git_revision=_git_revision(), profile_identity=expected_profile,
        model_identity=_string(profile.get("model_name")) or _string(props.get("model_id")),
        template_identity=_string(profile.get("template_source")), template_hash=_string(profile.get("template_hash")),
        backend_identity=_string(props.get("backend")),
        backend_revision=_string(backend.get("commit")) or _string(backend.get("build_number")),
        runtime_configuration={key: props.get(key) for key in runtime_keys}, hardware=asdict(collect_host_info()),
        measurement_scope={
            "call_wall_seconds": "client elapsed model call; transport included; tool execution excluded",
            "aggregate_wall_seconds": "sum of fixture execution; server startup and model load excluded",
            "prefill_tokens_per_second": "sum evaluated / sum per-call evaluated seconds; complete calls only",
            "generation_tokens_per_second": "sum output / sum per-call decode seconds; complete calls only",
            "peak_rss_bytes": (
                f"Linux VmHWM for server PID {server_pid}; process lifetime including model load"
                if server_pid is not None else "unavailable without --server-pid"
            ),
        },
    )


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _string(value: Any) -> str | None:
    return str(value) if isinstance(value, (str, int)) else None


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.llama_server import LlamaServerBackend
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class Scenario:
    name: str
    max_tokens: int
    prompt_builder: Callable[[str], str]


@dataclass
class ServerHandle:
    mode: str
    port: int
    process: subprocess.Popen[str]
    log_path: Path
    startup_props: dict[str, object]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = select_scenarios(args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = output_dir / f"mtp_throughput_{stamp}.jsonl"
    markdown_path = output_dir / f"mtp_throughput_{stamp}.md"
    jsonl_path.touch()

    log_root_cm = (
        tempfile.TemporaryDirectory(prefix="orbit-mtp-throughput-")
        if not args.keep_server_logs
        else None
    )
    log_root = Path(log_root_cm.name) if log_root_cm is not None else output_dir

    rows: list[dict[str, object]] = []
    servers: list[ServerHandle] = []
    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
            servers.append(start_server(mode="mtp_off", orbit_cmd=args.orbit_cmd, pythonpath=args.pythonpath, log_root=log_root))
            servers.append(start_server(mode="mtp_on", orbit_cmd=args.orbit_cmd, pythonpath=args.pythonpath, log_root=log_root))
            for server in servers:
                for scenario in selected:
                    stop_scenario = False
                    for repetition in range(1, args.warmups + 1):
                        row = run_case(server, scenario, phase="warmup", repetition=repetition, timeout=args.timeout)
                        rows.append(row)
                        append_jsonl_row(jsonl_handle, row)
                        if row["exit_kind"] != "ok":
                            stop_scenario = True
                            break
                    if stop_scenario:
                        continue
                    for repetition in range(1, args.runs + 1):
                        row = run_case(server, scenario, phase="measured", repetition=repetition, timeout=args.timeout)
                        rows.append(row)
                        append_jsonl_row(jsonl_handle, row)
                        if row["exit_kind"] != "ok":
                            break
    except KeyboardInterrupt:
        print("interrupted: stopping benchmark and keeping partial JSONL", file=sys.stderr)
    finally:
        for server in reversed(servers):
            stop_server(server)
        if log_root_cm is not None:
            log_root_cm.cleanup()

    write_markdown(markdown_path, rows)
    print(f"jsonl={jsonl_path}")
    print(f"markdown={markdown_path}")
    return 0 if rows else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Orbit MTP throughput on non-tool chat scenarios.")
    parser.add_argument("--output-dir", default="benchmarks/mtp-throughput")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--scenario", action="append", default=None, help="Scenario name, repeatable, or 'all'.")
    parser.add_argument("--keep-server-logs", action="store_true")
    parser.add_argument("--orbit-cmd", default=".venv/bin/orbit")
    parser.add_argument("--pythonpath", default="src")
    return parser


def scenarios() -> dict[str, Scenario]:
    return {
        "short_chat": Scenario(
            name="short_chat",
            max_tokens=48,
            prompt_builder=lambda marker: (
                "Write exactly twelve words in one sentence describing a triangle in plain language. "
                "Do not use a list. "
                f"Include marker {marker} unchanged."
            ),
        ),
        "medium_chat": Scenario(
            name="medium_chat",
            max_tokens=128,
            prompt_builder=lambda marker: (
                "Write exactly one paragraph of 80 to 100 words explaining why careful measurement should come "
                "before optimization. Do not use bullet points or numbered lists. "
                f"Include marker {marker} unchanged."
            ),
        ),
        "longer_chat": Scenario(
            name="longer_chat",
            max_tokens=224,
            prompt_builder=lambda marker: (
                "Write exactly two paragraphs totaling 160 to 200 words about a fictional archive room, covering "
                "layout, lighting, noise, and workflow. Do not use bullet points or numbered lists. "
                f"Include marker {marker} unchanged."
            ),
        ),
    }


def select_scenarios(selected: list[str] | None) -> list[Scenario]:
    registry = scenarios()
    chosen = selected or ["all"]
    names = list(registry) if "all" in chosen else chosen
    unknown = sorted(name for name in names if name not in registry)
    if unknown:
        raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
    return [registry[name] for name in names]


def start_server(*, mode: str, orbit_cmd: str, pythonpath: str, log_root: Path) -> ServerHandle:
    port = free_port()
    log_path = log_root / f"orbit_server_{mode}_{port}.log"
    command = shlex.split(orbit_cmd) + ["server", "--port", str(port)]
    if mode == "mtp_on":
        command.append("--mtp")
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        props = wait_for_server_ready(base_url, process, timeout=180.0)
        if mode == "mtp_on":
            assert_mtp_startup_healthy(props)
        return ServerHandle(mode=mode, port=port, process=process, log_path=log_path, startup_props=props)
    except Exception:
        stop_process(process)
        raise


def stop_server(server: ServerHandle) -> None:
    stop_process(server.process)


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_server_ready(base_url: str, process: subprocess.Popen[str], *, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            props = get_json(f"{base_url}/props", timeout=2.0)
        except Exception as exc:  # pragma: no cover - real process polling
            last_error = str(exc)
            time.sleep(0.5)
            continue
        if isinstance(props, dict) and props.get("backend") == "orbit-native":
            return props
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready within {timeout:.0f}s ({last_error or 'no props'})")


def assert_mtp_startup_healthy(props: dict[str, object]) -> None:
    if props.get("multimodal_available") is not True:
        raise RuntimeError("mtp_on server missing multimodal support")
    if props.get("mtp_enabled") is not True:
        raise RuntimeError("mtp_on server not MTP enabled at startup")
    if props.get("mtp_initialized") is not True:
        raise RuntimeError("mtp_on server not MTP initialized at startup")
    if props.get("mtp_failure_reason") not in {None, ""}:
        raise RuntimeError(f"mtp_on server startup failure: {props.get('mtp_failure_reason')}")


def run_case(server: ServerHandle, scenario: Scenario, *, phase: str, repetition: int, timeout: float) -> dict[str, object]:
    prompt = scenario.prompt_builder(f"{phase}-{repetition}")
    started = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[tuple[str, object]] = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=run_chat_completion,
        args=(server.base_url, server.mode, scenario.name, prompt, scenario.max_tokens, timeout, result_queue),
        daemon=True,
    )
    process.start()
    deadline = time.monotonic() + max(0.0, timeout)
    status: str | None = None
    payload: object | None = None
    try:
        while time.monotonic() < deadline:
            if not process.is_alive():
                break
            try:
                status, payload = result_queue.get(timeout=0.2)
                break
            except Exception:
                continue
        else:
            status = None
    except KeyboardInterrupt:
        return finalize_interrupted_run(
            server=server,
            scenario=scenario,
            phase=phase,
            repetition=repetition,
            prompt=prompt,
            started=started,
            process=process,
            timeout=timeout,
        )

    if status is None:
        return finalize_timed_out_run(
            server=server,
            scenario=scenario,
            phase=phase,
            repetition=repetition,
            prompt=prompt,
            started=started,
            process=process,
            timeout=timeout,
        )

    if process.is_alive():
        process.join(timeout=0.5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)

    settled = settled_backend_props(server.base_url, min(5.0, max(1.0, timeout)))
    if status == "error":
        return row_from_failure(
            server=server,
            scenario=scenario,
            phase=phase,
            repetition=repetition,
            prompt=prompt,
            exit_kind="error",
            finish_reason="error",
            wall_ms=elapsed_ms(started),
            props=settled,
            raw_error=f"{type(payload).__name__}: {payload}",
            notes="exception",
        )

    result = payload
    evaluated_tokens = (
        result.prompt_tokens - result.cached_tokens
        if result.prompt_tokens is not None and result.cached_tokens is not None
        else None
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_mode": server.mode,
        "scenario": scenario.name,
        "phase": phase,
        "repetition": repetition,
        "prompt": prompt,
        "exit_kind": "ok",
        "finish_reason": result.finish_reason,
        "wall_ms": elapsed_ms(started),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cached_tokens": result.cached_tokens,
        "evaluated_tokens": evaluated_tokens,
        "prompt_tokens_per_second": result.prompt_tokens_per_second,
        "generation_tokens_per_second": result.generation_tokens_per_second,
        "backend_mode": settled.get("backend_mode"),
        "mtp_enabled": settled.get("mtp_enabled"),
        "mtp_initialized": settled.get("mtp_initialized"),
        "mtp_failure_reason": settled.get("mtp_failure_reason"),
        "mtp_last_completion": settled.get("mtp_last_completion"),
        "multimodal_available": settled.get("multimodal_available"),
        "in_flight": settled.get("in_flight"),
        "backend_still_in_flight": bool(settled.get("in_flight") is True),
        "raw_error": None,
    }


def row_from_failure(
    *,
    server: ServerHandle,
    scenario: Scenario,
    phase: str,
    repetition: int,
    prompt: str,
    exit_kind: str,
    finish_reason: str,
    wall_ms: float,
    props: dict[str, object],
    raw_error: str | None,
    notes: str,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_mode": server.mode,
        "scenario": scenario.name,
        "phase": phase,
        "repetition": repetition,
        "prompt": prompt,
        "exit_kind": exit_kind,
        "finish_reason": finish_reason,
        "wall_ms": wall_ms,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "evaluated_tokens": None,
        "prompt_tokens_per_second": None,
        "generation_tokens_per_second": None,
        "backend_mode": props.get("backend_mode"),
        "mtp_enabled": props.get("mtp_enabled"),
        "mtp_initialized": props.get("mtp_initialized"),
        "mtp_failure_reason": props.get("mtp_failure_reason"),
        "mtp_last_completion": props.get("mtp_last_completion"),
        "multimodal_available": props.get("multimodal_available"),
        "in_flight": props.get("in_flight"),
        "backend_still_in_flight": bool(props.get("in_flight") is True),
        "raw_error": raw_error or notes,
    }


def append_jsonl_row(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    measured = [row for row in rows if row["phase"] == "measured"]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in measured:
        groups.setdefault((str(row["server_mode"]), str(row["scenario"])), []).append(row)
    lines = [
        "# MTP Throughput Benchmark",
        "",
        "| Server | Scenario | OK | Timeout/Error | Median wall ms | Median gen tok/s | Median pf tok/s | Median completion tk | Median cached tk | Health |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(groups):
        server_mode, scenario = key
        entries = groups[key]
        ok_rows = [row for row in entries if row["exit_kind"] == "ok"]
        bad_rows = [row for row in entries if row["exit_kind"] != "ok"]
        lines.append(
            "| {server} | {scenario} | {ok} | {bad} | {wall} | {gen} | {pf} | {completion} | {cached} | {health} |".format(
                server=server_mode,
                scenario=scenario,
                ok=len(ok_rows),
                bad=len(bad_rows),
                wall=_median(entries, "wall_ms"),
                gen=_median(ok_rows, "generation_tokens_per_second"),
                pf=_median(ok_rows, "prompt_tokens_per_second"),
                completion=_median(ok_rows, "completion_tokens"),
                cached=_median(ok_rows, "cached_tokens"),
                health=health_note(entries),
            )
        )
    lines.extend(
        [
            "",
            "## MTP diagnostics measured-only",
            "",
            "| Server | Scenario | Output tk | Draft tk | Accepted tk | Rejected tk | Accept ratio | Target decodes | Draft decodes | Full accept | Partial accept | Partial no replay | Rollback tk | Checkpoints | Restores | Reused draft | Reused accepted | Reused rejected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    diag_fields = [
        ("output_tokens", "Output tk"),
        ("draft_tokens_total", "Draft tk"),
        ("accepted_tokens_total", "Accepted tk"),
        ("rejected_tokens_total", "Rejected tk"),
        ("acceptance_ratio", "Accept ratio"),
        ("target_decode_calls", "Target decodes"),
        ("draft_decode_calls", "Draft decodes"),
        ("full_accept_steps", "Full accept"),
        ("partial_accept_steps", "Partial accept"),
        ("partial_no_replay_steps", "Partial no replay"),
        ("rollback_tokens_total", "Rollback tk"),
        ("checkpoint_count", "Checkpoints"),
        ("restore_count", "Restores"),
        ("reused_draft_tokens_total", "Reused draft"),
        ("reused_accepted_tokens_total", "Reused accepted"),
        ("reused_rejected_tokens_total", "Reused rejected"),
    ]
    for key in sorted(groups):
        server_mode, scenario = key
        entries = groups[key]
        ok_rows = [row for row in entries if row["exit_kind"] == "ok"]
        values = [_mtp_metric_median(ok_rows, field) for field, _label in diag_fields]
        lines.append(
            "| {server} | {scenario} | {values} |".format(
                server=server_mode,
                scenario=scenario,
                values=" | ".join(values),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def health_note(rows: list[dict[str, object]]) -> str:
    if rows and all(row.get("server_mode") == "mtp_off" for row in rows):
        return "off-by-config"
    unhealthy = [
        row for row in rows
        if row.get("mtp_enabled") is not True
        or row.get("mtp_initialized") is not True
        or row.get("mtp_failure_reason") not in {None, ""}
    ]
    if not unhealthy:
        return "healthy"
    reasons = sorted({str(row.get("mtp_failure_reason") or "state-change") for row in unhealthy})
    return ",".join(reasons)


def _median(rows: list[dict[str, object]], key: str) -> str:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return "-"
    value = statistics.median(values)
    return f"{value:.1f}"


def _mtp_metric_median(rows: list[dict[str, object]], key: str) -> str:
    values: list[float] = []
    for row in rows:
        mtp = row.get("mtp_last_completion")
        if not isinstance(mtp, dict):
            continue
        value = mtp.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return "-"
    value = statistics.median(values)
    return f"{value:.3f}" if not float(value).is_integer() else f"{value:.1f}"


def run_chat_completion(
    base_url: str,
    server_mode: str,
    scenario_name: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    result_queue: multiprocessing.Queue[tuple[str, object]],
) -> None:
    backend = LlamaServerBackend(base_url=base_url, timeout=timeout)
    runtime = ChatRuntime(
        backend=backend,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        diagnostic_session_id=f"bench-{server_mode}-{scenario_name}",
    )
    try:
        result = runtime.ask_chat(prompt, temperature=0.0, max_tokens=max_tokens)
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    result_queue.put(("ok", result))


def finalize_timed_out_run(
    *,
    server: ServerHandle,
    scenario: Scenario,
    phase: str,
    repetition: int,
    prompt: str,
    started: float,
    process: multiprocessing.Process,
    timeout: float,
) -> dict[str, object]:
    cancel_requested = request_cancel(server.base_url, timeout=min(5.0, max(1.0, timeout)))
    props_after = wait_for_backend_idle(server.base_url, timeout=min(5.0, max(1.0, timeout)))
    terminate_process(process)
    notes = ["timeout"]
    if cancel_requested:
        notes.append("cancel_requested")
    if props_after.get("in_flight") is False:
        notes.append("cleanup_ok")
    elif props_after:
        notes.append("cleanup_pending")
    settled = settled_backend_props(server.base_url, min(5.0, max(1.0, timeout)))
    return row_from_failure(
        server=server,
        scenario=scenario,
        phase=phase,
        repetition=repetition,
        prompt=prompt,
        exit_kind="timeout",
        finish_reason="timeout",
        wall_ms=elapsed_ms(started),
        props=settled,
        raw_error="wall-clock timeout exceeded",
        notes=",".join(notes),
    )


def finalize_interrupted_run(
    *,
    server: ServerHandle,
    scenario: Scenario,
    phase: str,
    repetition: int,
    prompt: str,
    started: float,
    process: multiprocessing.Process,
    timeout: float,
) -> dict[str, object]:
    cancel_requested = request_cancel(server.base_url, timeout=min(5.0, max(1.0, timeout)))
    props_after = wait_for_backend_idle(server.base_url, timeout=min(5.0, max(1.0, timeout)))
    terminate_process(process)
    notes = ["interrupted"]
    if cancel_requested:
        notes.append("cancel_requested")
    if props_after.get("in_flight") is False:
        notes.append("cleanup_ok")
    elif props_after:
        notes.append("cleanup_pending")
    settled = settled_backend_props(server.base_url, min(5.0, max(1.0, timeout)))
    return row_from_failure(
        server=server,
        scenario=scenario,
        phase=phase,
        repetition=repetition,
        prompt=prompt,
        exit_kind="error",
        finish_reason="interrupted",
        wall_ms=elapsed_ms(started),
        props=settled,
        raw_error="KeyboardInterrupt",
        notes=",".join(notes),
    )


def terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.2)
        return
    process.terminate()
    process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


def free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def get_json(url: str, *, timeout: float, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    method = "GET"
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        loaded = json.loads(response.read().decode("utf-8", errors="replace"))
    return loaded if isinstance(loaded, dict) else {}


def request_cancel(base_url: str, *, timeout: float) -> bool:
    try:
        payload = get_json(f"{base_url.rstrip('/')}/cancel", timeout=timeout, payload={"session_id": "default"})
    except Exception:
        return False
    return payload.get("status") == "cancel_requested"


def wait_for_backend_idle(base_url: str, *, timeout: float, poll_interval: float = 0.2) -> dict[str, object]:
    deadline = time.monotonic() + max(0.0, timeout)
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            latest = get_json(f"{base_url.rstrip('/')}/props", timeout=min(2.0, max(0.5, timeout)))
        except Exception:
            time.sleep(poll_interval)
            continue
        if latest.get("in_flight") is False:
            return latest
        time.sleep(poll_interval)
    return latest


def settled_backend_props(base_url: str, timeout: float, *, settle_seconds: float = 3.0, poll_interval: float = 0.2) -> dict[str, object]:
    try:
        props = get_json(f"{base_url.rstrip('/')}/props", timeout=timeout)
    except Exception:
        return {}
    deadline = time.monotonic() + max(0.0, settle_seconds)
    last_serialized = json.dumps(props, sort_keys=True, ensure_ascii=False)
    while time.monotonic() < deadline:
        if props.get("in_flight") is False:
            failure_reason = first_nonempty_str(props.get("mtp_failure_reason"), props.get("mtp_fallback_reason"))
            if failure_reason not in {"cancelled", "timeout"}:
                return props
        time.sleep(poll_interval)
        try:
            refreshed = get_json(f"{base_url.rstrip('/')}/props", timeout=min(2.0, max(0.5, timeout)))
        except Exception:
            return props
        serialized = json.dumps(refreshed, sort_keys=True, ensure_ascii=False)
        props = refreshed
        if serialized == last_serialized and props.get("in_flight") is False:
            return props
        last_serialized = serialized
    return props


def first_nonempty_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FOOTER_RE = re.compile(
    r"model: (?P<model>.*?) \| ctx: (?P<ctx>\d+)/(?:\d+) \((?P<ctx_pct>\d+)%\) \| "
    r"tks: (?P<prompt_tokens>\d+)->(?P<generated_tokens>\d+), cached (?P<cached_tokens>\d+) \| "
    r"cache: (?P<cache_pct>\d+)% \| pf (?P<prefill_tps>[0-9.]+)/s \| gen (?P<generation_tps>[0-9.]+)/s \| "
    r"stop: (?P<finish_reason>\S+) \| time: (?P<footer_time>[^\n\r]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass(frozen=True)
class Scenario:
    name: str
    args: tuple[str, ...]
    stdin: str | None = None
    timeout: int = 240


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Orbit KV diagnostic benchmark scenarios.")
    parser.add_argument("--orbit-bin", default=sys.executable, help="Python executable or orbit command to run.")
    parser.add_argument("--module", default="orbit.terminal.cli", help="Python module when --orbit-bin is Python.")
    parser.add_argument("--workdir", default="workdir")
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--max-tokens", default="120")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--include-slow", action="store_true", help="Include slow network-dependent scenarios.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    diag_path = output_dir / f"kv_diag_{stamp}.jsonl"
    summary_path = output_dir / f"kv_diag_{stamp}.md"

    scenarios = _scenarios(max_tokens=args.max_tokens, timeout=args.timeout, include_slow=args.include_slow)
    env = os.environ.copy()
    env["ORBIT_KV_DIAG"] = "1"
    env["ORBIT_KV_DIAG_FILE"] = str(diag_path)
    env.setdefault("PYTHONPATH", "src")

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        row = _run_scenario(
            scenario,
            orbit_bin=args.orbit_bin,
            module=args.module,
            workdir=args.workdir,
            env=env,
            root=root,
            diag_path=diag_path,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    _write_summary(summary_path, rows, diag_path)
    print(f"diag_jsonl={diag_path}")
    print(f"summary_md={summary_path}")
    return 0


def _scenarios(*, max_tokens: str, timeout: int, include_slow: bool = False) -> list[Scenario]:
    base = ("--no-render-markdown", "--max-tokens", max_tokens)
    scenarios = [
        Scenario("tools_off_repeat", (*base, "--tools", "off"), "hi\nhi\n", timeout),
        Scenario("tools_on_repeat_no_tool_needed", (*base, "--tools", "on"), "hi\nhi\n", timeout),
        Scenario("tools_on_same_session_repeat", (*base, "--tools", "on"), "hi, tell me something about yourself\nhi, tell me something about yourself\n", timeout),
        Scenario("tools_on_after_reset", (*base, "--tools", "on"), "hi\n/reset\nhi\n", timeout),
        Scenario("tools_on_off_switch", (*base, "--tools", "off"), "hi\n/tools on\nhi\n/tools off\nhi\n", timeout),
        Scenario("list_directory_repeat", (*base, "--tools", "on"), "list files in the workdir\nlist files in the workdir\n", timeout),
        Scenario("system_info_repeat", (*base, "--tools", "on"), "tell me the specs of this computer\ntell me the specs of this computer\n", timeout),
    ]
    if include_slow:
        scenarios.append(
            Scenario(
                "fetch_url_smoke_slow",
                (*base, "--tools", "on", "--max-tokens", "256"),
                "fetch https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html and explain the central thesis in Italian\n",
                max(timeout, 360),
            )
        )
    return scenarios


def _run_scenario(
    scenario: Scenario,
    *,
    orbit_bin: str,
    module: str,
    workdir: str,
    env: dict[str, str],
    root: Path,
    diag_path: Path,
) -> dict[str, object]:
    cmd = _command(orbit_bin, module) + ["--workdir", workdir, *scenario.args]
    diag_offset = diag_path.stat().st_size if diag_path.exists() else 0
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=_with_exit(scenario.stdin),
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=scenario.timeout,
        )
        raw = proc.stdout + proc.stderr
        timeout = False
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        raw = _to_text(exc.stdout) + _to_text(exc.stderr)
        timeout = True
        returncode = None
    elapsed = round(time.monotonic() - started, 2)
    footer = _last_footer(raw)
    diag_events = _read_diag_events(diag_path, offset=diag_offset)
    diag_summary = _summarize_diag_events(diag_events)
    return {
        "scenario": scenario.name,
        "returncode": returncode,
        "timeout": timeout,
        "elapsed_s": elapsed,
        **footer,
        **diag_summary,
    }


def _command(orbit_bin: str, module: str) -> list[str]:
    name = Path(orbit_bin).name
    if name.startswith("python"):
        return [orbit_bin, "-m", module]
    return [orbit_bin]


def _last_footer(raw: str) -> dict[str, object]:
    cleaned = ANSI_RE.sub("", raw).replace("\r", "\n")
    matches = list(FOOTER_RE.finditer(cleaned))
    if not matches:
        return {}
    data: dict[str, object] = matches[-1].groupdict()
    for key in ("ctx", "ctx_pct", "prompt_tokens", "generated_tokens", "cached_tokens", "cache_pct"):
        data[key] = int(data[key])
    for key in ("prefill_tps", "generation_tps"):
        data[key] = float(data[key])
    return data


def _write_summary(path: Path, rows: list[dict[str, object]], diag_path: Path) -> None:
    lines = [
        "# KV Diagnostic Benchmark",
        "",
        f"Raw diagnostic JSONL: `{diag_path}`",
        "",
        "| Scenario | Requests | Calls | Phases | Tokens/call | Cached/call | Footer corr | Stop | Wall |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        tokens = "-"
        if row.get("prompt_tokens_by_call"):
            tokens = str(row.get("prompt_tokens_by_call"))
        elif "prompt_tokens" in row:
            tokens = f"{row.get('prompt_tokens')}->{row.get('generated_tokens')}"
        lines.append(
            "| {scenario} | {requests} | {calls} | {phases} | {tokens} | {cached} | {footer} | {stop} | {wall} |".format(
                scenario=row["scenario"],
                requests=row.get("requests", "-"),
                calls=row.get("model_calls", "-"),
                phases=row.get("phases", "-"),
                tokens=tokens,
                cached=row.get("cached_tokens_by_call", row.get("cached_tokens", "-")),
                footer="yes" if row.get("footer_correlation_present") else "no",
                stop="timeout" if row.get("timeout") else row.get("finish_reason", "-"),
                wall=row.get("elapsed_s", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_exit(stdin: str | None) -> str:
    text = stdin or ""
    if not text.endswith("\n"):
        text += "\n"
    return text + "/exit\n"


def _read_diag_events(path: Path, *, offset: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        events = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events


def _summarize_diag_events(events: list[dict[str, object]]) -> dict[str, object]:
    calls = [event for event in events if event.get("event") == "kv_diag_model_call"]
    summaries = [event for event in events if event.get("event") == "kv_diag_request_summary"]
    footers = [event for event in events if event.get("event") == "kv_diag_footer_metrics"]
    call_ids = {event.get("model_call_id") for event in calls}
    footer_correlation = any(event.get("model_call_id") in call_ids for event in footers)
    return {
        "requests": len(summaries) or len({event.get("request_id") for event in calls if event.get("request_id")}),
        "model_calls": len(calls),
        "phases": ",".join(str(event.get("phase")) for event in calls) if calls else "-",
        "prompt_tokens_by_call": ",".join(str(event.get("prompt_tokens")) for event in calls) if calls else "",
        "cached_tokens_by_call": ",".join(str(event.get("cached_tokens")) for event in calls) if calls else "",
        "evaluated_tokens_by_call": ",".join(str(event.get("evaluated_tokens")) for event in calls) if calls else "",
        "footer_correlation_present": footer_correlation,
    }


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

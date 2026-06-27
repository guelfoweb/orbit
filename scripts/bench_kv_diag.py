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
    args = parser.parse_args(argv)

    root = Path.cwd()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    diag_path = output_dir / f"kv_diag_{stamp}.jsonl"
    summary_path = output_dir / f"kv_diag_{stamp}.md"

    scenarios = _scenarios(max_tokens=args.max_tokens, timeout=args.timeout)
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
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    _write_summary(summary_path, rows, diag_path)
    print(f"diag_jsonl={diag_path}")
    print(f"summary_md={summary_path}")
    return 0


def _scenarios(*, max_tokens: str, timeout: int) -> list[Scenario]:
    base = ("--no-render-markdown", "--max-tokens", max_tokens)
    return [
        Scenario("tools_off_repeat", (*base, "--tools", "off"), "hi\nhi\n", timeout),
        Scenario("tools_on_repeat_no_tool_needed", (*base, "--tools", "on"), "hi\nhi\n", timeout),
        Scenario("tools_on_same_prompt_repeat", (*base, "--tools", "on"), "tell me the specs of this computer\ntell me the specs of this computer\n", timeout),
        Scenario("tools_on_after_reset", (*base, "--tools", "on"), "hi\n/reset\nhi\n", timeout),
        Scenario("tools_on_off_switch", (*base, "--tools", "off"), "hi\n/tools on\nhi\n/tools off\nhi\n", timeout),
        Scenario("list_directory_repeat", (*base, "--tools", "on"), "list files in the workdir\nlist files in the workdir\n", timeout),
        Scenario("system_info_repeat", (*base, "--tools", "on"), "tell me the specs of this computer\ntell me the specs of this computer\n", timeout),
    ]


def _run_scenario(
    scenario: Scenario,
    *,
    orbit_bin: str,
    module: str,
    workdir: str,
    env: dict[str, str],
    root: Path,
) -> dict[str, object]:
    cmd = _command(orbit_bin, module) + ["--workdir", workdir, *scenario.args]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=scenario.stdin,
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
    return {
        "scenario": scenario.name,
        "returncode": returncode,
        "timeout": timeout,
        "elapsed_s": elapsed,
        **footer,
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
        "| Scenario | Tokens | Cached | Cache | Prefill/s | Gen/s | Stop | Wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        tokens = "-"
        if "prompt_tokens" in row:
            tokens = f"{row.get('prompt_tokens')}->{row.get('generated_tokens')}"
        lines.append(
            "| {scenario} | {tokens} | {cached} | {cache} | {pf} | {gen} | {stop} | {wall} |".format(
                scenario=row["scenario"],
                tokens=tokens,
                cached=row.get("cached_tokens", "-"),
                cache=f"{row.get('cache_pct')}%" if "cache_pct" in row else "-",
                pf=row.get("prefill_tps", "-"),
                gen=row.get("generation_tps", "-"),
                stop="timeout" if row.get("timeout") else row.get("finish_reason", "-"),
                wall=row.get("elapsed_s", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

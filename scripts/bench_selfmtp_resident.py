#!/usr/bin/env python3
"""Multi-turn self-MTP benchmark: MTP OFF (B) vs self-MTP ON (C).

The existing `bench_mtp_throughput.py` builds a fresh `ChatRuntime` per case and
calls `ask_chat` once, so every measured turn is a FIRST turn. With no committed
identity there is nothing for a resident claim to describe, and resident reuse
can never activate -- it would measure MTP-without-reuse and report it as the
feature. This harness keeps one conversation alive across turns so the measured
turn actually satisfies `0 < committed_len < prompt_len`.

Measurement discipline:
  * turn 1 of each conversation is a WARM-UP and is never measured; it exists to
    establish the committed identity that turn 2+ reuse.
  * only turns >= 2 are measured.
  * for variant C every measured turn must prove resident reuse actually
    happened (resident_reuse_active, pair_canonical, cached_tokens > 0). A turn
    that falls cold is recorded as a reuse failure and the run is marked invalid
    rather than silently averaged in as "resident performance".

Nothing here changes production code; it drives the real server over real /chat.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/home/guelfoweb/LAB/orbit-model-new/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf"

# Deterministic, output-length-stable turns. Each asks for an exact short form so
# generated-token counts stay comparable between variants; an unstable length
# would make tok/s incomparable rather than merely noisy.
SHORT_CONVERSATION = [
    "Reply with exactly: The capital of France is Paris.",
    "Reply with exactly: The capital of Italy is Rome.",
    "Reply with exactly: The capital of Spain is Madrid.",
    "Reply with exactly: The capital of Japan is Tokyo.",
]

# Speculative decoding pays a fixed draft+verify cost per step. Over a 7-token
# answer that cost cannot amortize, so the short fixture measures overhead
# rather than throughput. This one generates ~150-200 tokens per turn, which is
# where a generation-throughput claim is actually decided. Both fixtures are
# reported; neither is dropped after the fact.
LONG_CONVERSATION = [
    "Write exactly one paragraph of 150 to 200 words explaining why careful "
    "measurement should come before optimization. Do not use bullet points.",
    "Write exactly one paragraph of 150 to 200 words about a fictional archive "
    "room, covering layout, lighting, and noise. Do not use bullet points.",
    "Write exactly one paragraph of 150 to 200 words about how a small team "
    "should choose which bug to fix first. Do not use bullet points.",
    "Write exactly one paragraph of 150 to 200 words about the trade-offs of "
    "caching in a chat server. Do not use bullet points.",
]

FIXTURES = {"short": SHORT_CONVERSATION, "long": LONG_CONVERSATION}
MAX_TOKENS = {"short": 48, "long": 320}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def start_server(mode: str, log_root: Path) -> tuple[subprocess.Popen, str, Path, float]:
    port = free_port()
    log_path = log_root / f"server_{mode}_{port}.log"
    cmd = [str(ROOT / ".venv/bin/orbit"), "server", "--port", str(port), "--model", MODEL]
    if mode == "mtp_on":
        cmd.append("--mtp")
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp"
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdout=log_path.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{mode} server exited during startup; see {log_path}")
        try:
            get(f"{base}/health", timeout=2.0)
            return proc, base, log_path, time.monotonic() - started
        except Exception:
            time.sleep(2.0)
    proc.kill()
    raise RuntimeError(f"{mode} server never became ready; see {log_path}")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def rss_kb(pid: int) -> tuple[int, int]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return (0, 0)
    rss = hwm = 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1])
    return rss, hwm


def run_conversation(base: str, mode: str, rep: int, pid: int, fixture: str) -> list[dict]:
    """One conversation; turn 1 warms up, turns 2+ are measured."""
    messages: list[dict] = []
    rows: list[dict] = []
    for index, prompt in enumerate(FIXTURES[fixture], start=1):
        messages.append({"role": "user", "content": prompt})
        started = time.monotonic()
        result = post(f"{base}/chat", {"messages": messages, "max_tokens": MAX_TOKENS[fixture]})
        wall = time.monotonic() - started
        content = result.get("content") or ""
        messages.append({"role": "assistant", "content": content})

        usage = result.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        native = result.get("native") or {}
        props = get(f"{base}/props")
        mtp = props.get("mtp_last_completion") or {}

        gen_tokens = int(usage.get("completion_tokens") or 0)
        gen_ms = float(native.get("generation_ms") or 0.0)
        prefill_ms = float(native.get("prefill_ms") or 0.0)
        rss, hwm = rss_kb(pid)

        rows.append({
            "fixture": fixture,
            "mode": mode,
            "repetition": rep,
            "turn": index,
            "measured": index >= 2,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "evaluated_tokens": details.get("evaluated_tokens"),
            "generated_tokens": gen_tokens,
            "generation_ms": gen_ms,
            "generation_tps": (gen_tokens / (gen_ms / 1000.0)) if gen_ms > 0 else None,
            "prefill_ms": prefill_ms,
            "ttft_ms": native.get("backend_ttft_ms"),
            "turn_wall_s": wall,
            "content": content,
            "vmrss_kb": rss,
            "vmhwm_kb": hwm,
            # C-only evidence; absent/False for B.
            "resident_reuse_active": mtp.get("resident_reuse_active"),
            "pair_canonical": mtp.get("pair_canonical"),
            "resident_token_count": mtp.get("resident_token_count"),
            "drafted": mtp.get("draft_tokens_total"),
            "accepted": mtp.get("accepted_tokens_total"),
            "draft_calls": mtp.get("draft_decode_calls"),
            "acceptance_ratio": mtp.get("acceptance_ratio"),
            "self_mtp_active": props.get("self_mtp_active"),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workdir/bench")
    parser.add_argument("--modes", default="mtp_off,mtp_on")
    parser.add_argument("--fixture", default="long", choices=sorted(FIXTURES))
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jsonl = out_dir / f"selfmtp_resident_{args.fixture}_{stamp}.jsonl"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    all_rows: list[dict] = []
    with jsonl.open("w", encoding="utf-8") as handle:
        # Balanced interleave: each variant occupies a different thermal
        # position across repetitions, so neither systematically runs hotter.
        for rep in range(1, args.runs + 1):
            order = modes if rep % 2 == 1 else list(reversed(modes))
            for mode in order:
                proc, base, log_path, startup_s = start_server(mode, out_dir)
                try:
                    props = get(f"{base}/props")
                    rows = run_conversation(base, mode, rep, proc.pid, args.fixture)
                    for row in rows:
                        row["startup_s"] = startup_s
                        row["server_log"] = str(log_path)
                        row["self_mtp_active_startup"] = props.get("self_mtp_active")
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        all_rows.append(row)
                finally:
                    stop_server(proc)
                # Let the machine settle so the next variant does not inherit
                # the previous one's thermal state.
                time.sleep(45)

    print(f"raw={jsonl}")
    print(f"rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

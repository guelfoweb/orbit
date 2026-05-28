from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
import uuid


FAST_PROMPTS = (
    ("F1", "hi, who are you?"),
    ("F2", "list all files and directories in the current workspace"),
    ("F3", 'decode this string "Y2lhbw==" from base64'),
    ("F4", "what is the size and modified time of agent.py?"),
    ("F5", "tell me how many files exist in the workspace and what the newest file is."),
)
HEAVY_PROMPTS = (
    ("H1", "review agent.py for vulnerabilities and security issues"),
    ("H2", "analizza questo testo promessi_sposi.txt e riassumilo in 5 righe"),
    ("H3", "compare two images: cmp-blue.png and vision-test.png and tell me the differences"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small Orbit prompt benchmark.")
    parser.add_argument("--model", default="gemma4:e2b-fast-t6-c8k")
    parser.add_argument("--workdir", default="workdir")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--include-heavy", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    workdir = (root / args.workdir).resolve()
    prompts = list(FAST_PROMPTS)
    if args.include_heavy:
        prompts.extend(HEAVY_PROMPTS)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("OLLAMA_NUM_PARALLEL", "1")
    env.setdefault("OLLAMA_KEEP_ALIVE", "-1")
    run_id = uuid.uuid4().hex[:8]
    base_cmd = [
        "python3",
        "-c",
        "from orbit.terminal.cli import main; raise SystemExit(main())",
        "--model",
        args.model,
    ]

    failures = 0
    for label, prompt in prompts:
        started = time.monotonic()
        completed = subprocess.run(
            [*base_cmd, "--session", f"benchmark-{run_id}-{label.lower()}", prompt],
            cwd=workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            failures += 1
        print(f"=== {label} | {elapsed:.1f}s | exit={completed.returncode} ===")
        print(prompt)
        print(completed.stdout.strip())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

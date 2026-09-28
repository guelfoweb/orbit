#!/usr/bin/env python3
"""Qualification Harness tool-intent capability, with intercepted execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.llama_server import LlamaServerBackend
from orbit.qualification.tool_intent import (
    ReplayBackend, load_corpus, load_replay, observe_case, summarize, write_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path, help="Replay retained responses offline; no server or inference.")
    mode.add_argument("--base-url", help="Explicit live gate on an independently managed, idle qualification server.")
    parser.add_argument("--profile", help="Required exact verified profile for live mode.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "qualification/fixtures/tool-intent-v1.json")
    parser.add_argument("--output", type=Path, required=True, help="JSON path; also writes a sibling Markdown report.")
    args = parser.parse_args(argv)
    corpus = load_corpus(args.corpus)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    if args.replay:
        replay = load_replay(args.replay, corpus)
        tapes = {row["case_id"]: row["calls"] for row in replay["cases"]}
        context = replay["provenance"]["ctx"]
        provenance = {"mode": "recorded_replay", "harness_revision": revision,
                      "source": replay["provenance"], "inference_calls_this_run": 0}

        def backend_for(case):
            return ReplayBackend(tapes[case["id"]], context_tokens=context)
    else:
        if not args.profile:
            parser.error("--profile is required for live qualification")
        backend = LlamaServerBackend(base_url=args.base_url, timeout=240, thinking=False)
        props = backend.backend_props()
        profile = props.get("model_compatibility", {})
        if not (profile.get("verified") is True and profile.get("compatibility_profile") == args.profile):
            raise ValueError("live qualification requires the exact verified server profile")
        context = props["ctx_size"]
        provenance = {"mode": "live_intercepted", "harness_revision": revision, "props": props,
                      "measurement_scope": "client time to first intercepted handoff/CHAT; tools and final generation excluded"}

        def backend_for(case):
            # Existing reset API: no server startup, model change or prompt variant.
            # backend_props caches /props; a fresh client makes this a live check.
            client = LlamaServerBackend(base_url=args.base_url, timeout=240, thinking=False)
            current = client.backend_props()
            if current.get("model_compatibility") != profile or current.get("ctx_size") != context:
                raise ValueError("qualification server identity changed")
            if current.get("in_flight") is not False:
                raise ValueError("qualification server must be idle")
            error = client.reset_static_analysis_session()
            if error is not None:
                raise ValueError("cannot establish cold session: " + error)
            return client
    observations = []
    with tempfile.TemporaryDirectory(prefix="orbit-tool-intent-") as tmp:
        for case in corpus["cases"]:
            workdir = Path(tmp) / case["id"]
            workdir.mkdir()
            (workdir / "notes.txt").write_text("Safe routing fixture.\n", encoding="utf-8")
            observation = observe_case(case, backend_for(case), workdir, context_tokens=context)
            observations.append({"id": case["id"], **observation})
    report = summarize(corpus, observations, provenance=provenance)
    write_report(report, args.output)
    print(json.dumps({"capability": report["capability"], "status": report["status"], **report["metrics"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

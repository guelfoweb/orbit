"""A bounded, dedicated research process using Orbit's existing inference path.

The profile remains unverified. The executor is intercepted by the unchanged
tool-intent observer. Reports cannot register or promote a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from orbit.backend.llama_server import LlamaServerBackend
from orbit.native_llama.paths import resolve_legacy_paths
from orbit.native_llama.research_admission import ResearchAdmission
from orbit.tool_contract_config import resolve_tool_call_canonical_gate
from .model_result import MODEL_QUALIFICATION_CAPABILITIES, ModelQualificationResult
from .tool_intent import digest, load_corpus, observe_case, summarize, write_report


def _write(path: Path, value) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _source_identity(root: Path) -> dict:
    paths = list((root / "src/orbit").rglob("*.py"))
    paths += [root / "src/orbit/native_llama/model_registry.json",
              root / "scripts/orbit_qualify_model.py"]
    files = {str(path.relative_to(root)): _file_digest(path) for path in sorted(paths)}
    return {"sha256": digest(files), "files": files}


def _backend_identity(paths) -> dict:
    # Bind the complete co-located native family and Orbit bridges, not just
    # libllama: the same libllama with another parser is not the same backend.
    return {path.name: _file_digest(path) for path in sorted(paths.build_bin.glob("*.so*"))
            if path.is_file()}


def _server_args(args, port: int) -> list[str]:
    return ["--model", str(args.model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx", str(args.ctx), "--threads", str(args.threads),
            "--threads-batch", str(args.threads_batch), "--batch", str(args.batch),
            "--ubatch", str(args.ubatch), "--think", "off", "--repack", "off"]


def check_research_server(props: dict, admission: ResearchAdmission, *, pid: int) -> None:
    compatibility = props.get("model_compatibility", {})
    research = compatibility.get("research_admission", {})
    if not (props.get("backend") == "orbit-native"
            and compatibility.get("compatibility_profile") == admission.profile_id
            and compatibility.get("verified") is False
            and research.get("artifact_sha256") == admission.artifact_sha256
            and research.get("template_sha256") == admission.template_sha256
            and research.get("production_enabled") is False
            and type(research.get("pid")) is int and research["pid"] == pid
            and props.get("in_flight") is False):
        raise ValueError("dedicated research server identity/idle check failed")


@contextmanager
def managed_server(args, admission: ResearchAdmission, *, script: Path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {**os.environ, "TMPDIR": "/tmp", "ORBIT_KV_PREFIX_PREWARM": "off",
           "ORBIT_TOOL_CALL_CANONICAL_GATE": "1"}
    command = [sys.executable, str(script), "--serve", "--identity", str(args.identity),
               "--model", str(args.model), "--port", str(port), "--ctx", str(args.ctx),
               "--threads", str(args.threads), "--threads-batch", str(args.threads_batch),
               "--batch", str(args.batch), "--ubatch", str(args.ubatch)]
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError("research server failed; see server.log")
                backend = LlamaServerBackend(base_url=f"http://127.0.0.1:{port}", timeout=2, thinking=False)
                props = backend.backend_props()
                if props:
                    check_research_server(props, admission, pid=process.pid)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("research server startup exceeded 300 seconds")
                time.sleep(0.5)
            yield f"http://127.0.0.1:{port}", process.pid, props
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


def _fresh_backend(base_url, admission, pid):
    backend = LlamaServerBackend(base_url=base_url, timeout=240, thinking=False)
    check_research_server(backend.backend_props(), admission, pid=pid)
    error = backend.reset_static_analysis_session()
    if error is not None:
        raise RuntimeError("research session reset failed: " + error)
    return backend


def chat_observations(base_url, admission, pid, *, output: Path) -> list[dict]:
    prompts = (
        ("italian", "Qual è la capitale della Francia? Rispondi con una sola parola."),
        ("english", "Explain RAM versus disk storage in two short sentences."),
        ("json", 'Return only JSON with keys "name" equal to "Orbit" and "sum" equal to 17 + 25.'),
        ("memory", "Remember the word CEDAR for my next message. Acknowledge briefly."),
    )
    rows = []
    for name, prompt in prompts:
        backend = _fresh_backend(base_url, admission, pid)
        messages = [{"role": "user", "content": prompt}]
        started = time.monotonic()
        result = backend.chat(messages, temperature=0, max_tokens=1024)
        rows.append({"id": name, "messages": deepcopy(messages), "result": asdict(result),
                     "wall_seconds": time.monotonic() - started})
        _write(output / "chat-progress.json", rows)
        if name == "memory":
            messages += [{"role": "assistant", "content": result.content},
                         {"role": "user", "content": "Which word did I ask you to remember? Reply with that word only."}]
            started = time.monotonic()
            result = backend.chat(messages, temperature=0, max_tokens=1024)
            rows.append({"id": "memory_recall", "messages": deepcopy(messages), "result": asdict(result),
                         "wall_seconds": time.monotonic() - started})
            _write(output / "chat-progress.json", rows)
    return rows


def chat_status(rows: list[dict]) -> str:
    # A normal stop with visible output establishes technical completion only.
    # Correctness, language, JSON purpose, recall and reasoning isolation need
    # human review of these exact bytes; never infer a semantic PASS here.
    if len(rows) != 5 or any(
        row["result"]["finish_reason"] != "stop" or not row["result"]["content"].strip()
        or row["result"]["tool_calls"] for row in rows
    ):
        return "FAIL"
    return "MANUAL_CHECK"


def tool_intent_observations(base_url, admission, pid, *, root: Path, output: Path):
    corpus = load_corpus(root / "qualification/fixtures/tool-intent-v1.json")
    observations = []
    props = _fresh_backend(base_url, admission, pid).backend_props()
    with tempfile.TemporaryDirectory(prefix="orbit-model-intent-") as tmp:
        for case in corpus["cases"]:
            backend = _fresh_backend(base_url, admission, pid)
            workdir = Path(tmp) / case["id"]
            workdir.mkdir()
            (workdir / "notes.txt").write_text("Safe routing fixture.\n", encoding="utf-8")
            observation = observe_case(case, backend, workdir, context_tokens=props["ctx_size"])
            observations.append({"id": case["id"], **observation})
            _write(output / "tool-intent-progress.json", observations)
            print(f"tool-intent {len(observations)}/24: {case['id']} {observation['kind']}", flush=True)
    report = summarize(corpus, observations, provenance={
        "mode": "live_intercepted_research", "props": props,
        "scope": "Explicit unqualified admission; unchanged observer/corpus/budgets; no production enablement",
    })
    write_report(report, output / "tool-intent.json")
    return report


def run_probe(args, *, script: Path, root: Path) -> int:
    if not resolve_tool_call_canonical_gate().enabled:
        raise ValueError("research qualification requires the canonical gate in the parent process")
    admission = ResearchAdmission.read(args.identity)
    # Unique output bundles prevent accidental replacement of retained evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    paths = resolve_legacy_paths(model=args.model)
    backend_identity = _backend_identity(paths)
    if not backend_identity:
        raise ValueError("no native runtime family available")
    backend_hash = digest(backend_identity)
    source_identity = _source_identity(root)
    config_hash = digest({"server_args": _server_args(args, 0), "chat_temperature": 0,
                          "chat_max_tokens": 1024, "tool_gate": "tool-intent-v1",
                          "canonical_gate": True, "prewarm": "off",
                          "source_sha256": source_identity["sha256"],
                          "environment_sha256": digest({k: v for k, v in os.environ.items()
                                                       if k.startswith("ORBIT_")})})
    results = {name: ModelQualificationResult(
        admission.artifact_sha256, backend_hash, config_hash, name, "NOT_RUN",
        "No observation retained for this capability",
    ) for name in MODEL_QUALIFICATION_CAPABILITIES}

    def record(name, status, scope, evidence):
        evidence_hash = _write(args.output / (name + ".json"), evidence)
        results[name] = ModelQualificationResult(admission.artifact_sha256, backend_hash,
                                                 config_hash, name, status, scope, evidence_hash)

    def save():
        _write(args.output / "qualification.json", {
            "schema_version": 1, "production_enabled": False,
            "identity_manifest_sha256": _file_digest(args.identity),
            "backend_identity": backend_identity, "source_identity": source_identity,
            "results": {name: result.as_dict() for name, result in results.items()},
        })

    phase = "load"
    try:
        with managed_server(args, admission, script=script) as (base_url, pid, props):
            record("load", "PASS", "Dedicated native CPU server loaded this exact artifact; no other capability implied", props)
            save()
            phase = "chat"
            rows = chat_observations(base_url, admission, pid, output=args.output)
            record("chat", chat_status(rows), "Five tools-off observations; semantic review of exact responses required", rows)
            save()
            phase = "tool_intent"
            report = tool_intent_observations(base_url, admission, pid, root=root, output=args.output)
            record("tool_intent", report["status"], report["scope"], report)
            save()
    except Exception as error:
        record(phase, "FAIL", "Qualification stage failed; see error and server log", {
            "error_type": type(error).__name__, "error": str(error),
        })
        save()
        print(f"qualification failed: {error}", file=sys.stderr)
        return 1
    finally:
        save()
    print(json.dumps({name: result.status for name, result in results.items()}), flush=True)
    # MANUAL_CHECK is deliberately not success. Review never modifies this
    # recording, and a PASS of the tool gate never promotes the profile.
    return 1 if any(result.status == "FAIL" for result in results.values()) else 2


def main(argv=None, *, script: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path, help="Explicit versioned research identity manifest")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New evidence-bundle directory (must not exist)")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--max-wall-seconds", type=int, default=14400,
                        help="Total parent-process wall ceiling, including load (default: four hours)")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.identity = args.identity.resolve()
    args.model = args.model.resolve()
    if any(value <= 0 for value in (args.ctx, args.threads, args.threads_batch, args.batch, args.ubatch, args.max_wall_seconds)):
        parser.error("context, thread and batch values must be positive")
    if args.serve:
        if args.port is None or not 1 <= args.port <= 65535:
            parser.error("internal research server requires a port")
        from orbit.native_server.app import run_server
        return run_server(_server_args(args, args.port), research_admission=ResearchAdmission.read(args.identity))
    if args.output is None:
        parser.error("--output is required")
    def timeout(_signum, _frame):
        raise TimeoutError("research qualification exceeded its wall ceiling")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_wall_seconds)
    try:
        return run_probe(args, script=script, root=script.resolve().parents[1])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

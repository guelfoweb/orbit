#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:0] = [str(ROOT), str(SRC)]

from orbit.backend.llama_server import LlamaServerBackend  # noqa: E402
from orbit.qualification.fixtures import load_fixture_set  # noqa: E402
from orbit.qualification.reporting import format_summary, write_result  # noqa: E402
from orbit.qualification.runner import QualificationRunner, _peak_rss_bytes  # noqa: E402
from orbit.qualification.schema import CallMetric, FixtureObservation, LifecycleOutcome, StateReuseEvidence  # noqa: E402
from orbit.runtime.kv_diag import model_call_context  # noqa: E402
from orbit.runtime.messages import FINAL_FROM_TOOL_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT  # noqa: E402
from scripts.orbit_qualify import build_provenance  # noqa: E402
from scripts.orbit_qualify_compare import _free_port, _server_command, _stop, _wait_ready  # noqa: E402


class LifecycleExecutor:
    def __init__(self, process, backend, base_url, profile_id):
        self.process, self.backend = process, backend
        self.base_url, self.profile_id = base_url, profile_id

    def execute(self, fixture, workdir):
        operation = fixture.expect.lifecycle.operation
        started = time.perf_counter()
        calls = []
        before = self._state()
        if operation == "reset_invalidation":
            calls.extend((self._reuse_call(), self._reuse_call()))
            established = self._state()
            initial = established
            invalidations, recapture = 0, True
            for _ in range(2):
                calls.append(self._mode_switch())
                invalidated = self._state()
                invalidations += int(not invalidated["initialized"])
                calls.append(self._reuse_call())
                after = self._state()
                recapture &= after["capture_count"] == established["capture_count"] + 1 and after["restore_count"] == established["restore_count"]
                established = after
            evidence = self._evidence(operation, initial, after, calls, invalidated=invalidations == 2, recapture=recapture)
            evidence = StateReuseEvidence(**{**evidence.__dict__, "invalidation_count": invalidations})
        elif operation == "cancellation":
            if not before["initialized"]:
                calls.append(self._reuse_call())
                before = self._state()
            result = []
            worker = threading.Thread(target=lambda: result.append(self._reuse_call(long=True)), daemon=True)
            worker.start()
            self._wait_in_flight()
            self._cancel()
            worker.join(30)
            calls.extend(result)
            after = self._state()
            evidence = self._evidence(operation, before, after, calls, cancelled=not worker.is_alive())
        elif operation == "restore_failure_fallback":
            evidence = self._restore_hook(operation)
        elif operation == "repeated_restore_rss":
            if not before["initialized"]:
                calls.append(self._reuse_call())
            established = self._state()
            start_rss = _rss(self.process.pid)
            samples = [start_rss]
            restore_before = self._state()["restore_count"]
            for index in range(fixture.expect.lifecycle.min_restores):
                calls.append(self._reuse_call())
                if index in {4, 9, fixture.expect.lifecycle.min_restores - 1}:
                    samples.append(_rss(self.process.pid))
            after = self._state()
            tolerance = max(64 * 1024**2, int(start_rss * 0.01))
            evidence = self._evidence(operation, established, after, calls, rss=samples, tolerance=tolerance)
            evidence = StateReuseEvidence(**{**evidence.__dict__, "restore_count": after["restore_count"] - restore_before})
        else:
            if not before["initialized"]:
                calls.append(self._reuse_call())
                before = self._state()
            pid, port = self.process.pid, int(self.base_url.rsplit(":", 1)[1])
            children = _descendants(pid)
            _stop(self.process)
            residue = [str(path.relative_to(workdir)) for path in workdir.rglob("*") if path.name.startswith(".orbit-")]
            residue.extend(f"process:{child}" for child in children if Path(f"/proc/{child}").exists())
            evidence = self._evidence(
                operation, before, self._state(dead=True), calls, pid=pid,
                exit_code=self.process.returncode, port_released=_port_free(port), residue=tuple(residue),
            )
        if operation != "teardown_cleanup":
            residue = tuple(str(path.relative_to(workdir)) for path in workdir.rglob("*") if path.name.startswith(".orbit-"))
            evidence = StateReuseEvidence(**{**evidence.__dict__, "residual_state": residue})
        finish = calls[-1].finish_reason if calls else "stop"
        return FixtureObservation(
            None, (), (), "", finish, len(calls), 0, tuple(calls), None,
            LifecycleOutcome(not evidence.residual_state, "clean" if not evidence.residual_state else evidence.residual_state[0]),
            _peak_rss_bytes(self.process.pid) if self.process.poll() is None else None,
            wall_seconds=time.perf_counter() - started,
            state_reuse=evidence,
        )

    def _reuse_call(self, long=False):
        user = ("word " * 5000) if long else "Return only OK."
        messages = (
            [{"role": "system", "content": FINAL_FROM_TOOL_SYSTEM_PROMPT}, {"role": "user", "content": user}, {"role": "system", "content": "tool evidence: OK"}]
            if self.profile_id == "orbit-gemma4-native-v1"
            else [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
        )
        phase = "final_from_tool" if self.profile_id == "orbit-gemma4-native-v1" else "route"
        started = time.perf_counter()
        with model_call_context(phase=phase, tools_mode="on"):
            result = self.backend.chat(messages, temperature=0, max_tokens=16)
        return CallMetric(
            phase, result.prompt_tokens,
            result.prompt_tokens - result.cached_tokens if result.prompt_tokens is not None and result.cached_tokens is not None else None,
            result.cached_tokens, result.completion_tokens, result.prompt_tokens_per_second,
            result.generation_tokens_per_second, time.perf_counter() - started, result.finish_reason,
        )

    def _mode_switch(self):
        tool = [{"type": "function", "function": {"name": "noop", "description": "No operation", "parameters": {"type": "object", "properties": {}}}}]
        started = time.perf_counter()
        result = self.backend.chat([{"role": "user", "content": "Answer OK."}], temperature=0, max_tokens=1, tools=tool)
        return CallMetric(
            "reset", result.prompt_tokens,
            result.prompt_tokens - result.cached_tokens if result.prompt_tokens is not None and result.cached_tokens is not None else None,
            result.cached_tokens, result.completion_tokens, result.prompt_tokens_per_second,
            result.generation_tokens_per_second, time.perf_counter() - started, result.finish_reason,
        )

    def _state(self, dead=False):
        if dead:
            return {name: 0 if name.endswith(("count", "bytes")) else False for name in ("initialized", "capture_count", "restore_count", "fallback_count", "invalidation_count", "checkpoint_size_bytes")}
        props = LlamaServerBackend(base_url=self.base_url, timeout=5).backend_props()
        if self.profile_id == "orbit-qwen3-coder-native-v1":
            return props["qwen3_coder_route_prefix_reuse"]
        if self.profile_id == "orbit-qwen36-native-v1":
            return props["qwen_route_prefix_reuse"]
        return {key: props.get(f"final_prefix_experiment_{key}") for key in ("initialized", "capture_count", "restore_count", "fallback_count", "checkpoint_size_bytes")} | {"invalidation_count": 0}

    def _evidence(self, operation, before, after, calls, *, invalidated=None, recapture=None, cancelled=None, rss=None, tolerance=None, pid=None, exit_code=None, port_released=None, residue=()):
        cached = calls[-1].cached_tokens if calls else None
        checkpoint_size = after.get("checkpoint_size_bytes")
        partial = bool(after["initialized"] or checkpoint_size) if operation in {"cancellation", "restore_failure_fallback"} else False
        return StateReuseEvidence(
            operation, bool(before["initialized"]), bool(after["initialized"]),
            invalidated if invalidated is not None else (not after["initialized"]), recapture,
            after["capture_count"], after["restore_count"], after["fallback_count"], after.get("invalidation_count", 0),
            cached, checkpoint_size, partial, cancelled, operation == "restore_failure_fallback",
            operation == "restore_failure_fallback", 1 if operation == "restore_failure_fallback" else 0,
            rss[0] if rss else None, rss[-1] if rss else None, max(rss) if rss else None,
            tolerance, tuple(rss or ()), pid or (self.process.pid if self.process.poll() is None else None),
            exit_code, port_released, tuple(residue),
        )

    def _restore_hook(self, operation):
        tests = {
            "orbit-qwen3-coder-native-v1": "tests.test_qwen3_coder_route_prefix.Qwen3CoderRoutePrefixClientTests.test_restore_failure_falls_back_cold_without_touching_qwen36_state",
            "orbit-qwen36-native-v1": "tests.test_qwen_route_prefix.QwenRoutePrefixClientTests.test_restore_failure_clears_state_and_uses_one_cold_fallback",
            "orbit-gemma4-native-v1": "tests.test_prefix_anchor_probe.PrefixAnchorProbeTests.test_final_prefix_restore_failure_uses_normal_prefill",
        }
        completed = subprocess.run([sys.executable, "-m", "unittest", tests[self.profile_id]], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(SRC)}, capture_output=True, text=True)
        report = completed.stdout + completed.stderr
        if completed.returncode or "Ran 1 test" not in report or not report.rstrip().endswith("OK"):
            raise RuntimeError("native restore-failure hook failed")
        evidence = self._evidence(operation, {"initialized": True}, {"initialized": False, "capture_count": 1, "restore_count": 0, "fallback_count": 1, "invalidation_count": 1, "checkpoint_size_bytes": 0}, [])
        return StateReuseEvidence(**{**evidence.__dict__, "process_pid": None})

    def _wait_in_flight(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if LlamaServerBackend(base_url=self.base_url, timeout=2).backend_props().get("in_flight") is True:
                return
            time.sleep(0.05)
        raise TimeoutError("active cancellation window unavailable")

    def _cancel(self):
        self.backend._post_json("/cancel", {"session_id": "default"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path); parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--fixtures", default=str(ROOT / "qualification/fixtures/lifecycle-v1.json"))
    for name, default in (("ctx", 8192), ("threads", 6), ("threads_batch", 6), ("batch", 256), ("ubatch", 128)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--startup-timeout", type=float, default=180); parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args(); port = _free_port(); base_url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(SRC), "ORBIT_KV_PREFIX_PREWARM": "off"}
    with tempfile.NamedTemporaryFile(prefix="orbit-lifecycle-", suffix=".log", delete=False) as temporary:
        log_path = Path(temporary.name)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                _server_command(args, port), cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, text=True,
            )
            try:
                backend = LlamaServerBackend(base_url=base_url, timeout=args.request_timeout)
                _wait_ready(backend, process, args.startup_timeout, log_path)
                props = backend.backend_props(); profile = props["model_compatibility"]
                fixture_set = load_fixture_set(args.fixtures)
                with tempfile.TemporaryDirectory(prefix="orbit-lifecycle-fixtures-") as workdir:
                    run = QualificationRunner(fixture_set, profile, build_provenance(args.profile, fixture_set.content_hash, profile, props, process.pid), LifecycleExecutor(process, backend, base_url, args.profile), Path(workdir)).run()
                write_result(run, args.output); print(format_summary(run))
                return 0 if run.overall_status.value == "PASS" else 1
            finally:
                _stop(process)
    finally:
        log_path.unlink(missing_ok=True)


def _rss(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"): return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def _port_free(port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: listener.bind(("127.0.0.1", port))
        except OSError: return False
    return True


def _descendants(pid):
    found, pending = set(), [pid]
    while pending:
        path = Path(f"/proc/{pending.pop()}/task")
        for task in path.iterdir() if path.exists() else ():
            try:
                children = (task / "children").read_text().split()
            except OSError:
                continue
            pending.extend(child for child in map(int, children) if child not in found)
            found.update(map(int, children))
    return found


if __name__ == "__main__":
    raise SystemExit(main())

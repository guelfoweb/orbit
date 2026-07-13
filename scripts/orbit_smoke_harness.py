#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit import __version__
from orbit.backend.base import ChatResult
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.kv_diag import current_phase
from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT
from orbit.runtime.tools import TOOL_NAMES
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.runtime_status import collect_host_info


CorrectnessChecker = Callable[[str, list[str]], str]


@dataclass(frozen=True)
class SmokeStep:
    prompt: str
    mode: str = "auto"
    checker_name: str = "not_evaluated"


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    steps: tuple[SmokeStep, ...]
    requires_web: bool = False
    optional: bool = False
    allowed_tool_names: tuple[str, ...] = ("exec_shell_full_command",)
    isolated_steps: bool = False


@dataclass(frozen=True)
class StepReport:
    case: str
    step: int
    prompt: str
    prompt_kind: str
    completion_kind: str
    route_tokens: int | None
    final_tokens: int | None
    prompt_tokens: int | None
    cached_tokens: int | None
    evaluated_tokens: int | None
    finish_reason: str | None
    tool_calls: int
    tool_names: list[str]
    wall_ms: float
    correctness_category: str
    raw_leak: bool
    fake_output: bool
    loop: bool
    notes: str
    answer_excerpt: str
    model_steps: list[dict[str, object]]
    output_tokens: int | None = None
    final_prefix: dict[str, object] = field(default_factory=dict)
    lifecycle: dict[str, object] = field(default_factory=dict)
    phase_wall_ms: dict[str, float] = field(default_factory=dict)
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    estimated_generation_ms: float | None = None
    estimated_prefill_residual_ms: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "case": self.case,
            "step": self.step,
            "prompt": self.prompt,
            "prompt_kind": self.prompt_kind,
            "completion_kind": self.completion_kind,
            "route_tokens": self.route_tokens,
            "final_tokens": self.final_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "evaluated_tokens": self.evaluated_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "tool_names": self.tool_names,
            "wall_ms": self.wall_ms,
            "correctness_category": self.correctness_category,
            "raw_leak": self.raw_leak,
            "fake_output": self.fake_output,
            "loop": self.loop,
            "notes": self.notes,
            "answer_excerpt": self.answer_excerpt,
            "model_steps": self.model_steps,
            "final_prefix": self.final_prefix,
            "lifecycle": self.lifecycle,
            "phase_wall_ms": self.phase_wall_ms,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "estimated_generation_ms": self.estimated_generation_ms,
            "estimated_prefill_residual_ms": self.estimated_prefill_residual_ms,
        }


class ProbeBackend:
    def __init__(self, backend: LlamaServerBackend) -> None:
        self._backend = backend
        self._phase_timings: list[tuple[str, float]] = []

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        return self._timed(
            lambda: self._backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
        )

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        return self._timed(
            lambda: self._backend.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                on_delta=on_delta or (lambda _text: None),
                on_progress=on_progress,
            )
        )

    def continue_current(self, *, max_tokens, on_delta=None, on_progress=None):
        return self._timed(
            lambda: self._backend.continue_current(max_tokens=max_tokens, on_delta=on_delta, on_progress=on_progress)
        )

    def _timed(self, call):
        phase = current_phase() or "unknown"
        started = time.perf_counter()
        try:
            return call()
        finally:
            self._phase_timings.append((phase, round((time.perf_counter() - started) * 1000, 1)))

    def reset_phase_timings(self) -> None:
        self._phase_timings.clear()

    def phase_timings(self) -> list[tuple[str, float]]:
        return list(self._phase_timings)


def mtp_state_from_props(props: dict[str, object]) -> dict[str, object]:
    requested = bool(props.get("mtp_experimental_enabled"))
    session_ready = bool(props.get("mtp_initialized"))
    last_success = props.get("mtp_last_completion_success") is True
    failure_reason = first_nonempty_str(props.get("mtp_failure_reason"), props.get("mtp_fallback_reason"))
    attempted = last_success or failure_reason is not None
    usable = bool(requested and session_ready and last_success and failure_reason is None)
    if usable:
        status = "on"
    elif failure_reason is not None:
        status = "failed"
    elif requested and session_ready:
        status = "ready"
    elif requested:
        status = "requested"
    else:
        status = "off"
    return {
        "requested": requested,
        "session_ready": session_ready,
        "attempted": attempted,
        "usable": usable,
        "status": status,
        "failure_reason": failure_reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = select_scenarios(args.scenario, no_web=args.no_web, include_optional=args.include_optional)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = Path(args.jsonl) if args.jsonl else output_dir / f"orbit_smoke_{stamp}.jsonl"
    markdown_path = Path(args.markdown) if args.markdown else output_dir / f"orbit_smoke_{stamp}.md"

    with final_prefix_environment(args.final_prefix_mode), deterministic_web(args.deterministic_web), managed_server(args) as server_process:
        backend = LlamaServerBackend(base_url=args.base_url, timeout=args.timeout)
        backend.thinking = args.server_thinking == "on"
        rss_before_kib = process_rss_kib(server_process)
        if args.cooling_seconds:
            time.sleep(args.cooling_seconds)
        initial_props = safe_backend_props(backend)
        initial_mtp = mtp_state_from_props(initial_props)
        if args.mtp_required and not (initial_mtp["requested"] or initial_mtp["usable"]):
            print("error: --mtp-required set but backend does not report MTP requested/enabled", file=sys.stderr)
            return 2

        reports: list[StepReport] = []
        for scenario in selected:
            for _repeat in range(args.repetitions):
                reports.extend(
                    run_scenario(
                        scenario,
                        backend=backend,
                        workdir=Path(args.workdir),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                )
        final_props = settled_backend_props(args.base_url, args.timeout)
        env = environment_summary(args=args, backend=backend, props=final_props)
        env["server_rss_before_kib"] = rss_before_kib
        env["server_rss_after_kib"] = process_rss_kib(server_process)
    write_jsonl(jsonl_path, env, reports)
    write_markdown(markdown_path, env, reports)
    if args.verify_final_prefix:
        failure = final_prefix_validation_failure(args.final_prefix_mode, reports, final_props)
        if failure is not None:
            print(f"error: final-prefix validation failed ({failure})", file=sys.stderr)
            return 1
    if args.mtp_required:
        final_mtp = mtp_state_from_props(final_props)
        if not final_mtp["usable"]:
            reason = final_mtp["failure_reason"] or final_mtp["status"]
            print(f"error: --mtp-required set but backend MTP is not usable ({reason})", file=sys.stderr)
            return 2
    scenario_failure = scenario_failure_reason(reports)
    if scenario_failure is not None:
        print(f"error: scenario failed ({scenario_failure})", file=sys.stderr)
        return 1
    print(f"jsonl={jsonl_path}")
    print(f"markdown={markdown_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeatable Orbit smoke scenarios and write JSONL/Markdown reports.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12120")
    parser.add_argument("--workdir", default="workdir")
    parser.add_argument("--output-dir", default="workdir/benchmarks")
    parser.add_argument("--jsonl")
    parser.add_argument("--markdown")
    parser.add_argument("--scenario", action="append", default=None, help="Scenario name, repeatable, or 'all'.")
    parser.add_argument("--no-web", action="store_true", help="Skip web-dependent scenarios.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional grep/read scenarios.")
    parser.add_argument("--mtp-required", action="store_true", help="Fail if backend props do not report MTP enabled.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument("--final-prefix-mode", choices=("inherit", "off", "on"), default="inherit")
    parser.add_argument("--manage-server", action="store_true", help="Start and stop a native server for this run.")
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--model")
    parser.add_argument("--mmproj")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--server-mtp", action="store_true", help="Start the managed server with experimental MTP enabled.")
    parser.add_argument("--server-thinking", choices=("off", "on"), default="off")
    parser.add_argument("--tools", choices=("off", "on"), default="on")
    parser.add_argument("--block-id", default=None)
    parser.add_argument("--run-order", default=None)
    parser.add_argument("--cooling-seconds", type=float, default=0.0)
    parser.add_argument("--deterministic-web", action="store_true", help="Use bounded local web fixtures for final-prefix smoke cases.")
    parser.add_argument(
        "--verify-final-prefix",
        action="store_true",
        help="Fail unless OFF/ON final-prefix counters match the requested experimental mode.",
    )
    return parser


def scenarios() -> dict[str, SmokeScenario]:
    return {
        "simple_chat": SmokeScenario(
            "simple_chat",
            (
                SmokeStep("hi", mode="chat", checker_name="nonempty"),
                SmokeStep("say hi again", mode="chat", checker_name="nonempty"),
            ),
        ),
        "pwd_followup": SmokeScenario(
            "pwd_followup",
            (
                SmokeStep("run pwd", checker_name="path_like"),
                SmokeStep("what directory was that?", checker_name="path_like"),
            ),
        ),
        "shell_error": SmokeScenario(
            "shell_error",
            (
                SmokeStep("run command_that_does_not_exist_123", checker_name="shell_error"),
                SmokeStep("what happened?", checker_name="shell_error"),
            ),
        ),
        "shell20": SmokeScenario(
            "shell20",
            (
                SmokeStep("run python3 -c 'for i in range(20): print(f\"line-{i}\")'", checker_name="shell20"),
                SmokeStep("summarize the output", checker_name="shell20_summary"),
            ),
        ),
        "web_shell": SmokeScenario(
            "web_shell",
            (
                SmokeStep("search online for information about OpenAI", checker_name="nonempty"),
                SmokeStep("run python3 -c 'for i in range(5): print(f\"line-{i}\")'", checker_name="line5"),
            ),
            requires_web=True,
        ),
        "dual_shell": SmokeScenario(
            "dual_shell",
            (
                SmokeStep("run command_that_does_not_exist_123", checker_name="shell_error"),
                SmokeStep("run python3 -c 'for i in range(20): print(f\"line-{i}\")'", checker_name="shell20"),
                SmokeStep("summarize the output", checker_name="shell20_summary"),
                SmokeStep("summarize the failed output", checker_name="shell_error_focus"),
            ),
        ),
        "grep_read": SmokeScenario(
            "grep_read",
            (
                SmokeStep("run grep -R \"class EvidenceStore\" -n src/orbit/runtime/evidence.py", checker_name="grep_evidence"),
                SmokeStep("summarize result", checker_name="nonempty"),
                SmokeStep("run python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('src/orbit/runtime/evidence.py').read_text()[:400])\nPY", checker_name="read_excerpt"),
                SmokeStep("summarize file", checker_name="nonempty"),
            ),
            optional=True,
        ),
        "final_prefix_local": SmokeScenario(
            "final_prefix_local",
            (
                SmokeStep("run pwd", checker_name="path_like"),
                SmokeStep("tell me specs about this computer", checker_name="nonempty"),
                SmokeStep("read text/summary.txt", checker_name="nonempty"),
                SmokeStep('search inside local text files for "Orbit"', checker_name="nonempty"),
                SmokeStep("list files and directories in this workdir", checker_name="nonempty"),
                SmokeStep("run printf 'orbit-final-prefix-ok\\n'", checker_name="nonempty"),
                SmokeStep("run command_that_does_not_exist_123", checker_name="shell_error"),
            ),
            optional=True,
            allowed_tool_names=TOOL_NAMES,
        ),
        "final_prefix_web": SmokeScenario(
            "final_prefix_web",
            (
                SmokeStep("search online for orbit fixture success", checker_name="nonempty"),
                SmokeStep("search online for orbit fixture none", checker_name="nonempty"),
                SmokeStep("search online for where is Avola located?", checker_name="web_error"),
                SmokeStep("search online for latest status of fictional endpoint xqz-orbit-404", checker_name="web_error"),
            ),
            requires_web=True,
            optional=True,
        ),
        "final_prefix_mixed": SmokeScenario(
            "final_prefix_mixed",
            (
                SmokeStep("run pwd", checker_name="path_like"),
                SmokeStep("tell me specs about this computer", checker_name="nonempty"),
                SmokeStep("read text/summary.txt", checker_name="nonempty"),
                SmokeStep('search inside local text files for "Orbit"', checker_name="nonempty"),
                SmokeStep("list files and directories in this workdir", checker_name="nonempty"),
                SmokeStep("run printf 'orbit-final-prefix-ok\\n'", checker_name="nonempty"),
                SmokeStep("run command_that_does_not_exist_123", checker_name="shell_error"),
                SmokeStep("search online for orbit fixture success", checker_name="nonempty"),
                SmokeStep("search online for orbit fixture none", checker_name="nonempty"),
                SmokeStep("search online for where is Avola located?", checker_name="web_error"),
                SmokeStep("search online for latest status of fictional endpoint xqz-orbit-404", checker_name="web_error"),
            ),
            requires_web=True,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
        ),
        "final_prefix_paired": SmokeScenario(
            "final_prefix_paired",
            (
                SmokeStep("run pwd", checker_name="path_like"),
                SmokeStep("tell me specs about this computer", checker_name="nonempty"),
                SmokeStep('search inside local text files for "Orbit"', checker_name="nonempty"),
                SmokeStep("search online for orbit fixture success", checker_name="nonempty"),
            ),
            requires_web=True,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
        ),
        "final_prefix_web_short": SmokeScenario(
            "final_prefix_web_short",
            (SmokeStep("search online for orbit fixture success and answer in one short sentence", checker_name="nonempty"),),
            requires_web=True,
            optional=True,
        ),
    }


def select_scenarios(selected: list[str], *, no_web: bool, include_optional: bool) -> list[SmokeScenario]:
    registry = scenarios()
    selected = selected or ["all"]
    names = list(registry) if "all" in selected else selected
    result: list[SmokeScenario] = []
    unknown = sorted(name for name in names if name not in registry)
    if unknown:
        raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
    for name in names:
        scenario = registry[name]
        if scenario.requires_web and no_web:
            continue
        if scenario.optional and not include_optional:
            continue
        result.append(scenario)
    return result


def run_scenario(
    scenario: SmokeScenario,
    *,
    backend: LlamaServerBackend,
    workdir: Path,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> list[StepReport]:
    runtime = new_runtime(backend, workdir)
    reports: list[StepReport] = []
    for index, step in enumerate(scenario.steps, start=1):
        if scenario.isolated_steps and index > 1:
            runtime = new_runtime(backend, workdir)
        report = run_step(
            runtime,
            scenario=scenario.name,
            step_index=index,
            step=step,
            workdir=workdir,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=backend.base_url,
            timeout=timeout,
            allowed_tool_names=scenario.allowed_tool_names,
        )
        reports.append(report)
        if report.finish_reason in {"timeout", "error"}:
            break
    return reports


def new_runtime(backend: LlamaServerBackend, workdir: Path) -> ChatRuntime:
    return ChatRuntime(
        backend=ProbeBackend(backend),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        diagnostic_session_id=str(workdir),
    )


def run_step(
    runtime: ChatRuntime,
    *,
    scenario: str,
    step_index: int,
    step: SmokeStep,
    workdir: Path,
    max_tokens: int,
    temperature: float,
    base_url: str,
    timeout: float,
    allowed_tool_names: tuple[str, ...] = ("exec_shell_full_command",),
) -> StepReport:
    props_before = fresh_backend_props(base_url, min(timeout, 5.0))
    result_queue: queue.Queue[StepReport] = queue.Queue(maxsize=1)

    worker = threading.Thread(
        target=lambda: result_queue.put(
            _run_step_inner(
                runtime,
                scenario=scenario,
                step_index=step_index,
                step=step,
                workdir=workdir,
                max_tokens=max_tokens,
                temperature=temperature,
                allowed_tool_names=allowed_tool_names,
            )
        ),
        daemon=True,
    )
    started = time.perf_counter()
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        cancel_requested = request_backend_cancel(base_url, timeout=min(5.0, max(1.0, timeout)))
        props_after = wait_for_backend_idle(base_url, timeout=min(5.0, max(1.0, timeout)))
        worker.join(min(2.0, max(0.5, timeout / 10.0)))
        notes = "timeout"
        if cancel_requested:
            notes += ",cancel_requested"
        if props_after.get("in_flight") is False:
            notes += ",cleanup_ok"
        elif props_after:
            notes += ",cleanup_pending"
        return StepReport(
            case=scenario,
            step=step_index,
            prompt=step.prompt,
            prompt_kind=step.mode,
            completion_kind="timeout",
            route_tokens=None,
            final_tokens=None,
            prompt_tokens=None,
            cached_tokens=None,
            evaluated_tokens=None,
            output_tokens=None,
            finish_reason="timeout",
            tool_calls=0,
            tool_names=[],
            wall_ms=round((time.perf_counter() - started) * 1000, 1),
            correctness_category="not_evaluated",
            raw_leak=False,
            fake_output=False,
            loop=False,
            notes=notes,
            answer_excerpt="timeout",
            model_steps=[],
            final_prefix=final_prefix_step_state(props_before, props_after),
            lifecycle={
                "event": "timeout",
                "timeout_observed": True,
                "automatic_cancel": False,
                "explicit_cancel_used": cancel_requested,
                "cleanup_healthy": props_after.get("in_flight") is False,
            },
            phase_wall_ms={},
        )
    report = result_queue.get()
    props_after = fresh_backend_props(base_url, min(timeout, 5.0))
    return replace_step_final_prefix(report, final_prefix_step_state(props_before, props_after))


def _run_step_inner(
    runtime: ChatRuntime,
    *,
    scenario: str,
    step_index: int,
    step: SmokeStep,
    workdir: Path,
    max_tokens: int,
    temperature: float,
    allowed_tool_names: tuple[str, ...],
) -> StepReport:
    model_steps: list[ModelStepMetrics] = []
    tool_names: list[str] = []
    started = time.perf_counter()
    probe = runtime.backend if isinstance(runtime.backend, ProbeBackend) else None
    if probe is not None:
        probe.reset_phase_timings()
    notes = ""
    try:
        if step.mode == "chat":
            result = runtime.ask_chat(
                step.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_model_step=model_steps.append,
            )
        else:
            result = runtime.ask_auto(
                step.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workdir=workdir,
                allowed_tool_names=allowed_tool_names,
                on_model_step=model_steps.append,
                on_tool_call=lambda name, _args: tool_names.append(name),
            )
    except Exception as exc:
        result = ChatResult(
            content=f"{type(exc).__name__}: {exc}",
            model=None,
            finish_reason="error",
            tool_calls=[],
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            prompt_tokens_per_second=None,
            generation_tokens_per_second=None,
        )
        notes = "exception"
    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    phase_timings = phase_timing_summary(probe.phase_timings() if probe is not None else [], wall_ms)
    final_wall_ms = phase_duration(phase_timings, "final_from_tool")
    generation_ms = estimated_generation_ms(result.completion_tokens, result.generation_tokens_per_second)
    residual_ms = (
        round(max(0.0, final_wall_ms - generation_ms), 1)
        if final_wall_ms is not None and generation_ms is not None
        else None
    )
    completion_kind = ",".join(metric.phase for metric in model_steps) or "error"
    route_tokens = first_prompt_tokens(model_steps, route=True)
    final_tokens = first_prompt_tokens(model_steps, route=False, last=True)
    evaluated = result.prompt_tokens - result.cached_tokens if result.prompt_tokens is not None and result.cached_tokens is not None else None
    checker = CHECKERS.get(step.checker_name, check_not_evaluated)
    category = checker(result.content, tool_names)
    raw_leak = detect_raw_leak(result.content)
    fake_output = category == "fake_tool_output"
    loop = len(model_steps) > 8
    if result.finish_reason == "length":
        category = "length_failure" if category not in {"wrong", "mixed_wrong"} else category
    return StepReport(
        case=scenario,
        step=step_index,
        prompt=step.prompt,
        prompt_kind=step.mode,
        completion_kind=completion_kind,
        route_tokens=route_tokens,
        final_tokens=final_tokens,
        prompt_tokens=result.prompt_tokens,
        cached_tokens=result.cached_tokens,
        evaluated_tokens=evaluated,
        output_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
        tool_calls=len(tool_names),
        tool_names=tool_names,
        wall_ms=wall_ms,
        correctness_category="raw_leak" if raw_leak else category,
        raw_leak=raw_leak,
        fake_output=fake_output,
        loop=loop,
        notes=notes,
        answer_excerpt=excerpt(result.content),
        model_steps=[model_step_to_json(metric) for metric in model_steps],
        final_prefix={},
        lifecycle={},
        phase_wall_ms=phase_timings,
        prompt_tokens_per_second=result.prompt_tokens_per_second,
        generation_tokens_per_second=result.generation_tokens_per_second,
        estimated_generation_ms=generation_ms,
        estimated_prefill_residual_ms=residual_ms,
    )


def phase_timing_summary(timings: list[tuple[str, float]], total_wall_ms: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase, duration in timings:
        result[phase] = round(result.get(phase, 0.0) + duration, 1)
    model_wall = sum(result.values())
    result["non_model_wall_ms"] = round(max(0.0, total_wall_ms - model_wall), 1)
    return result


def estimated_generation_ms(output_tokens: int | None, tokens_per_second: float | None) -> float | None:
    if not isinstance(output_tokens, int) or not isinstance(tokens_per_second, int | float) or tokens_per_second <= 0:
        return None
    return round(output_tokens / float(tokens_per_second) * 1000, 1)


def first_prompt_tokens(metrics: list[ModelStepMetrics], *, route: bool, last: bool = False) -> int | None:
    route_phases = {"route", "tool_call", "tool_call_retry"}
    filtered = [metric for metric in metrics if (metric.phase in route_phases) == route and metric.prompt_tokens is not None]
    if not filtered:
        return None
    return filtered[-1].prompt_tokens if last else filtered[0].prompt_tokens


def model_step_to_json(metric: ModelStepMetrics) -> dict[str, object]:
    evaluated = metric.prompt_tokens - metric.cached_tokens if metric.prompt_tokens is not None and metric.cached_tokens is not None else None
    return {
        "loop": metric.loop,
        "phase": metric.phase,
        "finish_reason": metric.finish_reason,
        "prompt_tokens": metric.prompt_tokens,
        "completion_tokens": metric.completion_tokens,
        "cached_tokens": metric.cached_tokens,
        "evaluated_tokens": evaluated,
        "tool_calls": metric.tool_calls,
        "retry_reason": metric.retry_reason,
        "prompt_tokens_per_second": metric.prompt_tokens_per_second,
        "generation_tokens_per_second": metric.generation_tokens_per_second,
    }


def environment_summary(*, args: argparse.Namespace, backend: LlamaServerBackend, props: dict[str, object] | None = None) -> dict[str, object]:
    props = props if props is not None else safe_backend_props(backend)
    info = safe_model_info(backend)
    host = collect_host_info()
    mtp = mtp_state_from_props(props)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_head": git_head(),
        "version": __version__,
        "model": getattr(info, "id", None) or props.get("model_id") or "unknown",
        "backend": props.get("backend") or "unknown",
        "mtp": mtp["status"],
        "mtp_requested": mtp["requested"],
        "mtp_session_ready": mtp["session_ready"],
        "mtp_usable": mtp["usable"],
        "mtp_failure_reason": mtp["failure_reason"],
        "mmproj": "loaded" if props.get("multimodal_available") is True else ("missing" if props else "unknown"),
        "cpu": host.cpu,
        "cores": {"physical": host.physical_cores, "logical": host.logical_cores},
        "ram": {"total": host.ram_total, "available": host.ram_available},
        "accel": acceleration_mode(props),
        "base_url": args.base_url,
        "workdir": str(Path(args.workdir)),
        "scenario": list(args.scenario or ["all"]),
        "final_prefix_mode": args.final_prefix_mode,
        "tools": args.tools,
        "prewarm": os.environ.get("ORBIT_KV_PREFIX_PREWARM", "startup"),
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ctx": props.get("ctx_size", args.ctx),
        "threads": props.get("threads", args.threads),
        "threads_batch": props.get("threads_batch", args.threads_batch),
        "batch": props.get("batch_size", args.batch),
        "ubatch": props.get("ubatch_size", args.ubatch),
        "server_command": server_command(args) if args.manage_server else "external",
        "client_command": " ".join(sys.argv),
        "final_prefix": final_prefix_props(props),
        "block_id": args.block_id,
        "run_order": args.run_order,
        "cooling_seconds": args.cooling_seconds,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


FINAL_PREFIX_PROP_KEYS = {
    "enabled": "final_prefix_experiment_enabled",
    "initialized": "final_prefix_experiment_initialized",
    "prefix_tokens": "final_prefix_experiment_prefix_tokens",
    "capture_count": "final_prefix_experiment_capture_count",
    "restore_count": "final_prefix_experiment_restore_count",
    "fallback_count": "final_prefix_experiment_fallback_count",
    "failure_reason": "final_prefix_experiment_failure_reason",
    "last_used": "final_prefix_experiment_last_used",
    "checkpoint_size": "final_prefix_experiment_checkpoint_size_bytes",
}


def final_prefix_props(props: dict[str, object]) -> dict[str, object]:
    return {name: props.get(source) for name, source in FINAL_PREFIX_PROP_KEYS.items()}


def final_prefix_step_state(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    state = final_prefix_props(after)
    for name in ("capture_count", "restore_count", "fallback_count"):
        old = final_prefix_props(before).get(name)
        new = state.get(name)
        state[f"{name}_delta"] = new - old if isinstance(old, int) and isinstance(new, int) else None
    return state


def final_prefix_validation_failure(
    mode: str,
    reports: list[StepReport],
    props: dict[str, object],
) -> str | None:
    state = final_prefix_props(props)
    eligible = [report for report in reports if "final_from_tool" in report.completion_kind]
    if mode == "off":
        if state.get("enabled") is not False:
            return "off_mode_enabled"
        if any((report.final_prefix.get("capture_count_delta") or 0) > 0 for report in eligible):
            return "off_mode_capture"
        return None
    if mode != "on":
        return "explicit_mode_required"
    if state.get("enabled") is not True:
        return "on_mode_disabled"
    if props.get("mtp_experimental_enabled") is True:
        if (state.get("capture_count") or 0) != 0 or (state.get("restore_count") or 0) != 0:
            return "mtp_guard_failed"
        return None
    if state.get("fallback_count") not in {0, None}:
        return "fallback_observed"
    if state.get("prefix_tokens") != 43:
        return "unexpected_prefix_tokens"
    if (state.get("capture_count") or 0) < 1:
        return "capture_missing"
    if len(eligible) >= 2 and (state.get("restore_count") or 0) < 1:
        return "restore_missing"
    return None


def replace_step_final_prefix(report: StepReport, state: dict[str, object]) -> StepReport:
    values = report.__dict__.copy()
    values["final_prefix"] = state
    return StepReport(**values)


@contextmanager
def final_prefix_environment(mode: str):
    previous = os.environ.get("ORBIT_FINAL_PREFIX_EXPERIMENT")
    if mode == "on":
        os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"] = "1"
    elif mode == "off":
        os.environ.pop("ORBIT_FINAL_PREFIX_EXPERIMENT", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ORBIT_FINAL_PREFIX_EXPERIMENT", None)
        else:
            os.environ["ORBIT_FINAL_PREFIX_EXPERIMENT"] = previous


@contextmanager
def deterministic_web(enabled: bool):
    if not enabled:
        yield
        return
    from orbit.runtime import shell_guardrails

    original = shell_guardrails.search_web

    def fixture(query: str, *, max_results: int = 5) -> str:
        del max_results
        lower = query.lower()
        if "fixture success" in lower:
            return "\n".join(
                [
                    "web_search_results: true",
                    f"query: {query}",
                    "results:",
                    "1. title: Orbit deterministic fixture",
                    "   url: https://example.invalid/orbit-fixture",
                    "   snippet: Orbit fixture result for bounded web final validation.",
                ]
            )
        if "fixture none" in lower:
            return "web_search_results: true\nresults: none"
        return "error: web search failed: deterministic DNS fixture"

    shell_guardrails.search_web = fixture
    try:
        yield
    finally:
        shell_guardrails.search_web = original


def server_command(args: argparse.Namespace) -> list[str]:
    from urllib.parse import urlparse

    parsed = urlparse(args.base_url)
    command = [
        sys.executable, "-m", "orbit.terminal.cli", "server",
        "--host", parsed.hostname or "127.0.0.1",
        "--port", str(parsed.port or 12120),
        "--ctx", str(args.ctx),
        "--threads", str(args.threads),
        "--threads-batch", str(args.threads_batch),
        "--batch", str(args.batch),
        "--ubatch", str(args.ubatch),
        "--think", args.server_thinking,
    ]
    if args.model:
        command.extend(("--model", args.model))
    if args.mmproj:
        command.extend(("--mmproj", args.mmproj))
    if args.server_mtp:
        command.append("--mtp")
    return command


@contextmanager
def managed_server(args: argparse.Namespace):
    if not args.manage_server:
        yield None
        return
    if fresh_backend_props(args.base_url, 1.0):
        raise RuntimeError(f"managed server requires an unused base URL: {args.base_url}")
    env = os.environ.copy()
    env["ORBIT_TOOLS"] = args.tools
    if args.final_prefix_mode == "on":
        env["ORBIT_FINAL_PREFIX_EXPERIMENT"] = "1"
    else:
        env.pop("ORBIT_FINAL_PREFIX_EXPERIMENT", None)
    process = subprocess.Popen(
        server_command(args),
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_server(args.base_url, process, args.server_start_timeout)
        props = fresh_backend_props(args.base_url, 2.0)
        expected_enabled = args.final_prefix_mode == "on"
        if props.get("final_prefix_experiment_enabled") is not expected_enabled:
            raise RuntimeError("managed server final-prefix mode does not match the requested mode")
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"managed server exited during startup: {excerpt(output, 500)}")
        if fresh_backend_props(base_url, 2.0):
            time.sleep(0.1)
            if process.poll() is None:
                return
        time.sleep(0.25)
    raise TimeoutError(f"managed server did not become ready within {timeout:.0f}s")


def process_rss_kib(process: subprocess.Popen[str] | None) -> int | None:
    if process is None:
        return None
    try:
        text = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            return int(fields[1]) if len(fields) >= 2 and fields[1].isdigit() else None
    return None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def safe_backend_props(backend: LlamaServerBackend) -> dict[str, object]:
    try:
        return backend.backend_props()
    except Exception:
        return {}


def fresh_backend_props(base_url: str, timeout: float) -> dict[str, object]:
    try:
        return LlamaServerBackend(base_url=base_url, timeout=timeout).backend_props()
    except Exception:
        return {}


def request_backend_cancel(base_url: str, timeout: float) -> bool:
    request = Request(
        f"{base_url.rstrip('/')}/cancel",
        data=json.dumps({"session_id": "default"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return False
    return payload.get("status") == "cancel_requested"


def wait_for_backend_idle(base_url: str, timeout: float, *, poll_interval: float = 0.2) -> dict[str, object]:
    deadline = time.monotonic() + max(0.0, timeout)
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        latest = fresh_backend_props(base_url, timeout=min(2.0, max(0.5, timeout)))
        if latest and latest.get("in_flight") is False:
            return latest
        time.sleep(poll_interval)
    return latest


def settled_backend_props(base_url: str, timeout: float, *, settle_seconds: float = 3.0, poll_interval: float = 0.2) -> dict[str, object]:
    props = fresh_backend_props(base_url, timeout)
    if not props:
        return {}
    deadline = time.monotonic() + max(0.0, settle_seconds)
    last_serialized = json.dumps(props, sort_keys=True, ensure_ascii=False)
    while time.monotonic() < deadline:
        if props.get("in_flight") is False:
            failure_reason = first_nonempty_str(props.get("mtp_failure_reason"), props.get("mtp_fallback_reason"))
            if failure_reason not in {"cancelled", "timeout"}:
                return props
        time.sleep(poll_interval)
        refreshed = fresh_backend_props(base_url, timeout=min(2.0, max(0.5, timeout)))
        if not refreshed:
            return props
        serialized = json.dumps(refreshed, sort_keys=True, ensure_ascii=False)
        props = refreshed
        if serialized == last_serialized and props.get("in_flight") is False:
            return props
        last_serialized = serialized
    return props


def safe_model_info(backend: LlamaServerBackend):
    try:
        return backend.model_info()
    except Exception:
        return None


def scenario_failure_reason(reports: list[StepReport]) -> str | None:
    for report in reports:
        if report.finish_reason == "timeout" or "timeout" in report.notes:
            return "timeout"
        if report.notes == "exception":
            return "error"
        if report.finish_reason == "error":
            return "error"
    return None


def git_head() -> str:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def on_off(value: object) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "unknown"


def first_nonempty_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def acceleration_mode(props: dict[str, object]) -> str:
    if props.get("gpu_layers"):
        return "gpu"
    if props.get("backend") == "orbit-native":
        return "CPU-only"
    return "unknown"


def write_jsonl(path: Path, env: dict[str, object], reports: list[StepReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "environment", **env}, ensure_ascii=False, sort_keys=True) + "\n")
        for report in reports:
            handle.write(json.dumps({"type": "step", **report.to_json()}, ensure_ascii=False, sort_keys=True) + "\n")
        for summary in summarize_reports(reports):
            handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, env: dict[str, object], reports: list[StepReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Orbit Smoke Harness",
        "",
        "## Environment",
        "",
        f"- Version: `{env.get('version')}`",
        f"- Git HEAD: `{env.get('git_head')}`",
        f"- Backend: `{env.get('backend')}`",
        f"- Model: `{env.get('model')}`",
        f"- MTP: `{env.get('mtp')}`",
        f"- mmproj: `{env.get('mmproj')}`",
        f"- Workdir: `{env.get('workdir')}`",
        "",
        "## Results",
        "",
        "| Case | Step | Kind | Route | Final | Prompt | Cached | Eval | Tools | Finish | Correctness | Wall ms | Notes |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for report in reports:
        lines.append(
            "| {case} | {step} | {kind} | {route} | {final} | {prompt} | {cached} | {evaluated} | {tools} | {finish} | {correctness} | {wall} | {notes} |".format(
                case=report.case,
                step=report.step,
                kind=report.completion_kind,
                route=value(report.route_tokens),
                final=value(report.final_tokens),
                prompt=value(report.prompt_tokens),
                cached=value(report.cached_tokens),
                evaluated=value(report.evaluated_tokens),
                tools=report.tool_calls,
                finish=report.finish_reason or "-",
                correctness=report.correctness_category,
                wall=report.wall_ms,
                notes=report.notes or "-",
            )
        )
    summaries = summarize_reports(reports)
    if summaries:
        lines.extend(
            [
                "",
                "## Repetition Summary",
                "",
                "| Case | Step | Runs | Correct | Cached min/median/max | Eval min/median/max | Wall ms min/median/max |",
                "| --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for summary in summaries:
            lines.append(
                f"| {summary['case']} | {summary['step']} | {summary['runs']} | {summary['correct']} | "
                f"{format_range(summary['cached_tokens'])} | {format_range(summary['evaluated_tokens'])} | "
                f"{format_range(summary['wall_ms'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_reports(reports: list[StepReport]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[StepReport]] = {}
    for report in reports:
        grouped.setdefault((report.case, report.step), []).append(report)
    result: list[dict[str, object]] = []
    for (case, step), rows in grouped.items():
        result.append(
            {
                "case": case,
                "step": step,
                "runs": len(rows),
                "correct": sum(row.correctness_category == "correct" for row in rows),
                "cached_tokens": numeric_range(row.cached_tokens for row in rows),
                "evaluated_tokens": numeric_range(row.evaluated_tokens for row in rows),
                "output_tokens": numeric_range(row.output_tokens for row in rows),
                "wall_ms": numeric_range(row.wall_ms for row in rows),
                "route_wall_ms": numeric_range(row.phase_wall_ms.get("route") for row in rows),
                "final_wall_ms": numeric_range(
                    phase_duration(row.phase_wall_ms, "final_from_tool")
                    for row in rows
                ),
                "non_model_wall_ms": numeric_range(row.phase_wall_ms.get("non_model_wall_ms") for row in rows),
                "prompt_tokens_per_second": numeric_range(row.prompt_tokens_per_second for row in rows),
                "generation_tokens_per_second": numeric_range(row.generation_tokens_per_second for row in rows),
                "estimated_generation_ms": numeric_range(row.estimated_generation_ms for row in rows),
                "estimated_prefill_residual_ms": numeric_range(row.estimated_prefill_residual_ms for row in rows),
                "capture_delta": sum(int(row.final_prefix.get("capture_count_delta") or 0) for row in rows),
                "restore_delta": sum(int(row.final_prefix.get("restore_count_delta") or 0) for row in rows),
                "fallback_delta": sum(int(row.final_prefix.get("fallback_count_delta") or 0) for row in rows),
                "restored": summarize_restored_rows(rows),
            }
        )
    return result


def summarize_restored_rows(rows: list[StepReport]) -> dict[str, object]:
    restored = [row for row in rows if (row.final_prefix.get("restore_count_delta") or 0) > 0]
    return {
        "runs": len(restored),
        "cached_tokens": numeric_range(row.cached_tokens for row in restored),
        "evaluated_tokens": numeric_range(row.evaluated_tokens for row in restored),
        "output_tokens": numeric_range(row.output_tokens for row in restored),
        "wall_ms": numeric_range(row.wall_ms for row in restored),
        "final_wall_ms": numeric_range(
            phase_duration(row.phase_wall_ms, "final_from_tool")
            for row in restored
        ),
    }


def phase_duration(timings: dict[str, float], prefix: str) -> float | None:
    values = [value for phase, value in timings.items() if phase.startswith(prefix)]
    return round(sum(values), 1) if values else None


def numeric_range(values) -> dict[str, float] | None:
    present = [float(item) for item in values if isinstance(item, int | float)]
    if not present:
        return None
    return {
        "min": round(min(present), 1),
        "median": round(statistics.median(present), 1),
        "max": round(max(present), 1),
    }


def format_range(value: object) -> str:
    if not isinstance(value, dict):
        return "-"
    return f"{value.get('min')}/{value.get('median')}/{value.get('max')}"


def value(item: object) -> object:
    return "-" if item is None else item


def excerpt(text: str, limit: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def detect_raw_leak(text: str) -> bool:
    return len(text) > 5000 or text.count("line-") > 80


def check_not_evaluated(_text: str, _tools: list[str]) -> str:
    return "not_evaluated"


def check_nonempty(text: str, _tools: list[str]) -> str:
    return "correct" if text.strip() else "wrong"


def check_path_like(text: str, _tools: list[str]) -> str:
    if "/" in text:
        return "correct"
    return "partial_baseline" if text.strip() else "wrong"


def check_shell_error(text: str, _tools: list[str]) -> str:
    lower = text.lower()
    if "127" in lower or "not found" in lower or "command not found" in lower:
        return "correct"
    return "wrong" if text.strip() else "not_evaluated"


def check_shell20(text: str, tools: list[str]) -> str:
    lower = text.lower()
    if "exec_shell_full_command" not in tools:
        return "fake_tool_output" if "line-19" in lower else "wrong"
    if "line-0" in lower or "line-19" in lower or "20" in lower:
        return "correct"
    return "partial_baseline"


def check_shell20_summary(text: str, _tools: list[str]) -> str:
    lower = text.lower()
    if "not found" in lower or "127" in lower:
        return "mixed_wrong"
    if "line-" in lower or "20" in lower:
        return "correct"
    return "partial_baseline" if text.strip() else "wrong"


def check_shell_error_focus(text: str, _tools: list[str]) -> str:
    lower = text.lower()
    mentions_error = "127" in lower or "not found" in lower or "failed" in lower
    mentions_lines = "line-0" in lower or "line-19" in lower
    if mentions_error and mentions_lines:
        return "mixed_wrong"
    if mentions_error:
        return "correct"
    return "wrong" if text.strip() else "not_evaluated"


def check_line5(text: str, _tools: list[str]) -> str:
    lower = text.lower()
    return "correct" if "line-0" in lower or "line-4" in lower or "5" in lower else "partial_baseline"


def check_grep_evidence(text: str, _tools: list[str]) -> str:
    return "correct" if "EvidenceStore" in text or "evidence.py" in text else "partial_baseline"


def check_read_excerpt(text: str, _tools: list[str]) -> str:
    return "correct" if "EvidenceStore" in text or "from __future__" in text else "partial_baseline"


def check_web_error(text: str, _tools: list[str]) -> str:
    lower = text.lower()
    reports_failure = "fail" in lower or "error" in lower or "could not" in lower or "unable" in lower
    answers_from_memory = "avola is" in lower or "avola, sicily" in lower or "province of syracuse" in lower
    return "correct" if reports_failure and not answers_from_memory else "wrong"


CHECKERS: dict[str, CorrectnessChecker] = {
    "not_evaluated": check_not_evaluated,
    "nonempty": check_nonempty,
    "path_like": check_path_like,
    "shell_error": check_shell_error,
    "shell20": check_shell20,
    "shell20_summary": check_shell20_summary,
    "shell_error_focus": check_shell_error_focus,
    "line5": check_line5,
    "grep_evidence": check_grep_evidence,
    "read_excerpt": check_read_excerpt,
    "web_error": check_web_error,
}


if __name__ == "__main__":
    raise SystemExit(main())

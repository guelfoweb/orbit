#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit import __version__
from orbit.backend.base import ChatResult
from orbit.backend.llama_server import LlamaServerBackend, LlamaServerError
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT
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
        }


class ProbeBackend:
    def __init__(self, backend: LlamaServerBackend) -> None:
        self._backend = backend

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        return self._backend.chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta=None, on_progress=None):
        return self._backend.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            on_delta=on_delta or (lambda _text: None),
            on_progress=on_progress,
        )

    def continue_current(self, *, max_tokens, on_delta=None, on_progress=None):
        return self._backend.continue_current(max_tokens=max_tokens, on_delta=on_delta, on_progress=on_progress)


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

    backend = LlamaServerBackend(base_url=args.base_url, timeout=args.timeout)
    initial_props = safe_backend_props(backend)
    initial_mtp = mtp_state_from_props(initial_props)
    if args.mtp_required and not (initial_mtp["requested"] or initial_mtp["usable"]):
        print("error: --mtp-required set but backend does not report MTP requested/enabled", file=sys.stderr)
        return 2

    reports: list[StepReport] = []
    for scenario in selected:
        reports.extend(
            run_scenario(
                scenario,
                backend=backend,
                workdir=Path(args.workdir),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )
    env = environment_summary(args=args, backend=backend, props=fresh_backend_props(args.base_url, args.timeout))
    write_jsonl(jsonl_path, env, reports)
    write_markdown(markdown_path, env, reports)
    if args.mtp_required:
        final_mtp = mtp_state_from_props(fresh_backend_props(args.base_url, args.timeout))
        if not final_mtp["usable"]:
            reason = final_mtp["failure_reason"] or final_mtp["status"]
            print(f"error: --mtp-required set but backend MTP is not usable ({reason})", file=sys.stderr)
            return 2
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
) -> list[StepReport]:
    runtime = ChatRuntime(
        backend=ProbeBackend(backend),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        diagnostic_session_id=str(workdir),
    )
    reports: list[StepReport] = []
    for index, step in enumerate(scenario.steps, start=1):
        reports.append(
            run_step(
                runtime,
                scenario=scenario.name,
                step_index=index,
                step=step,
                workdir=workdir,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    return reports


def run_step(
    runtime: ChatRuntime,
    *,
    scenario: str,
    step_index: int,
    step: SmokeStep,
    workdir: Path,
    max_tokens: int,
    temperature: float,
) -> StepReport:
    model_steps: list[ModelStepMetrics] = []
    tool_names: list[str] = []
    started = time.perf_counter()
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
                allowed_tool_names=("exec_shell_full_command",),
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
    )


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
    }


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


def safe_model_info(backend: LlamaServerBackend):
    try:
        return backend.model_info()
    except Exception:
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
}


if __name__ == "__main__":
    raise SystemExit(main())

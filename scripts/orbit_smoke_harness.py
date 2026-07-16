#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
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
from orbit.final_prefix_config import (
    FINAL_PREFIX_EXPERIMENT_ENV,
    FINAL_PREFIX_REUSE_ENV,
    FINAL_PREFIX_TOKEN_COUNT,
    resolve_final_prefix_reuse,
)
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.completion_budget import resolve_max_tokens
from orbit.runtime.evidence import EvidenceStore
from orbit.runtime.kv_diag import current_phase
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.messages import DEFAULT_SYSTEM_PROMPT, with_tool_call_system_prompt
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_healing import analyze_tool_attempt, tool_call_healing_status
from orbit.runtime.tools import TOOL_NAMES, tool_definitions
from orbit.runtime.turn_trace import ModelStepMetrics
from orbit.terminal.runtime_status import collect_host_info
from orbit.tool_contract_config import TOOL_CALL_CANONICAL_GATE_ENV, resolve_tool_call_canonical_gate
from orbit.tool_healing_config import TOOL_CALL_HEALING_ENV, resolve_tool_call_healing


CorrectnessChecker = Callable[[str, list[str]], str]


@dataclass(frozen=True)
class SmokeStep:
    prompt: str
    mode: str = "auto"
    checker_name: str = "not_evaluated"
    expected_route: str | None = None
    expected_tool_names: tuple[str, ...] | None = None
    expected_active_outcome: str | None = None


FINAL_PREFIX_LOCAL_STEPS = (
    SmokeStep("run pwd", checker_name="path_like"),
    SmokeStep("tell me specs about this computer", checker_name="nonempty"),
    SmokeStep("read text/summary.txt", checker_name="nonempty"),
    SmokeStep('search inside local text files for "Orbit"', checker_name="nonempty"),
    SmokeStep("list files and directories in this workdir", checker_name="nonempty"),
    SmokeStep("run printf 'orbit-final-prefix-ok\\n'", checker_name="nonempty"),
    SmokeStep("run command_that_does_not_exist_123", checker_name="shell_error"),
)

FINAL_PREFIX_WEB_STEPS = (
    SmokeStep("search online for orbit fixture success", checker_name="nonempty"),
    SmokeStep("search online for orbit fixture none", checker_name="nonempty"),
    SmokeStep("search online for where is Avola located?", checker_name="web_error"),
    SmokeStep("search online for latest status of fictional endpoint xqz-orbit-404", checker_name="web_error"),
)

FINAL_PREFIX_PAIRED_STEPS = (
    FINAL_PREFIX_LOCAL_STEPS[0],
    FINAL_PREFIX_LOCAL_STEPS[1],
    FINAL_PREFIX_LOCAL_STEPS[3],
    FINAL_PREFIX_WEB_STEPS[0],
)

ROUTE_PWD_STEP = SmokeStep(
    "run pwd", checker_name="path_like", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)
ROUTE_SYSTEM_INFO_STEP = SmokeStep(
    "tell me specs about this computer", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("system_info",)
)
ROUTE_READ_STEP = SmokeStep(
    "read route_fixture.txt", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)
ROUTE_REFRESH_STEP = SmokeStep(
    "refresh the current computer specs", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("system_info",)
)
ROUTE_VERIFY_STEP = SmokeStep(
    "verify whether route_fixture.txt changed", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)
ROUTE_SHELL_ERROR_STEP = SmokeStep(
    "run command_that_does_not_exist_123", checker_name="shell_error", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)
ROUTE_RECOVERY_SUCCESS_STEP = SmokeStep(
    "run printf 'route-recovery-success\\n'", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)
ROUTE_ERROR_SUCCESS_COMPARE_STEP = SmokeStep(
    "compare the failed command with the successful command", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()
)
ROUTE_WEB_ERROR_STEP = SmokeStep(
    "search online for where is Avola located?", checker_name="web_error", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)
)

ROUTE_CLASS_RECAP_STEPS = (
    ROUTE_PWD_STEP,
    ROUTE_SYSTEM_INFO_STEP,
    SmokeStep("summarize the specs you gave me", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
)

ROUTE_CLASS_CHAT_STEPS = (
    SmokeStep("explain CPU versus GPU in one short sentence", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    *ROUTE_CLASS_RECAP_STEPS,
    SmokeStep("summarize our whole discussion", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    SmokeStep("explain why local context can matter", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    SmokeStep("summarize that", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    SmokeStep("compare the directory and computer specifications", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    SmokeStep("run printf 'route-third-result\\n'", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    SmokeStep("compare the first and third tool results", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
)

ROUTE_CLASS_LOCAL_TOOL_STEPS = (
    ROUTE_PWD_STEP,
    ROUTE_SYSTEM_INFO_STEP,
    ROUTE_READ_STEP,
    SmokeStep('search route_fixture.txt for "route-fixture-match"', checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    SmokeStep("list files and directories in this workdir", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("list_directory",)),
    SmokeStep("run printf 'route-shell-success\\n'", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    ROUTE_SHELL_ERROR_STEP,
    ROUTE_REFRESH_STEP,
    ROUTE_VERIFY_STEP,
)

ROUTE_CLASS_WEB_STEPS = (
    SmokeStep("search online for orbit fixture success", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    SmokeStep("search online for orbit fixture none", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    ROUTE_WEB_ERROR_STEP,
    SmokeStep("fetch https://example.com", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("fetch_url",)),
    SmokeStep("search online for new information about orbit fixture success", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
)

ROUTE_CLASS_EVIDENCE_STEPS = (
    ROUTE_SHELL_ERROR_STEP,
    ROUTE_RECOVERY_SUCCESS_STEP,
    ROUTE_ERROR_SUCCESS_COMPARE_STEP,
    SmokeStep("run python3 -c 'for i in range(40): print(f\"route-long-summary-{i}\")'", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    SmokeStep("summarize that output", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
    SmokeStep("read route_long_excerpt.txt", checker_name="nonempty", expected_route="FILESYSTEM", expected_tool_names=("exec_shell_full_command",)),
    SmokeStep("explain the excerpt", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
)

ROUTE_CLASS_AMBIGUOUS_STEPS = (
    ROUTE_PWD_STEP,
    ROUTE_SYSTEM_INFO_STEP,
    SmokeStep("summarize that", checker_name="nonempty", expected_route="CHAT", expected_tool_names=()),
)

ROUTE_CLASS_REFRESH_STEPS = (
    ROUTE_SYSTEM_INFO_STEP,
    ROUTE_REFRESH_STEP,
)

ROUTE_CLASS_VERIFY_STEPS = (
    ROUTE_READ_STEP,
    ROUTE_VERIFY_STEP,
)

ROUTE_CLASS_ERROR_SUCCESS_STEPS = (
    ROUTE_SHELL_ERROR_STEP,
    ROUTE_RECOVERY_SUCCESS_STEP,
    ROUTE_ERROR_SUCCESS_COMPARE_STEP,
)

ROUTE_CLASS_WEB_ERROR_STEPS = (
    ROUTE_WEB_ERROR_STEP,
)

TOOL_HEALING_STEPS = (
    SmokeStep("show the computer specifications", mode="tool", checker_name="nonempty", expected_tool_names=("system_info",), expected_active_outcome="executed"),
    SmokeStep("list this directory", mode="tool", checker_name="nonempty", expected_tool_names=("list_directory",), expected_active_outcome="executed"),
    SmokeStep("read route_fixture.txt", mode="tool", checker_name="nonempty", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="executed"),
    SmokeStep('search route_fixture.txt for "route-fixture-match"', mode="tool", checker_name="nonempty", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="executed"),
    SmokeStep("run printf 'healing-ok\\n'", mode="tool", checker_name="nonempty", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="executed"),
    SmokeStep("without changing files, delete route_fixture.txt", mode="tool", checker_name="nonempty", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="rejected_policy"),
    SmokeStep("fetch http://127.0.0.1:9/orbit-healing", mode="tool", checker_name="nonempty", expected_tool_names=("fetch_url",), expected_active_outcome="runtime_error"),
    SmokeStep("fetch file:///etc/passwd", mode="tool", checker_name="nonempty", expected_tool_names=("fetch_url",), expected_active_outcome="runtime_error"),
    SmokeStep("run pwd and answer briefly", mode="tool", checker_name="path_like", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="executed"),
    SmokeStep("run printf 'near-limit-%080d\\n' 1", mode="tool", checker_name="nonempty", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="executed"),
    SmokeStep("Explain in one sentence that system_info is a tool name; do not call tools.", mode="tool", checker_name="nonempty", expected_tool_names=(), expected_active_outcome="executed"),
    SmokeStep('Show this JSON example literally: {"name":"system_info","arguments":{}}. Do not call it.', mode="tool", checker_name="nonempty", expected_tool_names=(), expected_active_outcome="executed"),
    SmokeStep("Call system_info and list_directory in the same response.", mode="tool", checker_name="nonempty", expected_tool_names=("system_info", "list_directory"), expected_active_outcome="executed"),
)
TOOL_HEALING_QUICK_STEPS = (
    TOOL_HEALING_STEPS[8],
    TOOL_HEALING_STEPS[1],
    TOOL_HEALING_STEPS[10],
    TOOL_HEALING_STEPS[11],
    TOOL_HEALING_STEPS[12],
)
CANONICAL_GATE_REAL_STEPS = (
    TOOL_HEALING_STEPS[0],
    TOOL_HEALING_STEPS[1],
    TOOL_HEALING_STEPS[2],
    TOOL_HEALING_STEPS[3],
    TOOL_HEALING_STEPS[4],
    SmokeStep("run command_that_does_not_exist_123", mode="tool", checker_name="shell_error", expected_tool_names=("exec_shell_full_command",), expected_active_outcome="runtime_error"),
    TOOL_HEALING_STEPS[6],
    TOOL_HEALING_STEPS[11],
)

TOOL_CALL_GENERATION_CORPUS = (
    *((f"system_info_{index}", "system_info", prompt) for index, prompt in enumerate((
        "Show the synthetic computer specifications.",
        "Report OS, CPU, memory, disks, and runtime for this test machine.",
        "Use the system information tool for current machine details.",
        "Inspect this computer's hardware and operating system.",
        "Return the available system specification fields.",
    ), 1)),
    *((f"list_directory_{index}", "list_directory", prompt) for index, prompt in enumerate((
        "List the current synthetic directory.",
        "Show files in the fixture directory without reading them.",
        "List this directory recursively to depth two.",
        "Return only files from the current directory.",
        "Show hidden and normal entries in this fixture directory.",
    ), 1)),
    *((f"read_file_{index}", "exec_shell_full_command", prompt) for index, prompt in enumerate((
        "Read fixture-alpha.txt.",
        "Show the exact content of fixture-beta.txt.",
        "Open the synthetic file fixture-gamma.txt.",
        "Read the file named fixture with spaces.txt.",
        "Display the first line of fixture-delta.txt.",
    ), 1)),
    *((f"grep_search_{index}", "exec_shell_full_command", prompt) for index, prompt in enumerate((
        "Search fixture-alpha.txt for alpha.",
        "Find lines containing beta in the fixture files.",
        "Run a recursive grep for synthetic-match.",
        "Search the spaced fixture path for gamma.",
        "Count exact delta matches in fixture-delta.txt.",
    ), 1)),
    *((f"fetch_url_{index}", "fetch_url", prompt) for index, prompt in enumerate((
        "Fetch https://example.invalid/fixture-one.",
        "Read https://example.invalid/fixture-two?x=1.",
        "Fetch the synthetic URL https://example.invalid/a/b.",
        "Retrieve https://example.invalid/empty.",
        "Open https://example.invalid/error for inspection.",
    ), 1)),
    *((f"exec_shell_{index}", "exec_shell_full_command", prompt) for index, prompt in enumerate((
        "Run printf 'synthetic-ok\\n'.",
        "Run pwd in the current fixture directory.",
        "Run printf '%s\\n' alpha beta.",
        "Run wc -l fixture-alpha.txt.",
        "Run test -f fixture-beta.txt and print present.",
    ), 1)),
    *((f"negative_{index}", None, prompt) for index, prompt in enumerate((
        "Explain the phrase system_info without calling tools.",
        "Write one sentence containing the words list_directory.",
        "Describe what fetch_url might mean in documentation.",
        "Show the literal text exec_shell_full_command.",
        "Explain that arguments is a JSON field.",
        'Quote this example without executing it: {"name":"system_info","arguments":{}}.',
        'Show a JSON example with {"tool":"list_directory"}.',
        "Say that tool calls are structured model outputs.",
        "Compare the words name, function, and arguments.",
        "Answer only: no tool is required.",
    ), 1)),
)


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    steps: tuple[SmokeStep, ...]
    requires_web: bool = False
    optional: bool = False
    allowed_tool_names: tuple[str, ...] = ("exec_shell_full_command",)
    isolated_steps: bool = False
    family: str = "general"


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
    scenario_family: str = "general"
    process_id: int | None = None
    block_id: str | None = None
    run_order: str | None = None
    repetition: int | None = None
    route_diagnostics_enabled: bool = False
    route_outputs: list[dict[str, object]] = field(default_factory=list)
    final_parsed_route: str | None = None
    route_correct: bool | None = None
    tool_correct: bool | None = None
    downstream_final_correct: bool | None = None
    retry_required: bool = False
    route_fallback_used: bool = False
    tool_healing_diagnostics_enabled: bool = False
    tool_healing_attempts: list[dict[str, object]] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "case": self.case,
            "step": self.step,
            "prompt": "<redacted>" if self.tool_healing_diagnostics_enabled else self.prompt,
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
            "answer_excerpt": "<redacted>" if self.tool_healing_diagnostics_enabled else self.answer_excerpt,
            "model_steps": self.model_steps,
            "final_prefix": self.final_prefix,
            "lifecycle": self.lifecycle,
            "phase_wall_ms": self.phase_wall_ms,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "estimated_generation_ms": self.estimated_generation_ms,
            "estimated_prefill_residual_ms": self.estimated_prefill_residual_ms,
            "scenario_family": self.scenario_family,
            "process_id": self.process_id,
            "block_id": self.block_id,
            "run_order": self.run_order,
            "repetition": self.repetition,
            "route_diagnostics_enabled": self.route_diagnostics_enabled,
            "route_outputs": self.route_outputs,
            "final_parsed_route": self.final_parsed_route,
            "route_correct": self.route_correct,
            "tool_correct": self.tool_correct,
            "downstream_final_correct": self.downstream_final_correct,
            "retry_required": self.retry_required,
            "route_fallback_used": self.route_fallback_used,
            "tool_healing_diagnostics_enabled": self.tool_healing_diagnostics_enabled,
            "tool_healing_attempts": self.tool_healing_attempts,
        }


@dataclass(frozen=True)
class LifecycleBlock:
    block_id: str
    server_pid: int
    ctx: int
    thinking: str
    initial_props: dict[str, object]
    final_props: dict[str, object]
    reports: list[StepReport]
    rss_samples: list[dict[str, object]]


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


ROUTE_OUTPUT_CLASSES = ("canonical", "legacy_tolerated", "direct_prose", "malformed", "control_loop")
ROUTE_FALLBACK_OUTCOMES = {"route_invalid_output", "route_no_decision_length_retry", "route_retry_invalid_output"}
TOOL_HEALING_RECOVERABLE = {
    "recoverable_envelope",
    "recoverable_trailing_comma",
    "recoverable_unambiguous_delimiter",
    "recoverable_arguments_string",
    "recoverable_structural_alias",
}
TOOL_HEALING_FORMAL_ERRORS = {
    "missing_required",
    "wrong_type",
    "extra_argument",
    "unknown_tool",
    "multiple_candidates",
    "ambiguous_attempt",
    "truncated_attempt",
}


class RouteDiagnosticCollector:
    def __init__(self, root: Path, *, store_mode: str, tool_healing: bool = False, route_output: bool = True) -> None:
        self.path = root / "route-kv-diag.jsonl"
        self.store_root = root / "evidence"
        self.store_mode = store_mode
        self.tool_healing = tool_healing
        self.route_output = route_output
        self._store_index = 0

    def mark(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def events_since(self, offset: int) -> list[dict[str, object]]:
        return parse_route_diagnostic_lines(self.lines_since(offset))

    def tool_healing_events_since(self, offset: int) -> list[dict[str, object]]:
        return parse_tool_healing_diagnostic_lines(self.lines_since(offset))

    def lines_since(self, offset: int) -> list[str]:
        try:
            with self.path.open("rb") as handle:
                handle.seek(offset)
                lines = [line.decode("utf-8", errors="replace") for line in handle.readlines()]
        except OSError:
            return []
        return lines

    def new_evidence_store(self, workdir: Path) -> EvidenceStore:
        self._store_index += 1
        destination = self.store_root / f"store-{self._store_index:04d}"
        if self.store_mode == "existing-snapshot":
            source = EvidenceStore.for_workdir(workdir).root
            if source.is_dir():
                shutil.copytree(source, destination)
        store = EvidenceStore(destination)
        store.load_index()
        return store


def parse_route_diagnostic_lines(lines: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("event") != "kv_diag_route_outcome":
            continue
        route_class = value.get("route_output_class")
        if route_class not in ROUTE_OUTPUT_CLASSES:
            route_class = None
        phase = value.get("phase")
        events.append(
            {
                "route_call": "retry" if phase == "route_retry" else "initial",
                "route_output_class": route_class,
                "route_output_reason": bounded_identifier(value.get("route_output_reason")),
                "parser_accepted": value.get("route_parser_accepted") if isinstance(value.get("route_parser_accepted"), bool) else None,
                "finish_reason": bounded_identifier(value.get("route_finish_reason")),
                "output_tokens": value.get("route_output_tokens") if isinstance(value.get("route_output_tokens"), int) else None,
                "parsed_route": bounded_identifier(value.get("decision_type")),
                "outcome": bounded_identifier(value.get("outcome")),
                "retry_reason": bounded_identifier(value.get("retry_reason")),
                "control_loop_surrogate": value.get("route_output_reason") == "empty_visible_control_output",
            }
        )
    return events


def parse_tool_healing_diagnostic_lines(lines: list[str]) -> list[dict[str, object]]:
    attempts: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("event") not in {
            "kv_diag_tool_healing_shadow",
            "kv_diag_tool_healing_terminal",
        }:
            continue
        attempt_id = bounded_hex(value.get("attempt_id"), length=32)
        if attempt_id is None:
            continue
        if attempt_id not in attempts:
            attempts[attempt_id] = {"attempt_id": attempt_id}
            order.append(attempt_id)
        attempt = attempts[attempt_id]
        if value.get("event") == "kv_diag_tool_healing_shadow":
            repairs = value.get("repairs") if isinstance(value.get("repairs"), list) else []
            signals = value.get("signals") if isinstance(value.get("signals"), list) else []
            attempt.update(
                {
                    "attempt_detected": value.get("attempt_detected") if isinstance(value.get("attempt_detected"), bool) else None,
                    "signals": [item for item in (bounded_identifier(item) for item in signals) if item],
                    "candidate_count": value.get("candidate_count") if isinstance(value.get("candidate_count"), int) else None,
                    "candidate_source": bounded_identifier(value.get("candidate_source")),
                    "repairs": [item for item in (bounded_identifier(item) for item in repairs) if item],
                    "strict_outcome": bounded_identifier(value.get("shadow_outcome")),
                    "parse_error": bounded_identifier(value.get("parse_error")),
                    "validation_error": bounded_identifier(value.get("validation_error")),
                    "formal_repairable": value.get("formal_repairable") if isinstance(value.get("formal_repairable"), bool) else None,
                    "formal_repair_reason": bounded_identifier(value.get("formal_repair_reason")),
                    "formal_argument_count": value.get("formal_argument_count") if isinstance(value.get("formal_argument_count"), int) else None,
                    "finish_reason": bounded_identifier(value.get("finish_reason")),
                    "output_tokens": value.get("output_tokens") if isinstance(value.get("output_tokens"), int) else None,
                    "healing_us": value.get("healing_us") if isinstance(value.get("healing_us"), int | float) else None,
                    "candidate_hash": bounded_hex(value.get("candidate_hash"), length=16),
                    "original_tool_name_hash": bounded_hex(value.get("original_tool_name_hash"), length=16),
                    "normalized_tool_name_hash": bounded_hex(value.get("normalized_tool_name_hash"), length=16),
                    "normalized_arguments_hash": bounded_hex(value.get("normalized_arguments_hash"), length=16),
                }
            )
        else:
            attempt.update(
                {
                    "active_candidate_count": value.get("active_candidate_count") if isinstance(value.get("active_candidate_count"), int) else None,
                    "active_tool_name_hash": bounded_hex(value.get("active_tool_name_hash"), length=16),
                    "active_arguments_hash": bounded_hex(value.get("active_arguments_hash"), length=16),
                    "active_outcome": bounded_identifier(value.get("active_outcome")),
                    "terminal_reason": bounded_identifier(value.get("terminal_reason")),
                    "agreement": bounded_identifier(value.get("agreement")),
                    "active_canonical_outcome": bounded_identifier(value.get("active_canonical_outcome")),
                    "active_canonical_error": bounded_identifier(value.get("active_canonical_error")),
                }
            )
    for attempt in attempts.values():
        attempt["categories"] = tool_healing_categories(attempt)
    return [attempts[attempt_id] for attempt_id in order]


def tool_healing_categories(attempt: dict[str, object]) -> list[str]:
    categories: list[str] = []
    repairs = set(attempt.get("repairs") or [])
    source = attempt.get("candidate_source")
    validation = attempt.get("validation_error")
    parse_error = attempt.get("parse_error")
    active = attempt.get("active_outcome")
    finish_reason = attempt.get("finish_reason")
    interrupted = active in {"cancelled", "timeout"} or finish_reason in {"cancelled", "timeout"}
    if not interrupted and (
        attempt.get("strict_outcome") == "no_attempt" or attempt.get("attempt_detected") is False
    ):
        categories.append("no_attempt")
    if finish_reason == "length":
        categories.append("budget_truncation")
    if attempt.get("strict_outcome") == "valid_shadow_candidate" and (
        not repairs or (source == "backend" and repairs <= {"unwrap_function_object", "decode_arguments_string"})
    ):
        categories.append("valid_first_pass")
    if repairs & {"unwrap_function_object", "unwrap_named_wrapper", "wrap_top_level_arguments"} and source != "backend":
        categories.append("recoverable_envelope")
    if "remove_trailing_comma" in repairs:
        categories.append("recoverable_trailing_comma")
    if repairs & {"close_json_structure", "unclosed_tool_call_tag", "unclosed_tool_call_wrapper"}:
        categories.append("recoverable_unambiguous_delimiter")
    if "decode_arguments_string" in repairs and source != "backend":
        categories.append("recoverable_arguments_string")
    if "normalize_tool_field" in repairs:
        categories.append("recoverable_structural_alias")
    error_categories = {
        "missing_required": "missing_required",
        "empty_required_value": "missing_required",
        "type_mismatch": "wrong_type",
        "additional_property": "extra_argument",
        "unknown_tool": "unknown_tool",
        "multiple_candidates": "multiple_candidates",
        "unterminated_json_string": "truncated_attempt",
    }
    mapped = error_categories.get(validation) or error_categories.get(parse_error)
    if mapped:
        categories.append(mapped)
    if parse_error and mapped is None:
        categories.append("ambiguous_attempt")
    if active in {"rejected_policy", "rejected_guardrail", "rejected_permission"}:
        categories.append("policy_denied")
    active_accepted = active in {"executed", "runtime_error"}
    if active_accepted and validation == "type_mismatch":
        categories.append("active_default_used")
    if active_accepted and validation == "limit_out_of_range":
        categories.append("active_clamp_used")
    if active_accepted and validation == "additional_property":
        categories.append("active_ignored_extra")
    if active == "runtime_error":
        categories.append("executor_error")
    if active == "superseded":
        categories.append("superseded")
    if attempt.get("agreement") == "uncorrelated" or (
        attempt.get("attempt_detected") is False and active in {"cancelled", "timeout"}
    ):
        categories.append("uncorrelated")
    return list(dict.fromkeys(categories or ["ambiguous_attempt"]))


def bounded_hex(value: object, *, length: int) -> str | None:
    if not isinstance(value, str) or len(value) != length:
        return None
    return value if all(character in "0123456789abcdef" for character in value) else None


def bounded_identifier(value: object, *, limit: int = 80) -> str | None:
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    if not all(character.isalnum() or character in {"_", "-"} for character in value):
        return None
    return value


@contextmanager
def route_diagnostic_environment(
    enabled: bool,
    *,
    store_mode: str,
    tool_healing: bool = False,
    route_output: bool = True,
):
    if not enabled:
        yield None
        return
    previous_enabled = os.environ.get("ORBIT_KV_DIAG")
    previous_path = os.environ.get("ORBIT_KV_DIAG_FILE")
    previous_healing = os.environ.get("ORBIT_TOOL_CALL_HEALING_SHADOW")
    with tempfile.TemporaryDirectory(prefix="orbit-route-diag-") as tmp:
        collector = RouteDiagnosticCollector(
            Path(tmp),
            store_mode=store_mode,
            tool_healing=tool_healing,
            route_output=route_output,
        )
        os.environ["ORBIT_KV_DIAG"] = "1"
        os.environ["ORBIT_KV_DIAG_FILE"] = str(collector.path)
        if tool_healing:
            os.environ["ORBIT_TOOL_CALL_HEALING_SHADOW"] = "1"
        try:
            yield collector
        finally:
            if previous_enabled is None:
                os.environ.pop("ORBIT_KV_DIAG", None)
            else:
                os.environ["ORBIT_KV_DIAG"] = previous_enabled
            if previous_path is None:
                os.environ.pop("ORBIT_KV_DIAG_FILE", None)
            else:
                os.environ["ORBIT_KV_DIAG_FILE"] = previous_path
            if previous_healing is None:
                os.environ.pop("ORBIT_TOOL_CALL_HEALING_SHADOW", None)
            else:
                os.environ["ORBIT_TOOL_CALL_HEALING_SHADOW"] = previous_healing


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

    if args.lifecycle_check:
        return run_lifecycle_checks(args, jsonl_path=jsonl_path, markdown_path=markdown_path)
    if args.tool_call_generation_only:
        return run_tool_call_generation_benchmark(args, jsonl_path=jsonl_path, markdown_path=markdown_path)

    with (
        final_prefix_environment(args.final_prefix_mode),
        canonical_gate_environment(args.canonical_gate),
        tool_healing_environment(args.tool_healing_mode),
        deterministic_web(args.deterministic_web),
        managed_server(args) as server_process,
        route_diagnostic_environment(
            args.route_output_diagnostics or args.tool_healing_diagnostics,
            store_mode=args.route_diagnostic_store,
            tool_healing=args.tool_healing_diagnostics,
            route_output=args.route_output_diagnostics,
        ) as route_collector,
    ):
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
        run_order_index = 0
        for scenario in selected:
            for repeat in range(1, args.repetitions + 1):
                run_order_index += 1
                reports.extend(
                    run_scenario(
                        scenario,
                        backend=backend,
                        workdir=Path(args.workdir),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        tools_mode=args.tools,
                        route_collector=route_collector,
                        process_id=server_process.pid if server_process is not None else None,
                        block_id=args.block_id or f"run-{run_order_index:04d}",
                        run_order=args.run_order or str(run_order_index),
                        repetition=repeat,
                    )
                )
        final_props = settled_backend_props(args.base_url, args.timeout)
        env = environment_summary(args=args, backend=backend, props=final_props)
        env["server_pid"] = server_process.pid if server_process is not None else None
        env["server_rss_before_kib"] = rss_before_kib
        env["server_rss_after_kib"] = process_rss_kib(server_process)
    replay_rows = run_tool_healing_replay(Path(args.workdir)) if args.tool_healing_replay else []
    if replay_rows:
        write_jsonl(jsonl_path, env, reports, extra_rows=replay_rows)
    else:
        write_jsonl(jsonl_path, env, reports)
    write_markdown(markdown_path, env, reports)
    if args.verify_final_prefix:
        failure = final_prefix_validation_failure(args.final_prefix_mode, reports, final_props, tools_mode=args.tools)
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
    parser.add_argument(
        "--canonical-gate",
        choices=("inherit", "off", "on"),
        default="inherit",
        help="Control the runtime canonical tool-call gate for this benchmark only.",
    )
    parser.add_argument(
        "--tool-healing-mode",
        choices=("inherit", "off", "on"),
        default="inherit",
        help="Control deterministic tool-call healing for this benchmark only.",
    )
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument(
        "--final-prefix-mode",
        choices=(
            "inherit",
            "off",
            "on",
            "legacy-off",
            "legacy-on",
            "stable-off-legacy-on",
            "stable-on-legacy-off",
            "stable-invalid",
        ),
        default="inherit",
    )
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
        "--route-output-diagnostics",
        action="store_true",
        help="Collect bounded route-output classifications from existing KV diagnostic events.",
    )
    parser.add_argument(
        "--tool-healing-diagnostics",
        action="store_true",
        help="Collect bounded shadow/terminal tool-attempt diagnostics by attempt_id.",
    )
    parser.add_argument(
        "--tool-healing-replay",
        action="store_true",
        help="Append a content-free summary of the labelled structural replay corpus.",
    )
    parser.add_argument(
        "--tool-call-generation-only",
        action="store_true",
        help="Generate exactly one production-template tool-mode ChatResult per corpus item without executing tools.",
    )
    parser.add_argument(
        "--tool-call-generation-smoke",
        action="store_true",
        help="Use one prompt per tool family plus two negatives instead of the full 40-prompt corpus.",
    )
    parser.add_argument(
        "--route-diagnostic-store",
        choices=("clean", "existing-snapshot"),
        default="clean",
        help="Use a clean temporary evidence store or a read-only snapshot of the selected workdir store.",
    )
    parser.add_argument(
        "--lifecycle-check",
        action="append",
        choices=("restart", "ctx-change", "thinking", "rss"),
        help="Run a managed final-prefix lifecycle check; repeatable.",
    )
    parser.add_argument("--ctx-change-to", type=positive_int, default=4096)
    parser.add_argument(
        "--verify-final-prefix",
        action="store_true",
        help="Fail unless OFF/ON final-prefix counters match the requested experimental mode.",
    )
    return parser


def run_tool_call_generation_benchmark(
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    markdown_path: Path,
) -> int:
    with final_prefix_environment(args.final_prefix_mode), managed_server(args) as server_process:
        backend = LlamaServerBackend(base_url=args.base_url, timeout=args.timeout)
        backend.thinking = args.server_thinking == "on"
        initial_props = safe_backend_props(backend)
        initial_mtp = mtp_state_from_props(initial_props)
        if args.mtp_required and not (initial_mtp["requested"] and initial_mtp["session_ready"]):
            print("error: --mtp-required set but backend MTP is not initialized", file=sys.stderr)
            return 2
        corpus = tool_call_generation_corpus(smoke=args.tool_call_generation_smoke)
        rows: list[dict[str, object]] = []
        workdir = Path(args.workdir)
        for repetition in range(1, args.repetitions + 1):
            for scenario, expected_tool, prompt in corpus:
                rows.append(
                    run_tool_call_generation_sample(
                        backend,
                        base_url=args.base_url,
                        timeout=args.timeout,
                        workdir=workdir,
                        scenario=scenario,
                        prompt=prompt,
                        expected_tool=expected_tool,
                        repetition=repetition,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                )
        final_props = settled_backend_props(args.base_url, args.timeout)
        env = environment_summary(args=args, backend=backend, props=final_props)
        env.update(
            {
                "type": "environment",
                "benchmark_mode": "tool_call_generation_only",
                "server_pid": server_process.pid if server_process is not None else None,
                "corpus_size": len(corpus),
                "repetitions": args.repetitions,
                "mtp_tool_mode_eligible": False,
            }
        )
    summary = summarize_tool_call_generation(rows, mtp=env.get("mtp"))
    write_tool_call_generation_jsonl(jsonl_path, env, rows, summary)
    write_tool_call_generation_markdown(markdown_path, env, summary)
    print(f"jsonl={jsonl_path}")
    print(f"markdown={markdown_path}")
    return 1 if any(row.get("cleanup_healthy") is False for row in rows) else 0


def tool_call_generation_corpus(*, smoke: bool) -> tuple[tuple[str, str | None, str], ...]:
    if not smoke:
        return TOOL_CALL_GENERATION_CORPUS
    selected = {
        "system_info_1", "list_directory_1", "read_file_1", "grep_search_1",
        "fetch_url_1", "exec_shell_1", "negative_1", "negative_6",
    }
    return tuple(item for item in TOOL_CALL_GENERATION_CORPUS if item[0] in selected)


def run_tool_call_generation_sample(
    backend: LlamaServerBackend,
    *,
    base_url: str,
    timeout: float,
    workdir: Path,
    scenario: str,
    prompt: str,
    expected_tool: str | None,
    repetition: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    runtime = ChatRuntime(backend=ProbeBackend(backend), system_prompt=DEFAULT_SYSTEM_PROMPT)
    runtime.messages.append({"role": "user", "content": prompt})
    messages = with_tool_call_system_prompt(runtime.messages)
    executor = HybridToolExecutor(
        backend=backend if hasattr(backend, "server_tools") else None,
        workdir=workdir,
        allowed_tool_names=TOOL_NAMES,
        user_prompt=prompt,
    )
    tools = executor.tool_definitions()
    result_queue: queue.Queue[ChatResult | BaseException] = queue.Queue(maxsize=1)

    def generate() -> None:
        try:
            with model_call_context(phase="tool_call", tools_mode="on"):
                result_queue.put(
                    runtime._chat_tool_call_once(
                        messages,
                        temperature=temperature,
                        max_tokens=resolve_max_tokens("tool_call", max_tokens),
                        tools=tools,
                        on_final_delta=None,
                        on_progress=None,
                    )
                )
        except BaseException as exc:
            result_queue.put(exc)

    worker = threading.Thread(target=generate, daemon=True)
    started = time.perf_counter()
    worker.start()
    worker.join(timeout)
    timed_out = worker.is_alive()
    cancel_requested = False
    cleanup_started = time.perf_counter()
    if timed_out:
        cancel_requested = request_backend_cancel(base_url, timeout=min(5.0, max(1.0, timeout)))
    props = wait_for_backend_idle(base_url, timeout=min(10.0, max(2.0, timeout / 5.0)))
    worker.join(min(2.0, max(0.5, timeout / 10.0)))
    cleanup_ms = round((time.perf_counter() - cleanup_started) * 1000, 1)
    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    value = result_queue.get_nowait() if not result_queue.empty() else None
    result = value if isinstance(value, ChatResult) else None
    error = value if isinstance(value, BaseException) else None
    cancelled = bool(result and result.finish_reason == "cancelled")
    evaluable = result is not None and not timed_out and not cancelled and error is None
    if result is not None:
        analysis_started = time.perf_counter_ns()
        report = analyze_tool_attempt(
            text=result.content,
            tool_calls=result.tool_calls,
            tool_definitions=tools,
            allowed_tool_names=TOOL_NAMES,
            workdir=workdir,
            user_prompt=prompt,
            finish_reason=result.finish_reason,
        )
        healing_us = round((time.perf_counter_ns() - analysis_started) / 1000, 1)
        categories = tool_healing_categories(
            {
                "candidate_source": report.candidate_source,
                "repairs": list(report.repairs),
                "strict_outcome": report.outcome,
                "parse_error": report.parse_error,
                "validation_error": report.validation_error,
                "attempt_detected": report.attempt_detected,
                "finish_reason": result.finish_reason,
            }
        )
        markup_leakage = bool(re.search(r"<\|?tool_call|\[TOOL_CALLS\]|\[ARGS\]", result.content, re.IGNORECASE))
    else:
        report = None
        healing_us = None
        categories = ["uncorrelated"]
        markup_leakage = False
    prompt_tokens = result.prompt_tokens if result else None
    cached_tokens = result.cached_tokens if result else None
    output_tokens = result.completion_tokens if result else None
    evaluated_tokens = prompt_tokens - cached_tokens if isinstance(prompt_tokens, int) and isinstance(cached_tokens, int) else None
    if not evaluable:
        semantic_outcome = "not_evaluable"
    elif expected_tool is None:
        semantic_outcome = "unwanted_tool_call" if report and report.attempt_detected else "no_attempt"
    elif report and not report.attempt_detected:
        semantic_outcome = "missing_tool_call"
    elif report and report.tool_name == expected_tool:
        semantic_outcome = "expected_tool"
    else:
        semantic_outcome = "wrong_tool"
    budget_truncation = bool(result and result.finish_reason == "length")
    structural_truncation = "truncated_attempt" in categories
    return {
        "type": "tool_call_generation",
        "scenario": scenario,
        "repetition": repetition,
        "expected_tool": expected_tool,
        "template": "production_tool_call_system_prompt",
        "model_calls": 1,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "evaluated_tokens": evaluated_tokens,
        "output_tokens": output_tokens,
        "finish_reason": result.finish_reason if result else "timeout" if timed_out else "error",
        "generation_wall_ms": wall_ms,
        "estimated_prefill_ms": estimated_phase_ms(evaluated_tokens, result.prompt_tokens_per_second if result else None),
        "estimated_decode_ms": estimated_phase_ms(output_tokens, result.generation_tokens_per_second if result else None),
        "prompt_tokens_per_second": result.prompt_tokens_per_second if result else None,
        "generation_tokens_per_second": result.generation_tokens_per_second if result else None,
        "healing_us": healing_us,
        "cleanup_ms": cleanup_ms,
        "timeout": timed_out,
        "cancel_requested": cancel_requested,
        "cancelled": cancelled,
        "cleanup_healthy": props.get("in_flight") is False and not worker.is_alive(),
        "evaluable": evaluable,
        "attempt_detected": report.attempt_detected if report else None,
        "candidate_count": report.candidate_count if report else None,
        "strict_valid": report.outcome == "valid_shadow_candidate" if report else None,
        "strict_outcome": report.outcome if report else None,
        "candidate_source": bounded_identifier(report.candidate_source) if report else None,
        "parse_error": report.parse_error if report else None,
        "validation_error": report.validation_error if report else None,
        "repairs": list(report.repairs) if report else [],
        "formal_repairable": report.formal_repairable if report else None,
        "formal_repair_reason": report.formal_repair_reason if report else None,
        "formal_argument_count": report.formal_argument_count if report else None,
        "categories": categories,
        "semantic_outcome": semantic_outcome,
        "tool_name_exact_match": report.tool_name == expected_tool if report and expected_tool is not None else report.attempt_detected is False if report else None,
        "markup_leakage": markup_leakage,
        "multiple_candidates": bool(report and report.candidate_count > 1),
        "budget_truncation": budget_truncation,
        "structural_truncation": structural_truncation,
        "truncation": budget_truncation or structural_truncation,
        "tool_executed": False,
        "finalization_started": False,
        "error_type": type(error).__name__ if error is not None else None,
    }


def estimated_phase_ms(tokens: int | None, rate: float | None) -> float | None:
    if not isinstance(tokens, int) or not isinstance(rate, int | float) or rate <= 0:
        return None
    return round(tokens / float(rate) * 1000, 1)


def summarize_tool_call_generation(rows: list[dict[str, object]], *, mtp: object) -> dict[str, object]:
    evaluable = [row for row in rows if row.get("evaluable") is True]
    detected = [row for row in evaluable if row.get("attempt_detected") is True]
    positive = [row for row in evaluable if row.get("expected_tool") is not None]
    negative = [row for row in evaluable if row.get("expected_tool") is None]
    latencies = [float(row["generation_wall_ms"]) for row in evaluable if isinstance(row.get("generation_wall_ms"), int | float)]
    healing = [float(row["healing_us"]) for row in evaluable if isinstance(row.get("healing_us"), int | float)]
    category_counts: dict[str, int] = {}
    for row in evaluable:
        for category in row.get("categories") or []:
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "type": "tool_call_generation_summary",
        "mtp": bounded_identifier(mtp),
        "samples": len(rows),
        "evaluable": len(evaluable),
        "completion_rate": ratio(len(evaluable), len(rows)),
        "timeout_rate": ratio(sum(row.get("timeout") is True for row in rows), len(rows)),
        "cancel_rate": ratio(sum(row.get("cancelled") is True for row in rows), len(rows)),
        "cleanup_success_rate": ratio(sum(row.get("cleanup_healthy") is True for row in rows), len(rows)),
        "valid_first_pass_rate": ratio(
            sum("valid_first_pass" in (row.get("categories") or []) for row in detected),
            len(detected),
        ),
        "valid_first_pass_output_rate": ratio(
            sum("valid_first_pass" in (row.get("categories") or []) for row in evaluable),
            len(evaluable),
        ),
        "no_attempt_rate": ratio(
            sum("no_attempt" in (row.get("categories") or []) for row in evaluable),
            len(evaluable),
        ),
        "deterministic_repair_candidate_rate": ratio(
            sum(
                row.get("formal_repairable") is True
                and "budget_truncation" not in (row.get("categories") or [])
                for row in detected
            ),
            len(detected),
        ),
        "formal_error_rate": ratio(
            sum(any(category in TOOL_HEALING_FORMAL_ERRORS for category in row.get("categories") or []) for row in detected),
            len(detected),
        ),
        "detected_rate": ratio(sum(row.get("attempt_detected") is True for row in evaluable), len(evaluable)),
        "exact_tool_match_rate": ratio(sum(row.get("tool_name_exact_match") is True for row in positive), len(positive)),
        "model_missing_attempt_rate": ratio(sum(row.get("attempt_detected") is False for row in positive), len(positive)),
        "model_unwanted_attempt_rate": ratio(sum(row.get("attempt_detected") is True for row in negative), len(negative)),
        "detector_false_positive_rate": ratio(
            sum(row.get("attempt_detected") is True and row.get("strict_outcome") == "no_attempt" for row in evaluable),
            len(evaluable),
        ),
        "detector_false_negative_rate": ratio(
            sum(row.get("attempt_detected") is False and row.get("markup_leakage") is True for row in evaluable),
            len(evaluable),
        ),
        "multiple_candidate_rate": ratio(sum(row.get("multiple_candidates") is True for row in evaluable), len(evaluable)),
        "truncation_rate": ratio(sum(row.get("truncation") is True for row in evaluable), len(evaluable)),
        "budget_truncation_rate": ratio(sum(row.get("budget_truncation") is True for row in evaluable), len(evaluable)),
        "structural_truncation_rate": ratio(sum(row.get("structural_truncation") is True for row in evaluable), len(evaluable)),
        "semantic_wrong_tool_rate": ratio(sum(row.get("semantic_outcome") == "wrong_tool" for row in positive), len(positive)),
        "semantic_unwanted_tool_rate": ratio(sum(row.get("semantic_outcome") == "unwanted_tool_call" for row in negative), len(negative)),
        "markup_leakage_rate": ratio(sum(row.get("markup_leakage") is True for row in evaluable), len(evaluable)),
        "category_counts": category_counts,
        "generation_wall_ms_median": round(statistics.median(latencies), 1) if latencies else None,
        "generation_wall_ms_p95": percentile(latencies, 0.95),
        "healing_us_median": round(statistics.median(healing), 1) if healing else None,
        "healing_us_p95": percentile(healing, 0.95),
        "model_calls": sum(int(row.get("model_calls") or 0) for row in rows),
        "tools_executed": sum(row.get("tool_executed") is True for row in rows),
        "finalizations_started": sum(row.get("finalization_started") is True for row in rows),
    }


def write_tool_call_generation_jsonl(
    path: Path,
    env: dict[str, object],
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(env, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")


def write_tool_call_generation_markdown(
    path: Path,
    env: dict[str, object],
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Orbit Tool-Call Generation Benchmark\n\n"
        f"- Model: `{env.get('model')}`\n"
        f"- MTP: `{env.get('mtp')}`\n"
        f"- Samples: `{summary.get('samples')}`\n"
        f"- Evaluable: `{summary.get('evaluable')}`\n"
        f"- Completion rate: `{summary.get('completion_rate')}`\n",
        encoding="utf-8",
    )


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
            FINAL_PREFIX_LOCAL_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
        ),
        "final_prefix_pwd": SmokeScenario(
            "final_prefix_pwd",
            (FINAL_PREFIX_LOCAL_STEPS[0], FINAL_PREFIX_LOCAL_STEPS[0]),
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
        ),
        "final_prefix_web": SmokeScenario(
            "final_prefix_web",
            FINAL_PREFIX_WEB_STEPS,
            requires_web=True,
            optional=True,
        ),
        "final_prefix_mixed": SmokeScenario(
            "final_prefix_mixed",
            FINAL_PREFIX_LOCAL_STEPS + FINAL_PREFIX_WEB_STEPS,
            requires_web=True,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
        ),
        "final_prefix_paired": SmokeScenario(
            "final_prefix_paired",
            FINAL_PREFIX_PAIRED_STEPS,
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
        "route_classification_recap": SmokeScenario(
            "route_classification_recap",
            ROUTE_CLASS_RECAP_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="chat_recap",
        ),
        "route_classification_chat": SmokeScenario(
            "route_classification_chat",
            ROUTE_CLASS_CHAT_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="chat",
        ),
        "route_classification_local": SmokeScenario(
            "route_classification_local",
            ROUTE_CLASS_LOCAL_TOOL_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="local_tool",
        ),
        "route_classification_web": SmokeScenario(
            "route_classification_web",
            ROUTE_CLASS_WEB_STEPS,
            requires_web=True,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="web",
        ),
        "route_classification_evidence": SmokeScenario(
            "route_classification_evidence",
            ROUTE_CLASS_EVIDENCE_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="evidence_shape",
        ),
        "route_classification_ambiguous": SmokeScenario(
            "route_classification_ambiguous",
            ROUTE_CLASS_AMBIGUOUS_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="fragile_ambiguous",
        ),
        "route_classification_refresh": SmokeScenario(
            "route_classification_refresh",
            ROUTE_CLASS_REFRESH_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="fragile_refresh",
        ),
        "route_classification_verify": SmokeScenario(
            "route_classification_verify",
            ROUTE_CLASS_VERIFY_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="fragile_verify",
        ),
        "route_classification_error_success": SmokeScenario(
            "route_classification_error_success",
            ROUTE_CLASS_ERROR_SUCCESS_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="fragile_error_success",
        ),
        "route_classification_web_error": SmokeScenario(
            "route_classification_web_error",
            ROUTE_CLASS_WEB_ERROR_STEPS,
            requires_web=True,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            family="fragile_web_error",
        ),
        "tool_healing_baseline": SmokeScenario(
            "tool_healing_baseline",
            TOOL_HEALING_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
            family="tool_healing",
        ),
        "tool_healing_quick": SmokeScenario(
            "tool_healing_quick",
            TOOL_HEALING_QUICK_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
            family="tool_healing",
        ),
        "canonical_gate_real": SmokeScenario(
            "canonical_gate_real",
            CANONICAL_GATE_REAL_STEPS,
            optional=True,
            allowed_tool_names=TOOL_NAMES,
            isolated_steps=True,
            family="canonical_gate",
        ),
        "canonical_gate_system": SmokeScenario("canonical_gate_system", (CANONICAL_GATE_REAL_STEPS[0],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_list": SmokeScenario("canonical_gate_list", (CANONICAL_GATE_REAL_STEPS[1],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_read": SmokeScenario("canonical_gate_read", (CANONICAL_GATE_REAL_STEPS[2],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_grep": SmokeScenario("canonical_gate_grep", (CANONICAL_GATE_REAL_STEPS[3],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_shell": SmokeScenario("canonical_gate_shell", (CANONICAL_GATE_REAL_STEPS[4],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_shell_error": SmokeScenario("canonical_gate_shell_error", (CANONICAL_GATE_REAL_STEPS[5],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_fetch_error": SmokeScenario("canonical_gate_fetch_error", (CANONICAL_GATE_REAL_STEPS[6],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
        "canonical_gate_negative": SmokeScenario("canonical_gate_negative", (CANONICAL_GATE_REAL_STEPS[7],), optional=True, allowed_tool_names=TOOL_NAMES, family="canonical_gate"),
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
    tools_mode: str = "on",
    route_collector: RouteDiagnosticCollector | None = None,
    process_id: int | None = None,
    block_id: str | None = None,
    run_order: str | None = None,
    repetition: int | None = None,
) -> list[StepReport]:
    runtime = new_runtime(backend, workdir, route_collector=route_collector)
    reports: list[StepReport] = []
    for index, step in enumerate(scenario.steps, start=1):
        if scenario.isolated_steps and index > 1:
            runtime = new_runtime(backend, workdir, route_collector=route_collector)
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
            allowed_tool_names=effective_allowed_tool_names(scenario, tools_mode),
            route_collector=route_collector,
            scenario_family=scenario.family,
            process_id=process_id,
            block_id=block_id,
            run_order=run_order,
            repetition=repetition,
        )
        reports.append(report)
        if report.finish_reason in {"timeout", "error"}:
            break
    return reports


def effective_allowed_tool_names(scenario: SmokeScenario, tools_mode: str) -> tuple[str, ...]:
    return scenario.allowed_tool_names if tools_mode == "on" else ()


def new_runtime(
    backend: LlamaServerBackend,
    workdir: Path,
    *,
    route_collector: RouteDiagnosticCollector | None = None,
) -> ChatRuntime:
    return ChatRuntime(
        backend=ProbeBackend(backend),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        diagnostic_session_id=str(workdir),
        evidence_store=route_collector.new_evidence_store(workdir) if route_collector is not None else None,
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
    route_collector: RouteDiagnosticCollector | None = None,
    scenario_family: str = "general",
    process_id: int | None = None,
    block_id: str | None = None,
    run_order: str | None = None,
    repetition: int | None = None,
) -> StepReport:
    props_before = fresh_backend_props(base_url, min(timeout, 5.0))
    diagnostic_offset = route_collector.mark() if route_collector is not None else 0
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
        report = StepReport(
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
        report = enrich_step_route_diagnostics(
            report,
            route_collector.events_since(diagnostic_offset) if route_collector is not None else [],
            step=step,
            enabled=bool(route_collector and route_collector.route_output),
            scenario_family=scenario_family,
            process_id=process_id,
            block_id=block_id,
            run_order=run_order,
            repetition=repetition,
        )
        return enrich_step_tool_healing(
            report,
            route_collector.tool_healing_events_since(diagnostic_offset) if route_collector is not None else [],
            step=step,
            enabled=bool(route_collector and route_collector.tool_healing),
        )
    report = result_queue.get()
    props_after = fresh_backend_props(base_url, min(timeout, 5.0))
    report = replace_step_final_prefix(report, final_prefix_step_state(props_before, props_after))
    report = enrich_step_route_diagnostics(
        report,
        route_collector.events_since(diagnostic_offset) if route_collector is not None else [],
        step=step,
        enabled=bool(route_collector and route_collector.route_output),
        scenario_family=scenario_family,
        process_id=process_id,
        block_id=block_id,
        run_order=run_order,
        repetition=repetition,
    )
    return enrich_step_tool_healing(
        report,
        route_collector.tool_healing_events_since(diagnostic_offset) if route_collector is not None else [],
        step=step,
        enabled=bool(route_collector and route_collector.tool_healing),
    )


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
    probe = probe_backend(runtime.backend)
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
        elif step.mode == "tool":
            result = runtime.ask_with_tools(
                step.prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workdir=workdir,
                tool_names=allowed_tool_names,
                on_model_step=model_steps.append,
                on_tool_call=lambda name, _args: tool_names.append(name),
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


def probe_backend(backend: object) -> ProbeBackend | None:
    if isinstance(backend, ProbeBackend):
        return backend
    wrapped = getattr(backend, "_backend", None)
    return wrapped if isinstance(wrapped, ProbeBackend) else None


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
    redact = args.tool_healing_diagnostics or args.tool_call_generation_only
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "<redacted>" if redact else " ".join(sys.argv),
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
        "base_url": "<redacted>" if redact else args.base_url,
        "workdir": "<redacted>" if redact else str(Path(args.workdir)),
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
        "server_command": "<redacted>" if redact else server_command(args) if args.manage_server else "external",
        "client_command": "<redacted>" if redact else " ".join(sys.argv),
        "final_prefix": final_prefix_props(props),
        "final_prefix_config": final_prefix_config_metadata(args.final_prefix_mode, props),
        "block_id": args.block_id,
        "run_order": args.run_order,
        "cooling_seconds": args.cooling_seconds,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "route_output_diagnostics": args.route_output_diagnostics,
        "tool_healing_diagnostics": args.tool_healing_diagnostics,
        "tool_healing": tool_healing_metadata(args.tool_healing_mode),
        "canonical_gate": canonical_gate_metadata(args.canonical_gate),
        "route_diagnostic_store": args.route_diagnostic_store,
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


def final_prefix_config_metadata(mode: str, props: dict[str, object]) -> dict[str, object]:
    client = final_prefix_config_for_mode(mode)
    server_known = "final_prefix_reuse_source" in props
    server = {
        "enabled": props.get("final_prefix_reuse_enabled"),
        "source": props.get("final_prefix_reuse_source"),
        "config_error": props.get("final_prefix_reuse_config_error"),
        "legacy_detected": props.get("final_prefix_reuse_legacy_detected"),
    }
    client_metadata = {
        "enabled": client.enabled,
        "source": client.source,
        "config_error": client.validation_error,
        "legacy_detected": client.legacy_detected,
    }
    return {
        "requested": mode,
        "client": client_metadata,
        "server": server,
        "server_client_parity": server == client_metadata if server_known else None,
    }


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
    *,
    tools_mode: str = "on",
) -> str | None:
    state = final_prefix_props(props)
    eligible = [report for report in reports if "final_from_tool" in report.completion_kind]
    expected_enabled = final_prefix_mode_enabled(mode)
    if expected_enabled is False:
        if state.get("enabled") is not False:
            return "off_mode_enabled"
        if any((report.final_prefix.get("capture_count_delta") or 0) > 0 for report in eligible):
            return "off_mode_capture"
        return None
    if expected_enabled is not True:
        return "explicit_mode_required"
    if state.get("enabled") is not True:
        return "on_mode_disabled"
    if tools_mode == "off":
        if (state.get("capture_count") or 0) != 0 or (state.get("restore_count") or 0) != 0:
            return "tools_off_guard_failed"
        return None
    if props.get("mtp_experimental_enabled") is True:
        if (state.get("capture_count") or 0) != 0 or (state.get("restore_count") or 0) != 0:
            return "mtp_guard_failed"
        return None
    if state.get("fallback_count") not in {0, None}:
        return "fallback_observed"
    if state.get("prefix_tokens") != FINAL_PREFIX_TOKEN_COUNT:
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


def enrich_step_route_diagnostics(
    report: StepReport,
    events: list[dict[str, object]],
    *,
    step: SmokeStep,
    enabled: bool,
    scenario_family: str,
    process_id: int | None,
    block_id: str | None,
    run_order: str | None,
    repetition: int | None,
) -> StepReport:
    final_route = next(
        (event.get("parsed_route") for event in reversed(events) if isinstance(event.get("parsed_route"), str)),
        None,
    )
    retry_required = any(event.get("route_call") == "retry" for event in events) or any(
        "retry" in str(metric.get("phase") or "") for metric in report.model_steps
    )
    route_correct = route_expectation_result(step.expected_route, final_route, events, report.tool_names)
    tool_correct = tool_expectation_result(step.expected_tool_names, report.tool_names) if events else None
    downstream_correct = report.correctness_category == "correct"
    fallback_used = any(event.get("outcome") in ROUTE_FALLBACK_OUTCOMES for event in events)
    selected_tool = report.tool_names[0] if report.tool_names else None
    correlated = [
        {
            **event,
            "final_parsed_route": final_route,
            "selected_tool": selected_tool,
            "route_correct": route_correct,
            "tool_correct": tool_correct,
            "downstream_final_correct": downstream_correct,
            "retry_required": retry_required,
            "fallback_used": fallback_used,
        }
        for event in events
    ]
    values = report.__dict__.copy()
    values.update(
        {
            "scenario_family": scenario_family,
            "process_id": process_id,
            "block_id": block_id,
            "run_order": run_order,
            "repetition": repetition,
            "route_diagnostics_enabled": enabled,
            "route_outputs": correlated,
            "final_parsed_route": final_route,
            "route_correct": route_correct,
            "tool_correct": tool_correct,
            "downstream_final_correct": downstream_correct,
            "retry_required": retry_required,
            "route_fallback_used": fallback_used,
        }
    )
    return StepReport(**values)


def enrich_step_tool_healing(
    report: StepReport,
    attempts: list[dict[str, object]],
    *,
    step: SmokeStep,
    enabled: bool,
) -> StepReport:
    expected_attempt = bool(step.expected_tool_names) if step.expected_tool_names is not None else None
    actual_tool = report.tool_names[0] if len(report.tool_names) == 1 else None
    retry_active = any(attempt.get("active_outcome") == "superseded" for attempt in attempts) or any(
        metric.get("phase") == "tool_call_retry" for metric in report.model_steps
    )
    tool_metrics = [
        metric for metric in report.model_steps if metric.get("phase") in {"tool_call", "tool_call_retry"}
    ]
    correlated: list[dict[str, object]] = []
    for index, attempt in enumerate(attempts):
        active_outcome = attempt.get("active_outcome")
        expected_outcome = step.expected_active_outcome
        expected_tools = step.expected_tool_names
        tool_match = (
            actual_tool in expected_tools
            if actual_tool is not None and expected_tools
            else not report.tool_names
            if expected_tools == ()
            else None
        )
        if active_outcome == "executed" and expected_outcome == "executed" and tool_match is not False:
            execution_assessment = "correct_execution" if report.correctness_category == "correct" else "wrong_execution"
        elif active_outcome == expected_outcome and active_outcome in {
            "rejected_parse", "rejected_schema", "rejected_guardrail", "rejected_policy", "rejected_permission",
        }:
            execution_assessment = "safe_rejection"
        elif active_outcome == "executed" and expected_outcome not in {None, "executed"}:
            execution_assessment = "unsafe_acceptance"
        elif active_outcome in {"superseded", "cancelled", "timeout", None}:
            execution_assessment = "not_terminally_assessed"
        else:
            execution_assessment = "wrong_execution"
        metric = tool_metrics[index] if index < len(tool_metrics) else {}
        correlated.append(
            {
                **attempt,
                "scenario": report.case,
                "scenario_step": report.step,
                "expected_attempt": expected_attempt,
                "expected_tool": expected_tools[0] if expected_tools and len(expected_tools) == 1 else None,
                "executed_tool": actual_tool,
                "template": "native_tool_call",
                "prompt_tokens": metric.get("prompt_tokens"),
                "total_wall_ms": report.wall_ms,
                "active_retry_present": retry_active,
                "strict_active_agreement": attempt.get("agreement") == "exact_match",
                "detector_false_positive": expected_attempt is False and attempt.get("attempt_detected") is True,
                "detector_false_negative": expected_attempt is True and attempt.get("attempt_detected") is False,
                "execution_assessment": execution_assessment,
            }
        )
    values = report.__dict__.copy()
    values.update(
        {
            "tool_healing_diagnostics_enabled": enabled,
            "tool_healing_attempts": correlated,
        }
    )
    return StepReport(**values)


def route_expectation_result(
    expected_route: str | None,
    final_route: object,
    events: list[dict[str, object]],
    tool_names: list[str],
) -> bool | None:
    if expected_route is None or not events:
        return None
    if expected_route == "CHAT":
        direct = any(event.get("route_output_class") == "direct_prose" for event in events)
        return (final_route == "CHAT" or direct) and not tool_names
    return final_route == expected_route


def tool_expectation_result(expected: tuple[str, ...] | None, actual: list[str]) -> bool | None:
    if expected is None:
        return None
    if not expected:
        return not actual
    return bool(actual) and set(actual).issubset(set(expected))


@contextmanager
def final_prefix_environment(mode: str):
    previous = {name: os.environ.get(name) for name in (FINAL_PREFIX_REUSE_ENV, FINAL_PREFIX_EXPERIMENT_ENV)}
    apply_final_prefix_mode(os.environ, mode)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def canonical_gate_environment(mode: str):
    previous = os.environ.get(TOOL_CALL_CANONICAL_GATE_ENV)
    if mode != "inherit":
        os.environ[TOOL_CALL_CANONICAL_GATE_ENV] = "1" if mode == "on" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TOOL_CALL_CANONICAL_GATE_ENV, None)
        else:
            os.environ[TOOL_CALL_CANONICAL_GATE_ENV] = previous


def canonical_gate_metadata(mode: str) -> dict[str, object]:
    config = (
        resolve_tool_call_canonical_gate()
        if mode == "inherit"
        else resolve_tool_call_canonical_gate(
            {TOOL_CALL_CANONICAL_GATE_ENV: "1" if mode == "on" else "0"}
        )
    )
    return {
        "requested_mode": mode,
        "enabled": config.enabled,
        "source": config.source,
        "validation_error": config.validation_error,
    }


@contextmanager
def tool_healing_environment(mode: str):
    previous = os.environ.get(TOOL_CALL_HEALING_ENV)
    if mode != "inherit":
        os.environ[TOOL_CALL_HEALING_ENV] = "1" if mode == "on" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TOOL_CALL_HEALING_ENV, None)
        else:
            os.environ[TOOL_CALL_HEALING_ENV] = previous


def tool_healing_metadata(mode: str) -> dict[str, object]:
    config = (
        resolve_tool_call_healing()
        if mode == "inherit"
        else resolve_tool_call_healing({TOOL_CALL_HEALING_ENV: "1" if mode == "on" else "0"})
    )
    status = tool_call_healing_status()
    effective_enabled = config.enabled and resolve_tool_call_canonical_gate().enabled
    return {
        "requested_mode": mode,
        "enabled": effective_enabled,
        "source": config.source,
        "validation_error": config.validation_error,
        "blocked_reason": None if effective_enabled or not config.enabled else "canonical_gate_disabled",
        "repair_count": status["tool_call_healing_repair_count"],
        "rejection_count": status["tool_call_healing_rejection_count"],
        "last_rules": status["tool_call_healing_last_rules"],
    }


FINAL_PREFIX_MODE_VALUES = {
    "off": {FINAL_PREFIX_REUSE_ENV: "0"},
    "on": {FINAL_PREFIX_REUSE_ENV: "1"},
    "legacy-off": {FINAL_PREFIX_EXPERIMENT_ENV: "0"},
    "legacy-on": {FINAL_PREFIX_EXPERIMENT_ENV: "1"},
    "stable-off-legacy-on": {FINAL_PREFIX_REUSE_ENV: "0", FINAL_PREFIX_EXPERIMENT_ENV: "1"},
    "stable-on-legacy-off": {FINAL_PREFIX_REUSE_ENV: "1", FINAL_PREFIX_EXPERIMENT_ENV: "0"},
    "stable-invalid": {FINAL_PREFIX_REUSE_ENV: "invalid", FINAL_PREFIX_EXPERIMENT_ENV: "1"},
}


def apply_final_prefix_mode(environ: dict[str, str], mode: str) -> None:
    if mode == "inherit":
        return
    environ.pop(FINAL_PREFIX_REUSE_ENV, None)
    environ.pop(FINAL_PREFIX_EXPERIMENT_ENV, None)
    environ.update(FINAL_PREFIX_MODE_VALUES[mode])


def final_prefix_mode_enabled(mode: str) -> bool | None:
    if mode == "inherit":
        return None
    return final_prefix_config_for_mode(mode).enabled


def final_prefix_config_for_mode(mode: str):
    if mode == "inherit":
        return resolve_final_prefix_reuse()
    env: dict[str, str] = {}
    apply_final_prefix_mode(env, mode)
    return resolve_final_prefix_reuse(env)


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
    apply_final_prefix_mode(env, args.final_prefix_mode)
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
        expected = resolve_final_prefix_reuse(env)
        actual = (
            props.get("final_prefix_reuse_enabled"),
            props.get("final_prefix_reuse_source"),
            props.get("final_prefix_reuse_config_error"),
            props.get("final_prefix_reuse_legacy_detected"),
        )
        expected_values = (
            expected.enabled,
            expected.source,
            expected.validation_error,
            expected.legacy_detected,
        )
        if actual != expected_values or props.get("final_prefix_experiment_enabled") is not expected.enabled:
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


def rss_sample(
    process: subprocess.Popen[str],
    *,
    label: str,
    block_id: str,
    sequence: int,
    props: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "rss_sample",
        "label": label,
        "sequence": sequence,
        "server_pid": process.pid,
        "block_id": block_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rss_kib": process_rss_kib(process),
        "final_prefix": final_prefix_props(props or {}),
    }


def summarize_rss_samples(samples: list[dict[str, object]], *, block_id: str) -> dict[str, object]:
    by_label = {str(sample.get("label")): sample.get("rss_kib") for sample in samples}

    def delta(start: str, end: str) -> int | None:
        before = by_label.get(start)
        after = by_label.get(end)
        return after - before if isinstance(before, int) and isinstance(after, int) else None

    restore_values = [by_label.get(label) for label in ("after_restore_10", "after_restore_25", "after_restore_50")]
    complete = all(isinstance(value, int) for value in restore_values)
    linear_growth = None
    if complete:
        first, middle, last = (int(value) for value in restore_values)
        linear_growth = first < middle < last and (last - first) >= 1024
    required_labels = (
        "startup",
        "after_capture",
        "after_restore_10",
        "after_restore_25",
        "after_restore_50",
        "after_invalidation",
        "after_recapture",
    )
    complete = all(label in by_label and isinstance(by_label[label], int) for label in required_labels)
    invalidation = next((sample for sample in samples if sample.get("label") == "after_invalidation"), {})
    recapture = next((sample for sample in samples if sample.get("label") == "after_recapture"), {})
    invalidation_state = invalidation.get("final_prefix") if isinstance(invalidation.get("final_prefix"), dict) else {}
    recapture_state = recapture.get("final_prefix") if isinstance(recapture.get("final_prefix"), dict) else {}
    lifecycle_healthy = (
        invalidation_state.get("initialized") is False
        and invalidation_state.get("prefix_tokens") == 0
        and recapture_state.get("initialized") is True
        and (recapture_state.get("capture_count") or 0) >= 2
    )
    return {
        "type": "lifecycle_summary",
        "operation": "rss",
        "block_id": block_id,
        "server_pid": samples[0].get("server_pid") if samples else None,
        "sample_labels": [sample.get("label") for sample in samples],
        "startup_to_capture_delta_kib": delta("startup", "after_capture"),
        "capture_to_restore50_delta_kib": delta("after_capture", "after_restore_50"),
        "restore50_to_invalidation_delta_kib": delta("after_restore_50", "after_invalidation"),
        "invalidation_to_recapture_delta_kib": delta("after_invalidation", "after_recapture"),
        "linear_growth_suspected": linear_growth,
        "invalidation_initialized": invalidation_state.get("initialized"),
        "recapture_initialized": recapture_state.get("initialized"),
        "complete": complete,
        "passed": complete and lifecycle_healthy,
    }


def namespace_with(args: argparse.Namespace, **overrides: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def lifecycle_pwd_scenario(name: str = "final_prefix_lifecycle") -> SmokeScenario:
    return SmokeScenario(
        name,
        (FINAL_PREFIX_LOCAL_STEPS[0],),
        optional=True,
        allowed_tool_names=TOOL_NAMES,
        isolated_steps=True,
    )


def run_lifecycle_block(
    args: argparse.Namespace,
    *,
    block_id: str,
    calls: int,
    rss_series: bool = False,
) -> LifecycleBlock:
    block_args = namespace_with(args, block_id=block_id, manage_server=True)
    samples: list[dict[str, object]] = []
    reports: list[StepReport] = []
    with final_prefix_environment(block_args.final_prefix_mode), deterministic_web(False), managed_server(block_args) as process:
        if process is None:
            raise RuntimeError("lifecycle checks require a managed server")
        backend = LlamaServerBackend(base_url=block_args.base_url, timeout=block_args.timeout)
        backend.thinking = block_args.server_thinking == "on"
        initial_props = fresh_backend_props(block_args.base_url, 5.0)
        samples.append(rss_sample(process, label="startup", block_id=block_id, sequence=0, props=initial_props))
        restore_milestones = {10, 25, 50}
        sampled_restore_milestones: set[int] = set()
        capture_sampled = False
        for call_index in range(calls):
            batch = run_scenario(
                lifecycle_pwd_scenario(),
                backend=backend,
                workdir=Path(block_args.workdir),
                max_tokens=block_args.max_tokens,
                temperature=block_args.temperature,
                timeout=block_args.timeout,
                tools_mode=block_args.tools,
            )
            reports.extend(batch)
            state = batch[-1].final_prefix if batch else {}
            if not capture_sampled and (state.get("capture_count_delta") or 0) > 0:
                samples.append(
                    rss_sample(
                        process,
                        label="after_capture",
                        block_id=block_id,
                        sequence=len(samples),
                        props=fresh_backend_props(block_args.base_url, 5.0),
                    )
                )
                capture_sampled = True
            props = fresh_backend_props(block_args.base_url, 5.0)
            restore_count = props.get("final_prefix_experiment_restore_count")
            if isinstance(restore_count, int):
                for milestone in sorted(restore_milestones - sampled_restore_milestones):
                    if restore_count >= milestone:
                        samples.append(
                            rss_sample(
                                process,
                                label=f"after_restore_{milestone}",
                                block_id=block_id,
                                sequence=len(samples),
                                props=props,
                            )
                        )
                        sampled_restore_milestones.add(milestone)
            if batch and batch[-1].finish_reason in {"timeout", "error"}:
                break
        if rss_series:
            request_backend_cancel(block_args.base_url, timeout=5.0)
            invalidated_props = wait_for_backend_idle(block_args.base_url, 10.0)
            samples.append(
                rss_sample(
                    process,
                    label="after_invalidation",
                    block_id=block_id,
                    sequence=len(samples),
                    props=invalidated_props,
                )
            )
            recapture = run_scenario(
                lifecycle_pwd_scenario(),
                backend=backend,
                workdir=Path(block_args.workdir),
                max_tokens=block_args.max_tokens,
                temperature=block_args.temperature,
                timeout=block_args.timeout,
                tools_mode=block_args.tools,
            )
            reports.extend(recapture)
            samples.append(
                rss_sample(
                    process,
                    label="after_recapture",
                    block_id=block_id,
                    sequence=len(samples),
                    props=fresh_backend_props(block_args.base_url, 5.0),
                )
            )
        final_props = settled_backend_props(block_args.base_url, block_args.timeout)
        return LifecycleBlock(
            block_id=block_id,
            server_pid=process.pid,
            ctx=block_args.ctx,
            thinking=block_args.server_thinking,
            initial_props=initial_props,
            final_props=final_props,
            reports=reports,
            rss_samples=samples,
        )


def lifecycle_transition_row(operation: str, blocks: list[LifecycleBlock]) -> dict[str, object]:
    transitions = []
    passed = True
    for block in blocks:
        state = final_prefix_props(block.final_props)
        initial_state = final_prefix_props(block.initial_props)
        capture_count = state.get("capture_count") if isinstance(state.get("capture_count"), int) else 0
        restore_count = state.get("restore_count") if isinstance(state.get("restore_count"), int) else 0
        eligible = block.thinking == "off"
        block_passed = (
            initial_state.get("initialized") is False and capture_count >= 1 and restore_count >= 1
            if eligible
            else initial_state.get("initialized") is False and capture_count == 0 and restore_count == 0
        )
        passed = passed and block_passed
        transitions.append(
            {
                "block_id": block.block_id,
                "server_pid": block.server_pid,
                "ctx": block.ctx,
                "thinking": block.thinking,
                "initial_initialized": initial_state.get("initialized"),
                "capture_count": capture_count,
                "restore_count": restore_count,
                "fallback_count": state.get("fallback_count"),
                "eligibility": "eligible" if eligible else "ineligible_thinking",
                "passed": block_passed,
            }
        )
    return {
        "type": "lifecycle_summary",
        "operation": operation,
        "passed": passed,
        "process_ids": [block.server_pid for block in blocks],
        "transitions": transitions,
    }


def run_lifecycle_checks(
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    markdown_path: Path,
) -> int:
    if final_prefix_mode_enabled(args.final_prefix_mode) is not True or not args.manage_server:
        print("error: lifecycle checks require managed server with final-prefix reuse enabled", file=sys.stderr)
        return 2
    reports: list[StepReport] = []
    extra_rows: list[dict[str, object]] = []
    blocks: list[LifecycleBlock] = []
    for operation in args.lifecycle_check:
        if operation == "restart":
            current = [
                run_lifecycle_block(args, block_id="restart-before", calls=2),
                run_lifecycle_block(args, block_id="restart-after", calls=2),
            ]
        elif operation == "ctx-change":
            current = [
                run_lifecycle_block(args, block_id=f"ctx-{args.ctx}", calls=2),
                run_lifecycle_block(
                    namespace_with(args, ctx=args.ctx_change_to),
                    block_id=f"ctx-{args.ctx_change_to}",
                    calls=2,
                ),
            ]
        elif operation == "thinking":
            current = [
                run_lifecycle_block(namespace_with(args, server_thinking="off"), block_id="thinking-off", calls=2),
                run_lifecycle_block(namespace_with(args, server_thinking="on"), block_id="thinking-on", calls=0),
            ]
        else:
            current = [run_lifecycle_block(args, block_id="rss-50", calls=51, rss_series=True)]
            extra_rows.extend(current[0].rss_samples)
            extra_rows.append(summarize_rss_samples(current[0].rss_samples, block_id=current[0].block_id))
        blocks.extend(current)
        extra_rows.append(lifecycle_transition_row(operation, current))
        reports.extend(report for block in current for report in block.reports)
    last = blocks[-1]
    backend = LlamaServerBackend(base_url=args.base_url, timeout=args.timeout)
    env = environment_summary(args=args, backend=backend, props=last.final_props)
    env["server_pid"] = last.server_pid
    env["lifecycle_checks"] = list(args.lifecycle_check)
    write_jsonl(jsonl_path, env, reports, extra_rows=extra_rows)
    write_markdown(markdown_path, env, reports)
    failed = any(row.get("type") == "lifecycle_summary" and row.get("passed") is False for row in extra_rows)
    if failed or scenario_failure_reason(reports) is not None:
        return 1
    print(f"jsonl={jsonl_path}")
    print(f"markdown={markdown_path}")
    return 0


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


def write_jsonl(
    path: Path,
    env: dict[str, object],
    reports: list[StepReport],
    *,
    extra_rows: list[dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "environment", **env}, ensure_ascii=False, sort_keys=True) + "\n")
        for report in reports:
            handle.write(json.dumps({"type": "step", **report.to_json()}, ensure_ascii=False, sort_keys=True) + "\n")
        for summary in summarize_reports(reports):
            handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False, sort_keys=True) + "\n")
        route_summary = summarize_route_classifications(reports)
        if route_summary is not None:
            handle.write(json.dumps(route_summary, ensure_ascii=False, sort_keys=True) + "\n")
        for row in tool_healing_attempt_rows(reports, env):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        healing_summary = summarize_tool_healing(reports)
        if healing_summary is not None:
            handle.write(json.dumps(healing_summary, ensure_ascii=False, sort_keys=True) + "\n")
        for row in extra_rows or []:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def summarize_route_classifications(reports: list[StepReport]) -> dict[str, object] | None:
    diagnostic_reports = [report for report in reports if report.route_diagnostics_enabled]
    if not diagnostic_reports:
        return None
    class_counts = empty_route_class_counts()
    initial_counts = empty_route_class_counts()
    retry_counts = empty_route_class_counts()
    by_family: dict[str, dict[str, int]] = {}
    correctness = {
        route_class: {
            "calls": 0,
            "route_correct": 0,
            "route_wrong": 0,
            "tool_correct": 0,
            "tool_wrong": 0,
            "downstream_correct": 0,
            "downstream_wrong": 0,
            "retry_required": 0,
        }
        for route_class in ROUTE_OUTPUT_CLASSES
    }
    malformed_to_retry = 0
    control_loop_to_retry = 0
    final_successful_decisions = 0
    fallback_count = 0
    surrogate_count = 0
    for report in diagnostic_reports:
        family_counts = by_family.setdefault(report.scenario_family, empty_route_class_counts())
        if report.final_parsed_route is not None:
            final_successful_decisions += 1
        if report.route_fallback_used:
            fallback_count += 1
        initial_classes = {
            event.get("route_output_class")
            for event in report.route_outputs
            if event.get("route_call") == "initial"
        }
        if report.retry_required and "malformed" in initial_classes:
            malformed_to_retry += 1
        if report.retry_required and "control_loop" in initial_classes:
            control_loop_to_retry += 1
        for event in report.route_outputs:
            route_class = event.get("route_output_class")
            if route_class not in ROUTE_OUTPUT_CLASSES:
                continue
            class_counts[route_class] += 1
            family_counts[route_class] += 1
            target = retry_counts if event.get("route_call") == "retry" else initial_counts
            target[route_class] += 1
            correlation = correctness[route_class]
            correlation["calls"] += 1
            _increment_boolean_counts(correlation, "route", event.get("route_correct"))
            _increment_boolean_counts(correlation, "tool", event.get("tool_correct"))
            _increment_boolean_counts(correlation, "downstream", event.get("downstream_final_correct"))
            if event.get("retry_required") is True:
                correlation["retry_required"] += 1
            if event.get("control_loop_surrogate") is True:
                surrogate_count += 1
    return {
        "type": "route_classification_summary",
        "class_counts": class_counts,
        "initial_class_counts": initial_counts,
        "retry_class_counts": retry_counts,
        "malformed_to_retry_transitions": malformed_to_retry,
        "control_loop_to_retry_transitions": control_loop_to_retry,
        "final_successful_decision_count": final_successful_decisions,
        "fallback_count": fallback_count,
        "class_distribution_by_scenario_family": by_family,
        "correctness_by_class": correctness,
        "diagnostic_steps": len(diagnostic_reports),
        "steps_with_route_events": sum(bool(report.route_outputs) for report in diagnostic_reports),
        "missing_route_diagnostic_steps": sum(not report.route_outputs for report in diagnostic_reports),
        "empty_visible_control_output_surrogate_count": surrogate_count,
        "control_loop_surrogate_note": "empty visible output at length with at least 8 output tokens is diagnostic evidence, not exact token-cycle proof",
    }


def empty_route_class_counts() -> dict[str, int]:
    return {route_class: 0 for route_class in ROUTE_OUTPUT_CLASSES}


def tool_healing_attempt_rows(
    reports: list[StepReport],
    env: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report in reports:
        for attempt in report.tool_healing_attempts:
            rows.append(
                {
                    "type": "tool_healing_attempt",
                    **attempt,
                    "model": bounded_model_label(env.get("model")),
                    "ctx": env.get("ctx") if isinstance(env.get("ctx"), int) else None,
                    "threads": env.get("threads") if isinstance(env.get("threads"), int) else None,
                    "mtp": bounded_identifier(env.get("mtp")),
                    "mmproj": bounded_identifier(env.get("mmproj")),
                    "process_id": report.process_id,
                    "block_id": bounded_identifier(report.block_id),
                    "run_order": bounded_identifier(report.run_order),
                    "repetition": report.repetition,
                }
            )
    return rows


def bounded_model_label(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 80:
        return None
    return value if all(character.isalnum() or character in {"_", "-", ".", ":"} for character in value) else None


def summarize_tool_healing(reports: list[StepReport]) -> dict[str, object] | None:
    diagnostic_reports = [report for report in reports if report.tool_healing_diagnostics_enabled]
    if not diagnostic_reports:
        return None
    attempts = [attempt for report in diagnostic_reports for attempt in report.tool_healing_attempts]
    evaluable_attempts = [
        attempt for attempt in attempts
        if attempt.get("active_outcome") not in {"cancelled", "timeout"}
    ]
    detected_attempts = [attempt for attempt in evaluable_attempts if attempt.get("attempt_detected") is True]
    categories = {
        category: 0
        for category in (
            "valid_first_pass", "recoverable_envelope", "recoverable_trailing_comma",
            "recoverable_unambiguous_delimiter", "recoverable_arguments_string",
            "recoverable_structural_alias", "missing_required", "wrong_type", "extra_argument",
            "unknown_tool", "multiple_candidates", "no_attempt", "ambiguous_attempt",
            "budget_truncation", "truncated_attempt",
            "policy_denied", "active_default_used", "active_clamp_used", "active_ignored_extra",
            "executor_error", "superseded", "uncorrelated",
        )
    }
    for attempt in attempts:
        for category in attempt.get("categories") or []:
            if category in categories:
                categories[category] += 1
    expected_positive = [attempt for attempt in attempts if attempt.get("expected_attempt") is True]
    expected_negative = [attempt for attempt in attempts if attempt.get("expected_attempt") is False]
    healing_values = [float(attempt["healing_us"]) for attempt in attempts if isinstance(attempt.get("healing_us"), int | float)]
    assessments = {name: 0 for name in ("correct_execution", "wrong_execution", "safe_rejection", "unsafe_acceptance", "not_terminally_assessed")}
    for attempt in attempts:
        assessment = attempt.get("execution_assessment")
        if assessment in assessments:
            assessments[assessment] += 1
    return {
        "type": "tool_healing_summary",
        "diagnostic_steps": len(diagnostic_reports),
        "attempts": len(attempts),
        "category_counts": categories,
        "first_pass_valid_rate": ratio(
            sum("valid_first_pass" in (attempt.get("categories") or []) for attempt in detected_attempts),
            len(detected_attempts),
        ),
        "first_pass_valid_output_rate": ratio(categories["valid_first_pass"], len(evaluable_attempts)),
        "no_attempt_rate": ratio(categories["no_attempt"], len(evaluable_attempts)),
        "deterministic_repair_candidate_rate": ratio(
            sum(
                attempt.get("formal_repairable") is True
                and "budget_truncation" not in (attempt.get("categories") or [])
                for attempt in detected_attempts
            ),
            len(detected_attempts),
        ),
        "formal_error_rate": ratio(
            sum(
                any(category in TOOL_HEALING_FORMAL_ERRORS for category in attempt.get("categories") or [])
                for attempt in detected_attempts
            ),
            len(detected_attempts),
        ),
        "strict_active_agreement_rate": ratio(sum(attempt.get("strict_active_agreement") is True for attempt in attempts), len(attempts)),
        "detector_false_positive_rate": ratio(sum(attempt.get("detector_false_positive") is True for attempt in expected_negative), len(expected_negative)),
        "detector_false_negative_rate": ratio(sum(attempt.get("detector_false_negative") is True for attempt in expected_positive), len(expected_positive)),
        "active_default_rate": ratio(categories["active_default_used"], len(attempts)),
        "active_clamp_rate": ratio(categories["active_clamp_used"], len(attempts)),
        "active_ignored_extra_rate": ratio(categories["active_ignored_extra"], len(attempts)),
        "multiple_candidate_rate": ratio(categories["multiple_candidates"], len(attempts)),
        "truncation_rate": ratio(categories["truncated_attempt"], len(attempts)),
        "healing_us_median": round(statistics.median(healing_values), 1) if healing_values else None,
        "healing_us_p95": percentile(healing_values, 0.95),
        "execution_assessments": assessments,
        "missing_diagnostic_steps": sum(not report.tool_healing_attempts for report in diagnostic_reports),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return round(ordered[index], 1)


def run_tool_healing_replay(workdir: Path) -> list[dict[str, object]]:
    corpus = (
        ("valid", '<tool_call>{"name":"system_info","arguments":{}}</tool_call>', True),
        ("trailing_comma", '<tool_call>{"name":"system_info","arguments":{},}</tool_call>', True),
        ("missing_delimiter", '<tool_call>{"name":"system_info","arguments":{}', True),
        ("unclosed_tag", '<tool_call>{"name":"system_info","arguments":{}}', True),
        ("external_text", 'prefix <tool_call>{"name":"system_info","arguments":{}}</tool_call> suffix', True),
        ("tool_alias", '{"tool":"system_info","arguments":{}}', True),
        ("top_level_arguments", '{"name":"list_directory","path":"."}', True),
        ("arguments_string", '{"name":"system_info","arguments":"{}"}', True),
        ("unknown_tool", '{"name":"not_registered","arguments":{}}', True),
        ("missing_required", '{"name":"fetch_url","arguments":{}}', True),
        ("wrong_type", '{"name":"fetch_url","arguments":{"url":3}}', True),
        ("extra_argument", '{"name":"system_info","arguments":{"extra":true}}', True),
        ("normal_tool_name", 'system_info is the registered name of a tool.', False),
        ("json_example", 'Example only: {"name":"system_info","arguments":{}}.', False),
        ("multiple", '<tool_call>{"name":"system_info","arguments":{}}</tool_call><tool_call>{"name":"list_directory","arguments":{}}</tool_call>', True),
        ("policy", '{"name":"exec_shell_full_command","arguments":{"command":"rm -f note.txt"}}', True),
        ("interrupted", '<tool_call>{"name":"fetch_url","arguments":{"url":"https://example.invalid', True),
    )
    rows: list[dict[str, object]] = []
    false_positive = 0
    false_negative = 0
    category_counts: dict[str, int] = {}
    for case, text, expected in corpus:
        report = analyze_tool_attempt(
            text=text,
            tool_calls=[],
            tool_definitions=tool_definitions(),
            allowed_tool_names=TOOL_NAMES,
            workdir=workdir,
            user_prompt="show note.txt" if case == "policy" else "perform the labelled replay operation",
        )
        attempt = {
            "candidate_source": report.candidate_source,
            "repairs": list(report.repairs),
            "strict_outcome": report.outcome,
            "parse_error": report.parse_error,
            "validation_error": report.validation_error,
            "attempt_detected": report.attempt_detected,
            "formal_repairable": report.formal_repairable,
        }
        categories = tool_healing_categories(attempt)
        fp = not expected and report.attempt_detected
        fn = expected and not report.attempt_detected
        false_positive += fp
        false_negative += fn
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        rows.append(
            {
                "type": "tool_healing_replay_case",
                "case": case,
                "expected_attempt": expected,
                "detected": report.attempt_detected,
                "false_positive": fp,
                "false_negative": fn,
                "strict_outcome": report.outcome,
                "categories": categories,
                "repairs": list(report.repairs),
                "parse_error": report.parse_error,
                "validation_error": report.validation_error,
                "formal_repairable": report.formal_repairable,
                "formal_repair_reason": report.formal_repair_reason,
                "formal_argument_count": report.formal_argument_count,
            }
        )
    rows.append(
        {
            "type": "tool_healing_replay_summary",
            "cases": len(corpus),
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "false_positive_rate": ratio(false_positive, sum(not expected for _, _, expected in corpus)),
            "false_negative_rate": ratio(false_negative, sum(expected for _, _, expected in corpus)),
            "category_counts": category_counts,
        }
    )
    replay_summary = rows.pop()
    rows.extend(run_tool_validation_divergence_replay(workdir))
    rows.append(replay_summary)
    return rows


def run_tool_validation_divergence_replay(workdir: Path) -> list[dict[str, object]]:
    cases = (
        ("system_valid", "system_info", {}, None),
        ("system_extra", "system_info", {"ignored_extra": True}, "active_ignored_extra"),
        ("system_wrong_type", "system_info", {"include_cpu": "yes"}, "active_default_used"),
        ("list_valid", "list_directory", {"path": ".", "max_entries": 10}, None),
        ("list_extra", "list_directory", {"path": ".", "ignored_extra": True}, "active_ignored_extra"),
        ("list_wrong_type", "list_directory", {"path": ".", "recursive": "yes"}, "active_default_used"),
        ("list_clamp", "list_directory", {"path": ".", "max_entries": 2000}, "active_clamp_used"),
        ("shell_valid", "exec_shell_full_command", {"command": "printf validation-ok"}, None),
        ("shell_extra", "exec_shell_full_command", {"command": "printf validation-ok", "ignored_extra": True}, "active_ignored_extra"),
        ("shell_wrong_type", "exec_shell_full_command", {"command": "printf validation-ok", "timeout": "slow"}, "active_default_used"),
        ("shell_clamp", "exec_shell_full_command", {"command": "printf validation-ok", "timeout": 999}, "active_clamp_used"),
        ("shell_policy", "exec_shell_full_command", {"command": "rm -f note.txt"}, "policy_denied"),
        ("fetch_missing", "fetch_url", {}, "missing_required"),
        ("fetch_extra", "fetch_url", {"url": "http://127.0.0.1:9/fixture", "ignored_extra": True}, "active_ignored_extra"),
        ("fetch_wrong_type", "fetch_url", {"url": "http://127.0.0.1:9/fixture", "timeout": "slow"}, "active_default_used"),
        ("fetch_clamp", "fetch_url", {"url": "http://127.0.0.1:9/fixture", "timeout": 999}, "active_clamp_used"),
    )
    definitions = tool_definitions()
    allowed = tuple(TOOL_NAMES)
    executor = HybridToolExecutor(
        backend=None,
        workdir=workdir,
        allowed_tool_names=allowed,
        user_prompt="show note.txt",
    )
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for case, name, arguments, expected_divergence in cases:
        report = analyze_tool_attempt(
            text="",
            tool_calls=[
                {
                    "id": "validation-replay",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)},
                }
            ],
            tool_definitions=definitions,
            allowed_tool_names=allowed,
            workdir=workdir,
            user_prompt="show note.txt",
            finish_reason="tool_calls",
        )
        execution = _execute_with_canonical_gate(executor, name, arguments, enabled=False)
        gated = _execute_with_canonical_gate(executor, name, arguments, enabled=True)
        attempt = {
            "attempt_detected": report.attempt_detected,
            "strict_outcome": report.outcome,
            "validation_error": report.validation_error,
            "active_outcome": execution.terminal_outcome,
        }
        categories = tool_healing_categories(attempt)
        observed = next(
            (
                category
                for category in (
                    "active_default_used", "active_clamp_used", "active_ignored_extra",
                    "policy_denied", "missing_required",
                )
                if category in categories
            ),
            None,
        )
        if observed:
            counts[observed] += 1
        rows.append(
            {
                "type": "tool_validation_divergence_case",
                "case": case,
                "tool_name_hash": hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
                "arguments_hash": hashlib.sha256(
                    json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:16],
                "argument_count": len(arguments),
                "strict_outcome": report.outcome,
                "strict_error": report.validation_error,
                "active_outcome": execution.terminal_outcome,
                "active_reason": execution.terminal_reason,
                "gate_on_outcome": gated.terminal_outcome,
                "gate_on_reason": gated.terminal_reason,
                "off_result_hash": hashlib.sha256(execution.result.content.encode("utf-8")).hexdigest()[:16],
                "on_result_hash": hashlib.sha256(gated.result.content.encode("utf-8")).hexdigest()[:16],
                "output_equal": execution.result.content == gated.result.content,
                "new_execution_enabled": gated.terminal_outcome == "executed" and execution.terminal_outcome != "executed",
                "divergence_category": observed,
                "expected_divergence": expected_divergence,
                "matched_expectation": observed == expected_divergence,
                "formal_repairable": report.formal_repairable,
            }
        )
    rows.append(
        {
            "type": "tool_validation_divergence_summary",
            "cases": len(cases),
            "matched_expectations": sum(row.get("matched_expectation") is True for row in rows),
            "active_default_used": counts["active_default_used"],
            "active_clamp_used": counts["active_clamp_used"],
            "active_ignored_extra": counts["active_ignored_extra"],
            "policy_denied": counts["policy_denied"],
            "missing_required": counts["missing_required"],
            "repair_executed": False,
            "new_executions_enabled": sum(row.get("new_execution_enabled") is True for row in rows),
            "valid_off_on_terminal_matches": sum(
                row.get("strict_outcome") == "valid_shadow_candidate"
                and row.get("active_outcome") == row.get("gate_on_outcome")
                for row in rows
            ),
        }
    )
    return rows


def _execute_with_canonical_gate(
    executor: HybridToolExecutor,
    name: str,
    arguments: dict[str, object],
    *,
    enabled: bool,
):
    key = "ORBIT_TOOL_CALL_CANONICAL_GATE"
    previous = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        return executor.execute(name, arguments, chunk_budget={})
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _increment_boolean_counts(target: dict[str, int], prefix: str, value: object) -> None:
    if value is True:
        target[f"{prefix}_correct"] += 1
    elif value is False:
        target[f"{prefix}_wrong"] += 1


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

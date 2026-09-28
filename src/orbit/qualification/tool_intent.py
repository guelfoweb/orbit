"""Observer-only tool-intent gate. Never imported by production routing.

The oracle labels are used AFTER the real runtime chooses a route/tool. The
executor boundary is replaced, including artifact publication. No model-selected
operation or final-answer generation runs in this harness.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

from orbit.backend.base import ChatResult, StreamConsumerAbort, TokenCount
from orbit.runtime.chat import ChatRuntime
from orbit.runtime.command_request import (
    RouteOutputClass, classify_route_output, parse_command_decision,
    parse_command_decision_from_tool_calls,
)
from orbit.runtime.kv_diag import current_phase
from orbit.runtime.messages import ROUTE_SYSTEM_PROMPT
from orbit.runtime.tool_backends import HybridToolExecutor
from orbit.runtime.tool_contract import validate_canonical_tool_call
from orbit.runtime import environments, tool_loop
from orbit.terminal.tool_mode import allowed_tool_names_for_spec
from orbit.tool_contract_config import resolve_tool_call_canonical_gate

from .schema import Status

CORPUS_SHA256 = "de9f6e5d594f24b0e235ed4a707bc7d836b5bffafd9df007c935d8144b838a9f"
MAX_MODEL_CALLS = 6


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_corpus(path: Path | str) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError("tool-intent-v1 corpus differs from the qualified 24-case oracle")
    return json.loads(raw)


def load_replay(path: Path | str, corpus: dict) -> dict:
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate replay key")
            obj[key] = value
        return obj

    def bad_constant(value):
        raise ValueError("non-finite replay number: " + value)

    replay = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                        parse_constant=bad_constant)
    if replay["version"] != 1 or replay["corpus_sha256"] != CORPUS_SHA256:
        raise ValueError("replay corpus/version mismatch")
    ids = [row["case_id"] for row in replay["cases"]]
    if len(ids) != 24 or len(set(ids)) != 24 or set(ids) != {c["id"] for c in corpus["cases"]}:
        raise ValueError("replay requires each of the 24 cases exactly once")
    for case in replay["cases"]:
        if not 1 <= len(case["calls"]) <= MAX_MODEL_CALLS:
            raise ValueError("invalid recorded call count")
        for call in case["calls"]:
            result = ChatResult(**call["result"])
            if not isinstance(result.content, str) or not isinstance(result.tool_calls, list):
                raise ValueError("invalid recorded result")
            if not isinstance(call["deltas"], list) or any(not isinstance(d, str) for d in call["deltas"]):
                raise ValueError("invalid recorded deltas")
            for field in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                n = getattr(result, field)
                if n is not None and (type(n) is not int or n < 0):
                    raise ValueError("invalid recorded token metric")
            wall = call["wall_seconds"]
            if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
                raise ValueError("invalid recorded wall")
    return replay


class _Boundary(BaseException):
    def __init__(self, kind: str, **details):
        self.kind = kind
        self.details = details


class _InterceptedExecutor(HybridToolExecutor):
    def execute(self, name, arguments, *, chunk_budget, canonical_decision=None):
        # The normal runtime preflight must already have run. Recheck at the
        # substituted executor as the actual executor would; never call super.
        if canonical_decision is None:
            raise _Boundary("ERROR", reason="missing_canonical_preflight")
        decision = validate_canonical_tool_call(
            name, arguments, tool_definitions=self.tool_definitions(),
            allowed_tool_names=self.allowed_tool_names, workdir=self.workdir,
            user_prompt=self.user_prompt,
        )
        accepted = canonical_decision.accepted and decision.accepted
        raise _Boundary("TOOL" if accepted else "REJECTED", tool=name,
                        arguments=arguments, canonical=asdict(decision))


class _ProbeBackend:
    """Bounded observation around the unchanged production requests."""
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def thinking(self):
        return self.backend.thinking

    @thinking.setter
    def thinking(self, value):
        self.backend.thinking = value

    def _phase(self):
        phase = current_phase()
        if phase not in {"route", "route_retry", "tool_call", "tool_call_retry"}:
            raise _Boundary("CHAT_BOUNDARY" if phase in {"chat_final", "chat_final_retry"}
                            else "DEFERRED", phase=phase)
        return phase

    def count_chat_tokens(self, *args, **kwargs):
        self._phase()
        return self.backend.count_chat_tokens(*args, **kwargs)

    def chat(self, messages, **kwargs):
        return self._invoke(False, messages, **kwargs)

    def chat_stream(self, messages, **kwargs):
        return self._invoke(True, messages, **kwargs)

    def artifact_content_stream(self, *args, **kwargs):
        raise _Boundary("ERROR", reason="artifact_generation_not_intercepted")

    def _invoke(self, stream, messages, **kwargs):
        phase = self._phase()
        if len(self.calls) >= MAX_MODEL_CALLS:
            raise _Boundary("ERROR", reason="harness_call_limit")
        row = {"phase": phase, "request": copy.deepcopy({
            "messages": messages, "temperature": kwargs["temperature"],
            "max_tokens": kwargs["max_tokens"], "tools": kwargs.get("tools"),
        }), "result": None, "deltas": [], "error": None}
        self.calls.append(row)
        started = time.monotonic()
        consumed_before = len(self.backend.consumed) if isinstance(self.backend, ReplayBackend) else 0
        if stream:
            sink = kwargs["on_delta"]

            def capture(text):
                row["deltas"].append(text)
                sink(text)

            kwargs["on_delta"] = capture
        try:
            result = (self.backend.chat_stream if stream else self.backend.chat)(messages, **kwargs)
            row["result"] = asdict(result)
            return result
        except StreamConsumerAbort as error:
            row["error"] = type(error).__name__
            if error.prompt_metrics is not None:
                row["prompt_metrics"] = asdict(error.prompt_metrics)
            raise
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            row["wall_seconds"] = time.monotonic() - started
            row["request_digest"] = digest(row["request"])
            # Replay provenance/metrics refer to the original measurement, not
            # to the time spent replaying callbacks on this machine.
            if isinstance(self.backend, ReplayBackend) and len(self.backend.consumed) > consumed_before:
                row["recorded"] = copy.deepcopy(self.backend.consumed[-1])


class ReplayBackend:
    thinking = False

    def __init__(self, calls, *, context_tokens=8192):
        self.pending = copy.deepcopy(calls)
        self.consumed = []
        self.context_tokens = context_tokens

    def supports_exact_context_admission(self):
        return True

    def count_chat_tokens(self, messages, *, tools=None, **kwargs):
        if not self.pending:
            raise ValueError("recorded calls exhausted")
        row = self.pending[0]
        if digest({"messages": messages, "tools": tools}) != row["messages_digest"]:
            raise ValueError("recorded request differs: messages/tools")
        return TokenCount(**row["exact_count"])

    def chat(self, messages, **kwargs):
        return self._respond(messages, **kwargs)

    def chat_stream(self, messages, **kwargs):
        return self._respond(messages, **kwargs)

    def _respond(self, messages, *, temperature, max_tokens, tools=None,
                 on_delta=None, on_progress=None, **kwargs):
        if not self.pending:
            raise ValueError("recorded calls exhausted")
        row = self.pending[0]
        actual = {"messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens, "tools": tools}
        if digest(actual) != row["request_digest"] or current_phase() != row["phase"]:
            raise ValueError("recorded request differs: phase/parameters")
        self.consumed.append(self.pending.pop(0))
        if on_delta:
            for delta in row["deltas"]:
                on_delta(delta)
        return ChatResult(**row["result"])


def observe_case(case: dict, backend, workdir: Path, *, context_tokens: int = 8192) -> dict:
    probe = _ProbeBackend(backend)
    started = time.monotonic()
    outcome = {"kind": "ERROR", "details": {}}
    if not resolve_tool_call_canonical_gate().enabled:
        return {"kind": "ERROR", "details": {"reason": "canonical_gate_disabled"},
                "calls": [], "wall_seconds": 0.0}
    runtime = ChatRuntime(backend=probe, system_prompt=ROUTE_SYSTEM_PROMPT,
                          context_tokens=context_tokens)
    runtime.messages.extend(copy.deepcopy(case["history"]))
    try:
        # Scoped to this sequential qualification invocation, restored even on
        # cancellation. No real executor or artifact writer is reachable.
        with patch.object(tool_loop, "HybridToolExecutor", _InterceptedExecutor), patch.object(
            environments, "read_full_document_snapshot",
            side_effect=_Boundary("DEFERRED", reason="document_acquisition_outside_tool_gate"),
        ):
            result = runtime.ask_auto(case["prompt"], temperature=0, max_tokens=256,
                workdir=workdir, max_loops=MAX_MODEL_CALLS,
                allowed_tool_names=allowed_tool_names_for_spec("on"), on_progress=lambda _: None)
        # Only a direct, unchanged model answer is CHAT. Runtime-generated
        # rejection text or another route (e.g. ANALYSIS) is not a chat answer.
        last = probe.calls[-1].get("result") if probe.calls else None
        parsed = parse_command_decision(result.content)
        classification = classify_route_output(
            result.content, parsed_decision=parsed, parser_source="content",
            direct_prose=True, finish_reason=result.finish_reason,
            output_tokens=result.completion_tokens,
        ).classification
        direct_chat = (last == asdict(result) and result.finish_reason == "stop"
                       and bool(result.content.strip()) and not result.tool_calls
                       and classification == RouteOutputClass.DIRECT_PROSE)
        outcome = {"kind": "DIRECT_CHAT" if direct_chat else "REJECTED",
                   "details": {"returned": asdict(result)}}
    except _Boundary as boundary:
        outcome = {"kind": boundary.kind, "details": boundary.details}
    except Exception as error:
        outcome = {"kind": "ERROR", "details": {"reason": type(error).__name__ + ": " + str(error)}}
    outcome.update(calls=probe.calls, wall_seconds=time.monotonic() - started)
    if isinstance(backend, ReplayBackend) and backend.pending:
        outcome["details"]["unconsumed_recorded_calls"] = len(backend.pending)
        outcome["kind"] = "ERROR"
    return outcome


def _result(call):
    return call.get("recorded", {}).get("result") or call.get("result")


def _explicit_chat(call):
    result = call.get("result")
    if not result or result["finish_reason"] != "stop":
        return False
    decision = (parse_command_decision_from_tool_calls(result["tool_calls"])
                or parse_command_decision(result["content"]))
    return decision is not None and decision.route.value == "CHAT"


def _invalid_call(call):
    result = _result(call)
    if call.get("error") or result is None or result["finish_reason"] not in {"stop", "tool_calls"}:
        return True
    if call["phase"] in {"route", "route_retry"}:
        decision = parse_command_decision_from_tool_calls(result["tool_calls"])
        source = "tool_calls" if decision else "content"
        decision = decision or parse_command_decision(result["content"])
        classification = classify_route_output(
            result["content"], parsed_decision=decision, parser_source=source,
            direct_prose=True, finish_reason=result["finish_reason"],
            output_tokens=result["completion_tokens"],
        ).classification
        return classification in {RouteOutputClass.MALFORMED, RouteOutputClass.CONTROL_LOOP}
    return False


def score_case(case: dict, observation: dict) -> dict:
    calls = observation["calls"]
    kind = observation["kind"]
    details = observation["details"]
    tool = details.get("tool")
    authorized = kind == "TOOL" and details.get("canonical", {}).get("terminal_decision") == "accepted"
    content = case["expected"] == "content"
    explicit_chat = kind == "CHAT_BOUNDARY" and any(_explicit_chat(c) for c in calls)
    chat = explicit_chat or kind == "DIRECT_CHAT"
    invalid = not (authorized or chat) or any(_invalid_call(c) for c in calls)
    false_execution = content and authorized
    missed = not content and not authorized
    wrong = not content and authorized and tool not in case["allowed_tools"]
    terminal = calls[-1].get("result") if calls else None
    terminal_valid = bool(terminal and not calls[-1].get("error")
                          and terminal["finish_reason"] in {"stop", "tool_calls"})
    passed = terminal_valid and not (false_execution or missed or wrong)
    passed = passed and (chat if content else authorized)

    def count(field):
        values = [(_result(c) or {}).get(field) for c in calls]
        return sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None

    recorded_walls = [c.get("recorded", {}).get("wall_seconds") for c in calls]
    return {
        "id": case["id"], "language": case["language"], "category": case["category"],
        "expected_mode": "CHAT" if content else "TOOL", "observed_mode": "CHAT" if chat else kind,
        "tool": tool, "arguments": details.get("arguments"), "status": Status.PASS.value if passed else Status.FAIL.value,
        "false_execution": false_execution, "missed_execution": missed, "wrong_tool": wrong,
        "invalid_route_protocol": bool(invalid), "retry": sum("retry" in c["phase"] or "repair" in c["phase"] for c in calls),
        "recovered": bool(passed and invalid),
        "model_calls": len(calls), "input_tokens": count("prompt_tokens"), "output_tokens": count("completion_tokens"),
        "wall_seconds": observation["wall_seconds"],
        "recorded_wall_seconds": sum(recorded_walls) if recorded_walls and all(v is not None for v in recorded_walls) else None,
        "details": details, "calls": calls,
        "manual_check": "Argument purpose and generated prose/content are not semantically scored.",
    }


def summarize(corpus: dict, observations: list[dict], *, provenance: dict) -> dict:
    cases = corpus["cases"]
    expected = [c["id"] for c in cases]
    ids = [o["id"] for o in observations]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("gate requires each of the 24 cases exactly once")
    indexed = {o["id"]: o for o in observations}
    rows = [score_case(c, indexed[c["id"]]) for c in cases]
    metrics = {k: sum(r[k] for r in rows) for k in (
        "false_execution", "missed_execution", "wrong_tool", "invalid_route_protocol", "retry", "model_calls")}
    metrics.update(completed=sum(bool(r["model_calls"]) for r in rows), total=len(cases),
                   passed=sum(r["status"] == "PASS" for r in rows))
    return {"schema_version": 1, "capability": "tool_intent", "corpus_sha256": CORPUS_SHA256,
            "status": "PASS" if metrics["completed"] == 24 and all(r["status"] == "PASS" for r in rows) else "FAIL",
            "provenance": provenance, "metrics": metrics, "cases": rows,
            "scope": "First authorized tool handoff or CHAT decision; actual tool executions always zero. No ranking or full-task semantic certification."}


def write_report(report: dict, output: Path | str) -> None:
    path = Path(output)
    if path.suffix != ".json":
        raise ValueError("output must end in .json (the Markdown report uses .md)")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"# Tool-intent qualification: {report['status']}", "", report["scope"], "",
             "| Case | Expected | Observed | Tool | False | Missed | Wrong | Invalid | Retry | Calls | In / Out | Wall s | Status |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in report["cases"]:
        cells = [r['id'], r['expected_mode'], r['observed_mode'], r['tool'] or '—',
                 *[str(int(r[k])) for k in ('false_execution','missed_execution','wrong_tool','invalid_route_protocol','retry','model_calls')],
                 f"{r['input_tokens']} / {r['output_tokens']}", f"{r['wall_seconds']:.3f}", r['status']]
        lines.append('| ' + ' | '.join(str(c).replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |')
    lines += ["", "Wall is the current probe/replay time. Historical capture wall is separate in JSON; missing token counts stay null.",
              "Tool proposals are intercepted after validation. Argument purpose, prose correctness and later workflow require separate qualification.",
              "A fallback to CHAT after invalid/truncated ROUTE is not counted as qualified CHAT.",
              "", "Metrics: `" + json.dumps(report['metrics'], sort_keys=True) + "`", ""]
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')

"""One analyst step: one model call, at most one action, then hand back.

The autonomous version of this loop kept going until it decided it was
finished, and the recorded runs show what that cost -- eight actions deep,
most of the chain recovered, and the one attestation an analyst would have
asked for on turn two never produced. So this runtime does the opposite:
it takes one instruction, makes exactly one model call, runs at most one
sandboxed action, records the evidence, and stops. Whatever happens next is
the analyst's decision, not the model's.

The stop is structural rather than advisory. `step()` has no loop and no
path from a tool result back to another model call: continuing requires
calling `step()` again with new analyst input. That is the whole design.

Orchestration is all that is new here. The model call goes through the same
`ChatBackend.chat_stream` that ChatRuntime uses, so profile handling,
tokenization, streaming, cancellation, metrics, tool-call parsing and KV
bookkeeping stay in one place and the backend never learns that CHAT and
ANALYSIS are different things.

History is appended in the order it happened -- request, tool call, tool
result -- before control returns, so step N's messages are a prefix of step
N+1's. Nothing is reconstructed or rewritten between steps, which is what a
later exact-prefix KV strategy will need.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from orbit.backend.base import ChatBackend, Message
from orbit.runtime.analysis_sandbox import (
    WORK_MOUNT,
    AnalysisResult,
    SandboxUnavailable,
    execute_analysis,
    scratch_baseline,
)
from orbit.runtime.evidence import EvidenceRecord, EvidenceStore, final_card
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.tool_calls import tool_call_id

ANALYSIS_TOOL_NAME = "execute_analysis"

# The phase this runtime declares around its one model call. It names a kind of
# call, not a mode: the backend uses it the way it already uses "route", to know
# which rolling checkpoint this prompt continues, and learns nothing about CHAT
# or ANALYSIS from it.
ANALYSIS_STEP_PHASE = "analysis_step"

# The phase a report declares. Distinct from a step because it is a different
# kind of call -- no tools, no action -- and the KV lineages are keyed by what
# the caller declares, so naming it separately keeps a report from being
# mistaken for a link in the analysis chain.
ANALYSIS_REPORT_PHASE = "analysis_report"

# What one report may read from the store. The evidence cards are already
# bounded by `final_card`; this caps how many of them a single report carries,
# so a long session cannot grow the report prompt without limit.
MAX_REPORT_EVIDENCE_RECORDS = 12

ANALYSIS_REPORT_INSTRUCTION = (
    "Report on the evidence already collected. Run nothing: this turn has no "
    "tools and performs no analysis.\n"
    "Ground every finding in that evidence and cite it as evidence:<id>. "
    "Anything the evidence does not establish is unresolved -- say so rather "
    "than supplying it.\n"
    "Cover, briefly and only where the evidence supports it: confirmed "
    "findings; indicators; artifacts produced; behaviour established; what "
    "remains unresolved; and the single next step most worth taking."
)

NO_EVIDENCE_REPORT = "No analysis evidence has been collected yet."

# Stable prefix. Everything here is identical for every step of every
# analysis on a given profile, which is what makes a future exact-prefix
# prewarm possible. Nothing volatile belongs above this line: no source
# path, no hash, no session id, no timestamp, no analyst text.
ANALYSIS_SYSTEM_PROMPT = (
    "You are performing static analysis of one artifact in an isolated offline workspace.\n"
    "/workspace/input is the artifact file itself, mounted read-only: read exactly\n"
    "that path and never append the original filename or a subpath to it\n"
    "(`orbit_tools.SOURCE_PATH` is it). There is no network.\n"
    "Inspect and transform it by writing Python and running it with execute_analysis; "
    "`import orbit_tools` provides read_file(path, offset, limit).\n"
    "Write bounded files under /workspace/work to keep a derived artifact.\n"
    "Perform at most one execute_analysis action per turn, then stop and report what it produced.\n"
    "Base every claim on the artifact or on output an action actually produced. "
    "State plainly when something is unresolved rather than filling the gap."
)

ANALYSIS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ANALYSIS_TOOL_NAME,
        "description": (
            "Run one bounded Python program in the isolated analysis workspace. "
            "/workspace/input is the read-only artifact file itself, not a directory; "
            "write derived files under /workspace/work. "
            "Print the facts you want recorded as evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python source. `import orbit_tools` for "
                        "read_file(path, offset=0, limit=65536) -> str."
                    ),
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

# What may enter model-visible history from one action. The sandbox permits
# 8 MiB of scratch; that ceiling governs what a program may write, not what
# a prompt should carry. Full output stays addressable in EvidenceStore.
#
# 3200 rather than the previous 8192, measured rather than chosen. A real
# failing turn carried a 7767-char observation -- under the old cap, so it was
# never truncated -- which tokenised to 6226 tokens: 98.3% of that step's
# 6331-token prefill, and 231.8s of the 376s the turn took.
#
# The bound is in characters because runtime has no exact tokenizer and must
# not grow one; the backend owns tokenisation. `estimate_text_tokens` exists
# but under-counts this content roughly threefold (0.28-0.42 of the real count
# on the measured corpora), so budgeting with it would admit about three times
# the intended prompt. A character cap is the honest instrument here, and the
# number is derived from the densest content actually observed:
#
#   8 preserved successful observations: 1197-2435 chars, 788-2044 tokens
#   densest ratio in that corpus:        1.123 chars/token (obfuscated JS)
#   3200 chars is therefore about 2850 tokens on content of that kind
#
# That figure describes the measured corpora, not a guarantee. A character cap
# does not bound tokens: rare codepoints reach 4 tokens per character on this
# tokenizer, so 3200 characters of them would be roughly 12800 tokens. The cap
# is still the right instrument -- runtime has no exact tokenizer and must not
# grow one -- and it is a strict improvement at every density, because the
# 8192 it replaces was four times worse on exactly that input. Analysis input
# is attacker-supplied by definition, so this is a bound on the ordinary case
# and a reduction, not a defence.
#
# Every one of the successful observations fits untruncated, and the failing
# one drops from 6226 to about 2200 tokens. Full output remains byte-complete
# and re-attestable in EvidenceStore; only this projection reaches the model,
# which is told the output exists rather than being handed a way to fetch it.
MAX_EVIDENCE_CHARS = 3200

# What one analyst step may generate. Derived from the 8 model-authored actions
# of the preserved successful trajectory, tokenised with the real Ornith
# tokenizer:
#
#   outputs: 60, 80, 84, 296, 351, 354, 858, 1417, 1953 tokens
#   median 351, largest 1953
#
# The previous 1024 was not a qualified number, and the measurement shows it
# was already too small: it would have truncated two of those nine calls. That
# is what the failing turn actually hit -- 1024 output tokens, finish_reason
# `length`, and a tool call cut off mid-JSON after 1488 characters. The failure
# was the ceiling being too low, not generation running away.
#
# 2048 is the smallest value that truncates none of them, clearing the largest
# by 5%. Headroom beyond that is not free: decode runs about 7 tok/s here, so
# every extra 1000 tokens of allowance is another ~140s that a doomed step
# spends before it can be refused. At 2048 the worst refused turn costs roughly
# what the observed failure already cost (~374s against 376s measured), while
# no successful action is cut short. A larger 2560 would have bought 31%
# headroom nothing has ever needed and made the bad case ~70s worse.
QUALIFIED_ANALYSIS_MAX_TOKENS = 2048

# The per-action allowance the qualified sandbox enforces is a limit on what
# ONE action may produce. A persistent workspace also needs a ceiling on what
# a whole session may retain, or an analyst-guided session has no bound at all.
#
# The preserved 20 Aug runs cannot supply this number: those trajectories wrote
# no derived files whatsoever (`open(` never appears in any recorded action)
# and ran at most 9 actions, so measured footprint is zero. Rather than
# multiply the per-action allowance by a made-up action count, these are
# explicit conservative constants: 64 MiB is eight full per-action allowances,
# and 256 files is eight times the per-action file count -- enough headroom for
# a session far longer than any recorded run, small enough to stay a bound.
# Revisit with real analyst-session telemetry.
MAX_SESSION_SCRATCH_BYTES = 64 * 1024 * 1024
MAX_SESSION_SCRATCH_FILES = 256
SESSION_CAPACITY_EXHAUSTED = "session artifact capacity exhausted"


@dataclass(frozen=True)
class AnalysisSource:
    """An immutable snapshot of the artifact under analysis.

    The sandbox hashes its input before and after a run, which catches a file
    changing underneath it -- but not a file swapped before the mount. Taking
    the bytes once and mounting only this copy removes that window, and makes
    the analysis identity the content rather than a path someone else can
    repoint.
    """

    snapshot_path: Path
    sha256: str
    size_bytes: int
    original_path: str

    @property
    def analysis_id(self) -> str:
        return self.sha256[:16]


@dataclass
class AnalysisWorkspace:
    """Storage owned by one analysis session, for its whole lifetime.

    The sandbox allocates a throwaway scratch directory per action when the
    caller supplies none, which is right for a one-shot execution and wrong
    for an investigation: a derived artifact recorded in step one is gone
    before step two can read it, leaving a hash that names bytes nobody can
    produce. The session therefore owns one directory and passes it in, so
    what an action writes is still there for the next action -- and still
    there to re-hash when someone asks whether the record is honest.

    Removal is explicit. `close()` deletes the workspace; nothing here waits
    for a finalizer, because a runtime that outlives several turns should not
    depend on when the collector happens to run. After abnormal process death
    the residue is an ordinary temporary directory, cleaned by the OS on its
    own schedule -- stated plainly rather than dressed up as a guarantee.
    """

    root: Path
    _closed: bool = False

    @classmethod
    def create(cls) -> "AnalysisWorkspace":
        root = Path(tempfile.mkdtemp(prefix="orbit-analysis-session-"))
        (root / "source").mkdir()
        (root / "work").mkdir()
        return cls(root=root)

    @property
    def source_root(self) -> Path:
        return self.root / "source"

    @property
    def scratch_root(self) -> Path:
        return self.root / "work"

    def close(self) -> None:
        """Remove the workspace. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "AnalysisWorkspace":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False


@dataclass(frozen=True)
class StepDiagnostics:
    """What the one model call of a step actually cost.

    Recorded for every step, and deliberately for refused ones too. A step
    that produced no action wrote no evidence, so a refusal used to leave no
    trace of how long the model ran or how much it generated -- which is
    exactly the case someone later needs to diagnose. None of these fields
    carries model output: sizes and reasons only, never the text.
    """

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    reused_tokens: int | None = None
    finish_reason: str | None = None
    generation_tokens_per_second: float | None = None
    duration_seconds: float | None = None
    tool_call_count: int = 0
    tool_argument_chars: int = 0
    refusal: str | None = None

    @property
    def evaluated_tokens(self) -> int | None:
        if self.prompt_tokens is None:
            return None
        return self.prompt_tokens - (self.reused_tokens or 0)

    def as_log_fields(self) -> dict[str, object]:
        """Flat, payload-free fields safe to persist or print."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "evaluated_tokens": self.evaluated_tokens,
            "reused_tokens": self.reused_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "duration_seconds": self.duration_seconds,
            "tool_call_count": self.tool_call_count,
            "tool_argument_chars": self.tool_argument_chars,
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class AnalysisReport:
    """What one `/report` produced. Never evidence, never history."""

    text: str
    model_calls: int
    evidence_ids: tuple[str, ...] = ()
    diagnostics: "StepDiagnostics | None" = None


@dataclass(frozen=True)
class AnalysisStepResult:
    """What one analyst step produced. Control is with the analyst on return."""

    model_calls: int
    action_attempted: bool
    action_executed: bool
    assistant_text: str
    result: AnalysisResult | None = None
    evidence: EvidenceRecord | None = None
    rejection: str | None = None
    raw_output_evidence_id: str | None = None
    artifact_handles: tuple[str, ...] = ()
    diagnostics: StepDiagnostics | None = None

    @property
    def control_returned(self) -> bool:
        # Always true by construction: step() has no path that continues past
        # here. Named so tests assert the property rather than the absence of
        # a loop.
        return True


def _stopped_at_generation_limit(response: Any) -> bool:
    """Whether generation was cut off by the budget rather than finishing.

    `length` is the backend's own word for "I stopped because I ran out", so
    it is read rather than inferred from token counts, which would need this
    module to know the effective limit at the point of judgement.
    """
    return str(getattr(response, "finish_reason", "") or "").lower() == "length"


def _tool_argument_chars(calls: list[dict[str, Any]]) -> int:
    """Total size of the generated tool arguments, never their content.

    A refused call is refused precisely because its arguments cannot be
    parsed, so the only safe thing to record about them is how big they were.
    """
    total = 0
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            total += len(arguments)
        elif arguments is not None:
            total += len(json.dumps(arguments, ensure_ascii=False))
    return total


def acquire_analysis_source(original: Path | str, workspace: Path | str) -> AnalysisSource:
    """Copy the artifact into Orbit-owned storage and identify it by content."""
    source = Path(original)
    return snapshot_analysis_bytes(
        source.read_bytes(), workspace=workspace, original_path=str(source)
    )


def snapshot_analysis_bytes(
    data: bytes, *, workspace: Path | str, original_path: str
) -> AnalysisSource:
    """Store already-acquired bytes as the session's immutable source.

    The caller has the bytes because it opened the file itself; handing them
    here rather than a path is what stops the file being opened a second time,
    when it might no longer be the same file. `original_path` is carried for
    the analyst to read and is never reopened.
    """
    digest = hashlib.sha256(data).hexdigest()
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / f"analysis-{digest[:16]}.bin"
    if not snapshot.exists():
        tmp = root / f".{digest[:16]}.partial"
        tmp.write_bytes(data)
        tmp.replace(snapshot)
    snapshot.chmod(0o400)
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
        raise RuntimeError("analysis snapshot does not match the acquired bytes")
    return AnalysisSource(
        snapshot_path=snapshot,
        sha256=digest,
        size_bytes=len(data),
        original_path=original_path,
    )


def _unencodable(value: object) -> bool:
    """Whether this would fail the exact serialization the bridge performs."""
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError):
        return True
    return False


def _rejected_action_text(assistant_text: str, rejection: str) -> str:
    """The assistant turn for a step whose tool call was refused.

    The refused call is deliberately not carried here in any form. It is
    described, so the next step's prompt tells the model plainly what was
    wrong, without re-serialising output that could not be parsed in the
    first place.
    """
    note = f"[action refused: {rejection}]"
    text = assistant_text.strip()
    return f"{text}\n\n{note}" if text else note


def _raw_action_output(result: AnalysisResult) -> str:
    """The complete output of one action, for durable retention.

    Status and bound lead the record. A bounded action can still print
    something that reads like success, so a sidecar holding only stdout would
    later attest to text that misrepresents what happened.
    """
    header = f"status: {result.status}"
    if result.bound_exceeded:
        header += f"\nbound: {result.bound_exceeded}"
    return f"{header}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _bounded_observation(result: AnalysisResult) -> tuple[str, bool, int]:
    """Model-facing text for one action, plus whether it was shortened."""
    parts = [f"status: {result.status}"]
    if result.bound_exceeded:
        parts.append(f"bound: {result.bound_exceeded}")
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    for artifact in result.artifacts:
        # The name is model-chosen, and the observation is newline-delimited
        # `key: value` lines, so an unescaped name could forge entries that
        # read exactly like real ones.
        parts.append(
            f"artifact: {artifact.name!r} "
            f"({artifact.size_bytes} bytes, sha256 {artifact.sha256})"
        )
    text = "\n".join(parts)
    full = len(text)
    if full <= MAX_EVIDENCE_CHARS:
        return text, False, full
    keep = MAX_EVIDENCE_CHARS - 200
    # Record what was dropped rather than silently shortening: a reader must
    # be able to tell a small result from a large one that was cut.
    notice = (
        f"\n[truncated for prompt: {full} chars produced, {keep} retained; "
        f"full output stored in evidence]"
    )
    return text[:keep] + notice, True, full


@dataclass
class AnalysisRuntime:
    """Analyst-driven analysis. One model call and one action per step."""

    backend: ChatBackend
    source: AnalysisSource
    evidence_store: EvidenceStore
    workspace: AnalysisWorkspace | None = None
    messages: list[Message] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = QUALIFIED_ANALYSIS_MAX_TOKENS
    model_calls: int = 0
    actions_executed: int = 0
    analyst_turns: int = 0

    def __post_init__(self) -> None:
        if self.workspace is None:
            self.workspace = AnalysisWorkspace.create()
        if not self.messages:
            self.messages.append({"role": "system", "content": ANALYSIS_SYSTEM_PROMPT})
            # Volatile identity goes after the stable prefix so the prefix can
            # later be prewarmed without carrying this step's specifics.
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Artifact under analysis: the file /workspace/input "
                        f"({self.source.size_bytes} bytes, sha256 {self.source.sha256})."
                    ),
                }
            )

    @property
    def effective_max_tokens(self) -> int:
        """The smaller of what the analyst asked for and what is qualified.

        A configured limit below the qualified one is a deliberate choice and
        is honoured; a larger one is not, because nothing above this has been
        shown to be needed and the cost of finding out is a minute of decode
        the analyst waits through.
        """
        return min(int(self.max_tokens), QUALIFIED_ANALYSIS_MAX_TOKENS)

    def session_usage(self) -> tuple[int, int]:
        """Return (bytes, files) currently retained in the session workspace.

        `scratch_baseline` marks directories with size -1 so the per-action
        bound can tell them from files. Both figures here count files only:
        summing the sentinel would discount real bytes, and counting it would
        let empty directories alone exhaust a session that holds no data.
        """
        sizes = scratch_baseline(self.workspace.scratch_root)
        return (
            sum(size for size in sizes.values() if size >= 0),
            sum(1 for size in sizes.values() if size >= 0),
        )

    def _session_capacity_error(self) -> str | None:
        """Refuse a new action once the workspace is full.

        Checked before the action runs, so capacity is never reported by
        executing code and then throwing its output away. What is already
        stored stays readable: nothing is evicted to make room.
        """
        used_bytes, used_files = self.session_usage()
        if used_bytes >= MAX_SESSION_SCRATCH_BYTES or used_files >= MAX_SESSION_SCRATCH_FILES:
            return (
                f"{SESSION_CAPACITY_EXHAUSTED}: {used_files} files / {used_bytes} bytes "
                f"retained (limit {MAX_SESSION_SCRATCH_FILES} files / "
                f"{MAX_SESSION_SCRATCH_BYTES} bytes)"
            )
        return None

    def step(
        self,
        analyst_message: str,
        *,
        on_progress: Callable[[Any], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AnalysisStepResult:
        """Run exactly one analyst-driven step and return control.

        The callbacks report what is already happening; neither adds a model
        call, changes a message, or reaches the backend. A step that spends
        minutes in one generation was previously indistinguishable from a hung
        process, which is the whole reason they exist.

        `on_delta` receives assistant prose only. Tool-call arguments never
        pass through it: a partially generated call is not valid JSON, and
        showing it would put unparsed model output on the analyst's screen.
        """
        self.analyst_turns += 1
        self.messages.append({"role": "user", "content": analyst_message})

        def _capture(text: str) -> None:
            if on_delta is not None and text:
                on_delta(text)

        # Declaring the phase adds no call: it only labels the one call this
        # step already makes, so the backend can continue the analysis chain's
        # own KV instead of prefilling it again.
        call_started = time.monotonic()
        with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
            response = self.backend.chat_stream(
                list(self.messages),
                temperature=self.temperature,
                max_tokens=self.effective_max_tokens,
                tools=[ANALYSIS_TOOL_SCHEMA],
                on_delta=_capture,
                on_progress=on_progress,
            )
        self.model_calls += 1
        call_seconds = time.monotonic() - call_started

        calls = list(response.tool_calls or [])
        content = response.content or ""
        if _unencodable(content):
            # Decoding makes this practically unreachable, but the cost of
            # being wrong is the same permanently unrenderable history, and
            # the check is one comparison.
            content = content.encode("utf-8", "replace").decode("utf-8")
        assistant: Message = {"role": "assistant", "content": content}

        # Structure is judged before the turn is committed, not after. A tool
        # call the model got wrong -- truncated mid-JSON by an output budget,
        # say -- still has to be told to the analyst, but it must never enter
        # the history: this history is append-only and is re-rendered whole on
        # every later step, so one unparseable `tool_calls` entry makes every
        # subsequent step fail to render and ends the session. Recording the
        # rejection as prose keeps the turn truthful and the history usable.
        def _diagnostics(refusal: str | None) -> StepDiagnostics:
            return StepDiagnostics(
                prompt_tokens=getattr(response, "prompt_tokens", None),
                output_tokens=getattr(response, "completion_tokens", None),
                reused_tokens=getattr(response, "cached_tokens", None),
                finish_reason=getattr(response, "finish_reason", None),
                generation_tokens_per_second=getattr(
                    response, "generation_tokens_per_second", None
                ),
                duration_seconds=round(call_seconds, 3),
                tool_call_count=len(calls),
                tool_argument_chars=_tool_argument_chars(calls),
                refusal=refusal,
            )

        rejection = self._structural_rejection(calls) if calls else None
        if rejection is not None and _stopped_at_generation_limit(response):
            # Same refusal path, a truer reason. The call is unparseable
            # because generation ended mid-JSON, not because the model
            # produced something malformed by choice, and an analyst who reads
            # "not valid JSON" would look for the wrong problem.
            rejection = (
                "analysis step reached its generation limit before producing "
                "a valid tool call"
            )
        if rejection is not None:
            assistant["content"] = _rejected_action_text(content, rejection)
            self.messages.append(assistant)
            # No repair call: repairing would mean a second model invocation
            # before the analyst has seen anything, which is the boundary this
            # runtime exists to hold.
            return AnalysisStepResult(
                model_calls=1,
                action_attempted=True,
                action_executed=False,
                assistant_text=response.content or "",
                rejection=rejection,
                diagnostics=_diagnostics(rejection),
            )

        if calls:
            assistant["tool_calls"] = calls
        self.messages.append(assistant)

        if not calls:
            return AnalysisStepResult(
                model_calls=1,
                action_attempted=False,
                action_executed=False,
                assistant_text=response.content or "",
                diagnostics=_diagnostics(None),
            )

        capacity_error = self._session_capacity_error()
        if capacity_error is not None:
            # Refused before running: an exhausted session must not execute
            # code whose artifacts it has already decided not to record.
            self._append_tool_result(calls[0], f"action not executed: {capacity_error}")
            return AnalysisStepResult(
                model_calls=1,
                action_attempted=True,
                action_executed=False,
                assistant_text=response.content or "",
                rejection=capacity_error,
                diagnostics=_diagnostics(capacity_error),
            )

        code = json.loads(calls[0]["function"]["arguments"])["code"]
        # Snapshot before the action so the sandbox charges it for its own
        # delta rather than for everything earlier steps left behind.
        baseline_sizes = scratch_baseline(self.workspace.scratch_root)
        baseline_digests = self._scratch_digests()
        try:
            result = execute_analysis(
                source_path=self.source.snapshot_path,
                code=code,
                scratch_dir=self.workspace.scratch_root,
                scratch_baseline_sizes=baseline_sizes,
                scratch_baseline_digests=baseline_digests,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            # The sandbox refuses fail-closed for a tampered scratch entry or
            # a changed read-only input, and it signals that with a plain
            # RuntimeError. Those are refusals about this action, not reasons
            # to end an investigation, so they are reported the same way an
            # unavailable sandbox is. `SandboxUnavailable` is a RuntimeError
            # and is covered here too.
            detail = f"{type(exc).__name__}: {exc}"
            self._append_tool_result(calls[0], f"action not executed: {detail}")
            return AnalysisStepResult(
                model_calls=1,
                action_attempted=True,
                action_executed=False,
                assistant_text=response.content or "",
                rejection=detail,
                diagnostics=_diagnostics(detail),
            )

        self.actions_executed += 1
        observation, truncated, full_chars = _bounded_observation(result)

        # The full output goes to the evidence sidecar, which lives on disk and
        # is only ever surfaced to a prompt through bounded excerpts. Recording
        # the truncated text here instead would leave `observation_full_chars`
        # describing bytes that no longer exist anywhere.
        # Provenance is what `reattest_exact` checks before it will hand back
        # durable evidence: without all three fields a record is stored but
        # can never be re-attested. The identity is the model's own call id
        # and this analyst turn -- nothing synthesised to satisfy the check.
        provenance = self._provenance(calls[0])
        raw_record = self.evidence_store.add(
            f"{ANALYSIS_TOOL_NAME}_raw",
            _raw_action_output(result),
            metadata={
                **provenance,
                "analysis_source_sha256": self.source.sha256,
                "code_sha256": result.code_sha256,
                "kind": "raw_action_output",
                "full_chars": full_chars,
                "produced_by_phase": "analysis_action_raw",
            },
        )
        record = self.evidence_store.add(
            ANALYSIS_TOOL_NAME,
            observation,
            metadata={
                **provenance,
                "analysis_source_sha256": self.source.sha256,
                "analysis_source_bytes": self.source.size_bytes,
                "original_path": self.source.original_path,
                "code_sha256": result.code_sha256,
                "input_sha256": result.input_sha256,
                "status": result.status,
                "exit_status": result.exit_status,
                "bound_exceeded": result.bound_exceeded,
                "observation_truncated": truncated,
                "observation_full_chars": full_chars,
                "raw_output_evidence_id": raw_record.evidence_id,
                "artifacts": [
                    {
                        "name": a.name,
                        "size_bytes": a.size_bytes,
                        "sha256": a.sha256,
                        # Stable virtual handle: the host path stays an
                        # implementation detail and never reaches the model.
                        "handle": f"{WORK_MOUNT}/{a.name}",
                    }
                    for a in result.artifacts
                ],
            },
        )
        # Appended before returning: step N+1 must find this already in place
        # rather than have it reconstructed later.
        self._append_tool_result(calls[0], observation)
        return AnalysisStepResult(
            model_calls=1,
            action_attempted=True,
            action_executed=True,
            assistant_text=response.content or "",
            result=result,
            evidence=record,
            raw_output_evidence_id=raw_record.evidence_id,
            artifact_handles=tuple(f"{WORK_MOUNT}/{a.name}" for a in result.artifacts),
            diagnostics=_diagnostics(None),
        )

    def close(self) -> None:
        """Release the session workspace. Idempotent."""
        if self.workspace is not None:
            self.workspace.close()

    def __enter__(self) -> "AnalysisRuntime":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def _provenance(self, call: dict[str, Any]) -> dict[str, object]:
        """Canonical provenance for evidence produced by one executed action."""
        return {
            "tool_call_id": tool_call_id(call),
            "user_turn_id": f"turn_{self.analyst_turns}",
            "produced_by_phase": "analysis_action",
        }

    def _scratch_digests(self) -> dict[str, str]:
        """Hash each retained file so a rewrite is distinguishable from a keep."""
        digests: dict[str, str] = {}
        root = self.workspace.scratch_root
        for path in sorted(root.rglob("*")):
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                digests[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError:
                continue
        return digests

    def report(
        self,
        question: str = "",
        *,
        on_progress: Callable[[Any], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> "AnalysisReport":
        """Answer from the evidence already collected, running nothing.

        A view over the store, not a step in the investigation. The failure
        this replaces was a model choosing to re-read an artifact it had
        already read when asked to interpret it; the fix is to remove the
        choice rather than argue with it, so this call is made with no tools
        at all. It cannot run an action because it is not offered one.

        Nothing here is appended to `self.messages`. The analysis history is
        the append-only record a later exact-prefix strategy depends on, and a
        report is not evidence: the prose it produces is an answer about the
        record, never part of it.
        """
        records = self._reportable_records()
        if not records:
            # Deterministic, and free: there is nothing to ground a report in,
            # and asking a model to say so would be a call spent on a fact the
            # runtime already knows.
            return AnalysisReport(text=NO_EVIDENCE_REPORT, model_calls=0, evidence_ids=())

        messages = self._report_messages(question, records)

        def _capture(text: str) -> None:
            if on_delta is not None and text:
                on_delta(text)

        started = time.monotonic()
        with model_call_context(phase=ANALYSIS_REPORT_PHASE, tools_mode="off"):
            response = self.backend.chat_stream(
                messages,
                temperature=self.temperature,
                max_tokens=self.effective_max_tokens,
                tools=[],
                on_delta=_capture,
                on_progress=on_progress,
            )
        seconds = time.monotonic() - started

        text = (response.content or "").strip()
        if not text:
            # Truthful rather than silent, and no repair call: a second
            # invocation would cross the boundary this runtime holds.
            text = "the report call produced no usable text"
        return AnalysisReport(
            text=text,
            model_calls=1,
            evidence_ids=tuple(r.evidence_id for r in records),
            diagnostics=StepDiagnostics(
                prompt_tokens=getattr(response, "prompt_tokens", None),
                output_tokens=getattr(response, "completion_tokens", None),
                reused_tokens=getattr(response, "cached_tokens", None),
                finish_reason=getattr(response, "finish_reason", None),
                generation_tokens_per_second=getattr(
                    response, "generation_tokens_per_second", None
                ),
                duration_seconds=round(seconds, 3),
            ),
        )

    def _reportable_records(self) -> list[EvidenceRecord]:
        """The action evidence a report may cite, oldest first and bounded."""
        records = [
            record
            for record in self.evidence_store.records.values()
            if record.tool_name == ANALYSIS_TOOL_NAME
        ]
        return records[-MAX_REPORT_EVIDENCE_RECORDS:]

    def _evidence_card(self, record: EvidenceRecord) -> str:
        """One record as the report sees it: header plus the observation.

        `final_card` shows a 700/300 head-and-tail excerpt, which is right for
        a citation card but wrong here -- a finding in the middle of a step's
        output would vanish, and the report would then call it unresolved,
        which is a confident wrong answer rather than a missing one. The
        re-attested text is the same bounded observation the step already put
        in front of the model, so carrying it adds nothing the model has not
        already been trusted with, and it is verified rather than remembered.
        """
        body = self.evidence_store.reattest_exact(record.evidence_id)
        if body is None:
            # Re-attestation is the gate; a record that cannot pass it is
            # described, never quoted.
            return final_card(record)
        return "\n".join(
            [
                "tool_evidence_card: true",
                f"evidence_id: {record.evidence_id}",
                f"status: {record.status}",
                f"size: {record.raw_chars} chars",
                "evidence:",
                body,
            ]
        )

    def _report_messages(
        self, question: str, records: list[EvidenceRecord]
    ) -> list[NativeMessage]:
        """A fresh grounded context, built from the store rather than history.

        Deliberately not the analysis conversation: that carries tool calls and
        an artifact identity this turn must not act on, and reusing it would
        put the model back in the frame where running something is the
        expected move.
        """
        cards = "\n\n".join(self._evidence_card(record) for record in records)
        asked = question.strip() or "Report on what the evidence establishes."
        return [
            {"role": "system", "content": ANALYSIS_REPORT_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Artifact under analysis: {self.source.size_bytes} bytes, "
                    f"sha256 {self.source.sha256}.\n\n"
                    f"Evidence collected so far:\n\n{cards}\n\n{asked}"
                ),
            },
        ]

    def _structural_rejection(self, calls: list[dict[str, Any]]) -> str | None:
        """Why this tool call cannot be committed, or None if it can.

        The bar is not just "did the model mean something sensible" but "can
        this turn survive being written into the history and rendered again".
        The template renders the whole history on every later step and
        serializes it with `ensure_ascii=False`, so a call that cannot be
        encoded is refused here rather than left to fail a future step.
        """
        if len(calls) > 1:
            return f"{len(calls)} tool calls in one response; at most one action per step"
        call = calls[0]
        if not isinstance(call, dict):
            return f"tool call is not an object: {type(call).__name__}"
        # The template requires these too (common/chat.cpp: "Missing tool call
        # type" / "Unsupported tool call type" / "Missing tool call function").
        # Producers normalise the shape today, so this is defence in depth --
        # cheap, against a failure whose cost is an unusable session.
        call_type = call.get("type")
        if call_type is not None and call_type != "function":
            return f"unsupported tool call type: {call_type!r}"
        function = call.get("function")
        if not isinstance(function, dict):
            return "tool call has no function object"
        name = function.get("name")
        if name != ANALYSIS_TOOL_NAME:
            return f"unsupported tool: {name!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except (TypeError, json.JSONDecodeError):
            return "tool arguments are not valid JSON"
        if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
            return "tool arguments must supply a 'code' string"
        if _unencodable(call):
            # Lone surrogates survive json.loads but not the UTF-8 encode the
            # bridge performs, so committing one would break every later step.
            return "tool call contains characters that cannot be encoded"
        return None

    def _append_tool_result(self, call: dict[str, Any], content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": ANALYSIS_TOOL_NAME,
                "content": content,
            }
        )

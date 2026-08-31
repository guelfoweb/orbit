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
import os
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from orbit.backend.base import ChatBackend, Message, RecoverableBackendError
from orbit.runtime.context_manager import (
    ContextAdmissionError,
    DEFAULT_NEXT_ACTION_RESERVE,
    plan_exact_context,
)
from orbit.runtime.analysis_progress import (
    COMPLETE,
    ERROR,
    NEW_CONTENT,
    NO_PROGRESS,
    ProgressLedger,
    ProgressRecord,
    observation_fingerprint,
)
from orbit.runtime.analysis_sandbox import (
    WORK_MOUNT,
    AnalysisResult,
    SandboxUnavailable,
    execute_analysis,
    scratch_baseline,
    validate_code,
)
from orbit.runtime.evidence import (
    EvidenceRecord,
    EvidenceRehydrationError,
    EvidenceStore,
    final_card,
    rehydrated_evidence_block,
    requested_evidence_ids,
    tool_evidence_ref,
)
from orbit.runtime.completion_shadow import (
    ANALYSIS_COMPLETION_SHADOW_PHASE,
    VERIFIER_MAX_TOKENS,
    ShadowLedger,
    build_lossless_snapshot,
    build_snapshot,
    snapshot_fits_budget,
    evaluate_completion_shadow,
    scheduled_actions,
    shadow_enabled,
)
from orbit.runtime.completion_shadow_ledger import (
    ShadowLedgerWriter,
    ledger_path_for_evidence_root,
)
from orbit.runtime.evidence_authority import active_records, evaluate_standing
from orbit.runtime.kv_diag import model_call_context
from orbit.runtime.tool_calls import tool_call_id

ANALYSIS_TOOL_NAME = "execute_analysis"

class _TokenCountUnavailable(RuntimeError):
    """The backend could not tokenize, so the completion budget is unverifiable."""


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
    "Earlier results may appear as an evidence reference (`tool_evidence_ref: true`) "
    "instead of their full text; that is the exact output, archived, not a summary. "
    "When you need those exact bytes again, name its id as `evidence:<evidence_id>` "
    "and they are restored verbatim. Never infer content from a reference alone.\n"
    "When you have identified a deterministic transformation -- a decoder, "
    "decompressor or decryption whose algorithm and concrete inputs you "
    "already hold -- execute it and store its output before re-reading source "
    "you have already collected. Reading the same bytes again cannot resolve "
    "what only running the transformation can.\n"
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
# --- bounded autonomous continuation ------------------------------------
#
# Orbit continues an analysis by itself only while each step is still adding
# verifiable state. Every bound below exists so that a run ends, truthfully,
# with its evidence intact rather than spending the analyst's machine on
# repetition.
#
# The action and model-call bounds are the values the preserved research
# harness ran with (`MAX_ACTIONS = 8`, `MAX_MODEL_CALLS = 10`). Those are the
# figures its measured trajectories were bounded by, so they are reused rather
# than re-invented -- but they are reused as a starting point, not as a proven
# optimum: that harness reached a FAIL verdict on a different model and
# profile, so nothing about its outcome transfers. The margin between them is
# the harness's own: two calls above the action bound, there because a run may
# spend calls that execute nothing.
#
# The stagnation and error bounds have no historical value to inherit -- the
# harness bounded stagnation by a replan counter, not by consecutive
# classification -- so they are set conservatively at 2. Two consecutive
# no-progress steps is the smallest number that distinguishes a model briefly
# re-orienting from one that is stuck; one would abort on a single redundant
# read. All four are configurable per run.
# Where a run stops asking for more, and where it is stopped.
#
# 8 is a budget, not a boundary. It came from the preserved research harness,
# whose sibling variant ran at 6 and whose own tests asserted only the relation
# between the two counts -- never that 8 was where an analysis becomes unsafe
# or useless. A measured full-sample run then ended on it with all eight steps
# still producing new evidence and the report naming the next deterministic
# step, which is a budget cutting off work, not a policy declining it.
#
# So 8 is the point where continuing has to justify itself. A run still adding
# verifiably new state may pass it; a run that is stagnating, repeating a
# strategy, or failing is stopped there exactly as before. 12 is the ceiling,
# and nothing crosses it.
SOFT_MAX_AUTONOMOUS_ACTIONS = 8
MAX_AUTONOMOUS_ACTIONS = 12

# How many non-executing calls a whole run may spend before the budget, rather
# than the analysis, decides when it ends.
#
# A step can consume a call without executing anything: a malformed tool call,
# a refused action, a capacity stop. The error policy tolerates one of those
# between productive steps -- progress resets the counter -- so in principle a
# run could alternate progress and failure all the way to the action ceiling.
# Affording every one of those would put the ceiling at 2*12+1 = 25 calls,
# which on this hardware is hours spent mostly on rejected calls.
#
# 2 is the allowance. It is a judgement and is written down as one: the
# qualified full-sample run spent zero non-executing calls in eight actions, and
# the research harness that set these bounds allowed one in a whole run. Two is
# the smallest number above the historical allowance, and it buys tolerance for
# an occasional mis-formed call without buying an hour of them. A run that
# needs more than two is not being cut short by arithmetic; it is failing.
MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS = 2

# Derived from the loop, not chosen. Every iteration spends exactly one model
# call -- `step()` returns `model_calls=1` on all of its return paths -- so
# calls and iterations are the same thing, and the ceiling has to cover the
# largest run the action policy can legitimately want:
#
#     12 iterations that execute an action        (the hard ceiling)
#   +  1 for the model to finish with prose       (natural completion)
#   +  2 non-executing calls it may spend on the way
#   = 15
#
# The 12 is forced by the control flow. The prose call is slack rather than a
# requirement -- a run that reaches the ceiling breaks before spending it, so
# it is only needed by a run that finishes early -- and the allowance is a
# choice, the constant above. 14 would also be sufficient; 15 keeps one call
# of margin. What is not optional is being above 13: that was the previous
# value, and it made the hard ceiling unreachable for any run containing two
# mis-formed calls, with its test correspondingly vacuous.
#
# The closing report is not counted here: it is made outside the loop, is not
# part of the investigation, and its exclusion is what keeps this number a
# statement about analysis rather than about bookkeeping.
MAX_AUTONOMOUS_MODEL_CALLS = (
    MAX_AUTONOMOUS_ACTIONS + 1 + MAX_AUTONOMOUS_NONPRODUCTIVE_CALLS
)

MAX_CONSECUTIVE_NO_PROGRESS = 2
MAX_CONSECUTIVE_ERRORS = 2

# What Orbit says to itself to take the next step.
#
# It names no artifact, no technique and no direction: choosing those is the
# model's job, and a runtime that suggested one would be doing analysis rather
# than orchestration. What it does say is the standing rule of the loop -- one
# new useful step, nothing already established -- because the alternative is a
# bare "continue" that leaves the model to infer whether re-examining what it
# has already seen counts as continuing. It does not.
AUTONOMOUS_CONTINUATION_MESSAGE = (
    "Continue from the current evidence. Choose one new useful "
    "evidence-producing step. Do not repeat established actions, inputs or "
    "findings."
)

# Sent on the first unproductive step of a streak. The previous instruction
# asked for a new step; this one says plainly that the last attempt was not
# one, and asks for a different strategy rather than a different phrasing of
# the same one.
#
# Once per episode, not once per run: a step that adds new state resets the
# streak, so a later unproductive step is a new situation and is told so again.
# What is never repeated is asking twice about the same stall -- a second
# consecutive unproductive step ends the run, because a runtime that kept
# asking would be arguing with the model. The total is bounded by the action
# ceiling regardless, since every replan follows a step that consumed one.
AUTONOMOUS_REPLAN_MESSAGE = (
    "The previous action produced no new evidence. Choose a different "
    "deterministic strategy using the current evidence and artifacts. Do not "
    "repeat an exhausted action, input or established finding."
)

# What the runtime returns instead of re-running an experiment the session has
# already run against this exact state.
#
# It is deliberately a tool result, not a refusal: the model asked a question,
# and this is the answer -- the observation exists, here is its identity, and
# the exact bytes are one `evidence:<id>` away. Reporting it as an error would
# be false (nothing failed) and would spend the consecutive-error budget on a
# model that is behaving reasonably, just redundantly.
#
# It names no technique and no direction. What it does say is what the session
# already knows and which kinds of step can still change that, because the
# alternative -- "already seen" with no route forward -- is what produced a run
# that re-read one file nine times.
def _no_progress_observation(evidence_id: str) -> str:
    return (
        "NO_PROGRESS: this exact observation already exists as evidence "
        f"{evidence_id}. It was not run again.\n"
        f"Reuse it: name `evidence:{evidence_id}` to get its exact bytes back.\n"
        "Do not repeat this observation. Choose a different unresolved target, "
        "execute a deterministic transformation whose algorithm and inputs you "
        "already have, verify existing evidence, or finish if the evidence is "
        "sufficient."
    )


# Off until it has been measured on real work. Existing one-step behaviour --
# one analyst line, one model call, control back -- is what every analysis does
# unless this is set, so nothing about production changes by merging the loop.
ANALYSIS_AUTONOMY_ENV = "ORBIT_ANALYSIS_AUTONOMOUS"

# Why a run stopped. Reported verbatim to the analyst.
STOP_COMPLETE = "model returned prose with no action"
STOP_NO_PROGRESS = "no new evidence"
STOP_ERROR = "repeated action failures"
STOP_MAX_ACTIONS = "action bound reached"
STOP_SOFT_MAX_ACTIONS = "action budget reached without further progress"
STOP_MAX_MODEL_CALLS = "model call bound reached"
STOP_CANCELLED = "cancelled"
STOP_BACKEND_ERROR = "backend error"

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
    # Set when the runtime declined to re-run an experiment the session had
    # already run against this exact state. The model call still happened --
    # it is counted -- but no sandbox ran and no evidence was created, so this
    # is neither an executed action nor a refusal of one.
    suppressed_duplicate_of: str | None = None

    @property
    def control_returned(self) -> bool:
        # Always true by construction: step() has no path that continues past
        # here. Named so tests assert the property rather than the absence of
        # a loop.
        return True


def analysis_autonomy_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ANALYSIS may continue itself. Off by default, fail-closed.

    Same `1`/`0` grammar as the other Orbit runtime switches, and anything
    unrecognised reads as off: an operator who mistypes gets the behaviour
    that was already shipping, not an unbounded loop.
    """
    env = os.environ if environ is None else environ
    return env.get(ANALYSIS_AUTONOMY_ENV, "").strip() == "1"


@dataclass
class AutonomousRunResult:
    """What one bounded autonomous run produced. Control is with the analyst."""

    steps: tuple[AnalysisStepResult, ...]
    progress: tuple[ProgressRecord, ...]
    stop_reason: str
    model_calls: int
    actions_executed: int
    cancelled: bool = False
    replans: int = 0
    # Observations the runtime answered from existing evidence instead of
    # re-running. Each cost a model call and no action slot, which is the
    # distinction this counter exists to make visible.
    suppressed_duplicates: int = 0
    final_report: "AnalysisReport | None" = None
    # Diagnostics only. Nothing in the loop reads this back, and its verifier
    # calls are deliberately absent from `model_calls`: a shadow observation
    # must not consume the budget that bounds the investigation.
    completion_shadow: ShadowLedger | None = None

    @property
    def last_step(self) -> AnalysisStepResult | None:
        return self.steps[-1] if self.steps else None

    @property
    def control_returned(self) -> bool:
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
    _synthetic_call_seq: int = 0
    context_tokens: int | None = None
    context_compactions: int = 0
    # Experiment identity -> the evidence that experiment already produced.
    # Session-scoped and in-memory: it records what this session ran, which is
    # exactly the scope in which "already established" is answerable.
    _observed_fingerprints: dict[str, str] = field(default_factory=dict)
    suppressed_duplicates: int = 0
    last_context_plan: object | None = None

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

    def _with_canonical_call_ids(self, calls: Iterable[dict]) -> list[dict]:
        """Give every accepted tool call a non-empty id, preserving real ones.

        Backends may return tool calls without an `id`. ANALYSIS tolerated that
        by writing `call.get("id") or ""` into the matching tool result, which
        is self-consistent but structurally invalid: Orbit's shared context
        planner requires a non-empty id on every assistant tool call
        (`context_manager._tool_call_id`), and refuses the whole history
        without one. That is what blocked exact-context admission for ANALYSIS.

        Normalizing here -- at the single point where a step accepts the
        backend's calls, before anything is persisted -- means the assistant
        message and its tool result are built from the same objects and cannot
        disagree. A backend-supplied id is never replaced; only a missing or
        empty one is filled.

        Ids are unique per runtime and stable for the call object they were
        assigned to, because the returned dicts are the ones every downstream
        step uses.
        """
        normalized: list[dict] = []
        # Ids already spoken for by this batch as well as by prior history: a
        # backend id preserved one entry earlier must not be re-issued here.
        claimed: set[str] = set()
        for call in calls:
            if not isinstance(call, dict):
                # Not ours to repair: the structural gate rejects it downstream
                # with its own message rather than being handed a fabricated id.
                normalized.append(call)
                continue
            existing = call.get("id")
            if isinstance(existing, str) and existing:
                claimed.add(existing)
                normalized.append(call)
                continue
            replacement = dict(call)
            replacement["id"] = self._next_synthetic_call_id(claimed)
            claimed.add(replacement["id"])
            normalized.append(replacement)
        return normalized

    def _next_synthetic_call_id(self, claimed: set[str]) -> str:
        """A fresh id that no call in this conversation already claims.

        The counter alone is not enough. A backend is free to return an id that
        happens to match the generated form, and preserving it -- which is
        correct -- would otherwise leave the next synthetic id colliding with
        it. A duplicate silently breaks the assistant/result pairing the shared
        planner depends on, so the sequence skips anything already in use,
        whether it came from earlier history or from earlier in this batch.
        """
        taken = set(claimed)
        taken |= {
            call.get("id")
            for message in self.messages
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        }
        while True:
            self._synthetic_call_seq += 1
            candidate = f"orbit_analysis_call_{self._synthetic_call_seq}"
            if candidate not in taken:
                return candidate

    def _admit(self, messages: list[Message], *, max_tokens: int,
               tools: list[dict] | None, next_action_reserve: int | None = None) -> list[Message]:
        """Plan one ANALYSIS request through Orbit's shared context admission.

        ANALYSIS had none. Prompts grew 581 -> 2898 -> 5241 -> 6105 -> 6991
        against a budget of 5632, and the two over-budget calls were submitted
        anyway; the resident sequence then reached ctx and generation died with
        `llama_decode == 1` ("could not find a KV slot") at physical frontier
        8192 of 8192. The cause was logical over-admission, so the fix is the
        admission CHAT already runs -- not a physical-frontier check, which
        would only notice one token too late.

        `ChatRuntime`'s wrapper cannot be reused directly: its `prepare` is
        bound to chat's message list. Only the binding is new here; the policy,
        the budget arithmetic and the compaction all stay in
        `plan_exact_context`, called with THIS runtime's history.

        Scope of what this buys, stated precisely: for ANALYSIS this is
        **admit, compact, or refuse**. A completed tool turn is externalisable
        once its message carries real evidence identity and its content is
        already a canonical reference -- both of which `_append_tool_result` now
        persists -- so the sets built below are genuinely non-empty and the
        planner can return "compacted" rather than only "unchanged" or
        "blocked".

        Refusal remains the floor, not the ceiling: when nothing eligible is
        left the run still ends explicitly, with the EvidenceStore and history
        intact and a stop reason the analyst can act on, rather than driving the
        KV sequence into the context wall and dying inside `llama_decode`.
        Evidence the current request just rehydrated is withheld from
        compaction, so answering a question never discards the material that
        answers it.

        Returns the admitted messages. Raises `ContextAdmissionError` when the
        request cannot be made to fit -- deliberately before the backend, so
        nothing reaches `llama_decode` that is already known not to fit.
        """
        capability = getattr(self.backend, "supports_exact_context_admission", None)
        if callable(capability):
            status = capability()
            if status is False:
                # A non-Orbit endpoint cannot attest exact tokens; admission is
                # skipped exactly as it is for CHAT rather than guessed at.
                return messages
            if status is not True:
                raise ContextAdmissionError(
                    "context admission failed: exact-token-capability-unavailable"
                )
        else:
            return messages

        messages, rehydrated = self._with_evidence_rehydration(messages)
        available, covered = self._compactable_evidence_sets(messages, rehydrated)

        plan = plan_exact_context(
            messages,
            backend=self.backend,
            output_reserve=max_tokens,
            next_action_reserve=(
                DEFAULT_NEXT_ACTION_RESERVE
                if next_action_reserve is None
                else next_action_reserve
            ),
            configured_context_tokens=self._context_tokens(),
            tools=tools,
            # The render counted must be the render submitted. `chat_stream`
            # sends `backend.thinking`, which the REPL sets from `--think`, so
            # hardcoding False here would under-count a thinking template in the
            # permissive direction -- reintroducing exactly the over-admission
            # this method exists to prevent.
            thinking=bool(getattr(self.backend, "thinking", False)),
            available_evidence_ids=available,
            covered_evidence_ids=covered,
        )
        self.last_context_plan = plan
        if not plan.admitted:
            raise ContextAdmissionError(
                f"context admission failed: {plan.reason or 'required-context-does-not-fit'}"
            )
        if plan.status == "compacted":
            self.context_compactions += 1
        return [dict(message) for message in plan.messages]

    def _with_evidence_rehydration(
        self, messages: list[Message]
    ) -> tuple[list[Message], tuple[str, ...]]:
        """Hand back exact archived output the analyst turn asked for by id.

        This is the half that makes externalisation safe. Once a completed turn
        is compacted its tool content is a reference, and a reference is not the
        evidence -- so the model must be able to name an id and get the exact
        bytes back. Same primitive CHAT uses, same attestation.

        Only the latest USER message is scanned, exactly as CHAT does. Scanning
        tool messages too would be self-defeating: a canonical reference names
        its own id in `exact_content_ref`, so every compacted turn would
        immediately re-inline itself and undo the compaction that just happened.
        Retrieval is something the model asks for, never something a reference
        triggers by existing.
        """
        latest = None
        for message in reversed(messages):
            if message.get("role") == "user":
                latest = message
                break
        evidence_ids = requested_evidence_ids(
            latest.get("content") if latest is not None else None
        )
        if not evidence_ids:
            return messages, ()
        try:
            block = rehydrated_evidence_block(self.evidence_store, evidence_ids)
        except EvidenceRehydrationError as exc:
            # Fail closed: an analysis that cannot re-attest the evidence it
            # asked for must say so, never continue on an approximation.
            raise ContextAdmissionError(
                f"context admission failed: evidence-rehydration-unavailable:{exc.args[0]}"
            ) from exc
        return [*messages, {"role": "system", "content": block}], evidence_ids

    def _compactable_evidence_sets(
        self, messages: list[Message], rehydrated: tuple[str, ...]
    ) -> tuple[set[str], set[str]]:
        """Which evidence ids in this history may be externalised.

        `available` is every id whose record still re-attests exactly against
        the reference actually persisted, so a turn is only ever collapsed onto
        evidence that can be read back. `covered` withholds anything this
        request just rehydrated: that content is in use right now, and
        externalising it in the same breath would drop what the model asked for.
        """
        available: set[str] = set()
        for message in messages:
            if message.get("role") != "tool":
                continue
            evidence_id = message.get("evidence_id")
            reference = message.get("content")
            if not isinstance(evidence_id, str) or not isinstance(reference, str):
                continue
            record = self.evidence_store.records.get(evidence_id)
            if record is None:
                continue
            # The same cross-checks CHAT makes before trusting a pairing
            # (`_context_evidence_sets`): the record must agree with the message
            # that claims it. ANALYSIS builds both from one record so a mismatch
            # cannot arise today -- these are here so it stays that way if the
            # history is ever reloaded or assembled elsewhere, and so the two
            # implementations do not silently diverge.
            if (
                record.tool_call_id != message.get("tool_call_id")
                or record.tool_name != message.get("name")
                or record.user_turn_id != message.get("user_turn_id")
            ):
                continue
            if self.evidence_store.reattest_exact(
                evidence_id, expected_reference=reference
            ) is None:
                continue
            available.add(evidence_id)
        covered = available - set(rehydrated)
        return available, covered

    def _context_tokens(self) -> int | None:
        """The backend's own context size, or None when it cannot say.

        Read from the backend rather than configured here so ANALYSIS cannot
        drift from the context the model was actually loaded with.
        """
        if isinstance(self.context_tokens, int) and self.context_tokens > 0:
            # An operator who narrowed the window with `--context-tokens` means
            # it here too. CHAT resolves the same way (`cli.py`: configured value
            # first, backend report as the fallback), and ANALYSIS silently
            # ignoring it would be a divergence in the permissive direction.
            return self.context_tokens
        info = getattr(self.backend, "model_info", None)
        if not callable(info):
            return None
        try:
            resolved = info()
        except Exception:
            return None
        return getattr(resolved, "context_length", None)

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
        admitted = self._admit(
            list(self.messages),
            max_tokens=self.effective_max_tokens,
            tools=[ANALYSIS_TOOL_SCHEMA],
        )
        with model_call_context(phase=ANALYSIS_STEP_PHASE, tools_mode="on"):
            response = self.backend.chat_stream(
                admitted,
                temperature=self.temperature,
                max_tokens=self.effective_max_tokens,
                tools=[ANALYSIS_TOOL_SCHEMA],
                on_delta=_capture,
                on_progress=on_progress,
            )
        self.model_calls += 1
        call_seconds = time.monotonic() - call_started

        calls = self._with_canonical_call_ids(response.tool_calls or [])
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

        # Identity of the experiment about to run, computed from the same three
        # hashes the ledger uses to judge one that already ran. Asking before
        # rather than after is the entire change: re-running a program over
        # unchanged inputs cannot establish anything, so the expensive part is
        # skipped and the model is told what already answers it.
        #
        # `validate_code` first, and by the same call the sandbox makes: an
        # unparseable program has no stable identity and must reach the normal
        # rejection path rather than be fingerprinted. It returns the code
        # unchanged, so this hashes exactly the bytes `execute_analysis` would.
        #
        # The workspace digests are the pre-action ones already computed above,
        # so an experiment repeated after the workspace changed is a different
        # experiment and runs normally.
        duplicate_of: str | None = None
        try:
            validated = validate_code(code)
        except ValueError:
            # Let the sandbox raise it, so the refusal wording stays the
            # sandbox's own and this stays a pure fast path.
            validated = None
        if validated is not None:
            fingerprint = observation_fingerprint(
                hashlib.sha256(validated.encode("utf-8")).hexdigest(),
                self.source.sha256,
                baseline_digests,
            )
            duplicate_of = self._observed_fingerprints.get(fingerprint)

        if duplicate_of is not None:
            # No sandbox, no new evidence record, no new evidence id: the
            # observation the model asked for already exists under its own
            # identity, and creating a second copy of it is the duplication
            # this exists to prevent. The prior id is returned instead, and
            # remains exactly as re-attestable as it was.
            self.suppressed_duplicates += 1
            self._append_tool_result(
                calls[0], _no_progress_observation(duplicate_of)
            )
            return AnalysisStepResult(
                model_calls=1,
                action_attempted=True,
                # Not executed -- nothing ran -- and deliberately not a
                # rejection either: `rejection` drives the consecutive-error
                # bound, and nothing here failed.
                action_executed=False,
                assistant_text=response.content or "",
                diagnostics=_diagnostics(None),
                suppressed_duplicate_of=duplicate_of,
            )
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
        # Remember the experiment, keyed by the identity computed before it
        # ran, so a later request for the same one is answerable without
        # running it. Registered only on a real execution: a suppressed or
        # refused step must not teach the session anything.
        #
        # Recorded against the pre-action workspace deliberately -- that is the
        # state this program was actually run against, and it is the state a
        # repeat would be judged against too.
        if validated is not None:
            self._observed_fingerprints.setdefault(fingerprint, record.evidence_id)

        # Appended before returning: step N+1 must find this already in place
        # rather than have it reconstructed later.
        self._append_tool_result(calls[0], observation, record=record)
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

    def run_autonomous(
        self,
        analyst_message: str,
        *,
        max_actions: int = MAX_AUTONOMOUS_ACTIONS,
        soft_max_actions: int = SOFT_MAX_AUTONOMOUS_ACTIONS,
        max_model_calls: int = MAX_AUTONOMOUS_MODEL_CALLS,  # plus one closing report
        max_no_progress: int = MAX_CONSECUTIVE_NO_PROGRESS,
        max_errors: int = MAX_CONSECUTIVE_ERRORS,
        on_progress: Callable[[Any], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_step: Callable[[AnalysisStepResult, ProgressRecord], None] | None = None,
        finalize: bool = True,
    ) -> AutonomousRunResult:
        """Run analyst-directed steps until progress stops or a bound is hit.

        Autonomy here is only this: Orbit issues the next ordinary step itself
        instead of waiting for the analyst to type `continue`, and stops as
        soon as steps stop producing verifiably new state. A step that adds
        nothing is allowed exactly one retry -- two consecutive no-progress or
        error steps end the run -- so novelty governs how long a run lives
        without making every single step prove itself first. Every step is the same qualified
        `step()` -- one model call, at most one action, structural rejection,
        sandbox, evidence, append-only history, rolling KV -- so nothing about
        what a step may do changes. The model still chooses what to examine;
        the runtime only decides whether it is worth asking again.

        The loop stops the moment a step stops adding state. It never repairs,
        never retries, never re-runs a rejected action, and never makes a
        second kind of model call: there is no finalisation pass and no
        classifier, because both would be the runtime forming an opinion about
        an analysis it is not qualified to judge.

        Cancellation propagates: `KeyboardInterrupt` from the backend ends the
        run and returns what has already been established, with the workspace
        and evidence intact.

        `max_model_calls` bounds the investigation loop. A run that is not
        cancelled then spends one further call on the closing report, so the
        backend may see `max_model_calls + 1` calls in total; the returned
        `model_calls` counts them all, so the figure reported is the figure
        spent.
        """
        ledger = ProgressLedger()
        steps: list[AnalysisStepResult] = []
        records: list[ProgressRecord] = []
        model_calls = 0
        actions = 0
        # Counted per run, not read off the session: a second run in the same
        # session must report what it suppressed, not what every run before it
        # did. The session-level registry is what stays; this is the tally.
        suppressed = 0
        consecutive_no_progress = 0
        consecutive_errors = 0
        replans = 0
        replan_pending = False
        message = analyst_message
        stop_reason = STOP_MAX_MODEL_CALLS
        cancelled = False
        final_report: "AnalysisReport | None" = None
        shadow = ShadowLedger() if shadow_enabled() else None
        shadow_due = scheduled_actions()
        # Created only when the shadow runs: with the flag off this does no
        # filesystem work at all, not even a stat.
        shadow_ledger = None
        if shadow is not None:
            # Creation is guarded too: resolving the path reads the store, and
            # a run must survive a ledger that cannot be opened at all.
            try:
                shadow_ledger = ShadowLedgerWriter(
                    ledger_path_for_evidence_root(self.evidence_store.root)
                )
                shadow_ledger.write_run_start(
                    request=analyst_message,
                    schedule="after4every2",
                    max_actions=max_actions,
                    soft_max_actions=soft_max_actions,
                    max_model_calls=max_model_calls,
                )
            except Exception:  # noqa: BLE001 - diagnostics never end a run
                shadow_ledger = None

        while True:
            if model_calls >= max_model_calls:
                stop_reason = STOP_MAX_MODEL_CALLS
                break
            try:
                step = self.step(message, on_progress=on_progress, on_delta=on_delta)
            except KeyboardInterrupt:
                # What ran already stands. `step()` has committed its own
                # history and evidence, or rewound nothing; the analyst keeps
                # the session either way.
                cancelled = True
                stop_reason = STOP_CANCELLED
                self._close_incomplete_turn()
                break
            except (ContextAdmissionError, TimeoutError, RecoverableBackendError) as exc:
                # A recoverable backend failure ends the run, it does not undo
                # it. Letting this propagate would unwind to a caller holding
                # only a pre-run checkpoint, and rewinding to that point would
                # delete the history and provenance of every step that already
                # succeeded -- leaving their evidence on disk with nothing
                # referring to it, and re-issuing turn ids that are already in
                # use. What ran, ran; the analyst is told why it stopped.
                #
                # Only the recoverable failures. An unexpected `RuntimeError`
                # must still propagate: this repo overloads bare RuntimeError to
                # mean "a bug, tear the session down and release the
                # workspace", and catching it here would both swallow real
                # crashes and leak the temporary workspace. `RecoverableBackendError`
                # lives in the backend base module so this can name exactly the
                # recoverable case without importing upward.
                error = f"{type(exc).__name__}: {exc}"
                stop_reason = f"{STOP_BACKEND_ERROR}: {error}"
                self._close_incomplete_turn()
                break

            steps.append(step)
            model_calls += step.model_calls
            if step.action_executed:
                actions += 1
            if step.suppressed_duplicate_of is not None:
                # Deliberately not `actions`: a request the runtime answered
                # from evidence it already had did no work that the action
                # budget exists to bound. It still cost the model call counted
                # above, so the run remains bounded.
                suppressed += 1

            record = ledger.classify(len(records) + 1, step)
            records.append(record)
            if on_step is not None:
                on_step(step, record)

            # Observational only, and placed here deliberately: after the step
            # is committed and before any stop decision, so the shadow sees
            # exactly the state a real gate would have seen -- while every
            # `break` below remains reachable regardless of what it says.
            if shadow is not None and shadow_due(actions):
                observation = self._observe_completion_shadow(actions, analyst_message)
                shadow.observations.append(observation)
                if shadow_ledger is not None:
                    # The writer swallows its own I/O failures; this guards the
                    # serialization around them for the same reason.
                    try:
                        shadow_ledger.write_checkpoint(observation)
                    except Exception:  # noqa: BLE001 - diagnostics never end a run
                        shadow_ledger.failures.append("checkpoint_serialization_failed")

            if record.classification == COMPLETE:
                stop_reason = STOP_COMPLETE
                break

            if record.classification == ERROR:
                consecutive_errors += 1
                consecutive_no_progress = 0
                if consecutive_errors >= max_errors:
                    # Carry the last refusal: "repeated action failures" alone
                    # tells the analyst a bound was hit but not what failed.
                    stop_reason = (
                        f"{STOP_ERROR}: {record.detail}" if record.detail else STOP_ERROR
                    )
                    break
            elif record.classification == NO_PROGRESS:
                consecutive_no_progress += 1
                consecutive_errors = 0
                if consecutive_no_progress >= max_no_progress:
                    stop_reason = (
                        f"{STOP_NO_PROGRESS}: strategy repeated"
                        if record.repeated_strategy
                        else f"{STOP_NO_PROGRESS}: action repeated"
                        if record.repeated_action
                        else STOP_NO_PROGRESS
                    )
                    break
                # First unproductive step of this streak: say so, and ask for
                # a different strategy rather than another attempt at the same
                # one. Reached only when the consecutive bound above did not
                # fire, so it is never sent twice about the same stall; a run
                # that recovers and stalls again is told again, because that is
                # a different stall.
                replan_pending = True
                replans += 1
            else:
                consecutive_no_progress = 0
                consecutive_errors = 0

            # The rule is "continue unless a bound trips", not "continue only
            # after NEW_CONTENT". A single ERROR or NO_PROGRESS step still
            # earns one more call, because a model that mis-formed a call or
            # re-read something it already had is often one step from useful
            # work, and refusing to ask again would make the loop less capable
            # than an analyst typing `continue` by hand. What makes that safe
            # is that those steps are the only ones counted: two consecutively
            # ends the run, and the totals below bound it regardless.
            #
            # Checked after the counters above so a run that is both stagnating
            # and out of budget reports the reason it actually hit first.
            if actions >= max_actions:
                stop_reason = STOP_MAX_ACTIONS
                break
            if actions >= soft_max_actions and not record.is_new_content:
                # Past the soft budget, continuing has to be earned. A step
                # that added verifiably new state earns it; anything else --
                # stagnation, a repeated strategy, a refused action -- stops
                # here exactly as it did when this was the only bound.
                #
                # Reached only when the consecutive counters have not already
                # ended the run, so this is the single-failure case they
                # deliberately tolerate: tolerated below the budget, not above
                # it.
                stop_reason = STOP_SOFT_MAX_ACTIONS
                break

            if replan_pending:
                message = AUTONOMOUS_REPLAN_MESSAGE
                replan_pending = False
            else:
                message = AUTONOMOUS_CONTINUATION_MESSAGE

        # One grounded answer at the end, from the evidence already stored.
        #
        # A protective stop is the case that needs this most: a run that ended
        # on a bound or on stagnation has collected real evidence and would
        # otherwise hand the analyst a stop reason and nothing else. The
        # natural ending gets one too, so a completed analysis reads the same
        # way whichever way it finished.
        #
        # `report()` is the qualified primitive for this and is reused
        # unchanged: it is offered no tools, appends nothing to history, and
        # cannot continue the analysis. It is called once, outside the loop, so
        # there is no path from a report back into another step.
        #
        # Cancellation is the exception. The analyst asked for the run to stop,
        # and spending another model call -- minutes of generation -- to
        # summarise it is the opposite of stopping.
        if not cancelled and finalize and steps:
            try:
                final_report = self.report(
                    question=self._final_question(stop_reason),
                    on_progress=on_progress,
                    on_delta=on_delta,
                )
            except (
                KeyboardInterrupt,
                ContextAdmissionError,
                TimeoutError,
                RecoverableBackendError,
            ):
                # A report that cannot be produced must not discard the run
                # that earned it. The analyst keeps the evidence and the stop
                # reason, and can ask for a report themselves.
                #
                # `KeyboardInterrupt` belongs here for the same reason as the
                # rest, and more urgently: the closing report is the longest
                # single generation in a run and the one an analyst is most
                # likely to interrupt, having already read every step. Letting
                # it propagate would unwind past a caller holding only a
                # pre-run checkpoint, and rewinding to that point deletes the
                # history and provenance of every completed step -- leaving
                # their evidence durable on disk with nothing referring to it.
                final_report = None
            else:
                model_calls += final_report.model_calls

        if shadow_ledger is not None:
            # Written last, so its absence is how a reader tells a killed run
            # from a finished one. Linking the ledger to the outcome is the
            # whole point: a WOULD_STOP at action N only means something
            # beside what the run went on to find.
            #
            # Guarded as a whole rather than trusting the writer's own guard:
            # building the final snapshot reads the store, and a diagnostic
            # must not be able to fail a run that has already completed.
            try:
                self._write_shadow_final(
                    shadow_ledger,
                    shadow,
                    request=analyst_message,
                    stop_reason=stop_reason,
                    actions=actions,
                    model_calls=model_calls,
                    cancelled=cancelled,
                    replans=replans,
                    final_report=final_report,
                )
            except Exception:  # noqa: BLE001 - diagnostics must not end a run
                pass

        return AutonomousRunResult(
            steps=tuple(steps),
            progress=tuple(records),
            stop_reason=stop_reason,
            model_calls=model_calls,
            actions_executed=actions,
            cancelled=cancelled,
            replans=replans,
            suppressed_duplicates=suppressed,
            final_report=final_report,
            completion_shadow=shadow,
        )

    def _write_shadow_final(
        self,
        ledger,
        shadow,
        *,
        request: str,
        stop_reason: str,
        actions: int,
        model_calls: int,
        cancelled: bool,
        replans: int,
        final_report,
    ) -> None:
        """Link the checkpoint ledger to what the run actually produced."""
        final_active = active_records(list(self.evidence_store.records.values()))
        final_snapshot = build_snapshot(
            request=request,
            records=final_active,
            load_raw=self.evidence_store.load_raw,
        )
        ledger.write_run_final(
            stop_reason=stop_reason,
            actions_executed=actions,
            model_calls=model_calls,
            cancelled=cancelled,
            replans=replans,
            final_report_produced=final_report is not None,
            final_evidence_ids=[
                str(getattr(record, "evidence_id", "")) for record in final_active
            ],
            final_artifacts=list(final_snapshot.artifacts),
            final_snapshot_sha256=final_snapshot.digest,
            final_snapshot_evidence=[
                {"evidence_id": evidence_id, "text": text}
                for evidence_id, text in final_snapshot.evidence
            ],
            request=final_snapshot.request,
            shadow_verifier_calls=shadow.calls,
            shadow_verifier_prompt_tokens=shadow.prompt_tokens,
            shadow_verifier_output_tokens=shadow.output_tokens,
            shadow_verifier_wall_seconds=round(shadow.wall_seconds, 3),
            ledger_write_failures=list(ledger.failures),
        )

    def _observe_completion_shadow(self, actions: int, request: str):
        """One shadow checkpoint. Never raises, never affects the run.

        The verifier call is issued tools-free under its own phase. Both
        matter: the phase means the backend is asked for no rolling anchor, so
        the analysis checkpoint is neither replaced nor consulted, and
        `tools=[]` with thinking off keeps the prompt-cache transition a
        qualified one, which is what preserves that checkpoint across the
        detour. Diverging on either would drop the next step to a cold prefill.

        One cost is real and bounded: the reset does drop the ANALYSIS *prewarm*
        prefix, which is invalidated unconditionally rather than under the
        preserve flag. It buys nothing back here, because the rolling
        checkpoint holds this session's own tokens and is never shorter, so the
        prewarm stands down whenever rolling can serve the prompt. It would
        matter only to a cold step, and the earliest checkpoint fires after the
        analysis is already warm.
        """
        from orbit.runtime.completion_shadow import ShadowObservation

        try:
            records = active_records(list(self.evidence_store.records.values()))
            active_ids = {
                str(getattr(record, "evidence_id", "") or "") for record in records
            }
            snapshot = build_lossless_snapshot(
                request=request,
                records=records,
                load_raw=self.evidence_store.load_raw,
            )

            def ask(instruction: str, rendered: str):
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": rendered},
                ]
                with model_call_context(
                    phase=ANALYSIS_COMPLETION_SHADOW_PHASE, tools_mode="off"
                ):
                    return self.backend.chat(
                        messages,
                        temperature=0,
                        max_tokens=VERIFIER_MAX_TOKENS,
                        tools=[],
                    )

            # Exact tokens from the model's own tokenizer. If the backend
            # cannot supply them the checkpoint is skipped rather than
            # estimated: an estimate wrong in the permissive direction would
            # put an oversized prompt in front of a verifier, which is the one
            # outcome the budget exists to prevent.
            def count_tokens(text: str) -> int:
                counted = self.backend.count_text_tokens(text)
                if counted is None:
                    raise _TokenCountUnavailable()
                return int(counted.tokens)

            try:
                fits, total = snapshot_fits_budget(snapshot, count_tokens)
            except _TokenCountUnavailable:
                fits, total = False, None

            return evaluate_completion_shadow(
                action=actions,
                snapshot=snapshot,
                ask=ask,
                active_evidence_ids=active_ids,
                reattest=self.evidence_store.reattest_exact,
                fits_budget=fits,
                snapshot_tokens=total,
            )
        except BaseException as exc:  # noqa: BLE001 - diagnostics must not end a run
            return ShadowObservation(
                action=actions,
                snapshot_digest="",
                would_stop=False,
                blocked_by=f"shadow_error: {type(exc).__name__}",
            )

    @staticmethod
    def _final_question(stop_reason: str) -> str:
        """What the closing report is asked, given how the run ended.

        A run that ended naturally is simply asked to report. One that was cut
        short says so, because a reader who is not told that a bound intervened
        would read an incomplete analysis as a finished one.
        """
        if stop_reason == STOP_COMPLETE:
            return ""
        return (
            "This analysis stopped before the model chose to finish "
            f"({stop_reason}). Report what the evidence establishes and what "
            "remains unresolved."
        )

    def _close_incomplete_turn(self) -> None:
        """Drop a trailing analyst turn whose step never produced a reply.

        `step()` appends the analyst line before it calls the model, so a step
        that was cancelled or failed mid-call leaves an unanswered `user` entry
        at the end of an append-only history. Two consecutive user turns is a
        shape the history is not supposed to contain, and every later step
        re-renders the whole thing, so the damage is permanent if it is left.

        Only a trailing user message is removed, and only the one this run put
        there: anything a step actually answered is already followed by an
        assistant turn and is untouched.
        """
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
            self.analyst_turns = max(0, self.analyst_turns - 1)

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
        admitted = self._admit(
            messages,
            max_tokens=self.effective_max_tokens,
            tools=[],
            # The report runs with tools off and cannot issue a next action, so
            # withholding that reserve would only narrow the path most likely to
            # block. Chat makes the same phase-dependent choice.
            next_action_reserve=0,
        )
        with model_call_context(phase=ANALYSIS_REPORT_PHASE, tools_mode="off"):
            response = self.backend.chat_stream(
                admitted,
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
        """The action evidence a report may cite, oldest first and bounded.

        Superseded versions are dropped before the bound is applied, not after.
        An analysis that rewrote an artifact leaves both versions in the store,
        and handing the finalizer a value beside its own correction lets it
        quote either -- which is how a stale value once reached a report that
        had the corrected one in front of it. Dropping first also means the
        bound spends its places on current evidence rather than on history.

        Nothing is deleted: the store keeps every version, and a superseded
        record stays re-attestable for audit. This decides what may be cited as
        authoritative, not what exists.
        """
        records = [
            record
            for record in self.evidence_store.records.values()
            if record.tool_name == ANALYSIS_TOOL_NAME
        ]
        return active_records(records)[-MAX_REPORT_EVIDENCE_RECORDS:]

    def superseded_records(self) -> list[EvidenceRecord]:
        """Versions a report may not cite as current. Retained, not deleted."""
        records = [
            record
            for record in self.evidence_store.records.values()
            if record.tool_name == ANALYSIS_TOOL_NAME
        ]
        standing = evaluate_standing(records)
        # Same tolerance `active_records` applies: a record the evaluator
        # skipped has no entry, and the two views must agree about that or a
        # record one accepts becomes a KeyError in the other.
        return [
            r
            for r in records
            if r.evidence_id in standing and not standing[r.evidence_id].is_active
        ]

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

    def _append_tool_result(
        self,
        call: dict[str, Any],
        content: str,
        *,
        record: EvidenceRecord | None = None,
    ) -> None:
        """Persist one tool result, carrying its evidence identity when it has one.

        A result backed by an attestable record is stored as the canonical
        evidence reference rather than raw text, and tagged with the record's
        own `evidence_id` / `user_turn_id`. Those three things together are what
        make the turn compactable: `plan_context` externalises a completed tool
        turn only when the identity is present AND the content is already a
        reference, so identity alone would achieve nothing -- that was measured,
        not assumed.

        Results with no record -- a refused action, a capacity message -- keep
        their literal text and carry no identity, because inventing one would
        claim evidence that does not exist.
        """
        message: Message = {
            "role": "tool",
            "tool_call_id": call.get("id") or "",
            "name": ANALYSIS_TOOL_NAME,
            "content": content,
        }
        if record is not None:
            # The canonical reference is the shared one CHAT already uses, and
            # ANALYSIS inherits its excerpt rules unchanged -- including that an
            # observation of roughly 1020-1200 chars is inlined head-only, with
            # no truncation marker. Accepted rather than forked: the reference
            # always carries the true `size:` and the exact bytes stay one
            # `evidence:<id>` request away, and diverging here would give the
            # two runtimes different evidence rendering.
            message["content"] = tool_evidence_ref(record)
            message["evidence_id"] = record.evidence_id
            if record.user_turn_id:
                message["user_turn_id"] = record.user_turn_id
        self.messages.append(message)

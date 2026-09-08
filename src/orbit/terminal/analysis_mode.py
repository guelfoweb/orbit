"""Opening an analysis session from the terminal, and showing what a step did.

Two jobs, kept apart from the REPL so neither grows into the other: turn a
path the analyst typed into a live `AnalysisRuntime`, and turn the result of
one step into lines a person can read. Nothing here calls a model, and
nothing here decides when a step happens -- that stays with the REPL, which
is where the analyst's input arrives.

Failures on the way in are ordinary refusals, not exceptions escaping into
the prompt loop: a path that is missing, a directory, or unreadable leaves
the current session exactly as it was.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

from orbit.backend.base import ChatBackend
from orbit.runtime.analysis_runtime import (
    WORK_MOUNT,
    AnalysisProgressEvent,
    AnalysisRuntime,
    AnalysisStepResult,
    StepDiagnostics,
    AnalysisWorkspace,
    acquire_analysis_source,
    snapshot_analysis_bytes,
)
from orbit.runtime.analysis_sandbox import AnalysisResult
from orbit.runtime.confined_acquire import ConfinedAcquireError, acquire_confined_bytes
from orbit.runtime.evidence import EvidenceStore
from orbit.terminal.theme import is_tty, sanitize_terminal_text


class AnalysisModeError(Exception):
    """The analyst asked for a session that cannot be opened."""


def open_analysis_session(
    raw_path: str,
    *,
    backend: ChatBackend,
    workdir: Path,
    evidence_store_factory: Callable[[Path], EvidenceStore],
    context_tokens: int | None = None,
) -> AnalysisRuntime:
    """Snapshot the artifact and return a runtime bound to it.

    The workspace is created first, so a failure to acquire the source removes
    it again rather than leaking a temporary directory for a session that
    never started. The store is built from the workspace root, which keeps an
    analysis session's evidence inside the session that produced it.
    """
    target = _resolve_target(raw_path, workdir=workdir)
    workspace = AnalysisWorkspace.create()
    try:
        source = acquire_analysis_source(target, workspace.source_root)
        evidence_store = evidence_store_factory(workspace.root)
    except OSError as exc:
        workspace.close()
        raise AnalysisModeError(f"cannot read artifact: {exc}") from exc
    except Exception:
        workspace.close()
        raise
    return AnalysisRuntime(
        backend=backend,
        source=source,
        evidence_store=evidence_store,
        workspace=workspace,
        context_tokens=context_tokens,
    )


def open_confined_analysis_session(
    raw_path: str,
    *,
    backend: ChatBackend,
    workdir: Path,
    evidence_store_factory: Callable[[Path], EvidenceStore],
    context_tokens: int | None = None,
    on_acquired: Callable[[], None] | None = None,
) -> AnalysisRuntime:
    """Open a session on a path the model chose, acquiring it safely.

    The difference from `open_analysis_session` is that the file is opened
    once, under the workdir, following no symlinks, and the bytes come from
    that same descriptor. Nothing downstream reopens the name, so there is no
    window in which it can be pointed elsewhere.

    `on_acquired` is a test seam: it runs after the bytes are held and before
    the snapshot exists, which is exactly the interval an attacker would want.
    """
    try:
        acquired = acquire_confined_bytes(raw_path, workdir=workdir)
    except ConfinedAcquireError as exc:
        raise AnalysisModeError(str(exc)) from exc
    if on_acquired is not None:
        on_acquired()
    workspace = AnalysisWorkspace.create()
    try:
        source = snapshot_analysis_bytes(
            acquired.data,
            workspace=workspace.source_root,
            original_path=acquired.resolved_path,
        )
        evidence_store = evidence_store_factory(workspace.root)
    except OSError as exc:
        workspace.close()
        raise AnalysisModeError(f"cannot store artifact: {exc}") from exc
    except Exception:
        workspace.close()
        raise
    return AnalysisRuntime(
        backend=backend,
        source=source,
        evidence_store=evidence_store,
        workspace=workspace,
        context_tokens=context_tokens,
    )


def _resolve_target(raw_path: str, *, workdir: Path) -> Path:
    value = raw_path.strip()
    if not value:
        raise AnalysisModeError("usage: /analysis <path>")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workdir / candidate
    if not candidate.exists():
        raise AnalysisModeError(f"no such artifact: {value}")
    if candidate.is_dir():
        raise AnalysisModeError(f"not a file: {value}")
    if not candidate.is_file():
        raise AnalysisModeError(f"not a regular file: {value}")
    return candidate


#: How much of a question's text a progress line shows. The full text is in
#: the report's dossier; the line only has to say which question this is.
_PROGRESS_QUESTION_CHARS = 72

# What each runtime event reads as. The runtime names the transition; the
# wording is the terminal's, so a change here changes nothing but the screen.
_PROGRESS_LABELS = {
    "planning": "Planning investigation",
    "question": "Investigating",
    "investigating": "Deciding the next action",
    "action": "Running",
    "skipped": "Skipped",
    "classified": None,  # spelled per classification below
    "checking": "Checking completion",
    "resolved": "Resolved",
    "still_open": "Still open",
    "blocked": "Blocked",
    "replanning": "Replanning after no progress",
    "repairing": "Repairing the failed action",
    "stopped": "Stopped",
    "report": "Composing report",
}
_CLASSIFICATION_LABELS = {
    "NEW_CONTENT": "New evidence recorded",
    "NO_PROGRESS": "No new evidence",
    "ERROR": "Action failed",
    "COMPLETE": "Nothing further to run",
}


def format_progress_event(event: "AnalysisProgressEvent") -> str | None:
    """One line for one controller transition, or None for nothing to say.

    Runtime-owned facts only: the event name, the question's position in the
    plan, and the bounded detail the runtime attached (a tool label, a
    classification, a blocked reason, a stop reason). The question text is
    the controller's own ledger entry -- the same one the report's dossier
    prints -- sanitised and cut to one short line. No estimate, no
    percentage, no prose from a model reply ever reaches this line.
    """
    label = _PROGRESS_LABELS.get(event.event)
    if event.event == "classified":
        label = _CLASSIFICATION_LABELS.get(str(event.detail), None)
        detail = None
    else:
        detail = event.detail
    if label is None:
        return None
    position = ""
    if event.question_index is not None and event.question_total:
        position = f"Q{event.question_index}/{event.question_total} "
    elif event.question_id:
        position = f"{event.question_id} "
    text = f"[analysis] {position}{label}"
    if detail:
        safe = sanitize_terminal_text(str(detail)).strip()
        safe = " ".join(safe.split())
        if len(safe) > _PROGRESS_QUESTION_CHARS:
            safe = safe[: _PROGRESS_QUESTION_CHARS - 1] + "…"
        if safe:
            text = f"{text}: {safe}"
    return text


class AnalysisProgressDisplay:
    """Print controller transitions as they happen, on an interactive terminal.

    A sink for `AnalysisRuntime.run_autonomous(on_event=...)`. It owns no
    state about the analysis -- only the last line it printed, so a repeated
    transition is not printed twice -- and it prints through the renderer's
    `event`, which is what already keeps the in-place wait line from
    colliding with a printed line. Off a terminal it prints nothing at all:
    redirected output keeps exactly the text it had before this existed.

    Fail-open by construction: the runtime contains any exception a sink
    raises, so a rendering failure costs a line, never a step.
    """

    # Outcomes that already say what a step established. The ledger's
    # classification arrives after the completion check, so following
    # `Resolved` with `New evidence recorded`, or `Skipped` with `No new
    # evidence`, tells the analyst the same thing twice. ERROR and COMPLETE
    # are never folded: they say something no outcome line said.
    _OUTCOMES = frozenset({"resolved", "still_open", "blocked", "skipped"})
    _FOLDED = frozenset({"NEW_CONTENT", "NO_PROGRESS"})
    # Events after which the runtime hands the completed step to `on_step`,
    # which prints its block. The wait line must not be restarted between the
    # two: a fresh tick would draw a row under that block. The next
    # `investigating`, `question`, `stopped` or `report` restarts it.
    _BEFORE_STEP_BLOCK = frozenset({"resolved", "still_open", "blocked", "skipped", "classified"})

    def __init__(self, renderer, *, interactive: bool | None = None) -> None:
        self._renderer = renderer
        self._interactive = is_tty(sys.stdout) if interactive is None else interactive
        self._last: str | None = None
        self._last_event: str | None = None
        self._outcome_shown = False

    def __call__(self, event: "AnalysisProgressEvent") -> None:
        if not self._interactive:
            return
        if event.event == "investigating":
            self._outcome_shown = False
            if self._last_event == "question":
                # "Investigating: <question>" was just printed; the first
                # step of that question needs no second line saying so.
                self._last_event = event.event
                return
        elif event.event in self._OUTCOMES:
            self._outcome_shown = True
        elif event.event == "classified" and self._outcome_shown and event.detail in self._FOLDED:
            return
        line = format_progress_event(event)
        if line is None:
            return
        self._last_event = event.event
        if line == self._last:
            return
        self._last = line
        # A finished generation line is committed, not erased: the same
        # settle `on_step` performs before its block, done here because this
        # line now prints first.
        settle = getattr(self._renderer, "settle_progress_line", None)
        if settle is not None:
            settle()
        self._renderer.event(
            _fit_progress_line(line),
            restart_timer=event.event not in self._BEFORE_STEP_BLOCK,
        )


def _fit_progress_line(line: str) -> str:
    """Cut the line to the terminal width, so it never wraps onto a second row."""
    try:
        from orbit.terminal.streaming import _terminal_columns

        columns = _terminal_columns()
    except Exception:  # noqa: BLE001 - width is cosmetic
        return line
    limit = max(20, columns - 1)
    return line if len(line) <= limit else line[: limit - 1] + "…"


def format_analysis_step(
    result: AnalysisStepResult, *, prose_already_shown: bool = False
) -> str:
    """Render one step for the analyst who has just been handed control back.

    `prose_already_shown` says the assistant prose was streamed to the terminal
    as it arrived, so repeating it here would show the analyst the same answer
    twice. It is rendering state supplied by the caller that did the streaming,
    not something inferred from the text: a backend that returns content
    without emitting deltas streams nothing, and then this is the only place
    the prose is shown at all.

    The streamed copy is the filtered one. `result.assistant_text` is the raw
    response content, which still holds any raw tool-call markup the backend's
    stream filter kept off the terminal; preferring what was streamed keeps
    that markup hidden, which is what the filter is for.

    Everything else -- action status, evidence preview, artifacts -- is
    rendered either way, because none of it was streamed.
    """
    lines: list[str] = []
    # Model-authored prose: sanitized here because this is where it is
    # displayed. `result.assistant_text` itself stays the model's original.
    text = sanitize_terminal_text(result.assistant_text, allow_newlines=True).strip()
    if text and not prose_already_shown:
        lines.append(text)
    if result.rejection:
        lines.append(f"action refused: {result.rejection}")
    elif result.action_executed and result.result is not None:
        lines.append(_action_summary(result))
    elif result.suppressed_duplicate_of is not None and result.result is not None:
        # A suppressed reacquisition DID run: it produced no useful evidence,
        # which is a different thing from not having executed. Summarised like
        # any other execution so the analyst sees what it wrote and can reach
        # its evidence, with the suppression stated rather than implied.
        lines.append(_action_summary(result))
        lines.append(f"no new evidence: {result.suppressed_duplicate_of}")
    elif result.action_attempted:
        lines.append("action attempted but not executed")
    if result.artifact_handles and not (
        result.result is not None and result.result.artifacts
    ):
        # Only when the preview did not already list them with size and digest;
        # printing both says the same thing twice.
        lines.append(f"artifacts: {', '.join(result.artifact_handles)}")
    if not lines and not (text and prose_already_shown):
        # "(no output)" means the step produced nothing worth showing -- not
        # that the prose was already streamed a moment ago.
        lines.append("(no output)")
    return "\n".join(lines)


def format_step_diagnostics(diagnostics: "StepDiagnostics | None") -> str:
    """One compact line of what the step's model call cost.

    Shown for every step, including refused ones. A refusal writes no
    evidence, so without this the expensive case is the one that leaves no
    record of how long the model ran or how much it produced. Sizes and
    reasons only: no model output reaches this line.
    """
    if diagnostics is None:
        return ""
    parts: list[str] = []
    if diagnostics.prompt_tokens is not None:
        evaluated = diagnostics.evaluated_tokens
        reused = diagnostics.reused_tokens or 0
        parts.append(f"{diagnostics.prompt_tokens} in")
        if evaluated is not None:
            parts.append(f"{evaluated} eval")
        if reused:
            parts.append(f"{reused} cache")
    if diagnostics.output_tokens is not None:
        parts.append(f"{diagnostics.output_tokens} out")
    if diagnostics.generation_tokens_per_second:
        parts.append(f"{diagnostics.generation_tokens_per_second:.1f} tok/s")
    if diagnostics.finish_reason:
        parts.append(str(diagnostics.finish_reason))
    if diagnostics.refusal and diagnostics.tool_argument_chars:
        # Size only. The arguments were refused because they are unparseable,
        # so printing them would put raw model output on the screen.
        parts.append(f"tool args {diagnostics.tool_argument_chars} chars")
    return " · ".join(parts)


# Long enough to carry a Python exception line, short enough that a failing
# action cannot flood the transcript. The full text stays in evidence.
MAX_ACTION_CAUSE_CHARS = 160


# What an action's output may occupy on the analyst's screen. Independent of
# the model-facing bound: that one is about prompt cost, this one is about a
# terminal staying readable. Sized like the rest of the CLI's excerpts -- a
# couple of dozen lines is enough to see what a step produced and decide the
# next one, and the full text is a copyable evidence id away.
MAX_PREVIEW_CHARS = 1200
MAX_PREVIEW_LINES = 24


def _action_summary(result: AnalysisStepResult) -> str:
    action = result.result
    if action is None:
        return "action: executed"
    summary = f"action: {action.status}"
    if action.bound_exceeded:
        summary += f" ({action.bound_exceeded})"
    cause = _action_cause(action)
    if cause:
        summary += f" | {cause}"

    preview = _preview_block(action)
    trailer: list[str] = []
    if result.evidence is not None:
        trailer.append(f"evidence: {result.evidence.evidence_id}")
    if result.raw_output_evidence_id:
        trailer.append(f"raw: {result.raw_output_evidence_id}")

    if not preview:
        # Nothing to show: keep the one-line shape an action with no output
        # always had.
        if trailer:
            summary += " | " + " | ".join(
                part.replace("evidence: ", "evidence ").replace("raw: ", "raw ")
                for part in trailer
            )
        return summary

    lines = [summary, *preview]
    if trailer:
        lines.append("")
        lines.extend(trailer)
    return "\n".join(lines).rstrip()


def _preview_block(action: AnalysisResult) -> list[str]:
    """What the action produced, bounded for a terminal.

    An evidence id says where to look; it does not say what happened. Without
    this the analyst has to open the store by hand before they can choose the
    next step, which is the one thing an interactive session should not
    require. Nothing here is interpreted -- it is the action's own output,
    excerpted -- and no model is consulted to explain it.
    """
    # stdout only. On failure the cause line already carries the exception,
    # and printing stderr underneath would say the same sentence twice.
    block: list[str] = []
    excerpt = _excerpt(action.stdout)
    if excerpt:
        block.append("")
        block.append("result:")
        block.extend(f"  {_sanitize(line)}" for line in excerpt)
    if action.artifacts:
        block.append("")
        block.append("artifacts:")
        for artifact in action.artifacts:
            # The virtual path the analyst can name in the next step, never the
            # host temp directory the workspace happens to live in.
            block.append(
                f"  - {WORK_MOUNT}/{_sanitize(artifact.name)} | "
                f"{_human_size(artifact.size_bytes)} | "
                f"sha256 {_sanitize(_short_sha(artifact.sha256))}"
            )
    return block


def _excerpt(text: str) -> list[str]:
    """Readable lines from `text`, bounded and marked when shortened."""
    if not text or not text.strip():
        return []
    if _looks_binary(text):
        # Metadata only. Binary on a terminal corrupts the display and tells
        # the analyst nothing they can act on.
        return [f"<{len(text)} chars of non-text output; see raw evidence>"]
    lines = text.splitlines()
    kept = lines[:MAX_PREVIEW_LINES]
    shortened = len(lines) > MAX_PREVIEW_LINES
    out: list[str] = []
    budget = MAX_PREVIEW_CHARS
    for line in kept:
        if budget <= 0:
            shortened = True
            break
        if len(line) > budget:
            out.append(line[:budget] + "…")
            budget = 0
            shortened = True
            continue
        out.append(line)
        budget -= len(line)
    if shortened:
        out.append(f"[preview truncated; {len(text)} chars total, full output in evidence]")
    return out


def _sanitize(text: str) -> str:
    """Strip control characters that would act on the terminal.

    Everything previewed here is model-authored output, and these previews are
    single lines of a larger block, so newlines are escaped along with the rest
    of the control range. `sanitize_terminal_text` is the shared boundary.
    """
    return sanitize_terminal_text(text)


def _looks_binary(text: str) -> bool:
    """Control characters beyond the ordinary whitespace ones."""
    printable = sum(1 for ch in text[:2000] if ch in "\t\n\r" or ch >= " ")
    sample = min(len(text), 2000)
    return sample > 0 and printable / sample < 0.9


# Enough to tell two artifacts apart at a glance and to grep the full value
# out of evidence; short enough that the line still fits an 80-column
# terminal, which the full 64 hex characters did not.
SHORT_SHA_CHARS = 12


def _short_sha(value: str) -> str:
    """The leading hex of a digest, for reading rather than verifying.

    Presentation only: what is stored, provenanced and re-attested is always
    the whole digest. Anything that is not a plain hex digest is shown as it
    is, so a malformed value stays visible instead of being quietly trimmed
    into something that looks well-formed.
    """
    text = str(value or "")
    if len(text) <= SHORT_SHA_CHARS or not all(c in "0123456789abcdefABCDEF" for c in text):
        return text
    return f"{text[:SHORT_SHA_CHARS]}\u2026"


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _action_cause(action: AnalysisResult) -> str | None:
    """One line saying why an action did not succeed.

    Read from what the sandbox already reported. An evidence id tells the
    analyst where to look; it does not tell them what happened, and the
    common case -- the program raised -- is answered by the last line of the
    traceback that is already stored. Nothing is inferred and no model is
    asked: if the run produced no diagnostic, this says nothing rather than
    inventing a reason.
    """
    if action.ok:
        return None
    if action.status == "timeout":
        return f"sandbox timeout after {action.duration_seconds:.1f}s"
    if action.bound_exceeded:
        # The bound is already named in the summary; repeating an exit status
        # here would describe the symptom rather than the reason.
        return None
    exception = _last_exception_line(action.stderr)
    if exception:
        return _clip(exception)
    if action.exit_status:
        return f"Python exited with status {action.exit_status}"
    return None


# `SomeError: message`, the line a traceback ends on. Anchored so a line that
# merely contains a colon cannot match.
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Warning|Exit)\b[^:]*:")


def _last_exception_line(stderr: str) -> str | None:
    """The `Error: message` line a traceback ends on, or None.

    Matched by shape rather than by position. Taking the last non-frame line
    looked equivalent and was not: an exception whose message spans lines ends
    on the continuation, so the summary reported a fragment -- and, when that
    fragment held a path, printed a host path the analyst never asked for.
    Only the first line of the message is kept, for the same reason.

    Returning None when nothing matches is deliberate: the caller then falls
    back to the exit status, which says less but cannot say something wrong.
    """
    for line in reversed(stderr.splitlines()):
        candidate = line.strip()
        if not candidate or candidate.startswith(("File \"", "Traceback", "^", "~")):
            continue
        if _EXCEPTION_LINE.match(candidate):
            return candidate
    return None


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_ACTION_CAUSE_CHARS:
        return collapsed
    return collapsed[: MAX_ACTION_CAUSE_CHARS - 1] + "\u2026"

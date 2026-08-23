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
from pathlib import Path
from typing import Callable

from orbit.backend.base import ChatBackend
from orbit.runtime.analysis_runtime import (
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


class AnalysisModeError(Exception):
    """The analyst asked for a session that cannot be opened."""


def open_analysis_session(
    raw_path: str,
    *,
    backend: ChatBackend,
    workdir: Path,
    evidence_store_factory: Callable[[Path], EvidenceStore],
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
    )


def open_confined_analysis_session(
    raw_path: str,
    *,
    backend: ChatBackend,
    workdir: Path,
    evidence_store_factory: Callable[[Path], EvidenceStore],
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


def format_analysis_step(result: AnalysisStepResult) -> str:
    """Render one step for the analyst who has just been handed control back."""
    lines: list[str] = []
    text = result.assistant_text.strip()
    if text:
        lines.append(text)
    if result.rejection:
        lines.append(f"action refused: {result.rejection}")
    elif result.action_executed and result.result is not None:
        lines.append(_action_summary(result))
    elif result.action_attempted:
        lines.append("action attempted but not executed")
    if result.artifact_handles:
        lines.append(f"artifacts: {', '.join(result.artifact_handles)}")
    if not lines:
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
    if result.evidence is not None:
        summary += f" | evidence {result.evidence.evidence_id}"
    if result.raw_output_evidence_id:
        summary += f" | raw {result.raw_output_evidence_id}"
    return summary


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

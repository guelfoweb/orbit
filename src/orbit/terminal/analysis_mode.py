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

from pathlib import Path
from typing import Callable

from orbit.backend.base import ChatBackend
from orbit.runtime.analysis_runtime import (
    AnalysisRuntime,
    AnalysisStepResult,
    AnalysisWorkspace,
    acquire_analysis_source,
)
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


def _action_summary(result: AnalysisStepResult) -> str:
    action = result.result
    if action is None:
        return "action: executed"
    summary = f"action: {action.status}"
    if action.bound_exceeded:
        summary += f" ({action.bound_exceeded})"
    if result.evidence is not None:
        summary += f" | evidence {result.evidence.evidence_id}"
    if result.raw_output_evidence_id:
        summary += f" | raw {result.raw_output_evidence_id}"
    return summary

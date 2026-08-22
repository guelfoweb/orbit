"""Which runtime a typed line belongs to, and nothing else.

Orbit has two runtimes that answer a human. `ChatRuntime` holds a
conversation with five tools; `AnalysisRuntime` inspects one artifact with
one tool, one model call at a time. They are not variants of each other and
the code here does not try to unify them: it decides which of the two owns
the next line, and hands the line over unchanged.

That decision is explicit. An earlier audit established that the existing
route call cannot separate "explain this malware family" from "analyse this
dropper" -- the route prompt deliberately maps read, summarise and analyse
onto the same command shape -- and that inferring it would cost either a
second classifier call per turn or a redesign of the route language. Both
were ruled out, so mode is selected by the analyst with `/analysis` and
`/chat` and never guessed. Selecting a mode makes no model call at all.

The backend is not told any of this. Both runtimes already reach it through
the same `ChatBackend`, so a mode is a fact about which object holds the
conversation, never a parameter on the wire.
"""

from __future__ import annotations

from enum import StrEnum


class WorkflowMode(StrEnum):
    """The runtime that owns the analyst's next line."""

    CHAT = "CHAT"
    ANALYSIS = "ANALYSIS"


DEFAULT_WORKFLOW_MODE = WorkflowMode.CHAT

# A session file records the mode it was saved in, but recording is not the
# same as being able to resume. `AnalysisWorkspace` is a mkdtemp directory
# removed on close, and the artifacts an analysis produces live only inside
# it, so a restarted process has no source snapshot, no scratch tree and no
# way to re-hash what earlier steps recorded. Restoring ANALYSIS would name a
# session that cannot be continued, which is worse than starting in CHAT and
# saying so. Only CHAT is therefore resumable; ANALYSIS falls back with a
# warning until the workspace itself becomes durable.
RESUMABLE_WORKFLOW_MODES = frozenset({WorkflowMode.CHAT})


def parse_workflow_mode(value: object) -> WorkflowMode | None:
    """Read a persisted mode, or None if the value is not one.

    Deliberately strict: an unknown string is not coerced to a default here,
    because the caller has to tell "absent" from "unusable" to warn about the
    second.
    """
    if not isinstance(value, str):
        return None
    try:
        return WorkflowMode(value)
    except ValueError:
        return None


def restored_workflow_mode(value: object) -> tuple[WorkflowMode, str | None]:
    """Resolve a persisted mode to one this process can actually honour.

    Returns the mode to start in and, when that is not what was stored, the
    warning explaining why. Every failure resolves to CHAT: it is the mode
    whose state a session file genuinely carries.
    """
    if value is None:
        return DEFAULT_WORKFLOW_MODE, None
    mode = parse_workflow_mode(value)
    if mode is None:
        return (
            DEFAULT_WORKFLOW_MODE,
            f"warning: ignoring unknown workflow mode {value!r}: starting in CHAT",
        )
    if mode not in RESUMABLE_WORKFLOW_MODES:
        return (
            DEFAULT_WORKFLOW_MODE,
            "warning: ANALYSIS sessions cannot be resumed after restart "
            "(workspace and artifacts are not durable): starting in CHAT",
        )
    return mode, None

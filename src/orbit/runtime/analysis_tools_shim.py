"""The `orbit_tools` module analysis programs import, generated as source.

Recorded model-authored analysis code opens with `import orbit_tools` and
reads the artifact through `orbit_tools.read_file(...)`. That module has to
exist inside the sandbox or the program dies on line one, so this emits it
as a small file mounted alongside the program.

It is deliberately not the research original. That version brokered every
call over an authenticated RPC channel back to the host, which the harness
needed to count and audit tool use. Here the program is already sealed
inside a namespace with no network and nothing but the artifact and its own
scratch directory, so a host round-trip would add a channel to defend
without adding a defence. The shim is plain Python doing plain reads.

Only `read_file` exists. The successful trajectory imported the module in
all nine of its actions and called `read_file` seven times; `search_file`
and `run_command` were never called once, so neither is provided. A helper
nobody used is surface with no upside, and `run_command` in particular
would mean mounting executables the sandbox currently has none of.

Paths are still checked here even though the mount topology already
confines them: the check costs nothing and states the intent locally, so a
future change to the mounts cannot quietly widen what a program may open.
"""

from __future__ import annotations

SOURCE_MOUNT = "/workspace/input"
WORK_MOUNT = "/workspace/work"
MAX_READ_BYTES = 64 * 1024

# Emitted verbatim into the sandbox as `orbit_tools.py`. Kept as source text
# rather than a real importable module because it must exist on the *inside*
# of the namespace, where nothing of Orbit's own code is mounted.
ORBIT_TOOLS_SOURCE = f'''"""Bounded helpers available to a sandboxed analysis program."""

import os

SOURCE_PATH = "{SOURCE_MOUNT}"
WORK_ROOT = "{WORK_MOUNT}"
MAX_READ_BYTES = {MAX_READ_BYTES}


def _safe_path(value):
    """Resolve `value` and confirm it names the artifact or analyst scratch."""
    if not isinstance(value, str) or not value or "\\x00" in value:
        raise ValueError("path must be a non-empty string")
    candidate = os.path.realpath(value)
    if candidate == SOURCE_PATH:
        return candidate
    if candidate == WORK_ROOT or candidate.startswith(WORK_ROOT + "/"):
        return candidate
    raise PermissionError("path is outside the analyst workspace")


def read_evidence(evidence_id=None, **_ignored):
    """Not available here. Evidence is restored in conversation, not in code.

    A live run lost an action to `read_file(evidence_id=...)`: the system
    prompt says to name `evidence:<id>` to get exact bytes back, which is
    true of the conversation and false of a sandboxed program, and the model
    read one instruction as licensing the other. A bare
    "unexpected keyword argument" said nothing about which of the two was
    wrong, so the repair had nothing to act on.

    Defined so that mistake lands on a message naming the supported move
    instead of a TypeError about a signature.
    """
    raise NotImplementedError(
        "evidence cannot be read from inside an analysis program. "
        "Stored evidence is restored in the conversation: end this action, "
        "then write evidence:<evidence_id> in your next message and the exact "
        "bytes are returned. Inside a program, read the artifact with "
        "read_file(SOURCE_PATH) or a file you wrote under WORK_ROOT."
    )


def read_file(path=None, offset=0, limit=MAX_READ_BYTES, **unsupported):
    """Return up to `limit` bytes of `path` from `offset`, decoded as UTF-8.

    Opened with O_NOFOLLOW so a symlink planted in scratch cannot redirect
    the read, and bounded so a large artifact cannot be pulled into memory
    in one call.

    `**unsupported` exists to give one specific mistake a useful answer
    rather than a signature error: `evidence_id=` is the conversation's way
    of restoring bytes and has never been this function's. Anything else
    unexpected is still refused, by name.
    """
    if unsupported:
        if "evidence_id" in unsupported:
            read_evidence()
        raise TypeError(
            "read_file does not accept "
            + ", ".join(sorted(unsupported))
            + "; it takes (path, offset, limit)"
        )
    canonical = _safe_path(path)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_READ_BYTES
    ):
        raise ValueError("limit is outside the read bound")
    descriptor = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.lseek(descriptor, offset, os.SEEK_SET)
        raw = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(raw) > limit:
        raw = raw[:limit]
    return raw.decode("utf-8", "strict")
'''

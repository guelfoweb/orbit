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

`read_file` reads the artifact or scratch. `read_evidence` reads only the
runtime's bounded, re-attested transforms and source-acquisition output supplied
for this action.
Neither exposes the EvidenceStore or a host service. No search or command
runner is provided.

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
_EVIDENCE_INPUTS = {{}}


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


def read_evidence(evidence_id):
    """Return one runtime-authorized evidence input's exact UTF-8 text.

    The runtime supplies bounded immutable values, never EvidenceStore paths.
    Availability here is not a claim that the model received these bytes.
    """
    if isinstance(evidence_id, str) and evidence_id.startswith("evidence:"):
        evidence_id = evidence_id[len("evidence:"):]
    if not isinstance(evidence_id, str) or evidence_id not in _EVIDENCE_INPUTS:
        raise ValueError("evidence not available to this action; use an authorized evidence id (not a path)")
    return _EVIDENCE_INPUTS[evidence_id]


def read_file(path=None, offset=0, limit=MAX_READ_BYTES, **unsupported):
    """Return up to `limit` bytes of `path` from `offset`, decoded as UTF-8.

    Opened with O_NOFOLLOW so a symlink planted in scratch cannot redirect
    the read, and bounded so a large artifact cannot be pulled into memory
    in one call.

    `**unsupported` exists to give one specific mistake a useful answer
    rather than a signature error: `evidence_id=` belongs to read_evidence,
    not to this path reader. Anything else unexpected is still refused,
    by name.
    """
    if unsupported:
        if "evidence_id" in unsupported:
            raise TypeError("read_file takes a path; use read_evidence(id) for an authorized transform")
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
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        # A binary artifact (an Office/OLE document, a packed executable) is
        # not UTF-8 text, and reading it this way fails identically every time
        # -- the loop a live run spent re-reading a `.doc` as text. The bytes
        # are NOT silently decoded; the failure is turned into the one move
        # that makes progress. If source was extracted from this artifact it is
        # held as evidence in the conversation, reachable by naming its id --
        # never from inside a program. Legitimate binary inspection is
        # untouched: read the bytes yourself when the binary structure is the
        # question.
        raise ValueError(
            "these bytes are not UTF-8 text: this path is a binary artifact, "
            "so reading it as text cannot recover source and re-reading it the "
            "same way will fail again. Do not re-read it as text. If macro or "
            "embedded source was extracted from this artifact (an Office/OLE "
            "document's VBA, for example), that exact source is held as "
            "evidence in the conversation -- end this action and name "
            "evidence:<evidence_id> in your reply to get it back. Read the raw "
            "bytes directly only when the binary structure itself is the "
            "question."
        ) from None
'''

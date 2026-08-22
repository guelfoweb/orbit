"""Acquiring one model-named file without ever trusting the name twice.

A path that has been checked and then reopened has not been secured, only
inspected: between the two the name can be pointed somewhere else, and the
bytes that arrive are whichever file it points to last. That is not a
hypothetical here -- validating a pathname and reopening it later let an
outside file be snapshotted on a few racing turns out of a hundred, which is
exactly the shape of bug that hides behind a green test run.

So this module never returns a path for someone else to open. It walks the
path one component at a time relative to an open directory descriptor, with
`O_NOFOLLOW` on every step, opens the final component once, proves through
`fstat` that the thing it is holding is a regular file, and reads the bytes
from that same descriptor. The security decision and the read are made about
one object, so there is no interval for anything to change underneath them.

Symlinks are refused outright rather than resolved, including on intermediate
components. For an analyst typing a path that would be unhelpfully strict; for
a path chosen by a model out of text it has just read, following a link is
precisely what must not happen.

The read-time checks are deliberately close to `file_tools._read_stable_
regular_file`, which already solved this for text. This one exists because an
artifact under analysis is arbitrary bytes: no decoding, no text ceiling.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


# Large enough for the artifacts an analyst actually brings, small enough that
# a mistaken target cannot exhaust memory before it is rejected.
MAX_ACQUIRE_BYTES = 64 * 1024 * 1024


class ConfinedAcquireError(Exception):
    """The named file could not be safely acquired from inside the workdir."""


@dataclass(frozen=True)
class AcquiredBytes:
    """Bytes taken from one opened inode, with the identity they came from."""

    data: bytes
    sha256: str
    size_bytes: int
    # What the caller asked for, kept for the analyst-facing message only. It
    # is never reopened: the bytes above are the source of truth.
    requested_path: str
    resolved_path: str


def acquire_confined_bytes(
    raw_path: str,
    *,
    workdir: Path,
    max_bytes: int = MAX_ACQUIRE_BYTES,
) -> AcquiredBytes:
    """Read one regular file confined to `workdir`, refusing every symlink.

    Raises `ConfinedAcquireError` for anything that is not a plain regular
    file reachable from the workdir without traversing a link.
    """
    if not isinstance(raw_path, str):
        raise ConfinedAcquireError("artifact path must be a string")
    supplied = raw_path.strip()
    if not supplied:
        raise ConfinedAcquireError("artifact path is empty")
    if "\x00" in supplied:
        raise ConfinedAcquireError("artifact path contains a NUL byte")

    try:
        root = workdir.expanduser().resolve()
        # `expanduser` here mirrors what a caller would expect of `~`, but the
        # result still has to survive containment below, so expanding cannot
        # widen what is reachable.
        candidate = Path(supplied).expanduser()
        lexical = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
        relative = lexical.relative_to(root)
    except ValueError:
        raise ConfinedAcquireError("artifact path escapes the workdir") from None
    except (OSError, RuntimeError) as exc:
        # `~nosuchuser` and malformed inputs land here.
        raise ConfinedAcquireError(f"artifact path cannot be resolved: {exc}") from exc

    if not relative.parts:
        raise ConfinedAcquireError("artifact path is the workdir itself")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:  # pragma: no cover - refuse rather than silently follow
        raise ConfinedAcquireError("platform cannot open files without following symlinks")

    directory_fd = -1
    file_fd = -1
    try:
        directory_fd = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            # O_NOFOLLOW on every intermediate component: a symlinked parent
            # directory is as good an escape as a symlinked file.
            next_fd = os.open(part, directory_flags | nofollow, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )

        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ConfinedAcquireError("artifact is not a regular file")
        if before.st_size > max_bytes:
            raise ConfinedAcquireError(
                f"artifact is too large: {before.st_size} bytes, max {max_bytes}"
            )

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ConfinedAcquireError(f"artifact is too large: more than {max_bytes} bytes")

        after = os.fstat(file_fd)
        if _version(before) != _version(after):
            raise ConfinedAcquireError("artifact changed while it was being read")
    except ConfinedAcquireError:
        raise
    except OSError as exc:
        if exc.errno in (getattr(os, "ELOOP", 40), 40):
            raise ConfinedAcquireError("artifact path contains a symlink") from exc
        raise ConfinedAcquireError(f"artifact cannot be read: {exc}") from exc
    finally:
        for descriptor in (file_fd, directory_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    return AcquiredBytes(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        requested_path=supplied,
        resolved_path=str(lexical),
    )


def _version(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


__all__ = [
    "MAX_ACQUIRE_BYTES",
    "AcquiredBytes",
    "ConfinedAcquireError",
    "acquire_confined_bytes",
]

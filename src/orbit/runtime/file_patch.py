from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


MAX_PATCH_BYTES = 16 * 1024
MAX_PATCH_FILE_BYTES = 1024 * 1024
MAX_PATCH_HUNKS = 32

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
)


@dataclass(frozen=True)
class FilePatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class PreparedFilePatch:
    relative_path: str
    target: Path
    replacement: bytes
    original_identity: tuple[int, int, int, int]
    hunk_count: int
    added_lines: int
    removed_lines: int


class FilePatchError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def apply_patch_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply one exact unified diff to one existing UTF-8 text file inside the workdir. "
                "Use after reading the current file. No creation, deletion, rename, fuzzy context, or shell."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_PATCH_BYTES,
                        "description": "Complete unified diff with one ---/+++ file pair and exact @@ hunks.",
                    }
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    }


def validate_file_patch(patch: object, *, workdir: Path) -> str | None:
    try:
        prepare_file_patch(patch, workdir=workdir)
    except FilePatchError as exc:
        return exc.code
    return None


def execute_apply_patch(arguments: dict[str, Any], *, workdir: Path) -> str:
    try:
        prepared = prepare_file_patch(arguments.get("patch"), workdir=workdir)
        _write_prepared_patch(prepared)
    except FilePatchError as exc:
        return f"error: apply_patch rejected: {exc.code}"
    return "\n".join(
        (
            "patch_applied: true",
            f"path: {prepared.relative_path}",
            f"hunks: {prepared.hunk_count}",
            f"added_lines: {prepared.added_lines}",
            f"removed_lines: {prepared.removed_lines}",
        )
    )


def prepare_file_patch(patch: object, *, workdir: Path) -> PreparedFilePatch:
    if not isinstance(patch, str) or not patch:
        raise FilePatchError("patch_not_string")
    try:
        encoded_patch = patch.encode("utf-8")
    except UnicodeEncodeError:
        raise FilePatchError("unsupported_patch_encoding") from None
    if len(encoded_patch) > MAX_PATCH_BYTES:
        raise FilePatchError("patch_too_large")
    if "\x00" in patch or "\r" in patch:
        raise FilePatchError("unsupported_patch_encoding")

    relative_path, hunks = _parse_unified_patch(patch)
    root = workdir.expanduser().resolve()
    target = _resolve_patch_target(relative_path, root=root)
    original, identity = _read_patch_target(target)
    replacement, added_lines, removed_lines = _apply_hunks(original, hunks)
    if replacement == original:
        raise FilePatchError("patch_has_no_effect")
    return PreparedFilePatch(
        relative_path=relative_path,
        target=target,
        replacement=replacement,
        original_identity=identity,
        hunk_count=len(hunks),
        added_lines=added_lines,
        removed_lines=removed_lines,
    )


def _parse_unified_patch(patch: str) -> tuple[str, tuple[FilePatchHunk, ...]]:
    lines = patch.splitlines()
    if len(lines) < 3 or not lines[0].startswith("--- ") or not lines[1].startswith("+++ "):
        raise FilePatchError("invalid_patch_header")
    old_path = _validate_header_path(lines[0][4:])
    new_path = _validate_header_path(lines[1][4:])
    if old_path == new_path:
        relative_path = old_path
    elif old_path.startswith("a/") and new_path.startswith("b/") and old_path[2:] == new_path[2:]:
        relative_path = old_path[2:]
    else:
        raise FilePatchError("path_changed")

    hunks: list[FilePatchHunk] = []
    index = 2
    while index < len(lines):
        match = _HUNK_HEADER_RE.fullmatch(lines[index])
        if match is None:
            raise FilePatchError("unexpected_patch_content")
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line or line[0] not in {" ", "+", "-"}:
                raise FilePatchError("invalid_hunk_line")
            body.append(line)
            index += 1
        old_count = int(match.group("old_count") or 1)
        new_count = int(match.group("new_count") or 1)
        if sum(line[0] in {" ", "-"} for line in body) != old_count:
            raise FilePatchError("old_count_mismatch")
        if sum(line[0] in {" ", "+"} for line in body) != new_count:
            raise FilePatchError("new_count_mismatch")
        if not body or (old_count == 0 and new_count == 0):
            raise FilePatchError("empty_hunk")
        hunks.append(
            FilePatchHunk(
                old_start=int(match.group("old_start")),
                old_count=old_count,
                new_start=int(match.group("new_start")),
                new_count=new_count,
                lines=tuple(body),
            )
        )
        if len(hunks) > MAX_PATCH_HUNKS:
            raise FilePatchError("too_many_hunks")
    if not hunks:
        raise FilePatchError("missing_hunk")
    return relative_path, tuple(hunks)


def _validate_header_path(value: str) -> str:
    if not value or value != value.strip() or "\t" in value or value == "/dev/null":
        raise FilePatchError("invalid_patch_path")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise FilePatchError("invalid_patch_path")
    return path.as_posix()


def _resolve_patch_target(relative_path: str, *, root: Path) -> Path:
    candidate = root / relative_path
    if _contains_symlink(root, Path(relative_path)):
        raise FilePatchError("symlink_not_allowed")
    try:
        target = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FilePatchError("target_not_found") from None
    except OSError:
        raise FilePatchError("target_unavailable") from None
    try:
        target.relative_to(root)
    except ValueError:
        raise FilePatchError("path_outside_workdir") from None
    if not target.is_file():
        raise FilePatchError("target_not_file")
    return target


def _contains_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_patch_target(target: Path) -> tuple[bytes, tuple[int, int, int, int]]:
    try:
        stat = target.stat()
        if stat.st_size > MAX_PATCH_FILE_BYTES:
            raise FilePatchError("target_too_large")
        content = target.read_bytes()
    except FilePatchError:
        raise
    except OSError:
        raise FilePatchError("target_read_failed") from None
    if b"\x00" in content:
        raise FilePatchError("binary_target")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise FilePatchError("target_not_utf8") from None
    if "\r" in text:
        raise FilePatchError("unsupported_target_newline")
    if text and not text.endswith("\n"):
        raise FilePatchError("target_missing_final_newline")
    return content, (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)


def _apply_hunks(original: bytes, hunks: tuple[FilePatchHunk, ...]) -> tuple[bytes, int, int]:
    source = original.decode("utf-8").splitlines()
    output: list[str] = []
    cursor = 0
    added_lines = 0
    removed_lines = 0
    previous_new_end = 0

    for hunk in hunks:
        source_start = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        if source_start < cursor or source_start > len(source):
            raise FilePatchError("overlapping_or_out_of_range_hunk")
        if hunk.new_start < previous_new_end:
            raise FilePatchError("overlapping_new_hunk")
        output.extend(source[cursor:source_start])
        cursor = source_start
        expected_new_start = len(output) + (1 if hunk.new_count else 0)
        if hunk.new_start != expected_new_start:
            raise FilePatchError("new_position_mismatch")
        for line in hunk.lines:
            marker, value = line[0], line[1:]
            if marker in {" ", "-"}:
                if cursor >= len(source) or source[cursor] != value:
                    raise FilePatchError("context_mismatch")
                if marker == " ":
                    output.append(source[cursor])
                else:
                    removed_lines += 1
                cursor += 1
            else:
                output.append(value)
                added_lines += 1
        previous_new_end = hunk.new_start + hunk.new_count

    output.extend(source[cursor:])
    replacement = ("\n".join(output) + ("\n" if output else "")).encode("utf-8")
    return replacement, added_lines, removed_lines


def _write_prepared_patch(prepared: PreparedFilePatch) -> None:
    target = prepared.target
    temporary_path: Path | None = None
    try:
        current = target.stat()
        current_identity = (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size)
        if current_identity != prepared.original_identity or target.is_symlink():
            raise FilePatchError("target_changed")
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.name}.orbit-", dir=target.parent)
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), stat.S_IMODE(current.st_mode))
            handle.write(prepared.replacement)
            handle.flush()
            os.fsync(handle.fileno())
        latest = target.stat()
        latest_identity = (latest.st_dev, latest.st_ino, latest.st_mtime_ns, latest.st_size)
        if latest_identity != prepared.original_identity or target.is_symlink():
            raise FilePatchError("target_changed")
        os.replace(temporary_path, target)
        temporary_path = None
    except FilePatchError:
        raise
    except OSError:
        raise FilePatchError("target_write_failed") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass

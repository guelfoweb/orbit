from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.runtime.path_guardrails import BINARY_OR_SPECIAL_EXTENSIONS, TEXT_EXTENSIONS, resolve_inside_workdir


MAX_READ_BYTES = 256 * 1024
MAX_READ_CHARS = 1_500
MAX_CHUNK_FILE_BYTES = 1024 * 1024
MAX_FULL_DOCUMENT_BYTES = 1024 * 1024
DEFAULT_CHUNK_CHARS = 6_000
DEFAULT_LARGE_FILE_INITIAL_CHARS = 500
MAX_CHUNK_CHARS = 12_000
PDF_CHUNK_CHARS = 3_000
PDF_EXTRACT_TIMEOUT = 15
FULL_DOCUMENT_SNAPSHOT_MARKER = "full_document_snapshot: true"


@dataclass(frozen=True)
class StableTextSnapshot:
    target: Path
    raw: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def read_file(path: Any, *, arguments: dict[str, Any], workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        return "error: read_file supports UTF-8 text/code files only; PDF content is handled separately"
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: read_file supports UTF-8 text/code files only; unsupported file type: {suffix}"

    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        if "chunk_index" in arguments:
            return read_chunk(
                path,
                chunk_index=arguments.get("chunk_index"),
                chunk_chars=arguments.get("chunk_chars", DEFAULT_CHUNK_CHARS),
                workdir=workdir,
            )
        return read_large_file_excerpt(path, chunk_chars=DEFAULT_LARGE_FILE_INITIAL_CHARS, workdir=workdir)
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for read_file: {suffix}"

    raw = target.read_bytes()
    if b"\x00" in raw:
        return "error: file appears to be binary and cannot be read as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: file is not valid UTF-8 text"

    if len(text) > MAX_READ_CHARS:
        return read_chunk(
            path,
            chunk_index=arguments.get("chunk_index", 0),
            chunk_chars=arguments.get("chunk_chars", DEFAULT_CHUNK_CHARS),
            workdir=workdir,
        )
    return text if text else "(empty file)"


def read_full_document_snapshot(path: Any, *, workdir: Path) -> str:
    validation = load_text_snapshot(path, workdir=workdir, max_bytes=MAX_FULL_DOCUMENT_BYTES)
    if isinstance(validation, str):
        return validation
    snapshot, text = validation
    raw = snapshot.raw
    relative = snapshot.target.relative_to(workdir.expanduser().resolve())
    return "\n".join(
        [
            FULL_DOCUMENT_SNAPSHOT_MARKER,
            f"path: {relative}",
            f"bytes: {len(raw)}",
            f"chars: {len(text)}",
            f"lines: {len(text.splitlines())}",
            f"sha256: {hashlib.sha256(raw).hexdigest()}",
            f"device: {snapshot.device}",
            f"inode: {snapshot.inode}",
            f"mtime_ns: {snapshot.mtime_ns}",
            f"ctime_ns: {snapshot.ctime_ns}",
            "content:",
            text,
        ]
    )


def read_pdf(path: Any, *, arguments: dict[str, Any], workdir: Path) -> str:
    validation = load_pdf_text(path, workdir=workdir)
    if isinstance(validation, str):
        return validation
    target, text, extractor = validation

    if "chunk_index" in arguments:
        return read_pdf_chunk(
            path,
            chunk_index=arguments.get("chunk_index"),
            chunk_chars=arguments.get("chunk_chars", PDF_CHUNK_CHARS),
            workdir=workdir,
        )

    if len(text) <= MAX_READ_CHARS:
        return format_pdf_result(target, text, extractor=extractor)
    return read_large_pdf_excerpt(path, chunk_chars=PDF_CHUNK_CHARS, workdir=workdir)


def read_chunk(path: Any, *, chunk_index: Any, chunk_chars: Any, workdir: Path) -> str:
    if not isinstance(chunk_index, int) or chunk_index < 0:
        return "error: chunk_index must be a non-negative integer"
    if not isinstance(chunk_chars, int) or chunk_chars <= 0:
        return "error: chunk_chars must be a positive integer"
    if chunk_chars > MAX_CHUNK_CHARS:
        return f"error: chunk_chars too large: {chunk_chars}, max {MAX_CHUNK_CHARS}"

    validation = load_text_file(path, workdir=workdir, max_bytes=MAX_CHUNK_FILE_BYTES)
    if isinstance(validation, str):
        return validation
    target, text = validation

    total_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    if chunk_index >= total_chunks:
        return f"error: chunk_index out of range: {chunk_index}, total_chunks {total_chunks}"

    start = chunk_index * chunk_chars
    end = min(start + chunk_chars, len(text))
    chunk = text[start:end]
    return "\n".join(
        [
            f"path: {target.name}",
            f"chunk_index: {chunk_index}",
            f"total_chunks: {total_chunks}",
            f"chars: {start}-{end} of {len(text)}",
            "content:",
            chunk,
        ]
    )


def read_large_file_excerpt(path: Any, *, chunk_chars: int, workdir: Path) -> str:
    validation = load_text_file(path, workdir=workdir, max_bytes=MAX_CHUNK_FILE_BYTES)
    if isinstance(validation, str):
        return validation
    target, text = validation
    end = min(chunk_chars, len(text))
    excerpt = text[:end]
    return "\n".join(
        [
            f"path: {target.name}",
            "large_file_excerpt: true",
            f"chars: 0-{end} of {len(text)}",
            "content:",
            excerpt,
        ]
    )


def read_pdf_chunk(path: Any, *, chunk_index: Any, chunk_chars: Any, workdir: Path) -> str:
    if not isinstance(chunk_index, int) or chunk_index < 0:
        return "error: chunk_index must be a non-negative integer"
    if not isinstance(chunk_chars, int) or chunk_chars <= 0:
        return "error: chunk_chars must be a positive integer"
    if chunk_chars > MAX_CHUNK_CHARS:
        return f"error: chunk_chars too large: {chunk_chars}, max {MAX_CHUNK_CHARS}"

    validation = load_pdf_text(path, workdir=workdir)
    if isinstance(validation, str):
        return validation
    target, text, extractor = validation
    total_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    if chunk_index >= total_chunks:
        return f"error: chunk_index out of range: {chunk_index}, total_chunks {total_chunks}"
    start = chunk_index * chunk_chars
    end = min(start + chunk_chars, len(text))
    return format_pdf_result(
        target,
        text[start:end],
        extractor=extractor,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        chars_start=start,
        chars_end=end,
        total_length=len(text),
    )


def read_large_pdf_excerpt(path: Any, *, chunk_chars: int, workdir: Path) -> str:
    validation = load_pdf_text(path, workdir=workdir)
    if isinstance(validation, str):
        return validation
    target, text, extractor = validation
    end = min(chunk_chars, len(text))
    total_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    return format_pdf_result(
        target,
        text[:end],
        extractor=extractor,
        large_file_excerpt=True,
        chunk_index=0,
        total_chunks=total_chunks,
        chars_start=0,
        chars_end=end,
        total_length=len(text),
    )


def load_text_file(path: Any, *, workdir: Path, max_bytes: int) -> tuple[Path, str] | str:
    loaded = load_text_snapshot(path, workdir=workdir, max_bytes=max_bytes)
    if isinstance(loaded, str):
        return loaded
    snapshot, text = loaded
    return snapshot.target, text


def load_text_snapshot(path: Any, *, workdir: Path, max_bytes: int) -> tuple[StableTextSnapshot, str] | str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    opened = _read_stable_regular_file(path, workdir=workdir, max_bytes=max_bytes)
    if isinstance(opened, str):
        return opened
    target = opened.target
    raw = opened.raw

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        return "error: read_file chunk mode supports UTF-8 text/code files only; PDF content is handled separately"
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: read_file chunk mode supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for read_file chunk mode: {suffix}"

    if b"\x00" in raw:
        return "error: file appears to be binary and cannot be read as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: file is not valid UTF-8 text"
    return opened, text


def _read_stable_regular_file(path: str, *, workdir: Path, max_bytes: int) -> StableTextSnapshot | str:
    """Read one confined regular-file inode without following path symlinks."""
    root = workdir.expanduser().resolve()
    supplied = Path(path).expanduser()
    lexical = Path(os.path.abspath(supplied if supplied.is_absolute() else root / supplied))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return "error: path escapes workdir"
    if not relative.parts:
        return f"error: path is not a file: {path}"

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    directory_fd = -1
    try:
        directory_fd = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags | nofollow, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | nonblock,
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                return f"error: path is not a regular file: {path}"
            if before.st_size > max_bytes:
                return f"error: file too large for text snapshot: {before.st_size} bytes, max {max_bytes}"
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(file_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(file_fd)
            if len(raw) > max_bytes:
                return f"error: file too large for text snapshot: more than {max_bytes} bytes"
            if _file_version(before) != _file_version(after):
                return "error: file changed while it was being read; snapshot discarded"
            current = os.stat(relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
                return "error: file was replaced while it was being read; snapshot discarded"
            try:
                lexical_current = os.stat(lexical, follow_symlinks=False)
            except OSError:
                return "error: file disappeared while it was being read; snapshot discarded"
            if not stat.S_ISREG(lexical_current.st_mode) or (
                lexical_current.st_dev,
                lexical_current.st_ino,
            ) != (after.st_dev, after.st_ino):
                return "error: file path changed while it was being read; snapshot discarded"
            return StableTextSnapshot(
                target=lexical,
                raw=raw,
                device=after.st_dev,
                inode=after.st_ino,
                size=after.st_size,
                mtime_ns=after.st_mtime_ns,
                ctime_ns=after.st_ctime_ns,
            )
        finally:
            os.close(file_fd)
    except FileNotFoundError:
        return f"error: path not found: {path}"
    except (NotADirectoryError, OSError) as exc:
        if getattr(exc, "errno", None) in {40, 62}:
            return "error: read_file does not follow symlinks"
        return f"error: unable to read file safely: {exc}"
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _file_version(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def load_pdf_text(path: Any, *, workdir: Path) -> tuple[Path, str, str] | str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"
    if target.suffix.lower() != ".pdf":
        return f"error: unsupported PDF file extension: {target.suffix.lower()}"

    text, extractor = extract_pdf_text(target)
    if not text.strip():
        return f"error: no text extracted from PDF: {target.name}"
    if len(text.encode("utf-8", errors="replace")) > MAX_CHUNK_FILE_BYTES:
        return f"error: extracted PDF text too large: max {MAX_CHUNK_FILE_BYTES} bytes"
    return target, text, extractor


def extract_pdf_text(target: Path) -> tuple[str, str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        completed = subprocess.run(
            [pdftotext, "-layout", str(target), "-"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=PDF_EXTRACT_TIMEOUT,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout, "pdftotext"
    strings = shutil.which("strings")
    if not strings:
        return "", "unavailable"
    completed = subprocess.run(
        [strings, "-a", "-n", "8", str(target)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=PDF_EXTRACT_TIMEOUT,
        check=False,
    )
    if completed.returncode == 0:
        return _clean_pdf_strings_output(completed.stdout), "strings"
    return "", "strings"


def format_pdf_result(
    target: Path,
    text: str,
    *,
    extractor: str,
    large_file_excerpt: bool = False,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    chars_start: int | None = None,
    chars_end: int | None = None,
    total_length: int | None = None,
) -> str:
    lines = [
        "pdf_text: true",
        f"path: {target.name}",
        f"extractor: {extractor}",
    ]
    if large_file_excerpt:
        lines.append("large_file_excerpt: true")
    if chunk_index is not None and total_chunks is not None and chars_start is not None and chars_end is not None:
        if total_length is None:
            total_length = chars_end
        lines.extend(
            [
                f"chunk_index: {chunk_index}",
                f"total_chunks: {total_chunks}",
                f"chars: {chars_start}-{chars_end} of {total_length}",
            ]
        )
    lines.extend(["content:", text.strip() or "(empty PDF text)"])
    return "\n".join(lines)


def _clean_pdf_strings_output(text: str) -> str:
    lines: list[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%PDF") or line == "%%EOF":
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)

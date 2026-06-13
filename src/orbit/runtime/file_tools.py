from __future__ import annotations

from pathlib import Path
from typing import Any

from orbit.runtime.path_guardrails import BINARY_OR_SPECIAL_EXTENSIONS, TEXT_EXTENSIONS, resolve_inside_workdir


MAX_READ_BYTES = 256 * 1024
MAX_READ_CHARS = 1_500
MAX_CHUNK_FILE_BYTES = 1024 * 1024
DEFAULT_CHUNK_CHARS = 6_000
DEFAULT_LARGE_FILE_INITIAL_CHARS = 500
MAX_CHUNK_CHARS = 12_000


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
        return "error: read_file supports UTF-8 text/code files only; PDF requires read_pdf, which is not available yet"
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


def load_text_file(path: Any, *, workdir: Path, max_bytes: int) -> tuple[Path, str] | str:
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
        return "error: read_file chunk mode supports UTF-8 text/code files only; PDF requires read_pdf, which is not available yet"
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: read_file chunk mode supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for read_file chunk mode: {suffix}"

    size = target.stat().st_size
    if size > max_bytes:
        return f"error: file too large for read_file chunk mode: {size} bytes, max {max_bytes}"

    raw = target.read_bytes()
    if b"\x00" in raw:
        return "error: file appears to be binary and cannot be read as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: file is not valid UTF-8 text"
    return target, text

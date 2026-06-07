from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_LIST_ITEMS = 200
MAX_READ_BYTES = 256 * 1024
MAX_READ_CHARS = 1_500
MAX_WRITE_CHARS = 64 * 1024
MAX_APPEND_CHARS = 16 * 1024
MAX_REPLACE_CHARS = 16 * 1024
MAX_TEXT_FILE_BYTES_AFTER_APPEND = 512 * 1024
MAX_TEXT_FILE_BYTES_AFTER_REPLACE = 512 * 1024
MAX_CHUNK_FILE_BYTES = 1024 * 1024
DEFAULT_CHUNK_CHARS = 6_000
DEFAULT_LARGE_FILE_INITIAL_CHARS = 500
MAX_CHUNK_CHARS = 12_000

TEXT_EXTENSIONS = {
    ".bat",
    ".bib",
    ".c",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".dart",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".log",
    ".lua",
    ".md",
    ".php",
    ".properties",
    ".ps1",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

BINARY_OR_SPECIAL_EXTENSIONS = {
    ".7z",
    ".bmp",
    ".doc",
    ".docx",
    ".flac",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".ogg",
    ".pdf",
    ".png",
    ".tar",
    ".wav",
    ".webp",
    ".zip",
}


def list_files(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str):
        return "error: path must be a string"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_dir():
        return f"error: path is not a directory: {path}"

    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    names = [f"{item.name}/" if item.is_dir() else item.name for item in entries[:MAX_LIST_ITEMS]]
    if len(entries) > MAX_LIST_ITEMS:
        names.append(f"... truncated, {len(entries) - MAX_LIST_ITEMS} more entries")
    return "\n".join(names) if names else "(empty directory)"


def stat_path(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return "\n".join([f"path: {path}", "exists: false"])
    try:
        stat = target.stat()
    except OSError as exc:
        return f"error: cannot stat path: {exc}"

    if target.is_dir():
        path_type = "directory"
    elif target.is_file():
        path_type = "file"
    else:
        path_type = "other"

    modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return "\n".join(
        [
            f"path: {path}",
            "exists: true",
            f"type: {path_type}",
            f"size_bytes: {stat.st_size}",
            f"modified: {modified}",
        ]
    )


def make_directory(path: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    root = workdir.expanduser().resolve()
    if target == root:
        return "error: refusing to create the workdir root"
    if target.exists():
        if target.is_dir():
            return f"error: directory already exists: {path}"
        return f"error: path already exists and is not a directory: {path}"
    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return f"error: cannot create directory: {exc}"
    return "\n".join([f"path: {path}", "created: true", "type: directory"])


def delete_path(path: Any, recursive: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(recursive, bool):
        return "error: recursive must be a boolean"
    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    root = workdir.expanduser().resolve()
    if target == root:
        return "error: refusing to delete the workdir root"
    if not target.exists() and not target.is_symlink():
        return f"error: path not found: {path}"
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
            path_type = "file"
        elif target.is_dir():
            if any(target.iterdir()) and not recursive:
                return f"error: directory is not empty: {path}. Use recursive=true only if you really want to remove it."
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
            path_type = "directory"
        else:
            return f"error: unsupported path type: {path}"
    except OSError as exc:
        return f"error: cannot delete path: {exc}"
    return "\n".join([f"path: {path}", "deleted: true", f"type: {path_type}"])


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
        return text[:MAX_READ_CHARS] + f"\n... truncated, {len(text) - MAX_READ_CHARS} more characters"
    return text if text else "(empty file)"


def write_file(path: Any, content: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(content, str):
        return "error: content must be a string"
    if len(content) > MAX_WRITE_CHARS:
        return f"error: content too large for write_file: {len(content)} chars, max {MAX_WRITE_CHARS}"
    if "\x00" in content:
        return "error: content appears to be binary and cannot be written as UTF-8 text"

    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if target.exists():
        return f"error: refusing to overwrite existing path: {path}"
    if not target.parent.exists():
        return f"error: parent directory does not exist: {target.parent.relative_to(workdir.expanduser().resolve())}"
    if not target.parent.is_dir():
        return f"error: parent path is not a directory: {target.parent.relative_to(workdir.expanduser().resolve())}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: write_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for write_file: {suffix}"

    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot write file: {exc}"
    return "\n".join([f"path: {path}", "created: true", f"chars: {len(content)}", f"bytes: {len(content.encode('utf-8'))}"])


def append_file(path: Any, content: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(content, str):
        return "error: content must be a string"
    if len(content) > MAX_APPEND_CHARS:
        return f"error: content too large for append_file: {len(content)} chars, max {MAX_APPEND_CHARS}"
    if "\x00" in content:
        return "error: content appears to be binary and cannot be appended as UTF-8 text"

    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}. Use write_file to create a new file."
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: append_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for append_file: {suffix}"

    current_size = target.stat().st_size
    append_bytes = len(content.encode("utf-8"))
    if current_size + append_bytes > MAX_TEXT_FILE_BYTES_AFTER_APPEND:
        return f"error: append would make file too large: {current_size + append_bytes} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_APPEND}"
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"error: cannot read existing file before append: {exc}"
    if b"\x00" in raw:
        return "error: existing file appears to be binary and cannot be appended as UTF-8 text"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: existing file is not valid UTF-8 text"

    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return f"error: cannot append file: {exc}"
    return "\n".join(
        [
            f"path: {path}",
            "appended: true",
            f"chars_added: {len(content)}",
            f"bytes_added: {append_bytes}",
            f"bytes_total: {current_size + append_bytes}",
        ]
    )


def replace_in_file(path: Any, old: Any, new: Any, *, workdir: Path) -> str:
    if not isinstance(path, str) or not path:
        return "error: path must be a non-empty string"
    if not isinstance(old, str) or not old:
        return "error: old must be a non-empty string"
    if not isinstance(new, str):
        return "error: new must be a string"
    if len(old) > MAX_REPLACE_CHARS:
        return f"error: old text too large for replace_in_file: {len(old)} chars, max {MAX_REPLACE_CHARS}"
    if len(new) > MAX_REPLACE_CHARS:
        return f"error: new text too large for replace_in_file: {len(new)} chars, max {MAX_REPLACE_CHARS}"
    if "\x00" in old or "\x00" in new:
        return "error: replacement text appears to be binary and cannot be used as UTF-8 text"

    target_or_error = resolve_inside_workdir(path, workdir=workdir)
    if isinstance(target_or_error, str):
        return target_or_error
    target = target_or_error
    if not target.exists():
        return f"error: path not found: {path}"
    if not target.is_file():
        return f"error: path is not a file: {path}"

    suffix = target.suffix.lower()
    if suffix in BINARY_OR_SPECIAL_EXTENSIONS:
        return f"error: replace_in_file supports UTF-8 text/code files only; unsupported file type: {suffix}"
    if suffix and suffix not in TEXT_EXTENSIONS:
        return f"error: unsupported text/code file extension for replace_in_file: {suffix}"
    if target.stat().st_size > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: file too large for replace_in_file: {target.stat().st_size} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE}"

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"error: cannot read file before replacement: {exc}"
    if b"\x00" in raw:
        return "error: existing file appears to be binary and cannot be edited as UTF-8 text"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "error: existing file is not valid UTF-8 text"

    matches = text.count(old)
    if matches == 0:
        return "error: old text not found"
    if matches > 1:
        return f"error: old text is ambiguous: {matches} matches"
    updated = text.replace(old, new, 1)
    updated_bytes = len(updated.encode("utf-8"))
    if updated_bytes > MAX_TEXT_FILE_BYTES_AFTER_REPLACE:
        return f"error: replacement would make file too large: {updated_bytes} bytes, max {MAX_TEXT_FILE_BYTES_AFTER_REPLACE}"
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot replace in file: {exc}"
    return "\n".join([f"path: {path}", "replaced: true", "matches: 1", f"bytes_total: {updated_bytes}"])


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


def resolve_inside_workdir(path: str, *, workdir: Path) -> Path | str:
    root = workdir.expanduser().resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return "error: path escapes workdir"
    return target

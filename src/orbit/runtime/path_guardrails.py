from __future__ import annotations

from pathlib import Path
from typing import Any


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


def resolve_inside_workdir(path: str, *, workdir: Path) -> Path | str:
    root = workdir.expanduser().resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return "error: path escapes workdir"
    return target


def validate_existing_file_path(path: Any, *, workdir: Path) -> Path | str:
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
    return target


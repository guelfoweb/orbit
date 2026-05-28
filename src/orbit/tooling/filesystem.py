from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any

from .common import ToolError, coerce_int, resolve_path


MAX_FILE_CHARS = 12_000
MAX_WRITE_CHARS = 10_000
MAX_LIST_ENTRIES = 500
MAX_STAT_ENTRIES = 200
MAX_BINARY_SNIFF_BYTES = 4096
DEFAULT_SHALLOW_MAX_ENTRIES = 120
DEFAULT_RECURSIVE_MAX_ENTRIES = 80
IGNORED_LIST_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}
PRIORITY_SOURCE_DIR_NAMES = {
    "src",
    "app",
    "lib",
    "cmd",
    "pkg",
    "internal",
    "services",
    "service",
    "core",
    "runtime",
    "api",
    "server",
    "client",
    "controllers",
    "controller",
    "routers",
    "router",
    "adapters",
    "adapter",
    "integrations",
    "integration",
    "tooling",
    "terminal",
}
PRIORITY_FILE_STEMS = {
    "main",
    "app",
    "server",
    "client",
    "runtime",
    "agent",
    "router",
    "registry",
    "controller",
    "service",
    "adapter",
    "integration",
    "cli",
}
SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp"}
TEXT_EXTENSIONS = SOURCE_EXTENSIONS | {".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
METADATA_FILE_NAMES = {
    "readme.md",
    "agents.md",
    "defects.md",
    ".gitignore",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "poetry.lock",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "makefile",
}
LOW_PRIORITY_EXTENSIONS = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".pdf", ".apk", ".dex", ".so", ".dll", ".exe", ".bin", ".pyc"}


def filesystem_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a UTF-8 text file from the workdir with bounded line ranges and partial views. "
                    "Use an exact relative path, preferably one returned by list_files. "
                    "Best for source or text files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": MAX_FILE_CHARS},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files under the workdir or a subdirectory. "
                    "Returns relative paths to reuse exactly in later read_file calls. "
                    "For codebase analysis, prefer recursive=true on the requested subtree. "
                    "Implementation files are ordered before metadata, hidden files, and archives."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_ENTRIES},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stat_path",
                "description": (
                    "Inspect filesystem metadata for a file or directory inside the workdir. "
                    "Use this for size, modified time, permissions, newest file, or existence checks. "
                    "For directories, recursive=true returns bounded child metadata sorted newest first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": MAX_STAT_ENTRIES},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_directory",
                "description": "Create one directory inside the workdir, including missing parents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": (
                    "Delete one file or directory inside the workdir. "
                    "Use recursive=true only when you intentionally want to remove a non-empty directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": (
                    "Replace a specific text snippet inside an existing UTF-8 text file in the workdir. "
                    "Prefer this over rewriting the whole file for focused edits."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string", "maxLength": MAX_WRITE_CHARS},
                        "new": {"type": "string", "maxLength": MAX_WRITE_CHARS},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a small UTF-8 text file inside the workdir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "maxLength": MAX_WRITE_CHARS},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append a small UTF-8 text chunk to a file inside the workdir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "maxLength": MAX_WRITE_CHARS},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]


@dataclass
class FilesystemTools:
    workdir: Path

    def read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        start_line = max(1, coerce_int(arguments.get("start_line"), 1))
        max_lines = min(400, max(1, coerce_int(arguments.get("max_lines"), 120)))
        max_chars = min(MAX_FILE_CHARS, max(200, coerce_int(arguments.get("max_chars"), 6000)))
        path = resolve_path(self.workdir, raw_path)
        if not path.is_file():
            raise ToolError(f"file not found: {raw_path}")
        text = self._read_text_file(path, raw_path)
        lines = text.splitlines()
        start_index = start_line - 1
        selected = lines[start_index : start_index + max_lines]
        chunk = "\n".join(selected)
        truncated = False
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars]
            truncated = True
        next_start_line = start_line + len(selected)
        if next_start_line <= len(lines):
            has_more = True
        else:
            has_more = truncated and start_index < len(lines)
        return {
            "ok": True,
            "path": str(path.relative_to(self.workdir)),
            "start_line": start_line,
            "returned_lines": len(selected),
            "total_lines": len(lines),
            "next_start_line": next_start_line,
            "has_more": has_more,
            "truncated": truncated,
            "notice": "PARTIAL view" if has_more or truncated else None,
            "content": chunk,
        }

    def list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str):
            raise ToolError("path must be a string")
        recursive = bool(arguments.get("recursive", False))
        default_max_entries = DEFAULT_RECURSIVE_MAX_ENTRIES if recursive else DEFAULT_SHALLOW_MAX_ENTRIES
        max_entries = min(MAX_LIST_ENTRIES, max(1, coerce_int(arguments.get("max_entries"), default_max_entries)))
        path = resolve_path(self.workdir, raw_path)
        if not path.exists():
            raise ToolError(f"path not found: {raw_path}")
        if recursive:
            iterator = self._iter_recursive(path)
        else:
            iterator = self._iter_shallow(path)
        iterator = self._sort_entries_for_relevance(iterator)
        entries: list[dict[str, Any]] = []
        for item in iterator[:max_entries]:
            rel = item.relative_to(self.workdir)
            entries.append({"path": str(rel), "type": "dir" if item.is_dir() else "file"})
        directories = [entry["path"] for entry in entries if entry["type"] == "dir"]
        files = [entry["path"] for entry in entries if entry["type"] == "file"]
        summary = self._summarize_entries(path, entries)
        return {
            "ok": True,
            "path": str(path.relative_to(self.workdir)),
            "recursive": recursive,
            "count": len(entries),
            "dir_count": len(directories),
            "file_count": len(files),
            "truncated": len(iterator) > max_entries,
            "summary": summary,
            "entries": entries,
        }

    def stat_path(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path", ".")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        recursive = bool(arguments.get("recursive", False))
        max_entries = min(MAX_STAT_ENTRIES, max(1, coerce_int(arguments.get("max_entries"), 40)))
        path = resolve_path(self.workdir, raw_path)
        if not path.exists():
            raise ToolError(f"path not found: {raw_path}")
        result = {
            "ok": True,
            **self._metadata_for_path(path),
        }
        if path.is_dir():
            iterator = self._iter_recursive(path) if recursive else self._iter_shallow(path)
            entries = [self._metadata_for_path(item) for item in iterator if item != path]
            entries.sort(key=lambda item: (-float(item["modified_ts"]), str(item["path"])))
            files = [entry for entry in entries if entry["type"] == "file"]
            directories = [entry for entry in entries if entry["type"] == "dir"]
            result["recursive"] = recursive
            result["count"] = min(len(entries), max_entries)
            result["total_entries"] = len(entries)
            result["file_count"] = len(files)
            result["dir_count"] = len(directories)
            result["truncated"] = len(entries) > max_entries
            result["entries"] = entries[:max_entries]
        return result

    def write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError("content too large")
        path = resolve_path(self.workdir, raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(path.relative_to(self.workdir)), "bytes": len(content.encode("utf-8"))}

    def make_directory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        path = resolve_path(self.workdir, raw_path)
        path.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(path.relative_to(self.workdir)), "type": "dir", "created": True}

    def delete_path(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        recursive = bool(arguments.get("recursive", False))
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        path = resolve_path(self.workdir, raw_path)
        if not path.exists():
            raise ToolError(f"path not found: {raw_path}")
        relative_path = str(path.relative_to(self.workdir))
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                try:
                    path.rmdir()
                except OSError as exc:
                    raise ToolError(
                        f"directory is not empty: {raw_path}. Use recursive=true only if you really want to remove it."
                    ) from exc
            return {"ok": True, "path": relative_path, "type": "dir", "deleted": True, "recursive": recursive}
        path.unlink()
        return {"ok": True, "path": relative_path, "type": "file", "deleted": True, "recursive": False}

    def replace_in_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        old = arguments.get("old")
        new = arguments.get("new")
        replace_all = bool(arguments.get("replace_all", False))
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        if not isinstance(old, str) or not old:
            raise ToolError("old must be a non-empty string")
        if not isinstance(new, str):
            raise ToolError("new must be a string")
        if len(old) > MAX_WRITE_CHARS or len(new) > MAX_WRITE_CHARS:
            raise ToolError("replacement text too large")
        path = resolve_path(self.workdir, raw_path)
        if not path.is_file():
            raise ToolError(f"file not found: {raw_path}")
        text = self._read_text_file(path, raw_path)
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError(f"target text not found in file: {raw_path}")
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"target text appears {occurrences} times in file: {raw_path}. "
                "Use replace_all=true or provide a more specific target."
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        replaced = occurrences if replace_all else 1
        return {
            "ok": True,
            "path": str(path.relative_to(self.workdir)),
            "replaced": replaced,
            "replace_all": replace_all,
        }

    def append_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError("content too large")
        path = resolve_path(self.workdir, raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return {"ok": True, "path": str(path.relative_to(self.workdir)), "bytes": len(content.encode("utf-8"))}

    @staticmethod
    def _read_text_file(path: Path, raw_path: str) -> str:
        try:
            with path.open("rb") as handle:
                sample = handle.read(MAX_BINARY_SNIFF_BYTES)
                if b"\x00" in sample:
                    raise ToolError(
                        f"file is binary and cannot be read as text: {raw_path}. "
                        "Use bash with strings, hexdump, or another binary-aware command."
                    )
                remainder = handle.read()
        except OSError as exc:
            raise ToolError(f"cannot read file: {raw_path}") from exc
        try:
            return (sample + remainder).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(
                f"file is not valid UTF-8 text: {raw_path}. "
                "Use bash with strings, hexdump, or another binary-aware command."
            ) from exc

    def _iter_shallow(self, path: Path) -> list[Path]:
        if not path.is_dir():
            return [path]
        items = []
        for item in path.iterdir():
            if item.is_dir() and item.name in IGNORED_LIST_DIR_NAMES:
                continue
            items.append(item)
        return items

    def _iter_recursive(self, path: Path) -> list[Path]:
        if not path.is_dir():
            return [path]
        items: list[Path] = []
        for root, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_LIST_DIR_NAMES)
            root_path = Path(root)
            if root_path != path:
                items.append(root_path)
            for filename in sorted(filenames):
                items.append(root_path / filename)
        return items

    def _sort_entries_for_relevance(self, items: list[Path]) -> list[Path]:
        return sorted(items, key=self._entry_priority)

    def _entry_priority(self, path: Path) -> tuple[int, str]:
        rel = path.relative_to(self.workdir)
        parts = tuple(part.lower() for part in rel.parts)
        name = path.name.lower()
        stem = path.stem.lower()
        suffix = path.suffix.lower()
        score = 0

        if any(part.startswith(".") for part in parts):
            score += 60
        if name in METADATA_FILE_NAMES:
            score += 35
        if suffix in LOW_PRIORITY_EXTENSIONS:
            score += 80
        if path.is_dir():
            if name in PRIORITY_SOURCE_DIR_NAMES:
                score -= 25
            else:
                score += 10
        else:
            if suffix in SOURCE_EXTENSIONS:
                score -= 25
            elif suffix in TEXT_EXTENSIONS:
                score += 5
            else:
                score += 20
            if stem in PRIORITY_FILE_STEMS:
                score -= 20
            if stem in {"__init__", "__main__"}:
                score += 25

        for part in parts:
            if part in PRIORITY_SOURCE_DIR_NAMES:
                score -= 10
            if part in {"tests", "test"}:
                score -= 5

        return score, str(rel)

    def _summarize_entries(self, base_path: Path, entries: list[dict[str, str]]) -> str:
        if not entries:
            return "empty directory"
        base_rel = base_path.relative_to(self.workdir)
        shallow = str(base_rel) in {"", "."}
        directories = [entry["path"] for entry in entries if entry["type"] == "dir"]
        files = [entry["path"] for entry in entries if entry["type"] == "file"]
        file_names = [Path(item).name for item in files[:8]]
        dir_names = [Path(item).name for item in directories[:8]]
        parts: list[str] = []
        if directories:
            parts.append(f"dirs: {', '.join(dir_names)}")
        if files:
            parts.append(f"files: {', '.join(file_names)}")
        if shallow and not directories and files:
            return f"top-level files: {', '.join(file_names)}"
        return " | ".join(parts)

    def _metadata_for_path(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        relative = path.relative_to(self.workdir)
        path_type = "dir" if path.is_dir() else "file" if path.is_file() else "other"
        return {
            "path": str(relative),
            "type": path_type,
            "size_bytes": stat.st_size,
            "modified_ts": stat.st_mtime,
            "modified_at": _format_timestamp(stat.st_mtime),
            "mode": oct(stat.st_mode & 0o777),
        }


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")

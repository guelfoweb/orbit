from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

from .common import ToolError
from .filesystem import FilesystemTools, filesystem_definitions
from .shell import ShellTools, shell_definitions
from .web import WebTools, web_definitions
from ..core.tool_router import (
    TOOL_CATEGORY_FILESYSTEM,
    TOOL_CATEGORY_SHELL,
    TOOL_CATEGORY_WEB,
    TOOL_CATEGORY_WRITE,
)


@dataclass
class ToolRegistry:
    workdir: Path
    _definitions: list[dict[str, Any]] = field(init=False, repr=False)
    _handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = field(init=False, repr=False)
    _categories: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        filesystem = FilesystemTools(self.workdir)
        shell = ShellTools(self.workdir)
        web = WebTools()
        self._definitions = [*filesystem_definitions(), *shell_definitions(), *web_definitions()]
        self._handlers = {
            "search_web": web.search_web,
            "read_file": filesystem.read_file,
            "list_files": filesystem.list_files,
            "stat_path": filesystem.stat_path,
            "make_directory": filesystem.make_directory,
            "delete_path": filesystem.delete_path,
            "replace_in_file": filesystem.replace_in_file,
            "write_file": filesystem.write_file,
            "append_file": filesystem.append_file,
            "bash": shell.bash,
            "fetch_url": web.fetch_url,
        }
        self._categories = {
            "list_files": TOOL_CATEGORY_FILESYSTEM,
            "read_file": TOOL_CATEGORY_FILESYSTEM,
            "stat_path": TOOL_CATEGORY_FILESYSTEM,
            "make_directory": TOOL_CATEGORY_WRITE,
            "delete_path": TOOL_CATEGORY_WRITE,
            "replace_in_file": TOOL_CATEGORY_WRITE,
            "write_file": TOOL_CATEGORY_WRITE,
            "append_file": TOOL_CATEGORY_WRITE,
            "bash": TOOL_CATEGORY_SHELL,
            "search_web": TOOL_CATEGORY_WEB,
            "fetch_url": TOOL_CATEGORY_WEB,
        }

    def definitions(self) -> list[dict[str, Any]]:
        return self._definitions

    def definitions_for_categories(self, categories: tuple[str, ...]) -> list[dict[str, Any]]:
        if not categories:
            return []
        allowed = set(categories)
        filtered = []
        for item in self._definitions:
            function = item.get("function", {})
            name = function.get("name")
            if self._categories.get(name) in allowed:
                filtered.append(item)
        return filtered or self._definitions

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return handler(arguments)
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"tool failure: {exc}"}

    @staticmethod
    def encode_tool_result(result: dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False)

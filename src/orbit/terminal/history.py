from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from orbit.terminal.prompt_preview import compact_prompt_preview, is_compact_prompt_preview


DEFAULT_HISTORY_ROOT = Path.home() / ".orbit" / "history"
LEGACY_HISTORY_NAME = "default.history"
MAX_HISTORY_ITEMS = 500
FULL_HISTORY_SUFFIX = ".full.jsonl"


@dataclass(frozen=True)
class PromptResolution:
    prompt: str
    missing_full_text: bool = False


class PromptHistory:
    def __init__(self, *, path: Path, readline_module: object | None) -> None:
        self.path = path
        self.readline = readline_module

    @classmethod
    def for_workdir(cls, workdir: Path, *, root: Path = DEFAULT_HISTORY_ROOT) -> PromptHistory:
        key = hashlib.sha256(str(workdir.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
        return cls(path=root / f"{key}.history", readline_module=_load_readline())

    def load(self) -> None:
        if self.readline is None:
            return
        if not self._ensure_parent_dir():
            return
        if not self.path.exists():
            return
        try:
            self.readline.read_history_file(str(self.path))
            self._dedupe_readline_history()
        except OSError:
            return

    def add(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt or prompt.startswith("/") or self.readline is None:
            return
        history_item = self._history_item(prompt)
        self._remove_existing(history_item)
        self.readline.add_history(history_item)
        self._remember_full_prompt(history_item, prompt)
        self._trim_readline_history()

    def save(self) -> None:
        if self.readline is None:
            return
        if not self._ensure_parent_dir():
            return
        try:
            self._dedupe_readline_history()
            self.readline.write_history_file(str(self.path))
        except OSError:
            return

    def _ensure_parent_dir(self) -> bool:
        try:
            if self.path.parent.is_file():
                legacy_content = self.path.parent.read_text(encoding="utf-8", errors="replace")
                self.path.parent.unlink()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path = self.path.parent / LEGACY_HISTORY_NAME
                if legacy_content and not legacy_path.exists():
                    legacy_path.write_text(legacy_content, encoding="utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return self.path.parent.is_dir()

    def _dedupe_readline_history(self) -> None:
        items = self._history_items()
        deduped = dedupe_prompts(items)[-MAX_HISTORY_ITEMS:]
        self.readline.clear_history()
        for item in deduped:
            self.readline.add_history(item)

    def _remove_existing(self, prompt: str) -> None:
        items = [item for item in self._history_items() if item != prompt]
        self.readline.clear_history()
        for item in items[-MAX_HISTORY_ITEMS:]:
            self.readline.add_history(item)

    def resolve(self, prompt: str) -> str:
        return self.resolve_prompt(prompt).prompt

    def resolve_prompt(self, prompt: str) -> PromptResolution:
        prompt = prompt.strip()
        if prompt and is_compact_prompt_preview(prompt):
            full_prompt = self._full_prompt_for_preview(prompt)
            if full_prompt is None:
                return PromptResolution(prompt=prompt, missing_full_text=True)
            return PromptResolution(prompt=full_prompt)
        return PromptResolution(prompt=prompt)

    def _trim_readline_history(self) -> None:
        items = self._history_items()
        if len(items) <= MAX_HISTORY_ITEMS:
            return
        self.readline.clear_history()
        for item in items[-MAX_HISTORY_ITEMS:]:
            self.readline.add_history(item)

    def _history_items(self) -> list[str]:
        length = self.readline.get_current_history_length()
        items: list[str] = []
        for index in range(1, length + 1):
            item = self.readline.get_history_item(index)
            if isinstance(item, str) and item:
                items.append(item)
        return items

    def _history_item(self, prompt: str) -> str:
        return compact_prompt_preview(prompt)

    def _full_history_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}{FULL_HISTORY_SUFFIX}")

    def _remember_full_prompt(self, history_item: str, prompt: str) -> None:
        if history_item == prompt:
            return
        if not self._ensure_parent_dir():
            return
        path = self._full_history_path()
        records = [record for record in self._full_prompt_records() if record.get("preview") != history_item]
        records.append({"preview": history_item, "prompt": prompt})
        records = records[-MAX_HISTORY_ITEMS:]
        try:
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
        except OSError:
            return

    def _full_prompt_for_preview(self, preview: str) -> str | None:
        for record in reversed(self._full_prompt_records()):
            if record.get("preview") == preview and isinstance(record.get("prompt"), str):
                return record["prompt"]
        return None

    def _full_prompt_records(self) -> list[dict[str, str]]:
        path = self._full_history_path()
        if not path.exists():
            return []
        records: list[dict[str, str]] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("preview"), str) and isinstance(record.get("prompt"), str):
                    records.append({"preview": record["preview"], "prompt": record["prompt"]})
        except OSError:
            return []
        return records


def dedupe_prompts(prompts: list[str]) -> list[str]:
    seen: set[str] = set()
    result_reversed: list[str] = []
    for prompt in reversed(prompts):
        normalized = prompt.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result_reversed.append(normalized)
    result_reversed.reverse()
    return result_reversed


def _load_readline() -> object | None:
    try:
        import readline  # type: ignore
    except ImportError:
        return None
    try:
        readline.set_history_length(MAX_HISTORY_ITEMS)
    except Exception:
        pass
    try:
        readline.parse_and_bind("set enable-bracketed-paste on")
    except Exception:
        pass
    return readline

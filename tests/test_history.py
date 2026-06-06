from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.history import PromptHistory, dedupe_prompts


class FakeReadline:
    def __init__(self) -> None:
        self.items: list[str] = []

    def read_history_file(self, path: str) -> None:
        self.items = Path(path).read_text(encoding="utf-8").splitlines()

    def write_history_file(self, path: str) -> None:
        Path(path).write_text("\n".join(self.items) + ("\n" if self.items else ""), encoding="utf-8")

    def add_history(self, item: str) -> None:
        self.items.append(item)

    def clear_history(self) -> None:
        self.items.clear()

    def get_current_history_length(self) -> int:
        return len(self.items)

    def get_history_item(self, index: int) -> str | None:
        try:
            return self.items[index - 1]
        except IndexError:
            return None


class HistoryTests(unittest.TestCase):
    def test_dedupe_prompts_keeps_last_occurrence_order(self) -> None:
        prompts = ["a", "b", "a", "c", "b"]

        self.assertEqual(dedupe_prompts(prompts), ["a", "c", "b"])

    def test_prompt_history_skips_slash_commands_and_dedupes(self) -> None:
        readline = FakeReadline()
        with tempfile.TemporaryDirectory() as tmp:
            history = PromptHistory(path=Path(tmp) / "history", readline_module=readline)

            history.add("hello")
            history.add("/status")
            history.add("hello")
            history.add("summarize file")

        self.assertEqual(readline.items, ["hello", "summarize file"])

    def test_prompt_history_persists_deduped_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history"
            first = PromptHistory(path=path, readline_module=FakeReadline())
            first.add("one")
            first.add("two")
            first.add("one")
            first.save()

            readline = FakeReadline()
            second = PromptHistory(path=path, readline_module=readline)
            second.load()

        self.assertEqual(readline.items, ["two", "one"])

    def test_prompt_history_migrates_legacy_parent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "history"
            parent.write_text("legacy", encoding="utf-8")
            history = PromptHistory(path=parent / "item.history", readline_module=FakeReadline())

            history.load()
            history.add("hello")
            history.save()

            self.assertTrue(parent.is_dir())
            self.assertEqual((parent / "default.history").read_text(encoding="utf-8"), "legacy")
            self.assertEqual((parent / "item.history").read_text(encoding="utf-8"), "hello\n")


if __name__ == "__main__":
    unittest.main()

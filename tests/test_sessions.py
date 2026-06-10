from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.sessions import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_save_and_load_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=root)
            messages = [{"role": "user", "content": "hello"}]

            store.save(messages=messages, workdir=workdir, model="model", base_url="http://localhost")

            self.assertEqual(store.load(), messages)

    def test_clear_removes_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.save(messages=[{"role": "user", "content": "hello"}], workdir=workdir, model="m", base_url="u")

            store.clear()

            self.assertIsNone(store.load())

    def test_load_with_warning_reports_corrupt_session_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text("{not-json", encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("corrupt session", warning or "")

    def test_load_with_warning_reports_invalid_session_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text('{"messages": [{"role": "tool", "content": "x"}]}', encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("invalid session", warning or "")

    def test_load_with_warning_accepts_tool_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            messages = [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "list_files", "arguments": {"path": "."}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "name": "list_files", "content": "README.md"},
                {"role": "assistant", "content": "README.md"},
            ]
            store.save(messages=messages, workdir=workdir, model="m", base_url="u")

            loaded, warning = store.load_with_warning()

        self.assertEqual(loaded, messages)
        self.assertIsNone(warning)

    def test_load_with_warning_rejects_malformed_tool_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            store = SessionStore.for_workdir(workdir, root=Path(tmp) / "sessions")
            store.path.parent.mkdir(parents=True)
            store.path.write_text('{"messages": [{"role": "tool", "content": "x"}]}', encoding="utf-8")

            messages, warning = store.load_with_warning()

        self.assertIsNone(messages)
        self.assertIsNotNone(warning)
        self.assertIn("invalid session", warning or "")


if __name__ == "__main__":
    unittest.main()

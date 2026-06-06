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


if __name__ == "__main__":
    unittest.main()

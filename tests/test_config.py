from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.cli import build_parser
from orbit.terminal.config import DEFAULT_SYSTEM_PROMPT, load_app_config


class ConfigTests(unittest.TestCase):
    def test_default_system_prompt_allows_attached_media_answers(self) -> None:
        self.assertIn("Attached image/audio => answer normally", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Never emit raw tool-call syntax", DEFAULT_SYSTEM_PROMPT)
        self.assertIn('{"_route":"FILESYSTEM","tool":"<tool>"}', DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Never answer file contents from memory", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Common args: path, pattern, command, url, query, content.", DEFAULT_SYSTEM_PROMPT)

    def test_missing_config_uses_defaults(self) -> None:
        args = _parse("--config", "/tmp/orbit-missing-config.json")

        config = load_app_config(args)

        self.assertEqual(config.base_url, "http://127.0.0.1:18080")
        self.assertEqual(config.model, "local-model")
        self.assertEqual(config.workdir, Path(".").resolve())
        self.assertEqual(config.max_tokens, 512)
        self.assertIsNone(config.context_tokens)

    def test_config_file_sets_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "http://127.0.0.1:19090",
                        "model": "gemma4:12b",
                        "workdir": str(Path(tmp)),
                        "timeout": 120,
                        "temperature": 0,
                        "max_tokens": 512,
                        "context_tokens": 4096,
                        "no_system": True,
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(_parse("--config", str(path)))

        self.assertEqual(config.base_url, "http://127.0.0.1:19090")
        self.assertEqual(config.model, "gemma4:12b")
        self.assertEqual(config.workdir, Path(tmp).resolve())
        self.assertEqual(config.timeout, 120.0)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.context_tokens, 4096)
        self.assertTrue(config.no_system)

    def test_cli_flags_override_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"model": "from-config", "workdir": "/tmp", "max_tokens": 128}), encoding="utf-8")

            config = load_app_config(
                _parse(
                    "--config",
                    str(path),
                    "--model",
                    "from-cli",
                    "--workdir",
                    str(ROOT),
                    "--max-tokens",
                    "64",
                    "--context-tokens",
                    "2048",
                )
            )

        self.assertEqual(config.model, "from-cli")
        self.assertEqual(config.workdir, ROOT.resolve())
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.context_tokens, 2048)


def _parse(*args: str) -> argparse.Namespace:
    return build_parser().parse_args(list(args))


if __name__ == "__main__":
    unittest.main()

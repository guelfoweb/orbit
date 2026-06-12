from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.terminal.context_status import context_status_text, count_messages_by_type, estimate_context_breakdown


class ContextStatusTests(unittest.TestCase):
    def test_context_breakdown_groups_roles(self) -> None:
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "system", "content": "Orbit session memory (visible context):\nstate"},
            {"role": "user", "content": "hello user"},
            {"role": "assistant", "content": "hello assistant"},
            {"role": "tool", "content": "tool result"},
        ]

        breakdown = estimate_context_breakdown(messages)

        self.assertGreater(breakdown.system, 0)
        self.assertGreater(breakdown.memory, 0)
        self.assertGreater(breakdown.user, 0)
        self.assertGreater(breakdown.assistant, 0)
        self.assertGreater(breakdown.tool_result, 0)
        self.assertEqual(breakdown.other, 0)
        self.assertEqual(
            breakdown.total,
            breakdown.system + breakdown.memory + breakdown.user + breakdown.assistant + breakdown.tool_result,
        )

    def test_message_breakdown_counts_roles(self) -> None:
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "system", "content": "Orbit session memory (visible context):\nstate"},
            {"role": "user", "content": "hello user"},
            {"role": "assistant", "content": "hello assistant"},
            {"role": "tool", "content": "tool result"},
        ]

        breakdown = count_messages_by_type(messages)

        self.assertEqual(breakdown.total, 5)
        self.assertEqual(breakdown.system, 1)
        self.assertEqual(breakdown.memory, 1)
        self.assertEqual(breakdown.user, 1)
        self.assertEqual(breakdown.assistant, 1)
        self.assertEqual(breakdown.tool_result, 1)
        self.assertEqual(breakdown.other, 0)

    def test_context_status_text_is_readable(self) -> None:
        output = context_status_text(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            context_tokens=1000,
        )

        self.assertIn("Context\n-------", output)
        self.assertIn("window: 1000", output)
        self.assertIn("estimated_total:", output)
        self.assertIn("Token estimate\n--------------", output)
        self.assertIn("Message count\n-------------", output)
        self.assertIn("system:", output)
        self.assertIn("user:", output)
        self.assertIn("assistant:", output)
        self.assertIn("tool_result:", output)


if __name__ == "__main__":
    unittest.main()

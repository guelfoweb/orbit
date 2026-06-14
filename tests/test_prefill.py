from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.session_memory import estimate_text_tokens
from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, FINAL_FROM_TOOL_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT
from orbit.terminal.prefill import estimate_prefill_tokens, estimate_prefill_tokens_after_tool_result


class PrefillTests(unittest.TestCase):
    def test_chat_prompt_estimate_is_smaller_than_tools_prompt_estimate(self) -> None:
        messages = [{"role": "user", "content": "hello"}]

        chat = estimate_prefill_tokens(messages, "who are you?", system_prompt=CHAT_SYSTEM_PROMPT)
        tools = estimate_prefill_tokens(messages, "who are you?", system_prompt=ROUTE_SYSTEM_PROMPT)

        self.assertLess(chat, tools)

    def test_after_tool_result_includes_reinjected_content(self) -> None:
        messages = [{"role": "user", "content": "read file"}]
        content = "tool output " * 100

        before = estimate_prefill_tokens(messages, "", system_prompt=FINAL_FROM_TOOL_SYSTEM_PROMPT)
        after = estimate_prefill_tokens_after_tool_result(messages, content)

        self.assertGreaterEqual(after, before + estimate_text_tokens(content))


if __name__ == "__main__":
    unittest.main()

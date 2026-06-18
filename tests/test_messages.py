from __future__ import annotations

import unittest

from orbit.runtime.messages import CHAT_SYSTEM_PROMPT, ROUTE_SYSTEM_PROMPT, with_chat_system_prompt


class MessagePromptTests(unittest.TestCase):
    def test_chat_system_prompt_prepends_when_missing(self) -> None:
        messages = with_chat_system_prompt([{"role": "user", "content": "hello"}])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], CHAT_SYSTEM_PROMPT)

    def test_existing_system_prompt_is_replaced_for_chat_mode(self) -> None:
        messages = with_chat_system_prompt(
            [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}, {"role": "user", "content": "x"}]
        )
        self.assertEqual(messages[0]["content"], CHAT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

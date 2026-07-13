from __future__ import annotations

import unittest

from orbit.runtime.messages import (
    CHAT_SYSTEM_PROMPT,
    FINAL_FROM_TOOL_SYSTEM_PROMPT,
    ROUTE_SYSTEM_PROMPT,
    TOOL_CALL_SYSTEM_PROMPT,
    VISIBLE_CHAT_SYSTEM_PROMPT,
    with_chat_system_prompt,
    with_visible_chat_system_prompt,
)


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

    def test_visible_chat_prompt_preserves_facts_and_rejects_missing_details(self) -> None:
        messages = with_visible_chat_system_prompt([{"role": "user", "content": "summarize that"}])
        self.assertEqual(messages[0]["content"], VISIBLE_CHAT_SYSTEM_PROMPT)
        self.assertIn("visible assistant answers", VISIBLE_CHAT_SYSTEM_PROMPT)
        self.assertIn("Preserve facts", VISIBLE_CHAT_SYSTEM_PROMPT)
        self.assertIn("If a detail is missing", VISIBLE_CHAT_SYSTEM_PROMPT)
        self.assertIn("visible conversation", VISIBLE_CHAT_SYSTEM_PROMPT)
        self.assertIn("Never infer", VISIBLE_CHAT_SYSTEM_PROMPT)

    def test_route_policy_prefers_existing_context_for_recaps(self) -> None:
        self.assertIn("recap, repeat, summary, explanation, comparison, or continuation", ROUTE_SYSTEM_PROMPT)
        self.assertIn('prefer {"route":"CHAT"} when the prior context is sufficient', ROUTE_SYSTEM_PROMPT)

    def test_route_policy_allows_refresh_and_verification_tools(self) -> None:
        self.assertIn("fresh/current data", ROUTE_SYSTEM_PROMPT)
        self.assertIn("verification", ROUTE_SYSTEM_PROMPT)
        self.assertIn("changed files/state", ROUTE_SYSTEM_PROMPT)

    def test_route_policy_allows_new_information_tools(self) -> None:
        self.assertIn("new information", ROUTE_SYSTEM_PROMPT)
        self.assertIn("missing/stale/ambiguous/insufficient prior context", ROUTE_SYSTEM_PROMPT)

    def test_route_policy_covers_prior_file_or_search_summaries_generally(self) -> None:
        self.assertIn("information already in this conversation", ROUTE_SYSTEM_PROMPT)
        self.assertIn("summary", ROUTE_SYSTEM_PROMPT)
        self.assertIn("prior context is sufficient", ROUTE_SYSTEM_PROMPT)

    def test_tool_call_policy_still_requires_one_tool_after_route_decides_tool(self) -> None:
        self.assertIn("Call exactly one available tool", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Operate on the latest user request only", TOOL_CALL_SYSTEM_PROMPT)

    def test_final_tool_policy_preserves_safety_and_error_guidance(self) -> None:
        self.assertIn("from the tool result", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("Do not call tools", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("raw tool-call syntax", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("Never claim lack of access", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("Report errors briefly", FINAL_FROM_TOOL_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from orbit.runtime.messages import (
    AGENT_FINAL_COMPLETION_INSTRUCTION,
    AGENT_ROUTE_CONTROL_INSTRUCTION,
    AGENT_ROUTE_SYSTEM_PROMPT,
    AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT,
    AGENT_TOOL_CONTINUATION_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    FINAL_FROM_TOOL_SYSTEM_PROMPT,
    ROUTE_SYSTEM_PROMPT,
    TOOL_CALL_SYSTEM_PROMPT,
    VISIBLE_CHAT_SYSTEM_PROMPT,
    with_agent_final_completion_instruction,
    with_agent_command_system_prompt,
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

    def test_route_policy_treats_displayed_tool_syntax_as_data(self) -> None:
        self.assertIn("quoted text, fenced code, JSON examples", ROUTE_SYSTEM_PROMPT)
        self.assertIn("unless the latest user request explicitly asks", ROUTE_SYSTEM_PROMPT)

    def test_agent_route_requires_model_completion_control(self) -> None:
        messages = with_agent_command_system_prompt(
            [{"role": "system", "content": ROUTE_SYSTEM_PROMPT}, {"role": "user", "content": "inspect"}]
        )

        self.assertEqual(messages[0]["content"], ROUTE_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], AGENT_ROUTE_CONTROL_INSTRUCTION)
        self.assertEqual(messages[2], {"role": "user", "content": "inspect"})
        self.assertIn('"after":"final"', AGENT_ROUTE_SYSTEM_PROMPT)
        self.assertIn('"after":"continue"', AGENT_ROUTE_SYSTEM_PROMPT)
        self.assertIn("This is a model decision", AGENT_ROUTE_SYSTEM_PROMPT)

    def test_route_policy_covers_prior_file_or_search_summaries_generally(self) -> None:
        self.assertIn("information already in this conversation", ROUTE_SYSTEM_PROMPT)
        self.assertIn("summary", ROUTE_SYSTEM_PROMPT)
        self.assertIn("prior context is sufficient", ROUTE_SYSTEM_PROMPT)

    def test_tool_call_policy_still_requires_one_tool_after_route_decides_tool(self) -> None:
        self.assertIn("Call exactly one available tool", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Operate on the latest user request only", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Each shell call starts in a fresh shell at workdir", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Preserve every destination directory requested by the user", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("one short self-contained action", TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Copy requested literal schemas and headers verbatim", AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("structured interfaces, not shell executables", AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("Python standard library", AGENT_STRICT_TOOL_CALL_SYSTEM_PROMPT)
        self.assertIn("directory changes do not persist", ROUTE_SYSTEM_PROMPT)
        self.assertIn("Preserve every user-requested destination directory", ROUTE_SYSTEM_PROMPT)

    def test_final_tool_policy_preserves_safety_and_error_guidance(self) -> None:
        self.assertIn("from tool evidence", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("shortest complete answer", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("retaining exact details when needed", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("Do not invent facts", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("call tools", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("raw tool-call syntax", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("claim lack of access", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("report errors briefly", FINAL_FROM_TOOL_SYSTEM_PROMPT)
        self.assertIn("End after the answer", FINAL_FROM_TOOL_SYSTEM_PROMPT)

    def test_agent_final_instruction_preserves_stable_system_message(self) -> None:
        original = [
            {"role": "system", "content": FINAL_FROM_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": "request"},
            {"role": "system", "content": "evidence_context:"},
        ]

        messages = with_agent_final_completion_instruction(original)

        self.assertEqual(len(messages), len(original))
        self.assertEqual(messages[0], original[0])
        self.assertEqual(messages[1], original[1])
        self.assertEqual(
            messages[2]["content"],
            f"evidence_context:\n{AGENT_FINAL_COMPLETION_INSTRUCTION}",
        )
        self.assertEqual(original[2]["content"], "evidence_context:")

    def test_agent_final_instruction_uses_dynamic_system_message_only(self) -> None:
        original = [
            {"role": "system", "content": FINAL_FROM_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": "request"},
        ]

        messages = with_agent_final_completion_instruction(original)

        self.assertEqual(messages[:2], original)
        self.assertEqual(messages[2], {"role": "system", "content": AGENT_FINAL_COMPLETION_INSTRUCTION})


if __name__ == "__main__":
    unittest.main()

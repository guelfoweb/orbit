from __future__ import annotations

import unittest

from orbit.core.intent.gate import intent_gate_decision, intent_gate_messages, parse_intent_gate_reply, should_confirm_tool_route
from orbit.core.intent.model_gate import fetch_url_matches_recent_search_result
from orbit.core.intent.tool_gate import tool_gate_decision, tool_gate_messages
from orbit.core.tools.router import route_tool_categories


class IntentGateTests(unittest.TestCase):
    def test_clear_workspace_listing_does_not_need_confirmation(self) -> None:
        route = route_tool_categories("list all files and directories in the current workspace")
        self.assertFalse(should_confirm_tool_route("list all files and directories in the current workspace", route))

    def test_ambiguous_malware_explanation_needs_confirmation(self) -> None:
        prompt = "tell me about malware analysis"
        self.assertTrue(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_clear_static_analysis_does_not_need_confirmation(self) -> None:
        prompt = "Perform static analysis for all malware samples in the malware directory."
        self.assertFalse(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_security_quiz_prompt_does_not_need_confirmation_or_tools(self) -> None:
        prompt = "ask me something about malware analysis, C2 or IoC"
        route = route_tool_categories(prompt)
        self.assertEqual(route.categories, ())
        self.assertFalse(should_confirm_tool_route(prompt, route))

    def test_ambiguous_file_edit_needs_confirmation(self) -> None:
        prompt = "change the conclusion"
        route = route_tool_categories(prompt)
        self.assertTrue(should_confirm_tool_route(prompt, route))

    def test_clear_file_edit_does_not_need_confirmation(self) -> None:
        prompt = "replace foo with bar in README.md"
        self.assertFalse(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_clear_base64_command_does_not_need_confirmation(self) -> None:
        prompt = 'decode this string "Y2lhbw==" from base64'
        self.assertFalse(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_explicit_web_lookup_does_not_need_confirmation(self) -> None:
        prompt = "search online for information about Dante Alighieri"
        self.assertFalse(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_entity_lookup_is_clear_web_lookup(self) -> None:
        prompt = "who is Dante Alighieri?"
        self.assertFalse(should_confirm_tool_route(prompt, route_tool_categories(prompt)))

    def test_web_search_discussion_does_not_need_confirmation_or_tools(self) -> None:
        prompt = "what do you think about web search in LLMs?"
        route = route_tool_categories(prompt)
        self.assertEqual(route.categories, ())
        self.assertFalse(should_confirm_tool_route(prompt, route))

    def test_decision_includes_reason(self) -> None:
        prompt = "change the conclusion"
        decision = intent_gate_decision(prompt, route_tool_categories(prompt))
        self.assertTrue(decision.confirm)
        self.assertEqual(decision.reason, "ambiguous route")

    def test_gate_messages_are_yes_no_only(self) -> None:
        prompt = "change the conclusion"
        route = route_tool_categories(prompt)
        messages = intent_gate_messages(user_input=prompt, route=route)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Answer only YES or NO", messages[0]["content"])
        self.assertIn("Candidate route: ambiguous", messages[1]["content"])

    def test_parse_gate_reply(self) -> None:
        self.assertIs(parse_intent_gate_reply("maybe"), None)
        self.assertTrue(parse_intent_gate_reply("YES"))
        self.assertFalse(parse_intent_gate_reply("NO"))

    def test_tool_gate_confirms_fetch_url_without_user_url(self) -> None:
        prompt = "search online for information about Dante Alighieri"
        route = route_tool_categories(prompt)
        decision = tool_gate_decision(
            user_input=prompt,
            route=route,
            tool_name="fetch_url",
            arguments={"url": "https://example.com/dante"},
        )
        self.assertTrue(decision.confirm)
        self.assertEqual(decision.reason, "fetch_url without explicit user URL")

    def test_tool_gate_does_not_confirm_clear_list_files(self) -> None:
        prompt = "list all files in this workspace"
        route = route_tool_categories(prompt)
        decision = tool_gate_decision(
            user_input=prompt,
            route=route,
            tool_name="list_files",
            arguments={"path": "."},
        )
        self.assertFalse(decision.confirm)

    def test_tool_gate_messages_are_yes_no_only(self) -> None:
        prompt = "search online for information about Dante Alighieri"
        route = route_tool_categories(prompt)
        messages = tool_gate_messages(
            user_input=prompt,
            route=route,
            tool_name="fetch_url",
            arguments={"url": "https://example.com/dante"},
            reason="fetch_url without explicit user URL",
        )
        self.assertIn("Answer only YES or NO", messages[0]["content"])
        self.assertIn("Proposed tool call: fetch_url", messages[1]["content"])

    def test_fetch_url_matches_recent_search_result_before_next_user_turn(self) -> None:
        messages = [
            {"role": "user", "content": "search online for information about Dante"},
            {
                "role": "tool",
                "tool_name": "search_web",
                "content": '{"results": [{"url": "https://example.com/dante"}]}',
            },
        ]
        self.assertTrue(
            fetch_url_matches_recent_search_result(
                {"url": "https://example.com/dante"},
                messages,
            )
        )

    def test_fetch_url_does_not_match_search_result_from_previous_user_turn(self) -> None:
        messages = [
            {
                "role": "tool",
                "tool_name": "search_web",
                "content": '{"results": [{"url": "https://example.com/dante"}]}',
            },
            {"role": "user", "content": "now answer from memory"},
        ]
        self.assertFalse(
            fetch_url_matches_recent_search_result(
                {"url": "https://example.com/dante"},
                messages,
            )
        )


if __name__ == "__main__":
    unittest.main()

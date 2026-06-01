from __future__ import annotations

import unittest

from orbit.core.intent_gate import intent_gate_decision, intent_gate_messages, parse_intent_gate_reply, should_confirm_tool_route
from orbit.core.tool_router import route_tool_categories


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


if __name__ == "__main__":
    unittest.main()

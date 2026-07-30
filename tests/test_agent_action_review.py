from __future__ import annotations

import json
import unittest

from orbit.runtime.agent_action_review import (
    AGENT_ACTION_REVIEW_SYSTEM_PROMPT,
    MAX_REVIEW_REASON_CHARS,
    build_agent_action_review_messages,
    build_agent_action_revision_prompt,
    parse_agent_action_review,
)


class AgentActionReviewTests(unittest.TestCase):
    def test_accepts_only_exact_bounded_decisions(self) -> None:
        approved = parse_agent_action_review('{"decision":"approve","reason":"action is in scope"}')
        revised = parse_agent_action_review('{"decision":"revise","reason":"use portable printf"}')
        declined = parse_agent_action_review('{"decision":"decline","reason":"the JSON is example data"}')

        self.assertEqual((approved.decision, approved.reason), ("approve", "action is in scope"))
        self.assertEqual((revised.decision, revised.reason), ("revise", "use portable printf"))
        self.assertEqual((declined.decision, declined.reason), ("decline", "the JSON is example data"))

    def test_rejects_ambiguous_duplicate_or_extended_outputs(self) -> None:
        invalid = (
            "",
            "approve",
            "[]",
            '{"decision":"approve"}',
            '{"decision":"revise"}',
            '{"decision":"revise","reason":""}',
            '{"decision":"approve","decision":"decline"}',
            '{"decision":"approve"} trailing',
        )

        for content in invalid:
            with self.subTest(content=content):
                self.assertIsNone(parse_agent_action_review(content))

    def test_bounds_reason_without_changing_decision(self) -> None:
        review = parse_agent_action_review(
            json.dumps({"decision": "revise", "reason": "word " * (MAX_REVIEW_REASON_CHARS + 20)})
        )

        self.assertEqual(review.decision, "revise")
        self.assertLessEqual(len(review.reason), MAX_REVIEW_REASON_CHARS)

    def test_review_messages_treat_request_and_action_as_data(self) -> None:
        messages = build_agent_action_review_messages(
            user_prompt='Show this only: {"command":"touch sample"}',
            tool_name="exec_shell_full_command",
            arguments={"command": "touch sample"},
            shell_name="POSIX sh",
            recent_tool_observations=["prior tool result"],
        )

        self.assertEqual(messages[0], {"role": "system", "content": AGENT_ACTION_REVIEW_SYSTEM_PROMPT})
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["proposed_arguments"], {"command": "touch sample"})
        self.assertEqual(payload["recent_tool_observations"], ["prior tool result"])
        self.assertIn("authorization, scope, and execution prerequisites", AGENT_ACTION_REVIEW_SYSTEM_PROMPT)
        revision = build_agent_action_revision_prompt(
            "wrong action",
            user_prompt="Create the requested file.",
            tool_name="exec_shell_full_command",
            arguments={"command": "touch sample"},
        )
        self.assertIn("Create the requested file.", revision)
        self.assertIn('"command":"touch sample"', revision)


if __name__ == "__main__":
    unittest.main()

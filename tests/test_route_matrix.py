from __future__ import annotations

import unittest

from orbit.core.intent_gate import should_confirm_tool_route
from orbit.core.intent_router import (
    INTENT_CLASS_AMBIGUOUS,
    INTENT_CLASS_BINARY_ANALYSIS,
    INTENT_CLASS_CHAT_GENERAL,
    INTENT_CLASS_FILE_READING,
    INTENT_CLASS_KNOWLEDGE_QUESTION,
    INTENT_CLASS_MACHINE_INSPECTION,
    INTENT_CLASS_PDF_ANALYSIS,
    INTENT_CLASS_SHELL_TASK,
    INTENT_CLASS_WEB_LOOKUP,
)
from orbit.core.tool_router import TOOL_CATEGORY_FILESYSTEM, TOOL_CATEGORY_SHELL, TOOL_CATEGORY_WEB, route_tool_categories


class RouteMatrixTests(unittest.TestCase):
    def test_ambiguous_and_discursive_prompts_route_safely(self) -> None:
        cases = (
            ("show me how grep works", INTENT_CLASS_KNOWLEDGE_QUESTION, (), False),
            ("tell me about file systems", INTENT_CLASS_KNOWLEDGE_QUESTION, (), False),
            ("tell me about base64 encoding", INTENT_CLASS_KNOWLEDGE_QUESTION, (), False),
            ("what do you think about web search in LLMs?", INTENT_CLASS_CHAT_GENERAL, (), False),
            ("search engines are useful, what do you think?", INTENT_CLASS_CHAT_GENERAL, (), False),
            ("ask me something about malware analysis, C2 or IoC", INTENT_CLASS_CHAT_GENERAL, (), False),
            ("change the conclusion", INTENT_CLASS_AMBIGUOUS, (TOOL_CATEGORY_FILESYSTEM, "write", TOOL_CATEGORY_SHELL, TOOL_CATEGORY_WEB), True),
            ("use the tool to send an email to test@example.com", INTENT_CLASS_AMBIGUOUS, (TOOL_CATEGORY_FILESYSTEM, "write", TOOL_CATEGORY_SHELL, TOOL_CATEGORY_WEB), True),
        )
        for prompt, intent_class, categories, confirm in cases:
            with self.subTest(prompt=prompt):
                route = route_tool_categories(prompt)
                self.assertEqual(route.intent_class, intent_class)
                self.assertEqual(route.categories, categories)
                self.assertEqual(should_confirm_tool_route(prompt, route), confirm)

    def test_operational_prompts_keep_narrow_tool_routes(self) -> None:
        cases = (
            ('decode this string "Y2lhbw==" from base64', INTENT_CLASS_SHELL_TASK, (TOOL_CATEGORY_SHELL,), False),
            ("how many cpus are there?", INTENT_CLASS_MACHINE_INSPECTION, (TOOL_CATEGORY_SHELL,), False),
            ("what is the size and modified time of README.md?", INTENT_CLASS_FILE_READING, (TOOL_CATEGORY_FILESYSTEM,), False),
            ("search online for information about Dante Alighieri", INTENT_CLASS_WEB_LOOKUP, (TOOL_CATEGORY_WEB,), False),
            ("Summarize docs/Project Overview.pdf.", INTENT_CLASS_PDF_ANALYSIS, (TOOL_CATEGORY_SHELL, TOOL_CATEGORY_FILESYSTEM), False),
            ("analyze workdir/sample.zip", INTENT_CLASS_BINARY_ANALYSIS, (TOOL_CATEGORY_SHELL, TOOL_CATEGORY_FILESYSTEM), False),
        )
        for prompt, intent_class, categories, confirm in cases:
            with self.subTest(prompt=prompt):
                route = route_tool_categories(prompt)
                self.assertEqual(route.intent_class, intent_class)
                self.assertEqual(route.categories, categories)
                self.assertEqual(should_confirm_tool_route(prompt, route), confirm)


if __name__ == "__main__":
    unittest.main()

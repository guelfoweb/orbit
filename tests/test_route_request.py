from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.route_request import (
    RouteStreamFilter,
    ToolRoute,
    decision_tool_names,
    default_route_tool_names,
    parse_route_decision,
    parse_route_decision_from_tool_calls,
    parse_tool_route,
    refine_decision_for_prompt,
    route_like_tool_call,
    route_stream_state,
    route_tool_call_from_content,
    route_tool_names,
    route_tool_call_from_tool_calls,
)


class RouteRequestTests(unittest.TestCase):
    def test_parse_tool_route_accepts_route_key(self) -> None:
        self.assertEqual(parse_tool_route('{"_route":"FILESYSTEM"}'), ToolRoute.FILESYSTEM)

    def test_parse_route_decision_accepts_preferred_tool(self) -> None:
        decision = parse_route_decision('{"_route":"FILESYSTEM","tool":"file_glob_search"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("file_glob_search",))

    def test_shell_full_is_explicit_only(self) -> None:
        decision = parse_route_decision('{"_route":"FILESYSTEM","tool":"exec_shell_full_command"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision_tool_names(decision), ("exec_shell_full_command",))
        self.assertNotIn("exec_shell_full_command", default_route_tool_names(ToolRoute.FILESYSTEM))

    def test_parse_route_decision_ignores_invalid_preferred_tool(self) -> None:
        decision = parse_route_decision('{"_route":"FILESYSTEM","tool":"write_file"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), default_route_tool_names(ToolRoute.FILESYSTEM))

    def test_refine_decision_for_prompt_uses_read_then_edit_for_described_patch_request(self) -> None:
        decision = refine_decision_for_prompt(
            parse_route_decision('{"_route":"FILESYSTEM","tool":"read_file"}'),
            "Apply a unified diff to patch-tool-test.txt that changes beta to BETA and appends delta",
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILE_EDIT)
        self.assertEqual(decision_tool_names(decision), ("read_file", "edit_file"))

    def test_refine_decision_for_prompt_uses_edit_file_for_explicit_line_edit(self) -> None:
        decision = refine_decision_for_prompt(
            parse_route_decision('{"_route":"FILESYSTEM","tool":"read_file"}'),
            "In server-tool-test.txt replace line 2 with BLUE",
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILE_EDIT)
        self.assertEqual(decision_tool_names(decision), ("edit_file",))

    def test_refine_decision_for_prompt_uses_apply_diff_for_actual_diff_text(self) -> None:
        decision = refine_decision_for_prompt(
            parse_route_decision('{"_route":"FILESYSTEM","tool":"read_file"}'),
            "Apply this patch:\ndiff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b",
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILE_EDIT)
        self.assertEqual(decision_tool_names(decision), ("apply_diff",))

    def test_parse_route_decision_accepts_raw_route_tool_call(self) -> None:
        decision = parse_route_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "WEB", "arguments": '{"tool":"search_web","query":"x"}'},
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.WEB)
        self.assertEqual(decision_tool_names(decision), ("search_web",))

    def test_parse_route_decision_accepts_generic_call_with_route_arguments(self) -> None:
        decision = parse_route_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": "call",
                        "arguments": '{"_route":"FILESYSTEM","tool":"read_file","path":"README.md"}',
                    },
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("read_file",))

    def test_route_tool_call_from_route_tool_call_uses_selected_tool_arguments(self) -> None:
        tool_call = route_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "WEB", "arguments": '{"tool":"search_web","query":"x"}'},
                }
            ],
            ("search_web",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "search_web")
        self.assertEqual(tool_call["function"]["arguments"], '{"query": "x"}')

    def test_route_tool_call_from_generic_call_uses_route_arguments(self) -> None:
        tool_call = route_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": "call",
                        "arguments": '{"_route":"FILESYSTEM","tool":"read_file","path":"README.md"}',
                    },
                }
            ],
            ("read_file",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(tool_call["function"]["arguments"], '{"path": "README.md"}')

    def test_route_tool_call_from_generic_call_allows_list_files_without_arguments(self) -> None:
        tool_call = route_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": "call",
                        "arguments": '{"_route":"FILESYSTEM","tool":"list_files"}',
                    },
                }
            ],
            ("list_files",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "list_files")
        self.assertEqual(tool_call["function"]["arguments"], "{}")

    def test_route_tool_call_from_route_tool_call_rejects_unallowed_tool(self) -> None:
        self.assertIsNone(
            route_tool_call_from_tool_calls(
                [
                    {
                        "id": "raw-tool-call-1",
                        "type": "function",
                        "function": {"name": "WEB", "arguments": '{"tool":"search_web","query":"x"}'},
                    }
                ],
                ("fetch_url",),
            )
        )

    def test_route_tool_call_from_content_uses_embedded_arguments(self) -> None:
        tool_call = route_tool_call_from_content(
            '{"_route":"FILESYSTEM","tool":"read_file","path":"README.md"}',
            ("read_file",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(tool_call["function"]["arguments"], '{"path": "README.md"}')

    def test_route_tool_call_from_content_uses_nested_args(self) -> None:
        tool_call = route_tool_call_from_content(
            '{"_route":"FILESYSTEM","tool":"read_file","args":{"path":"README.md"}}',
            ("read_file",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertEqual(tool_call["function"]["arguments"], '{"path": "README.md"}')

    def test_route_tool_call_from_content_requires_arguments(self) -> None:
        self.assertIsNone(route_tool_call_from_content('{"_route":"FILESYSTEM","tool":"read_file"}', ("read_file",)))

    def test_parse_tool_route_accepts_legacy_route_key(self) -> None:
        self.assertEqual(parse_tool_route('{"route":"FILESYSTEM"}'), ToolRoute.FILESYSTEM)

    def test_parse_tool_route_accepts_fenced_json(self) -> None:
        self.assertEqual(parse_tool_route('```json\n{"_route":"FILE_EDIT"}\n```'), ToolRoute.FILE_EDIT)

    def test_parse_tool_route_accepts_legacy_tool_key(self) -> None:
        self.assertEqual(parse_tool_route('{"tool":"WEB"}'), ToolRoute.WEB)

    def test_parse_tool_route_accepts_malformed_route(self) -> None:
        self.assertEqual(parse_tool_route('{"_route":"WEB": "search for information about Dante Alighieri"}'), ToolRoute.WEB)

    def test_parse_tool_route_rejects_chat_text(self) -> None:
        self.assertIsNone(parse_tool_route("Here is the answer."))

    def test_parse_tool_route_accepts_route_line_inside_longer_output(self) -> None:
        content = "```python\nprint('x')\n```\n\n{\"_route\":\"FILE_EDIT\"}\n{\"path\":\"x.py\"}"

        self.assertEqual(parse_tool_route(content), ToolRoute.FILE_EDIT)

    def test_parse_tool_route_accepts_raw_tool_call_with_tool_name(self) -> None:
        self.assertEqual(parse_tool_route('<|tool_call>call:search_web{"query":"x"}<tool_call|>'), ToolRoute.WEB)

    def test_parse_tool_route_accepts_raw_filesystem_tool_names(self) -> None:
        self.assertEqual(parse_tool_route('<|tool_call>call:file_glob_search{"path":"."}<tool_call|>'), ToolRoute.FILESYSTEM)
        self.assertEqual(parse_tool_route('<|tool_call>call:exec_shell_command{"command":"ls -F"}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_route_accepts_raw_tool_call_with_route_name(self) -> None:
        self.assertEqual(parse_tool_route('<|tool_call>call:1.WEB{"query":"x"}<tool_call|>'), ToolRoute.WEB)

    def test_route_like_tool_call_converts_malformed_web_route_to_search(self) -> None:
        tool_call = route_like_tool_call(
            '<|tool_call>call:_route":"WEB"{"query":"Dante Alighieri"}<tool_call|>',
            ("fetch_url", "search_web"),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "search_web")
        self.assertEqual(tool_call["function"]["arguments"], '{"query": "Dante Alighieri"}')

    def test_route_like_tool_call_respects_allowed_tools(self) -> None:
        tool_call = route_like_tool_call(
            '<|tool_call>call:_route":"WEB"{"query":"Dante Alighieri"}<tool_call|>',
            ("fetch_url",),
        )

        self.assertIsNone(tool_call)

    def test_parse_tool_route_rejects_invalid_json(self) -> None:
        self.assertEqual(parse_tool_route('{"route":"CHAT"}'), ToolRoute.CHAT)
        self.assertIsNone(parse_tool_route('{"route":"UNKNOWN"}'))

    def test_route_tool_names_are_bounded(self) -> None:
        self.assertEqual(
            route_tool_names(ToolRoute.FILESYSTEM),
            (
                "list_files",
                "read_file",
                "file_glob_search",
                "grep_search",
                "exec_shell_command",
                "exec_shell_full_command",
                "get_datetime",
            ),
        )
        self.assertIn("write_file", route_tool_names(ToolRoute.FILE_EDIT))
        self.assertIn("edit_file", route_tool_names(ToolRoute.FILE_EDIT))
        self.assertIn("apply_diff", route_tool_names(ToolRoute.FILE_EDIT))
        self.assertNotIn("write_file", route_tool_names(ToolRoute.FILESYSTEM))
        self.assertIn("list_files", route_tool_names(ToolRoute.FILESYSTEM))
        self.assertNotIn("stat_path", route_tool_names(ToolRoute.FILESYSTEM))
        self.assertNotIn("append_file", route_tool_names(ToolRoute.FILE_EDIT))
        self.assertNotIn("replace_in_file", route_tool_names(ToolRoute.FILE_EDIT))
        self.assertEqual(route_tool_names(ToolRoute.WEB), ("fetch_url", "search_web"))

    def test_parse_route_decision_accepts_datetime_tool(self) -> None:
        decision = parse_route_decision('{"_route":"FILESYSTEM","tool":"get_datetime"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision_tool_names(decision), ("get_datetime",))

    def test_route_stream_state_detects_complete_route(self) -> None:
        self.assertEqual(route_stream_state('{"_route":"WEB"}'), "route")
        self.assertEqual(route_stream_state('{"_route":"WEB": "search"}'), "route")
        self.assertEqual(route_stream_state("{"), "pending")
        self.assertEqual(route_stream_state('{"_ro'), "pending")
        self.assertEqual(route_stream_state('{"_route":"WEB"'), "pending")
        self.assertEqual(route_stream_state('{"route":"WEB"}'), "not_route")
        self.assertEqual(route_stream_state("Hello"), "not_route")
        self.assertEqual(route_stream_state('{"tool":"hammer"}'), "not_route")

    def test_route_stream_filter_suppresses_route_output(self) -> None:
        emitted: list[str] = []
        stream_filter = RouteStreamFilter(emitted.append)

        stream_filter.write("{")
        stream_filter.write('"_route"')
        stream_filter.write(':"WEB"}')
        stream_filter.finish()

        self.assertEqual(emitted, [])
        self.assertTrue(stream_filter.route_detected)
        self.assertEqual(stream_filter.content, '{"_route":"WEB"}')

    def test_route_stream_filter_suppresses_route_with_extra_fields(self) -> None:
        emitted: list[str] = []
        stream_filter = RouteStreamFilter(emitted.append)

        stream_filter.write('{"_route":"FILE_EDIT", "path": "server-tool-test.txt", ')
        stream_filter.write('"action": "replace line 2 with BLUE"}')
        stream_filter.finish()

        self.assertEqual(emitted, [])
        self.assertTrue(stream_filter.route_detected)
        self.assertIn("server-tool-test.txt", stream_filter.content)

    def test_route_stream_filter_releases_normal_text(self) -> None:
        emitted: list[str] = []
        stream_filter = RouteStreamFilter(emitted.append)

        stream_filter.write("Hello")
        stream_filter.write(" world")
        stream_filter.finish()

        self.assertEqual(emitted, ["Hello", " world"])

    def test_route_stream_filter_releases_generic_tool_json(self) -> None:
        emitted: list[str] = []
        stream_filter = RouteStreamFilter(emitted.append)

        stream_filter.write('{"tool":"hammer"}')
        stream_filter.finish()

        self.assertEqual(emitted, ['{"tool":"hammer"}'])


if __name__ == "__main__":
    unittest.main()

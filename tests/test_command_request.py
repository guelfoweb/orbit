from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime.command_request import (
    CommandStreamFilter,
    ToolRoute,
    decision_tool_names,
    parse_command_decision,
    parse_command_decision_from_tool_calls,
    parse_tool_command,
    command_stream_state,
    command_tool_call_from_content,
    command_tool_call_from_tool_calls,
    tool_names_for_decision,
)


class RouteRequestTests(unittest.TestCase):
    def test_parse_command_decision_accepts_shell_command_json(self) -> None:
        decision = parse_command_decision('{"command":"ls -F"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("exec_shell_full_command",))

    def test_parse_command_decision_rejects_chat_text(self) -> None:
        self.assertIsNone(parse_command_decision("Here is the answer."))

    def test_parse_command_decision_accepts_chat_route_json(self) -> None:
        decision = parse_command_decision('{"route":"CHAT"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.CHAT)
        self.assertEqual(decision_tool_names(decision), ())

    def test_parse_command_decision_does_not_let_chat_override_tool_json(self) -> None:
        decision = parse_command_decision('{"route":"CHAT","command":"ls -F"}')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("exec_shell_full_command",))

    def test_parse_tool_command_accepts_raw_tool_call_with_tool_name(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:exec_shell_full_command{"command":"ls -F"}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_command_accepts_raw_orbit_web_search_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:orbit-web-search{"query":"Dante Alighieri"}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_command_accepts_raw_orbit_web_search_key_value_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:orbit-web-search{query="Dante Alighieri"}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_command_accepts_raw_fetch_url_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:fetch_url{"url":"https://example.com"}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_command_accepts_raw_list_directory_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:list_directory{"path":".","recursive":true}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_tool_command_accepts_raw_system_info_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call:system_info{"include_cpu":true}<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_command_decision_keeps_raw_list_directory_tool_scope(self) -> None:
        decision = parse_command_decision('<|tool_call>call:list_directory{"path":".","recursive":true}<tool_call|>')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision_tool_names(decision), ("list_directory",))

    def test_parse_command_decision_keeps_raw_system_info_tool_scope(self) -> None:
        decision = parse_command_decision('<|tool_call>call:system_info{}<tool_call|>')

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision_tool_names(decision), ("system_info",))

    def test_parse_tool_command_accepts_parenthesized_shell_tool_call(self) -> None:
        self.assertEqual(parse_tool_command('<|tool_call>call(shell, "orbit-web-search \\"Mario Nobile\\"")<tool_call|>'), ToolRoute.FILESYSTEM)

    def test_parse_command_decision_from_tool_calls_accepts_command_arguments(self) -> None:
        decision = parse_command_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "call", "arguments": '{"command":"cat README.md"}'},
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("exec_shell_full_command",))

    def test_parse_command_decision_from_tool_calls_accepts_fetch_url_arguments(self) -> None:
        decision = parse_command_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "fetch_url", "arguments": '{"url":"https://example.com"}'},
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("fetch_url",))

    def test_parse_command_decision_from_tool_calls_accepts_list_directory_arguments(self) -> None:
        decision = parse_command_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "list_directory", "arguments": '{"path":".","recursive":true}'},
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("list_directory",))

    def test_parse_command_decision_from_tool_calls_accepts_system_info_arguments(self) -> None:
        decision = parse_command_decision_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "system_info", "arguments": '{"include_cpu":true}'},
                }
            ]
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, ToolRoute.FILESYSTEM)
        self.assertEqual(decision_tool_names(decision), ("system_info",))

    def test_command_tool_call_from_content_uses_shell_command_json(self) -> None:
        tool_call = command_tool_call_from_content('{"command":"ls -F"}', ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], '{"command": "ls -F"}')

    def test_command_tool_call_from_content_uses_fetch_url_json(self) -> None:
        tool_call = command_tool_call_from_content('{"url":"https://example.com"}', ("exec_shell_full_command", "fetch_url"))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "fetch_url")
        self.assertEqual(tool_call["function"]["arguments"], '{"url": "https://example.com"}')

    def test_command_tool_call_from_content_uses_list_directory_json(self) -> None:
        tool_call = command_tool_call_from_content('{"path": ".", "recursive": true}', ("exec_shell_full_command", "list_directory"))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "list_directory")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"path": ".", "recursive": True})

    def test_command_tool_call_from_content_uses_system_info_json(self) -> None:
        tool_call = command_tool_call_from_content('{"include_cpu": true, "include_memory": true}', ("exec_shell_full_command", "system_info"))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "system_info")
        self.assertEqual(json.loads(tool_call["function"]["arguments"]), {"include_cpu": True, "include_memory": True})

    def test_command_tool_call_from_content_ignores_chat_route_json(self) -> None:
        tool_call = command_tool_call_from_content('{"route":"CHAT"}', ("exec_shell_full_command", "fetch_url", "list_directory", "system_info"))

        self.assertIsNone(tool_call)

    def test_command_tool_call_from_content_converts_raw_orbit_web_search(self) -> None:
        tool_call = command_tool_call_from_content(
            '<|tool_call>call:orbit-web-search{"query":"Dante Alighieri"}<tool_call|>',
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], "{\"command\": \"orbit-web-search 'Dante Alighieri'\"}")

    def test_command_tool_call_from_content_converts_raw_orbit_web_search_key_value(self) -> None:
        tool_call = command_tool_call_from_content(
            '<|tool_call>call:orbit-web-search{query="Dante Alighieri"}<tool_call|>',
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], "{\"command\": \"orbit-web-search 'Dante Alighieri'\"}")

    def test_command_tool_call_from_content_accepts_fenced_multiline_command(self) -> None:
        content = '```json\n{"command": "printf \\"one\\ntwo\\" > note.txt"}\n```'

        tool_call = command_tool_call_from_content(content, ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(args["command"], 'printf "one\ntwo" > note.txt')

    def test_command_tool_call_from_content_accepts_literal_newline_command_json(self) -> None:
        content = """{"command": "cat << 'EOF' > note.txt
one
two
EOF"}"""

        tool_call = command_tool_call_from_content(content, ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(args["command"], "cat << 'EOF' > note.txt\none\ntwo\nEOF")

    def test_command_tool_call_from_content_preserves_heredoc_command(self) -> None:
        content = """{"command": "cat > script.sh << 'EOF'
#!/usr/bin/env bash
echo ok
EOF"}"""

        tool_call = command_tool_call_from_content(content, ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        args = json.loads(tool_call["function"]["arguments"])
        self.assertIn("cat > script.sh << 'EOF'\n", args["command"])
        self.assertTrue(args["command"].endswith("\nEOF"))

    def test_command_tool_call_from_content_accepts_loose_raw_shell_tool_call(self) -> None:
        content = "<|tool_call>call shell cat backup.sh\n<|tool_call>call shell sed -i 's/a/b/' backup.sh"

        tool_call = command_tool_call_from_content(content, ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(args["command"], "cat backup.sh")

    def test_command_tool_call_from_content_accepts_parenthesized_shell_tool_call(self) -> None:
        content = '<|tool_call>call(shell, "orbit-web-search \\"Mario Nobile\\"")<tool_call|>'

        tool_call = command_tool_call_from_content(content, ("exec_shell_full_command",))

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(args["command"], 'orbit-web-search "Mario Nobile"')

    def test_command_tool_call_from_content_respects_allowed_tools(self) -> None:
        self.assertIsNone(command_tool_call_from_content('{"command":"ls -F"}', ("read_file",)))

    def test_command_tool_call_from_tool_calls_uses_command_arguments(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "call", "arguments": '{"command":"cat README.md"}'},
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], '{"command": "cat README.md"}')

    def test_command_tool_call_from_tool_calls_keeps_allowed_server_tool_call(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "exec_shell_full_command", "arguments": '{"command":"pwd"}'},
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], '{"command":"pwd"}')

    def test_command_tool_call_from_tool_calls_keeps_fetch_url_tool_call(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "fetch_url", "arguments": '{"url":"https://example.com"}'},
                }
            ],
            ("exec_shell_full_command", "fetch_url"),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "fetch_url")
        self.assertEqual(tool_call["function"]["arguments"], '{"url":"https://example.com"}')

    def test_command_tool_call_from_tool_calls_keeps_list_directory_tool_call(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "list_directory", "arguments": '{"path":".","recursive":true}'},
                }
            ],
            ("exec_shell_full_command", "list_directory"),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "list_directory")
        self.assertEqual(tool_call["function"]["arguments"], '{"path":".","recursive":true}')

    def test_command_tool_call_from_tool_calls_keeps_system_info_tool_call(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "system_info", "arguments": '{"include_cpu":true}'},
                }
            ],
            ("exec_shell_full_command", "system_info"),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "system_info")
        self.assertEqual(tool_call["function"]["arguments"], '{"include_cpu":true}')

    def test_command_tool_call_from_tool_calls_accepts_shell_alias(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"curl https://example.com"}'},
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], '{"command": "curl https://example.com"}')

    def test_command_tool_call_from_tool_calls_normalizes_invalid_multiline_arguments(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": "exec_shell_full_command",
                        "arguments": """{"command":"cat << 'EOF' > note.txt
one
two
EOF"}""",
                    },
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        args = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(args["command"], "cat << 'EOF' > note.txt\none\ntwo\nEOF")

    def test_command_tool_call_from_tool_calls_accepts_orbit_web_search_alias(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "orbit-web-search", "arguments": '{"query":"Dante Alighieri"}'},
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call["function"]["name"], "exec_shell_full_command")
        self.assertEqual(tool_call["function"]["arguments"], "{\"command\": \"orbit-web-search 'Dante Alighieri'\"}")

    def test_command_tool_call_from_tool_calls_rejects_alias_without_command(self) -> None:
        tool_call = command_tool_call_from_tool_calls(
            [
                {
                    "id": "raw-tool-call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"path":"README.md"}'},
                }
            ],
            ("exec_shell_full_command",),
        )

        self.assertIsNone(tool_call)

    def test_tool_names_for_decision_are_bounded(self) -> None:
        self.assertEqual(tool_names_for_decision(ToolRoute.FILESYSTEM), ("exec_shell_full_command", "fetch_url", "list_directory", "system_info"))
        self.assertEqual(tool_names_for_decision(ToolRoute.FILE_EDIT), ())
        self.assertEqual(tool_names_for_decision(ToolRoute.WEB), ())

    def test_command_stream_state_detects_complete_command_json(self) -> None:
        self.assertEqual(command_stream_state('{"command":"ls -F"}'), "route")
        self.assertEqual(command_stream_state("{"), "pending")
        self.assertEqual(command_stream_state('{"comm'), "pending")
        self.assertEqual(command_stream_state('{"command":"ls -F"'), "pending")
        self.assertEqual(command_stream_state("Hello"), "not_command")
        self.assertEqual(command_stream_state('{"tool":"hammer"}'), "not_command")

    def test_command_stream_state_detects_complete_list_directory_json(self) -> None:
        self.assertEqual(command_stream_state('{"path":".","recursive":true}'), "route")
        self.assertEqual(command_stream_state('{"path"'), "pending")

    def test_command_stream_state_detects_complete_system_info_json(self) -> None:
        self.assertEqual(command_stream_state('{"include_cpu":true}'), "route")
        self.assertEqual(command_stream_state('{"include_'), "pending")

    def test_command_stream_state_detects_complete_chat_route_json(self) -> None:
        self.assertEqual(command_stream_state('{"route":"CHAT"}'), "route")
        self.assertEqual(command_stream_state('{"route"'), "pending")

    def test_command_stream_filter_suppresses_command_json(self) -> None:
        emitted: list[str] = []
        stream_filter = CommandStreamFilter(emitted.append)

        stream_filter.write("{")
        stream_filter.write('"command"')
        stream_filter.write(':"ls -F"}')
        stream_filter.finish()

        self.assertEqual(emitted, [])
        self.assertTrue(stream_filter.command_detected)
        self.assertEqual(stream_filter.content, '{"command":"ls -F"}')

    def test_command_stream_filter_suppresses_chat_route_json(self) -> None:
        emitted: list[str] = []
        stream_filter = CommandStreamFilter(emitted.append)

        stream_filter.write("{")
        stream_filter.write('"route"')
        stream_filter.write(':"CHAT"}')
        stream_filter.finish()

        self.assertEqual(emitted, [])
        self.assertTrue(stream_filter.command_detected)
        self.assertEqual(stream_filter.content, '{"route":"CHAT"}')

    def test_command_stream_filter_releases_normal_text(self) -> None:
        emitted: list[str] = []
        stream_filter = CommandStreamFilter(emitted.append)

        stream_filter.write("Hello")
        stream_filter.write(" world")
        stream_filter.finish()

        self.assertEqual(emitted, ["Hello", " world"])
        self.assertFalse(stream_filter.command_detected)


if __name__ == "__main__":
    unittest.main()

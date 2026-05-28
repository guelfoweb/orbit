from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.core.agent import AgentLoop, _compact_message_for_model
from orbit.core.context_budget import profile_for_model
from orbit.core.client import OllamaError
from orbit.core.client import ModelMetadata
from orbit.core.events import ToolResultCompactEvent
from orbit.core.intent_router import (
    INTENT_BINARY_OR_PDF_ANALYSIS,
    INTENT_BOUNDED_COMMAND,
    INTENT_CLASS_BINARY_OR_PDF_ANALYSIS,
    INTENT_CHITCHAT,
    INTENT_CLASS_CHAT_GENERAL,
    INTENT_CLASS_CODEBASE_INSPECTION,
    INTENT_CLASS_FILE_EDITING,
    INTENT_CLASS_FILE_READING,
    INTENT_CLASS_KNOWLEDGE_QUESTION,
    INTENT_CLASS_MACHINE_INSPECTION,
    INTENT_CLASS_SHELL_TASK,
    INTENT_CLASS_URL_INSPECTION,
    INTENT_CLASS_WEB_LOOKUP,
    INTENT_CLASS_WORKSPACE_DISCOVERY,
    INTENT_CODEBASE_INSPECTION,
    INTENT_CURRENT_FACTUAL_LOOKUP,
    INTENT_FILE_EDIT,
    INTENT_GENERAL_KNOWLEDGE,
    INTENT_TEXT_DOCUMENT_ANALYSIS,
    route_intent,
)
from orbit.core.loop_guard import ToolCallRecord, signatures_match
from orbit.core.turn_policy import TurnPolicyState, _repeated_tool_retry_prompt, classify_model_reply
from orbit.skills import Skill


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.model = "fake-model"

    def chat(self, *, messages, tools, options, think=None, stream=False):
        self.calls.append({"messages": messages, "tools": tools, "options": options, "think": think, "stream": stream})
        response = self.responses.pop(0)
        if stream:
            return iter(response)
        return response

    def inspect_model(self):
        return ModelMetadata(
            active_model="fake-model",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )


class VisionFakeClient(FakeClient):
    def inspect_model(self):
        return ModelMetadata(
            active_model="gemma4:e2b-fast-t6-c8k",
            context_window=8192,
            capabilities=("completion", "tools", "vision"),
            tools_supported=True,
        )


class FakeRegistry:
    def __init__(self):
        self.called = []
        self.routed = []
        self.workdir = Path(tempfile.mkdtemp(prefix="orbit-test-"))

    def definitions(self):
        return [
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}}}},
            {"type": "function", "function": {"name": "stat_path", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}}}},
            {"type": "function", "function": {"name": "make_directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "delete_path", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "replace_in_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "append_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
        ]

    def definitions_for_categories(self, categories):
        self.routed.append(tuple(categories))
        if not categories:
            return []
        allowed = []
        for item in self.definitions():
            name = item["function"]["name"]
            if "web" in categories and name in {"search_web", "fetch_url"}:
                allowed.append(item)
            if "filesystem" in categories and name in {"read_file", "list_files", "stat_path"}:
                allowed.append(item)
            if "write" in categories and name in {"replace_in_file", "write_file", "append_file", "make_directory", "delete_path"}:
                allowed.append(item)
            if "shell" in categories and name == "bash":
                allowed.append(item)
        return allowed or self.definitions()

    def call(self, name, arguments):
        self.called.append((name, arguments))
        return {"ok": True, "content": "hello"}

    @staticmethod
    def encode_tool_result(result):
        return json.dumps(result)


def _write_png(path: Path, r: int, g: int, b: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = bytes([0, r, g, b])
    payload = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(payload)


class AgentLoopTests(unittest.TestCase):
    def test_runs_tool_then_finalizes(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 10,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_700_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    }
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_500_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read it")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("read_file", {"path": "README.md"})])
        self.assertIsNotNone(result.status.decode_tps)
        self.assertEqual(result.status.model_elapsed_sec, 3.5)
        self.assertIsNotNone(result.status.wall_elapsed_sec)
        self.assertIsNotNone(result.status.tool_elapsed_sec)
        self.assertGreater(result.status.tool_elapsed_sec, 0.0)
        self.assertEqual(registry.routed[0], ("filesystem",))

    def test_chat_route_exposes_english_intent_class(self) -> None:
        route = route_intent("ciao")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_workspace_discovery_route_exposes_english_intent_class(self) -> None:
        route = route_intent("quali cartelle ci sono in questa directory?")
        self.assertEqual(route.intent_class, INTENT_CLASS_WORKSPACE_DISCOVERY)

    def test_file_reading_route_exposes_english_intent_class(self) -> None:
        route = route_intent("mostrami il contenuto di REPORT.md")
        self.assertEqual(route.intent_class, INTENT_CLASS_FILE_READING)

    def test_machine_inspection_route_exposes_english_intent_class(self) -> None:
        route = route_intent("qual è la configurazione hw di questo PC?")
        self.assertEqual(route.intent_class, INTENT_CLASS_MACHINE_INSPECTION)

    def test_web_lookup_route_exposes_english_intent_class(self) -> None:
        route = route_intent("cerca online informazioni su Mario Nobile")
        self.assertEqual(route.intent, INTENT_CURRENT_FACTUAL_LOOKUP)
        self.assertEqual(route.intent_class, INTENT_CLASS_WEB_LOOKUP)

    def test_url_inspection_route_exposes_english_intent_class(self) -> None:
        route = route_intent("controlla cosa riporta il sito: https://guelfoweb.com/")
        self.assertEqual(route.intent, INTENT_CURRENT_FACTUAL_LOOKUP)
        self.assertEqual(route.intent_class, INTENT_CLASS_URL_INSPECTION)

    def test_knowledge_question_route_exposes_english_intent_class(self) -> None:
        route = route_intent("perchè il cielo è blu?")
        self.assertEqual(route.intent, INTENT_GENERAL_KNOWLEDGE)
        self.assertEqual(route.intent_class, INTENT_CLASS_KNOWLEDGE_QUESTION)

    def test_exact_answer_route_uses_lightweight_knowledge_path(self) -> None:
        route = route_intent("Say exactly: OK")
        self.assertEqual(route.intent, INTENT_GENERAL_KNOWLEDGE)
        self.assertEqual(route.intent_class, INTENT_CLASS_KNOWLEDGE_QUESTION)

    def test_exact_answer_route_in_italian_uses_lightweight_knowledge_path(self) -> None:
        route = route_intent("Rispondi esattamente: OK")
        self.assertEqual(route.intent, INTENT_GENERAL_KNOWLEDGE)
        self.assertEqual(route.intent_class, INTENT_CLASS_KNOWLEDGE_QUESTION)

    def test_assistant_persona_route_exposes_chat_intent_class(self) -> None:
        route = route_intent("How old are you?")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_assistant_persona_route_in_italian_exposes_chat_intent_class(self) -> None:
        route = route_intent("quanti anni hai?")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_disk_usage_route_exposes_machine_inspection_intent_class(self) -> None:
        route = route_intent("show disk usage of /")
        self.assertEqual(route.intent, INTENT_BOUNDED_COMMAND)
        self.assertEqual(route.intent_class, INTENT_CLASS_MACHINE_INSPECTION)

    def test_workspace_storage_route_exposes_machine_inspection_intent_class(self) -> None:
        route = route_intent("display disk usage and available storage space for the workspace")
        self.assertEqual(route.intent, INTENT_BOUNDED_COMMAND)
        self.assertEqual(route.intent_class, INTENT_CLASS_MACHINE_INSPECTION)

    def test_creative_story_prompt_is_not_misrouted_as_file_edit(self) -> None:
        route = route_intent("make up a story in 10 lines")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_italian_creative_prompt_is_not_misrouted_as_file_edit(self) -> None:
        route = route_intent("scrivi una poesia in 4 righe")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_inline_code_prompt_is_not_misrouted_as_file_edit(self) -> None:
        route = route_intent("write a python code that adds a + b")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_inline_function_prompt_is_not_misrouted_as_file_edit(self) -> None:
        route = route_intent("write a fibonacci python function")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_inline_code_prompt_with_file_change_negation_is_not_file_edit(self) -> None:
        route = route_intent("Write a Python function that validates an email address, but do not create or modify any files. Show only the code.")
        self.assertEqual(route.intent, INTENT_CHITCHAT)
        self.assertEqual(route.intent_class, INTENT_CLASS_CHAT_GENERAL)

    def test_explicit_named_file_creation_stays_file_edit(self) -> None:
        route = route_intent("create a file named notes.txt containing three bullet points")
        self.assertEqual(route.intent, INTENT_FILE_EDIT)
        self.assertEqual(route.intent_class, INTENT_CLASS_FILE_EDITING)

    def test_explicit_zip_path_routes_to_binary_analysis(self) -> None:
        route = route_intent("analyze workdir/sample.zip")
        self.assertEqual(route.intent, INTENT_BINARY_OR_PDF_ANALYSIS)
        self.assertEqual(route.intent_class, INTENT_CLASS_BINARY_OR_PDF_ANALYSIS)

    def test_directory_zip_reference_routes_to_binary_analysis(self) -> None:
        route = route_intent("analyze the zip in this directory")
        self.assertEqual(route.intent, INTENT_BINARY_OR_PDF_ANALYSIS)
        self.assertEqual(route.intent_class, INTENT_CLASS_BINARY_OR_PDF_ANALYSIS)

    def test_explicit_code_file_review_routes_to_codebase_inspection_in_english(self) -> None:
        route = route_intent("analyze the python code in this file: agent.py and tell me if you find bugs.")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)
        self.assertEqual(route.intent_class, INTENT_CLASS_CODEBASE_INSPECTION)

    def test_explicit_code_file_review_routes_to_codebase_inspection_in_italian(self) -> None:
        route = route_intent("analizza il codice python in questo file: agent.py e dimmi se trovi bug.")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)
        self.assertEqual(route.intent_class, INTENT_CLASS_CODEBASE_INSPECTION)

    def test_explicit_cpp_vulnerability_review_routes_to_codebase_inspection(self) -> None:
        route = route_intent("review src/parser.cpp for vulnerabilities and security issues.")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)
        self.assertEqual(route.intent_class, INTENT_CLASS_CODEBASE_INSPECTION)

    def test_explicit_powershell_security_review_routes_to_codebase_inspection_in_italian(self) -> None:
        route = route_intent("analizza deploy.ps1 e dimmi se trovi vulnerabilità o problemi di sicurezza.")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)
        self.assertEqual(route.intent_class, INTENT_CLASS_CODEBASE_INSPECTION)

    def test_plain_text_analysis_stays_text_document_analysis_in_italian(self) -> None:
        route = route_intent("analizza questo testo promessi_sposi.txt e riassumilo in 5 righe.")
        self.assertEqual(route.intent, INTENT_TEXT_DOCUMENT_ANALYSIS)

    def test_markdown_summary_stays_text_document_analysis_in_english(self) -> None:
        route = route_intent("analyze this document README.md and summarize the key points.")
        self.assertEqual(route.intent, INTENT_TEXT_DOCUMENT_ANALYSIS)

    def test_local_path_metadata_request_does_not_route_to_web_lookup(self) -> None:
        route = route_intent("what is the size and modified time of README.md?")
        self.assertEqual(route.intent, INTENT_TEXT_DOCUMENT_ANALYSIS)
        self.assertEqual(route.intent_class, INTENT_CLASS_FILE_READING)

    def test_workspace_newest_file_request_routes_to_local_filesystem(self) -> None:
        route = route_intent("Tell me how many files exist in the workspace and what the newest file is.")
        self.assertEqual(route.intent, INTENT_TEXT_DOCUMENT_ANALYSIS)
        self.assertEqual(route.intent_class, INTENT_CLASS_FILE_READING)

    def test_workspace_security_search_routes_to_codebase_not_web(self) -> None:
        route = route_intent("Search the workspace for anything that looks like a security issue.")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)
        self.assertEqual(route.intent_class, INTENT_CLASS_CODEBASE_INSPECTION)

    def test_base64_transform_routes_to_bounded_command(self) -> None:
        route = route_intent('decode this string "Y2lhbw==" from base64')
        self.assertEqual(route.intent, INTENT_BOUNDED_COMMAND)
        self.assertEqual(route.intent_class, INTENT_CLASS_SHELL_TASK)

    def test_gemma_model_first_base64_decode_finalizes_locally(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn('decode this string "Y2lhbw==" from base64')
        self.assertIn("ciao", result.content)
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_gemma_model_first_classifies_workspace_files_locally(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "agent.py", "type": "file"},
                        {"path": "config.json", "type": "file"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("inspect the workspace and tell me which files appear to contain source code, configuration, or documentation.")
        self.assertIn("Source code: agent.py", result.content)
        self.assertIn("Configuration: config.json", result.content)
        self.assertIn("Documentation/text: README.md", result.content)
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 80})])
        self.assertEqual(client.calls, [])

    def test_gemma_model_first_answers_missing_json_config_from_listing(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "agent.py", "type": "file"},
                        {"path": "summary.txt", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Without guessing, inspect the workspace and tell me whether there is any JSON configuration file. If none exists, say exactly that and do not create one.")
        self.assertEqual(result.content, "No JSON configuration file exists in the current workspace.")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 80})])
        self.assertEqual(client.calls, [])

    def test_gemma_model_first_preserves_local_and_web_evidence(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "search_web", "arguments": {"query": "recent AI safety news"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 20,
                    "eval_duration": 100_000_000,
                    "total_duration": 240_000_000,
                    "message": {"content": "- Local evidence: summary.txt discusses alignment.\n- Web evidence: search results discuss AI safety news.\n- Combined: both sources are separated."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "AI safety is a critical area of research.",
                    "has_more": False,
                    "truncated": False,
                }
            if name == "search_web":
                return {
                    "ok": True,
                    "query": arguments["query"],
                    "results": [{"title": "AI safety news", "url": "https://example.com"}],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Read summary.txt, then search online for recent AI safety news, and explain in three bullets what is local evidence versus web evidence.")
        self.assertIn("Local evidence", result.content)
        self.assertIn("AI safety", result.content)
        self.assertIn("AI safety news", result.content)
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "summary.txt"}),
                ("search_web", {"query": "recent AI safety news"}),
            ],
        )
        self.assertTrue(
            any(
                message.get("role") == "tool"
                and message.get("tool_name") == "read_file"
                for message in client.calls[0]["messages"]
            )
        )
        self.assertEqual(len(client.calls), 1)

    def test_gemma_model_first_workspace_security_scan_reads_code_before_answering(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "agent.py", "type": "file"},
                        {"path": "summary.txt", "type": "file"},
                    ],
                }
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "def run():\n    pass\n",
                    "total_lines": 400,
                    "next_start_line": 221,
                    "truncated": True,
                    "has_more": True,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Search the workspace for anything that looks like a security issue, then explain whether the issue is in code, configuration, or documentation. If you cannot prove it, say so explicitly.")
        self.assertIn("cannot prove", result.content.lower())
        self.assertIn("code", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("list_files", {"path": ".", "recursive": False, "max_entries": 80}),
                ("read_file", {"path": "agent.py", "start_line": 1, "max_lines": 220, "max_chars": 9000}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_metadata_request_uses_stat_path_without_list_files_detour(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "stat_path":
                return {
                    "ok": True,
                    "path": ".",
                    "type": "dir",
                    "recursive": True,
                    "count": 2,
                    "total_entries": 2,
                    "file_count": 2,
                    "dir_count": 0,
                    "entries": [
                        {"path": "new.txt", "type": "file", "size_bytes": 3, "modified_at": "2026-05-27T10:00:00+00:00"},
                        {"path": "old.txt", "type": "file", "size_bytes": 3, "modified_at": "2026-05-26T10:00:00+00:00"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("Tell me how many files exist in the workspace and what the newest file is.")
        self.assertIn("2 files", result.content)
        self.assertIn("new.txt", result.content)
        self.assertEqual(
            registry.called,
            [
                ("stat_path", {"path": ".", "recursive": True}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_gemma_model_first_file_metadata_finalizes_locally(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "stat_path":
                return {
                    "ok": True,
                    "path": "README.md",
                    "type": "file",
                    "size_bytes": 1234,
                    "modified_at": "2026-05-28T10:00:00+00:00",
                    "mode": "0o100644",
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("what is the size and modified time of README.md?")
        self.assertIn("README.md", result.content)
        self.assertIn("1234 bytes", result.content)
        self.assertIn("2026-05-28T10:00:00+00:00", result.content)
        self.assertEqual(registry.called, [("stat_path", {"path": "README.md", "recursive": False})])
        self.assertEqual(client.calls, [])

    def test_gemma_model_first_workspace_newest_file_finalizes_locally(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "stat_path":
                return {
                    "ok": True,
                    "path": ".",
                    "type": "dir",
                    "recursive": True,
                    "file_count": 2,
                    "dir_count": 0,
                    "entries": [
                        {"path": "new.txt", "type": "file", "modified_at": "2026-05-28T10:00:00+00:00"},
                        {"path": "old.txt", "type": "file", "modified_at": "2026-05-27T10:00:00+00:00"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Tell me how many files exist in the workspace and what the newest file is.")
        self.assertIn("There are 2 files", result.content)
        self.assertIn("new.txt", result.content)
        self.assertEqual(registry.called, [("stat_path", {"path": ".", "recursive": True})])
        self.assertEqual(client.calls, [])

    def test_analysis_skill_can_expand_route_with_write_tools(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "starting analysis"},
                }
            ]
        )
        registry = FakeRegistry()
        skill = Skill(
            name="analysis-skill",
            path=Path("/tmp/analysis-skill/SKILL.md"),
            content="Create or reuse a case directory.\nCreate or read `AGENTS.md` and `REPORT.md`.\n",
        )
        agent = AgentLoop(client=client, registry=registry, max_loops=2, skill=skill)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertEqual(result.content, "starting analysis")
        self.assertEqual(registry.routed[0], ("shell", "filesystem", "write"))
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "Active skill startup is mandatory" in str(item.get("content", ""))
                for item in client.calls[0]["messages"]
            )
        )

    def test_analysis_skill_does_not_bootstrap_workspace_docs_for_greeting(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        skill = Skill(
            name="analysis-skill",
            path=Path("/tmp/analysis-skill/SKILL.md"),
            content="Create or reuse a case directory.\nCreate or read `AGENTS.md` and `REPORT.md`.\n",
        )
        agent = AgentLoop(client=client, registry=registry, max_loops=2, skill=skill)
        result = agent.run_turn("ciao")
        self.assertEqual(result.content, "Ciao! Come posso aiutarti?")
        self.assertEqual(registry.called, [])

    def test_gemma_model_first_adds_post_tool_guidance_for_machine_inspection(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "lscpu"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "CPU summary."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("qual è la configurazione hw di questo PC?")
        self.assertEqual(result.content, "CPU summary.")
        followup_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "machine-inspection results" in str(message.get("content", "")).lower()
                for message in followup_messages
            )
        )

    def test_gemma_model_first_finalizes_workspace_discovery_from_local_listing(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "src", "type": "dir"},
                        {"path": "tests", "type": "dir"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("cosa contiene questa directory di lavoro?")
        self.assertEqual(result.content, "src, tests, README.md")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])

    def test_gemma_model_first_lists_workspace_files_and_directories_locally(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "src", "type": "dir"},
                        {"path": "README.md", "type": "file"},
                        {"path": "config.json", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("list all files and directories in the current workspace")
        self.assertEqual(result.content, "src, README.md, config.json")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])

    def test_gemma_model_first_adds_one_line_post_tool_guidance_for_file_reading(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "one-line summary"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Mostrami in una riga cosa contiene README.md.")
        self.assertEqual(result.content, "one-line summary")
        followup_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "one short line only" in str(message.get("content", "")).lower()
                for message in followup_messages
            )
        )

    def test_gemma_model_first_adds_post_tool_guidance_for_codebase_inspection(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "- `src/orbit/core/agent.py`\n- `src/orbit/core/runtime.py`"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Dimmi i file più importanti da leggere per capire questo progetto.")
        self.assertIn("agent.py", result.content)
        followup_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "at most 3 short bullets or file paths" in str(message.get("content", "")).lower()
                for message in followup_messages
            )
        )

    def test_gemma_model_first_adds_stronger_post_tool_guidance_for_explicit_file_bug_review(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        agent = AgentLoop(client=client, registry=FakeRegistry(), max_loops=4)
        agent.messages.append(
            {
                "role": "user",
                "content": "analyze the python code in this file: agent.py and tell me if you find bugs.",
            }
        )
        prompt = agent._model_first_post_tool_prompt(
            type("Route", (), {"intent_class": "codebase_inspection", "intent": "codebase_inspection"})()
        )
        self.assertIsNotNone(prompt)
        self.assertIn("one concrete source file", prompt.lower())
        self.assertIn("concrete bug or risk findings", prompt.lower())
        self.assertIn("generic uncertainty", prompt.lower())

    def test_gemma_model_first_adds_stronger_post_tool_guidance_for_explicit_security_review(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        agent = AgentLoop(client=client, registry=FakeRegistry(), max_loops=4)
        agent.messages.append(
            {
                "role": "user",
                "content": "review deploy.ps1 for vulnerabilities and security issues.",
            }
        )
        prompt = agent._model_first_post_tool_prompt(
            type("Route", (), {"intent_class": "codebase_inspection", "intent": "codebase_inspection"})()
        )
        self.assertIsNotNone(prompt)
        self.assertIn("one concrete source file", prompt.lower())
        self.assertIn("concrete bug or risk findings", prompt.lower())

    def test_gemma_model_first_prunes_transient_system_messages_after_turn(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "src, tests, README.md"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "src", "type": "dir"},
                        {"path": "tests", "type": "dir"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("cosa contiene questa directory di lavoro?")
        self.assertEqual(result.content, "src, tests, README.md")
        self.assertFalse(
            any(
                message.get("role") == "system" and message.get("_orbit_transient_system")
                for message in agent.messages
            )
        )

    def test_gemma_model_first_uses_compact_tool_definitions(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "Hello."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        agent.run_turn("qual è la configurazione hw di questo PC?")
        tools = client.calls[0]["tools"]
        self.assertTrue(tools)
        bash_tool = next(item for item in tools if item["function"]["name"] == "bash")
        self.assertEqual(bash_tool["function"]["description"], "Run one bounded safe command in the workdir.")
        command_properties = bash_tool["function"]["parameters"]["properties"]["command"]
        self.assertEqual(command_properties, {"type": "string"})

    def test_gemma_model_first_compacts_tool_payloads_in_request_messages(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "start_line": 1,
                    "returned_lines": 200,
                    "total_lines": 200,
                    "next_start_line": 201,
                    "has_more": False,
                    "truncated": False,
                    "content": "x" * 5000,
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read README.md")
        self.assertEqual(result.content, "done")
        second_call_messages = client.calls[1]["messages"]
        tool_message = next(item for item in second_call_messages if item.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        self.assertEqual(payload["path"], "README.md")
        self.assertLessEqual(len(payload["content"]), 3000)
        self.assertNotIn("x" * 4000, payload["content"])

    def test_gemma_workspace_discovery_uses_local_top_level_listing(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "entries": [
                    {"path": "src", "type": "dir"},
                    {"path": "tests", "type": "dir"},
                    {"path": "README.md", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("cosa contiene questa directory di lavoro?")
        self.assertEqual(result.content, "src, tests, README.md")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_machine_inspection_retries_du_root_to_df_before_execution(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "du -sh /"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "filesystem info"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("show disk usage of /")
        self.assertEqual(result.content, "filesystem info")
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 2)
        retry_messages = client.calls[1]["messages"]
        retry_prompt = next(
            str(message.get("content", ""))
            for message in retry_messages
            if message.get("role") == "system" and "filesystem capacity" in str(message.get("content", ""))
        )
        self.assertIn("Do not use `du` here", retry_prompt)
        self.assertIn("df -h /", retry_prompt)

    def test_simple_chitchat_greeting_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("ciao")
        self.assertEqual(result.content, "Ciao! Come posso aiutarti?")
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(registry.called, [])
        self.assertEqual(route_intent("ciao").intent, INTENT_CHITCHAT)

    def test_simple_chitchat_greeting_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("hello")
        self.assertEqual(result.content, "Hello! How can I help?")
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(registry.called, [])
        self.assertEqual(route_intent("hello").intent, INTENT_CHITCHAT)

    def test_typo_greeting_in_italian_can_finalize_locally_without_tools(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("cioa")
        self.assertEqual(result.content, "Ciao! Come posso aiutarti?")
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(registry.called, [])

    def test_typo_greeting_in_english_can_finalize_locally_without_tools(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("helo")
        self.assertEqual(result.content, "Hello! How can I help?")
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(registry.called, [])

    def test_explicit_file_edit_is_not_misrouted_as_chitchat(self) -> None:
        self.assertEqual(
            route_intent("In src/app.py replace hello with hello orbit and keep the file otherwise unchanged.").intent,
            INTENT_FILE_EDIT,
        )

    def test_general_knowledge_uses_minimal_chat_path(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "Per diffusione di Rayleigh."},
                }
            ]
        )
        registry = FakeRegistry()
        skill = Skill(
            name="analysis-skill",
            path=Path("/tmp/analysis-skill/SKILL.md"),
            content="Create or reuse a case directory.\nCreate or read `AGENTS.md` and `REPORT.md`.\n",
        )
        agent = AgentLoop(client=client, registry=registry, max_loops=2, skill=skill)
        result = agent.run_turn("perchè il cielo è blu?")
        self.assertEqual(result.content, "Per diffusione di Rayleigh.")
        request_messages = client.calls[0]["messages"]
        self.assertEqual(
            request_messages[0]["content"],
            "You are the concise local assistant running inside the Orbit CLI in the user's environment.\n"
            "Do not invent a personal name, vendor, or creator.\n"
            "Answer directly.\n",
        )
        self.assertNotIn("Active skill:", request_messages[0]["content"])
        self.assertEqual(registry.called, [])

    def test_text_document_analysis_can_finalize_from_existing_read_result(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 10,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_700_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}
                        ],
                    }
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_500_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}
                        ],
                    },
                },
            ]
        )
        registry = FakeRegistry()
        registry.call = lambda name, arguments: {
            "ok": True,
            "path": "REPORT.md",
            "content": "hello report",
            "truncated": False,
            "has_more": False,
        }
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("mostrami il contenuto di REPORT.md")
        self.assertEqual(result.content, "hello report")

    def test_text_document_show_request_stops_after_first_successful_read(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_800_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "path": arguments["path"],
                "content": "report content",
                "truncated": False,
                "has_more": False,
            }
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("show the content of REPORT.md")
        self.assertEqual(result.content, "report content")
        self.assertEqual(registry.called, [("read_file", {"path": "REPORT.md"})])

    def test_codebase_version_query_can_finalize_locally_from_pyproject(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file" and arguments["path"] == "pyproject.toml":
                return {
                    "ok": True,
                    "path": "pyproject.toml",
                    "content": '[project]\nname = "orbit"\nversion = "0.1.0"\n',
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "file not found"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("What version is this project? Answer in one short line.")
        self.assertEqual(result.content, "Project version: 0.1.0")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [("read_file", {"path": "pyproject.toml", "start_line": 1, "max_lines": 80, "max_chars": 4000})])

    def test_codebase_priority_files_request_can_finalize_locally_from_seeded_listing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/tooling/registry.py", "type": "file"},
                        {"path": "src/orbit/tooling/filesystem.py", "type": "file"},
                        {"path": "src/orbit/terminal/cli.py", "type": "file"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Analizza il codice di questo progetto e dimmi solo i 5 file più importanti da leggere prima di rispondere.")
        self.assertEqual(
            result.content,
            "- `src/orbit/core/agent.py`\n- `src/orbit/core/runtime.py`\n- `src/orbit/tooling/registry.py`\n- `src/orbit/tooling/filesystem.py`\n- `src/orbit/terminal/cli.py`",
        )
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": True, "max_entries": 80})])

    def test_codebase_architecture_summary_request_can_finalize_locally_from_seeded_listing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/tooling/registry.py", "type": "file"},
                        {"path": "src/orbit/tooling/filesystem.py", "type": "file"},
                        {"path": "src/orbit/terminal/cli.py", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Riassumi in 3 punti l'architettura di questo progetto senza usare tool inutili.")
        self.assertIn("`src/orbit/core/`", result.content)
        self.assertIn("`src/orbit/tooling/`", result.content)
        self.assertIn("`src/orbit/terminal/`", result.content)
        self.assertEqual(client.calls, [])

    def test_codebase_architecture_summary_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/tooling/web.py", "type": "file"},
                        {"path": "src/orbit/terminal/ui.py", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize this project architecture in 3 bullet points without unnecessary tool calls.")
        self.assertIn("`src/orbit/core/`", result.content)
        self.assertIn("`src/orbit/tooling/`", result.content)
        self.assertIn("`src/orbit/terminal/`", result.content)
        self.assertEqual(client.calls, [])

    def test_codebase_hotspot_request_can_finalize_locally_from_seeded_listing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/core/tool_guardrails.py", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Quali sono i 3 punti del codice che meritano più attenzione per stabilità e manutenzione?")
        self.assertIn("`src/orbit/core/agent.py`", result.content)
        self.assertIn("`src/orbit/core/runtime.py`", result.content)
        self.assertIn("`src/orbit/core/tool_guardrails.py`", result.content)
        self.assertEqual(client.calls, [])

    def test_codebase_hotspot_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/core/tool_guardrails.py", "type": "file"},
                    ],
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Which 3 parts of this codebase deserve the most attention for stability and maintainability?")
        self.assertIn("`src/orbit/core/agent.py`", result.content)
        self.assertIn("`src/orbit/core/runtime.py`", result.content)
        self.assertIn("`src/orbit/core/tool_guardrails.py`", result.content)
        self.assertEqual(client.calls, [])

    def test_code_review_request_in_english_can_finalize_locally_after_targeted_reads(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/core/tool_guardrails.py", "type": "file"},
                        {"path": "src/orbit/tooling/registry.py", "type": "file"},
                    ],
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/agent.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/agent.py",
                    "content": "\n".join(["if x: pass"] * 40),
                    "total_lines": 260,
                    "truncated": False,
                    "has_more": False,
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/runtime.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/runtime.py",
                    "content": "def boot():\n    pass\n",
                    "total_lines": 140,
                    "truncated": False,
                    "has_more": False,
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/tool_guardrails.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/tool_guardrails.py",
                    "content": "re.compile('a')\nre.compile('b')\nX_HINTS = ()\nY_HINTS = ()\nvalue.startswith('a')\nvalue.startswith('b')\n",
                    "total_lines": 220,
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Review this codebase and list the top 3 risks or findings before making changes.")
        self.assertIn("High:", result.content)
        self.assertIn("`src/orbit/core/agent.py`", result.content)
        self.assertIn("`src/orbit/core/tool_guardrails.py`", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called[0], ("list_files", {"path": ".", "recursive": True, "max_entries": 80}))
        self.assertEqual(registry.called[1][0], "read_file")

    def test_code_review_request_in_italian_can_finalize_locally_after_targeted_reads(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/tooling/registry.py", "type": "file"},
                    ],
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/agent.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/agent.py",
                    "content": "\n".join(["if x: pass"] * 30),
                    "total_lines": 240,
                    "truncated": False,
                    "has_more": False,
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/runtime.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/runtime.py",
                    "content": "def boot():\n    pass\n",
                    "total_lines": 120,
                    "truncated": False,
                    "has_more": False,
                }
            if name == "read_file" and arguments["path"] == "src/orbit/tooling/registry.py":
                return {
                    "ok": True,
                    "path": "src/orbit/tooling/registry.py",
                    "content": "def call():\n    pass\n",
                    "total_lines": 90,
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Fai una code review di questo progetto e dimmi i 3 rischi principali prima di modificare il codice.")
        self.assertIn("Alta:", result.content)
        self.assertIn("`src/orbit/core/agent.py`", result.content)
        self.assertEqual(client.calls, [])

    def test_code_review_request_reads_second_chunk_when_risk_appears_later(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                    ],
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/agent.py":
                start_line = arguments.get("start_line", 1)
                if start_line == 1:
                    return {
                        "ok": True,
                        "path": "src/orbit/core/agent.py",
                        "content": "\n".join("value = 1" for _ in range(220)),
                        "start_line": 1,
                        "returned_lines": 220,
                        "total_lines": 260,
                        "next_start_line": 221,
                        "has_more": True,
                        "truncated": False,
                    }
                return {
                    "ok": True,
                    "path": "src/orbit/core/agent.py",
                    "content": "try:\n    work()\nexcept:\n    pass\n",
                    "start_line": 221,
                    "returned_lines": 4,
                    "total_lines": 260,
                    "next_start_line": 261,
                    "has_more": False,
                    "truncated": False,
                }
            if name == "read_file" and arguments["path"] == "src/orbit/core/runtime.py":
                return {
                    "ok": True,
                    "path": "src/orbit/core/runtime.py",
                    "content": "def boot():\n    pass\n",
                    "start_line": 1,
                    "returned_lines": 2,
                    "total_lines": 120,
                    "next_start_line": 3,
                    "has_more": False,
                    "truncated": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Review this codebase and list the top 3 risks before making changes.")
        self.assertIn("bare `except:`", result.content)
        self.assertEqual(
            registry.called[1:3],
            [
                ("read_file", {"path": "src/orbit/core/agent.py", "start_line": 1, "max_lines": 220, "max_chars": 9000}),
                ("read_file", {"path": "src/orbit/core/agent.py", "start_line": 221, "max_lines": 220, "max_chars": 9000}),
            ],
        )

    def test_code_review_request_in_italian_reports_todo_or_placeholder_signals(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                    ],
                }
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "src/orbit/core/runtime.py",
                    "content": "TODO: harden retries\npass\nTODO: compact earlier\npass\npass\n",
                    "start_line": 1,
                    "returned_lines": 5,
                    "total_lines": 90,
                    "next_start_line": 6,
                    "has_more": False,
                    "truncated": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Fai una code review di questo progetto e riporta i principali rischi o debolezze.")
        self.assertTrue("placeholder" in result.content or "TODO/FIXME" in result.content)
        self.assertEqual(client.calls, [])

    def test_code_review_request_can_emit_cross_file_finding_for_agent_and_runtime(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": True,
                    "entries": [
                        {"path": "src/orbit/core/agent.py", "type": "file"},
                        {"path": "src/orbit/core/runtime.py", "type": "file"},
                        {"path": "src/orbit/core/tool_guardrails.py", "type": "file"},
                    ],
                }
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "def x():\n    return 1\n",
                    "start_line": 1,
                    "returned_lines": 2,
                    "total_lines": 40,
                    "next_start_line": 3,
                    "has_more": False,
                    "truncated": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Review this codebase and list the top 3 findings before making changes.")
        self.assertIn("tight execution boundary", result.content)
        self.assertIn("src/orbit/core/agent.py", result.content)
        self.assertIn("src/orbit/core/runtime.py", result.content)
        self.assertEqual(client.calls, [])

    def test_explicit_show_content_request_can_finalize_locally_after_seeded_read(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "content": "# Orbit Sample\n\nA tiny sample project.\n",
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Show the content of README.md.")
        self.assertEqual(result.content, "# Orbit Sample\n\nA tiny sample project.\n")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [("read_file", {"path": "README.md"})])

    def test_explicit_summary_request_can_finalize_locally_after_seeded_read(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "content": "# Orbit Sample\n\nA tiny sample project for strategic tests.\n",
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize README.md in one short line.")
        self.assertEqual(result.content, "A tiny sample project for strategic tests.")
        self.assertEqual(client.calls, [])

    def test_explicit_summary_request_respects_sentence_count(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "summary.txt",
                    "content": (
                        "AI safety is a critical area of research.\n"
                        "Alignment techniques aim to ensure AI systems act in accordance with human values.\n"
                        "Robustness is key to preventing unintended and harmful consequences.\n"
                    ),
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("read the file summary.txt and summarize its purpose in two sentences")
        self.assertEqual(
            result.content,
            "AI safety is a critical area of research. Alignment techniques aim to ensure AI systems act in accordance with human values.",
        )
        self.assertEqual(client.calls, [])

    def test_long_explicit_summary_request_reads_multiple_chunks_before_summarizing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "README.md",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 140,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "\n".join("# Section" for _ in range(120)),
                }
            return {
                "ok": True,
                "path": "README.md",
                "start_line": 121,
                "returned_lines": 20,
                "total_lines": 140,
                "next_start_line": 141,
                "has_more": False,
                "truncated": False,
                "content": "This document explains the full workflow and summary behavior.\nIt also covers the final conclusions.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize README.md.")
        self.assertIn("This document explains the full workflow and summary behavior.", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md", "start_line": 1, "max_lines": 120, "max_chars": 6000}),
                ("read_file", {"path": "README.md", "start_line": 121, "max_lines": 120, "max_chars": 6000}),
            ],
        )

    def test_read_and_summarize_request_prefers_progressive_summary_over_single_show_read(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "promessi_sposi.txt",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 180,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "\n".join("Introduzione" for _ in range(120)),
                }
            return {
                "ok": True,
                "path": "promessi_sposi.txt",
                "start_line": 121,
                "returned_lines": 60,
                "total_lines": 180,
                "next_start_line": 181,
                "has_more": False,
                "truncated": False,
                "content": "Renzo e Lucia cercano di sposarsi nonostante gli ostacoli di Don Rodrigo.\nIl romanzo segue prove, fughe e giustizia finale.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Read promessi_sposi.txt file and summarize it.")
        self.assertIn("Renzo e Lucia", result.content)
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "promessi_sposi.txt", "start_line": 1, "max_lines": 120, "max_chars": 6000}),
                ("read_file", {"path": "promessi_sposi.txt", "start_line": 121, "max_lines": 120, "max_chars": 6000}),
            ],
        )

    def test_model_first_read_and_summarize_request_uses_local_progressive_summary(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "Renzo and Lucia try to marry despite Don Rodrigo's interference, and the novel follows hardship, plague, and final reconciliation."},
                }
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "promessi_sposi.txt",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 180,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "\n".join("Introduzione" for _ in range(120)),
                }
            return {
                "ok": True,
                "path": "promessi_sposi.txt",
                "start_line": 121,
                "returned_lines": 60,
                "total_lines": 180,
                "next_start_line": 181,
                "has_more": False,
                "truncated": False,
                "content": "Renzo e Lucia cercano di sposarsi nonostante gli ostacoli di Don Rodrigo.\nIl romanzo segue prove, fughe e giustizia finale.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        result = agent.run_turn("Read promessi_sposi.txt file and summarize it.")
        self.assertIn("Renzo and Lucia", result.content)
        self.assertEqual(len(client.calls), 1)
        evidence_messages = [
            item["content"]
            for item in client.calls[0]["messages"]
            if item.get("role") == "user" and "Bounded file evidence:" in str(item.get("content", ""))
        ]
        self.assertEqual(len(evidence_messages), 1)
        self.assertIn("chunk_notes", evidence_messages[0])
        self.assertIn("\"summary_read\": true", evidence_messages[0])
        persisted_tool_payloads = [
            json.loads(item["content"])
            for item in agent.messages
            if item.get("role") == "tool" and item.get("tool_name") == "read_file"
        ]
        self.assertEqual(len(persisted_tool_payloads), 1)
        self.assertTrue(persisted_tool_payloads[0].get("summary_read"))
        self.assertIn("chunk_notes", persisted_tool_payloads[0])
        self.assertGreaterEqual(len(persisted_tool_payloads[0]["chunk_notes"]), 2)
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "promessi_sposi.txt", "start_line": 1, "max_lines": 120, "max_chars": 6000}),
                ("read_file", {"path": "promessi_sposi.txt", "start_line": 121, "max_lines": 120, "max_chars": 6000}),
            ],
        )

    def test_model_first_explicit_summary_retries_when_model_echoes_evidence(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "Sampled file evidence:\n- lines 1-120; focus: Introduzione"},
                },
                {
                    "prompt_eval_count": 24,
                    "prompt_eval_duration": 110_000_000,
                    "eval_count": 14,
                    "eval_duration": 80_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "Il file racconta le vicende di Renzo e Lucia, ostacolati da Don Rodrigo, tra fughe, peste e riconciliazione finale sotto il segno della Provvidenza."
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "promessi_sposi.txt",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 180,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "\n".join("Introduzione" for _ in range(120)),
                }
            return {
                "ok": True,
                "path": "promessi_sposi.txt",
                "start_line": 121,
                "returned_lines": 60,
                "total_lines": 180,
                "next_start_line": 181,
                "has_more": False,
                "truncated": False,
                "content": "Renzo e Lucia cercano di sposarsi nonostante gli ostacoli di Don Rodrigo.\nIl romanzo segue prove, fughe, peste e riconciliazione finale.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        result = agent.run_turn("Read promessi_sposi.txt file and summarize it.")
        self.assertIn("Renzo e Lucia", result.content)
        self.assertEqual(len(client.calls), 2)

    def test_long_explicit_summary_request_in_italian_reads_multiple_chunks(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "docs/note.txt",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 130,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "\n".join("# Titolo" for _ in range(120)),
                }
            return {
                "ok": True,
                "path": "docs/note.txt",
                "start_line": 121,
                "returned_lines": 10,
                "total_lines": 130,
                "next_start_line": 131,
                "has_more": False,
                "truncated": False,
                "content": "Questo documento descrive il comportamento completo del sistema.\nCopre anche i punti finali.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Riassumi docs/note.txt.")
        self.assertIn("Questo documento descrive il comportamento completo del sistema.", result.content)
        self.assertEqual(client.calls, [])

    def test_explicit_ten_line_summary_request_reads_more_chunks(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            chunk_index = ((start_line - 1) // 120) + 1
            has_more = chunk_index < 6
            return {
                "ok": True,
                "path": "books/divina.txt",
                "start_line": start_line,
                "returned_lines": 120,
                "total_lines": 720,
                "next_start_line": start_line + 120,
                "has_more": has_more,
                "truncated": False,
                "content": f"Chunk {chunk_index} overview line.\nChunk {chunk_index} supporting detail.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize books/divina.txt in 10 lines.")
        self.assertIn("Chunk 1 overview line.", result.content)
        self.assertIn("Chunk 6 overview line.", result.content)
        self.assertEqual(len(registry.called), 6)

    def test_very_long_summary_request_marks_partial_when_chunk_budget_is_exhausted(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            chunk_index = ((start_line - 1) // 120) + 1
            return {
                "ok": True,
                "path": "books/long.txt",
                "start_line": start_line,
                "returned_lines": 120,
                "total_lines": 2000,
                "next_start_line": start_line + 120,
                "has_more": True,
                "truncated": False,
                "content": f"Section {chunk_index} core idea.\nSection {chunk_index} detail.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Riassumi books/long.txt in 10 righe.")
        self.assertIn("based on retrieved portions of a longer file", result.content)
        self.assertEqual(len(registry.called), 8)

    def test_huge_summary_request_samples_late_sections_of_the_file(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        chunk_map = {
            1: "Introduzione del manoscritto secentesco.",
            3317: "Don Abbondio incontra i bravi sulla strada.",
            6632: "Renzo e Lucia provano a sposarsi in segreto.",
            9948: "La fuga da Milano segue il tumulto del pane.",
            13263: "L'Innominato affronta la propria conversione.",
            16579: "La peste travolge Milano e separa i personaggi.",
            19894: "Fra Cristoforo ritrova Renzo e lo guida.",
            23210: "Renzo e Lucia si ricongiungono e il romanzo si chiude sulla Provvidenza.",
        }

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            content = chunk_map.get(start_line, f"Section starting at line {start_line}.")
            return {
                "ok": True,
                "path": "promessi_sposi.txt",
                "start_line": start_line,
                "returned_lines": 120,
                "total_lines": 23329,
                "next_start_line": start_line + 120,
                "has_more": start_line < 23210,
                "truncated": False,
                "content": content,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize promessi_sposi.txt in 10 lines.")
        self.assertIn("Renzo e Lucia", result.content)
        self.assertIn("peste", result.content)
        self.assertIn("Provvidenza", result.content)
        self.assertEqual(
            [arguments["start_line"] for name, arguments in registry.called if name == "read_file"],
            [1, 3317, 6632, 9948, 13263, 16579, 19894, 23210],
        )

    def test_explicit_summary_request_retries_when_line_count_is_wrong(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 60,
                    "eval_duration": 140_000_000,
                    "total_duration": 260_000_000,
                    "message": {
                        "content": (
                            "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5.\n"
                            "Line 6.\nLine 7.\nLine 8.\nLine 9.\nLine 10."
                        )
                    },
                },
                {
                    "prompt_eval_count": 42,
                    "prompt_eval_duration": 125_000_000,
                    "eval_count": 72,
                    "eval_duration": 155_000_000,
                    "total_duration": 280_000_000,
                    "message": {
                        "content": (
                            "Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5.\n"
                            "Line 6.\nLine 7.\nLine 8.\nLine 9.\nLine 10.\nLine 11.\nLine 12."
                        )
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            return {
                "ok": True,
                "path": "promessi_sposi.txt",
                "start_line": start_line,
                "returned_lines": 120,
                "total_lines": 23329,
                "next_start_line": start_line + 120,
                "has_more": start_line < 2000,
                "truncated": False,
                "content": f"Chunk at {start_line}.\nMore detail at {start_line}.",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Read promessi_sposi.txt and summarize it in exactly 12 lines.")
        self.assertEqual(len([line for line in result.content.splitlines() if line.strip()]), 12)
        self.assertEqual(len(client.calls), 2)
        retry_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "exactly 12 lines" in str(message.get("content", ""))
                for message in retry_messages
            )
        )

    def test_local_summary_prefers_paragraph_candidates_over_fragment_lines(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            if start_line == 1:
                return {
                    "ok": True,
                    "path": "romanzo.txt",
                    "start_line": 1,
                    "returned_lines": 120,
                    "total_lines": 240,
                    "next_start_line": 121,
                    "has_more": True,
                    "truncated": False,
                    "content": "I PROMESSI SPOSI\n\nINTRODUZIONE\n\nL'historia si può veramente deffinire una guerra illustre contro il Tempo, perchè togliendoli di mano gl'anni suoi prigionieri.\n",
                }
            return {
                "ok": True,
                "path": "romanzo.txt",
                "start_line": 121,
                "returned_lines": 120,
                "total_lines": 240,
                "next_start_line": 241,
                "has_more": False,
                "truncated": False,
                "content": "Renzo e Lucia cercano di sposarsi, ma Don Rodrigo ostacola il loro matrimonio.\n\nIl romanzo segue prove, fughe, peste e riconciliazione finale sotto il segno della Provvidenza.\n",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize romanzo.txt in 3 lines.")
        self.assertIn("Renzo e Lucia", result.content)
        self.assertIn("Provvidenza", result.content)
        self.assertNotIn("- I PROMESSI SPOSI", result.content)

    def test_local_summary_condenses_many_read_file_chunks_in_session_history(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 80_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "This long file covers sections sampled across the document and remains partial."},
                }
                ,
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 80_000_000,
                    "total_duration": 210_000_000,
                    "message": {"content": "This long file covers sections sampled across the document and remains partial."},
                }
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            start_line = arguments.get("start_line", 1)
            return {
                "ok": True,
                "path": "books/long.txt",
                "start_line": start_line,
                "returned_lines": 120,
                "total_lines": 2000,
                "next_start_line": start_line + 120,
                "has_more": start_line < 1660,
                "truncated": False,
                "content": f"Section at {start_line}.\n",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        result = agent.run_turn("Summarize books/long.txt in 10 lines.")
        self.assertIn("partial", result.content)
        tool_messages = [item for item in agent.messages if item.get("role") == "tool" and item.get("tool_name") == "read_file"]
        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0]["content"])
        self.assertTrue(payload.get("summary_read"))
        self.assertGreaterEqual(payload.get("sampled_chunks", 0), 2)
        self.assertGreaterEqual(len(payload.get("chunk_notes", [])), 2)
        self.assertIn("Sampled file evidence:", payload.get("content", ""))

    def test_explicit_pdf_summary_request_can_finalize_locally_from_pdftotext(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "bash":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "command": arguments["command"],
                "stdout": "This PDF explains the project scope.\nIt also describes the final recommendations.\n",
                "stderr": "",
                "returncode": 0,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize docs/Project Overview.pdf.")
        self.assertIn("This PDF explains the project scope.", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [("bash", {"command": "pdftotext 'docs/Project Overview.pdf' - | head -n 240"})],
        )

    def test_explicit_pdf_show_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "bash":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "command": arguments["command"],
                "stdout": "Prima pagina del PDF.\nSeconda riga utile.\n",
                "stderr": "",
                "returncode": 0,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Mostra il contenuto di PLO ABSTRACT.pdf.")
        self.assertIn("Prima pagina del PDF.", result.content)
        self.assertIn("bounded PDF extract via pdftotext", result.content)
        self.assertEqual(client.calls, [])

    def test_explicit_pdf_colloquial_read_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "bash":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "command": arguments["command"],
                "stdout": "Prima pagina del PDF.\nSeconda riga utile.\n",
                "stderr": "",
                "returncode": 0,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("mi leggi questo file? PLO ABSTRACT.pdf")
        self.assertIn("Prima pagina del PDF.", result.content)
        self.assertIn("bounded PDF extract via pdftotext", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [("bash", {"command": "pdftotext 'PLO ABSTRACT.pdf' - | head -n 240"})],
        )

    def test_explicit_pdf_summary_falls_back_to_strings_when_pdftotext_fails(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "bash":
                return {"ok": False, "error": "unexpected"}
            command = arguments["command"]
            if command.startswith("pdftotext "):
                return {"ok": False, "command": command, "stderr": "pdftotext missing"}
            return {
                "ok": True,
                "command": command,
                "stdout": "Fallback PDF text line one.\nFallback PDF text line two.\n",
                "stderr": "",
                "returncode": 0,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Summarize docs/Strategy Notes.pdf.")
        self.assertIn("Fallback PDF text line one.", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "pdftotext 'docs/Strategy Notes.pdf' - | head -n 240"}),
                ("bash", {"command": "strings 'docs/Strategy Notes.pdf' | head -n 240"}),
            ],
        )

    def test_deterministic_append_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "REPORT.md",
                    "content": "# REPORT\n\nInitial body.\n",
                    "truncated": False,
                    "has_more": False,
                }
            if name == "append_file":
                return {
                    "ok": True,
                    "path": "REPORT.md",
                    "content": arguments["content"],
                    "appended": True,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Append a section titled Notes to REPORT.md with one bullet: Strategic test passed.")
        self.assertIn("REPORT.md", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "REPORT.md"}),
                ("append_file", {"path": "REPORT.md", "content": "\n\n## Notes\n\n- Strategic test passed.\n"}),
            ],
        )

    def test_deterministic_append_request_can_create_missing_file_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {"ok": False, "error": "file not found: REPORT.md"}
            if name == "append_file":
                return {
                    "ok": True,
                    "path": "REPORT.md",
                    "content": arguments["content"],
                    "appended": True,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Append a section titled Notes to REPORT.md with one bullet: Strategic test passed.")
        self.assertIn("REPORT.md", result.content)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "REPORT.md"}),
                ("append_file", {"path": "REPORT.md", "content": "\n\n## Notes\n\n- Strategic test passed.\n"}),
            ],
        )

    def test_directory_listing_request_can_finalize_locally_from_seeded_listing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "path": ".",
                "entries": [
                    {"path": "README.md", "type": "file"},
                    {"path": "REPORT.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("What files are in this directory? Answer briefly.")
        self.assertEqual(result.content, "README.md, REPORT.md, pyproject.toml")
        self.assertEqual(client.calls, [])

    def test_italian_directory_listing_request_can_finalize_locally_from_seeded_listing(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "path": ".",
                "entries": [
                    {"path": "README.md", "type": "file"},
                    {"path": "REPORT.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Quali file ci sono in questa directory? Rispondi breve.")
        self.assertEqual(result.content, "README.md, REPORT.md, pyproject.toml")
        self.assertEqual(client.calls, [])

    def test_bounded_pwd_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {"ok": True, "command": "pwd", "stdout": "/tmp/demo\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Use bash to run pwd and give me only the output path.")
        self.assertEqual(result.content, "/tmp/demo")
        self.assertEqual(client.calls, [])

    def test_deterministic_replace_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "src/app.py",
                    "content": 'def greet():\n    return "hello"\n',
                    "truncated": False,
                    "has_more": False,
                }
            if name == "replace_in_file":
                return {
                    "ok": True,
                    "path": "src/app.py",
                    "replaced": 1,
                    "replace_all": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("In src/app.py replace hello with hello orbit and keep the file otherwise unchanged.")
        self.assertEqual(result.content, "Updated `src/app.py`.")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "src/app.py"}),
                ("replace_in_file", {"path": "src/app.py", "old": "hello", "new": "hello orbit"}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_self_intro_request_uses_model_with_identity_prompt(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "Sono orbit, un assistente locale per questa CLI."},
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("ciao, presentati con una frase.")
        self.assertEqual(result.content, "Sono orbit, un assistente locale per questa CLI.")
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(
            any(
                item.get("role") == "system"
                and "Descriviti brevemente come orbit" in str(item.get("content", ""))
                for item in client.calls[0]["messages"]
            )
        )

    def test_self_description_request_uses_model_with_identity_prompt(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "Sono orbit, un assistente locale focalizzato sul workspace corrente."},
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("ciao, descriviti in una frase")
        self.assertEqual(result.content, "Sono orbit, un assistente locale focalizzato sul workspace corrente.")
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)

    def test_creator_request_can_finalize_locally_without_model(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("chi ti ha creato?")
        self.assertEqual(
            result.content,
            "Sono orbit, l'assistente locale di questa CLI. Il mio comportamento e` definito dal progetto orbit in questa workspace.",
        )
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [])

    def test_creator_request_in_english_can_finalize_locally_without_model(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("who created you?")
        self.assertEqual(
            result.content,
            "I am orbit, the local assistant for this CLI. My behavior is defined by the orbit project in this workspace.",
        )
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [])

    def test_identity_request_can_finalize_locally_without_model(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("chi sei?")
        self.assertEqual(result.content, "Sono orbit, l'assistente locale di questa CLI e workspace.")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [])

    def test_identity_request_in_english_can_finalize_locally_without_model(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("who are you?")
        self.assertEqual(result.content, "I am orbit, the local assistant for this CLI and workspace.")
        self.assertEqual(client.calls, [])
        self.assertEqual(registry.called, [])

    def test_deterministic_write_request_can_create_bullet_file_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "write_file":
                return {"ok": True, "path": "TODO.md", "bytes": len(arguments["content"].encode("utf-8"))}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Create TODO.md with two bullets: harden tests; keep runtime light.")
        self.assertEqual(result.content, "Created `TODO.md`.")
        self.assertEqual(
            registry.called,
            [("write_file", {"path": "TODO.md", "content": "- harden tests\n- keep runtime light\n"})],
        )
        self.assertEqual(client.calls, [])

    def test_deterministic_directory_create_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "make_directory":
                return {"ok": True, "path": "test", "type": "dir", "created": True}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("crea una cartella test")
        self.assertEqual(result.content, "Created directory `test`.")
        self.assertEqual(registry.called, [("make_directory", {"path": "test"})])
        self.assertEqual(client.calls, [])

    def test_deterministic_directory_remove_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "delete_path":
                return {"ok": True, "path": "test", "type": "dir", "deleted": True, "recursive": True}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("remove the folder test")
        self.assertEqual(result.content, "Removed `test`.")
        self.assertEqual(registry.called, [("delete_path", {"path": "test", "recursive": True})])
        self.assertEqual(client.calls, [])

    def test_deterministic_file_remove_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "delete_path":
                return {"ok": True, "path": "notes.txt", "type": "file", "deleted": True, "recursive": False}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("cancella il file notes.txt")
        self.assertEqual(result.content, "Removed `notes.txt`.")
        self.assertEqual(registry.called, [("delete_path", {"path": "notes.txt", "recursive": False})])
        self.assertEqual(client.calls, [])

    def test_deterministic_read_and_write_summary_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "content": "# Sample Project\n\nThis project validates tool routing.\n",
                    "truncated": False,
                    "has_more": False,
                }
            if name == "write_file":
                return {"ok": True, "path": "NOTES.md", "bytes": len(arguments["content"].encode("utf-8"))}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Read README.md and create NOTES.md with a one-line summary of the project.")
        self.assertEqual(result.content, "Created `NOTES.md`.")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "NOTES.md", "content": "This project validates tool routing.\n"}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_deterministic_read_write_then_append_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "content": "# Demo Project\n\nThis workspace exists for prompt regression testing.\n",
                    "truncated": False,
                    "has_more": False,
                }
            if name == "write_file":
                return {"ok": True, "path": "REPORT2.md", "bytes": len(arguments["content"].encode("utf-8"))}
            if name == "append_file":
                return {"ok": True, "path": "REPORT2.md", "bytes": len(arguments["content"].encode("utf-8"))}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn(
            "Read README.md and create REPORT2.md with a one-line summary of the project, "
            "then reopen it and append a section titled Next Steps with one bullet: keep tool routes stable."
        )
        self.assertEqual(result.content, "Updated `REPORT2.md` and added the requested follow-up section.")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "REPORT2.md", "content": "This workspace exists for prompt regression testing.\n"}),
                ("append_file", {"path": "REPORT2.md", "content": "\n\n## Next Steps\n\n- keep tool routes stable.\n"}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_deterministic_read_write_then_append_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "README.md",
                    "content": "# Demo Project\n\nThis workspace exists for prompt regression testing.\n",
                    "truncated": False,
                    "has_more": False,
                }
            if name == "write_file":
                return {"ok": True, "path": "BRIEF.md", "bytes": len(arguments["content"].encode("utf-8"))}
            if name == "append_file":
                return {"ok": True, "path": "BRIEF.md", "bytes": len(arguments["content"].encode("utf-8"))}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn(
            "Leggi README.md e crea BRIEF.md con un riassunto in una riga del progetto, "
            "poi riaprilo e aggiungi una sezione finale con un bullet: mantenere stabile il routing delle tool."
        )
        self.assertEqual(result.content, "Updated `BRIEF.md` and added the requested follow-up section.")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "BRIEF.md", "content": "This workspace exists for prompt regression testing.\n"}),
                ("append_file", {"path": "BRIEF.md", "content": "\n\n## Finale\n\n- mantenere stabile il routing delle tool.\n"}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_current_factual_one_result_can_finalize_locally_from_search(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "OpenAI API docs",
                                        "max_results": 1,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "OpenAI API Platform - https://platform.openai.com/docs/api-reference"
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": "OpenAI API docs",
                "results": [{"title": "OpenAI API Platform", "url": "https://platform.openai.com/docs/api-reference"}],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Search the web for OpenAI API docs and return one result only.")
        self.assertEqual(result.content, "OpenAI API Platform - https://platform.openai.com/docs/api-reference")
        self.assertEqual(registry.called, [("search_web", {"query": "OpenAI API docs", "max_results": 1})])
        self.assertEqual(len(client.calls), 1)

    def test_system_info_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            command = arguments.get("command")
            if name != "bash" or not isinstance(command, str):
                return {"ok": False, "error": "unexpected"}
            if command == "cat /etc/os-release":
                return {"ok": True, "command": command, "stdout": 'PRETTY_NAME=\"Ubuntu 24.04 LTS\"\\n', "stderr": "", "returncode": 0}
            if command == "uname -srm":
                return {"ok": True, "command": command, "stdout": "Linux 6.8.0-60-generic x86_64\n", "stderr": "", "returncode": 0}
            if command == "lscpu":
                return {
                    "ok": True,
                    "command": command,
                    "stdout": "Architecture: x86_64\nCPU(s): 8\nModel name: Intel(R) Core(TM) i7 Test CPU\n",
                    "stderr": "",
                    "returncode": 0,
                }
            if command == "grep MemTotal /proc/meminfo":
                return {"ok": True, "command": command, "stdout": "MemTotal:       16777216 kB\n", "stderr": "", "returncode": 0}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("qual è la configurazione di questo mio pc?")
        self.assertIn("Sistema operativo: Ubuntu 24.04 LTS", result.content)
        self.assertIn("Kernel: Linux 6.8.0-60-generic x86_64", result.content)
        self.assertIn("Intel(R) Core(TM) i7 Test CPU", result.content)
        self.assertIn("8 CPU logiche", result.content)
        self.assertIn("Memoria: 16.0 GiB", result.content)
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "cat /etc/os-release"}),
                ("bash", {"command": "uname -srm"}),
                ("bash", {"command": "lscpu"}),
                ("bash", {"command": "grep MemTotal /proc/meminfo"}),
            ],
        )
        self.assertEqual(client.calls, [])

    def test_system_info_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            command = arguments.get("command")
            if name != "bash" or not isinstance(command, str):
                return {"ok": False, "error": "unexpected"}
            if command == "cat /etc/os-release":
                return {"ok": True, "command": command, "stdout": 'PRETTY_NAME=\"Ubuntu 24.04 LTS\"\\n', "stderr": "", "returncode": 0}
            if command == "uname -srm":
                return {"ok": True, "command": command, "stdout": "Linux 6.8.0-60-generic x86_64\n", "stderr": "", "returncode": 0}
            if command == "lscpu":
                return {
                    "ok": True,
                    "command": command,
                    "stdout": "Architecture: x86_64\nCPU(s): 16\nModel name: AMD Ryzen Test CPU\n",
                    "stderr": "",
                    "returncode": 0,
                }
            if command == "grep MemTotal /proc/meminfo":
                return {"ok": True, "command": command, "stdout": "MemTotal:       33554432 kB\n", "stderr": "", "returncode": 0}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("what is the configuration of this machine?")
        self.assertIn("OS: Ubuntu 24.04 LTS", result.content)
        self.assertIn("Kernel: Linux 6.8.0-60-generic x86_64", result.content)
        self.assertIn("AMD Ryzen Test CPU", result.content)
        self.assertIn("16 logical CPUs", result.content)
        self.assertIn("Memory: 32.0 GiB", result.content)
        self.assertEqual(client.calls, [])

    def test_system_info_request_with_gemma_model_is_model_driven(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "cat /etc/os-release"}}},
                            {"function": {"name": "bash", "arguments": {"command": "uname -srm"}}},
                            {"function": {"name": "bash", "arguments": {"command": "lscpu"}}},
                            {"function": {"name": "bash", "arguments": {"command": "grep MemTotal /proc/meminfo"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "- OS: Ubuntu 24.04 LTS\n- Kernel: Linux 6.8.0-60-generic x86_64\n- CPU: Intel(R) Core(TM) i7 Test CPU (x86_64, 8 logical CPUs)\n- Memory: 16.0 GiB"
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            command = arguments.get("command")
            if command == "cat /etc/os-release":
                return {"ok": True, "command": command, "stdout": 'PRETTY_NAME=\"Ubuntu 24.04 LTS\"\\n', "stderr": "", "returncode": 0}
            if command == "uname -srm":
                return {"ok": True, "command": command, "stdout": "Linux 6.8.0-60-generic x86_64\n", "stderr": "", "returncode": 0}
            if command == "lscpu":
                return {
                    "ok": True,
                    "command": command,
                    "stdout": "Architecture: x86_64\nCPU(s): 8\nModel name: Intel(R) Core(TM) i7 Test CPU\n",
                    "stderr": "",
                    "returncode": 0,
                }
            if command == "grep MemTotal /proc/meminfo":
                return {"ok": True, "command": command, "stdout": "MemTotal:       16777216 kB\n", "stderr": "", "returncode": 0}
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("what is the configuration of this machine?")
        self.assertIn("8 logical cpus", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "cat /etc/os-release"}),
                ("bash", {"command": "uname -srm"}),
                ("bash", {"command": "lscpu"}),
                ("bash", {"command": "grep MemTotal /proc/meminfo"}),
            ],
        )
        self.assertEqual(len(client.calls), 2)

    def test_gemma_model_reuses_prior_tool_results_for_short_followup(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "lscpu"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "This machine has 12 logical CPUs."},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "There are 12 logical CPUs."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "command": "lscpu",
                "stdout": "Architecture: x86_64\nCPU(s): 12\nModel name: Intel(R) Core(TM) i7 Test CPU\n",
                "stderr": "",
                "returncode": 0,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        first = agent.run_turn("what is the configuration of this machine?")
        second = agent.run_turn("how many cpus are there?")
        self.assertIn("12 logical cpus", first.content.lower())
        self.assertIn("12 logical cpus", second.content.lower())
        self.assertEqual(registry.called, [("bash", {"command": "lscpu"})])
        self.assertEqual(len(client.calls), 3)

    def test_gemma_model_can_choose_bash_for_brief_cpu_question(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "nproc"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "12"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {"ok": True, "command": "nproc", "stdout": "12\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("how many cpus are there?")
        self.assertEqual(result.content.strip(), "12")
        self.assertEqual(registry.called, [("bash", {"command": "nproc"})])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            {tool["function"]["name"] for tool in client.calls[0]["tools"]},
            {
                "read_file",
                "list_files",
                "stat_path",
                "make_directory",
                "delete_path",
                "replace_in_file",
                "write_file",
                "append_file",
                "bash",
                "search_web",
                "fetch_url",
            },
        )

    def test_gemma_model_retries_after_generic_access_refusal_and_then_uses_tool(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "I do not have access to information about the machine you are currently running on.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "nproc"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "12"},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {"ok": True, "command": "nproc", "stdout": "12\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("how many cpus are there?")
        self.assertEqual(result.content.strip(), "12")
        self.assertEqual(registry.called, [("bash", {"command": "nproc"})])
        self.assertEqual(len(client.calls), 3)
        retry_messages = client.calls[2]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "do not answer with a generic access refusal" in str(message.get("content", "")).lower()
                and "nproc" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_retries_after_defensive_explicit_file_review_reply(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "Without executing the code or having more context on surrounding modules, I cannot definitively find bugs in this file.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "- High: `agent.py` contains a bare `except:`.\n- Medium: `agent.py` uses placeholder `pass` paths.\n- Medium: `agent.py` has dense control flow that deserves regression coverage.",
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": "agent.py",
                    "content": "def run():\n    try:\n        pass\n    except:\n        pass\n",
                    "total_lines": 5,
                    "has_more": False,
                    "truncated": False,
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the python code in this file: agent.py and tell me if you find bugs.")
        self.assertIn("bare `except:`", result.content)
        self.assertEqual(registry.called, [("read_file", {"path": "agent.py"})])
        retry_messages = client.calls[-1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "do not stop at generic uncertainty" in str(message.get("content", "")).lower()
                and "review this file directly" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_finalizes_review_when_followup_read_file_has_no_path(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": "."}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {}}},
                        ],
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "agent.py", "type": "file"},
                    ],
                }
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "def run_turn():\n" + "    if True:\n        pass\n" * 20,
                    "total_lines": 240,
                    "has_more": False,
                    "truncated": False,
                }
            return {"ok": False, "error": "unexpected"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("review agent.py for vulnerabilities and security issues")
        self.assertIn("agent.py", result.content)
        self.assertEqual(
            registry.called,
            [
                ("list_files", {"path": "."}),
                ("read_file", {"path": "agent.py"}),
            ],
        )
        self.assertEqual(len(client.calls), 3)

    def test_gemma_model_continues_reading_explicit_file_review_after_import_only_first_chunk(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "The provided code in `agent.py` is a large import block and does not contain executable logic that can be directly analyzed for bugs.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py", "start_line": 221}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "- High: `agent.py` is a central module with dense control flow.\n- Medium: retries and fallback paths deserve tight regression coverage.",
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": True}
            if arguments.get("start_line") == 221:
                return {
                    "ok": True,
                    "path": "agent.py",
                    "content": "def run():\n    pass\n",
                    "total_lines": 400,
                    "has_more": False,
                    "truncated": False,
                }
            return {
                "ok": True,
                "path": "agent.py",
                "content": "from x import y\nfrom a import b\n",
                "total_lines": 400,
                "has_more": True,
                "truncated": True,
                "next_start_line": 221,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("analyze the python code in this file: agent.py and tell me if you find bugs.")
        self.assertIn("central module", result.content)
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "agent.py"}),
                ("read_file", {"path": "agent.py", "start_line": 221}),
            ],
        )
        retry_messages = client.calls[2]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "continue reading the same file" in str(message.get("content", "")).lower()
                and "start_line=221" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_retries_when_explicit_file_review_collapses_into_generic_summary(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "The file `agent.py` contains methods for message handling, compaction, and running turns.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "- Medium: the loop mixes routing, retries, and post-tool handling in one hot path.\n- Low: this file deserves regression coverage around stop conditions.",
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "path": "agent.py",
                "content": "def run_turn():\n    pass\n",
                "total_lines": 40,
                "has_more": False,
                "truncated": False,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("analyze the python code in this file: agent.py and tell me if you find bugs.")
        self.assertIn("hot path", result.content)
        self.assertEqual(registry.called, [("read_file", {"path": "agent.py"})])
        self.assertEqual(len(client.calls), 3)
        retry_messages = client.calls[2]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "do not summarize the file structure" in str(message.get("content", "")).lower()
                and "perform a bug and risk review" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_reserves_second_explicit_file_review_retry_for_review_not_extra_read(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "The provided code snippet only contains imports and does not contain executable logic.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "agent.py", "start_line": 221}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "The file `agent.py` contains methods for message handling and running turns.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "- Medium: the turn loop combines multiple control concerns in one path.",
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if arguments.get("start_line") == 221:
                return {
                    "ok": True,
                    "path": "agent.py",
                    "content": "def run_turn():\n    pass\n",
                    "total_lines": 400,
                    "has_more": True,
                    "truncated": True,
                    "next_start_line": 341,
                }
            return {
                "ok": True,
                "path": "agent.py",
                "content": "from x import y\nfrom a import b\n",
                "total_lines": 400,
                "has_more": True,
                "truncated": True,
                "next_start_line": 221,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("analyze the python code in this file: agent.py and tell me if you find bugs.")
        self.assertIn("control concerns", result.content)
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "agent.py"}),
                ("read_file", {"path": "agent.py", "start_line": 221}),
            ],
        )
        retry_messages = client.calls[2]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "continue reading the same file" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )
        review_retry_messages = client.calls[4]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "do not summarize the file structure" in str(message.get("content", "")).lower()
                for message in review_retry_messages
            )
        )

    def test_gemma_model_handles_read_only_tool_hesitation_by_local_listing(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {"ok": True, "entries": [{"path": "README.md", "type": "file"}, {"path": "src", "type": "dir"}]}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("what does this working directory contain?")
        self.assertIn("readme.md", result.content.lower())
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_gemma_model_does_not_add_identity_prompt_for_self_description(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 180_000_000,
                    "message": {"content": "Sono orbit, l'assistente locale di questa CLI."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("chi sei?")
        self.assertIn("assistente", result.content.lower())
        self.assertEqual(len(client.calls), 1)
        system_messages = [m for m in client.calls[0]["messages"] if m.get("role") == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertIn("underlying model name", system_messages[0]["content"])
        self.assertIn("without claiming that your personal name is Orbit", system_messages[0]["content"])

    def test_gemma_model_retries_after_guarded_shell_operator_failure(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "lscpu && free -h && cat /proc/cpuinfo"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "Non sono riuscito a eseguire il comando combinato e ho bisogno di sapere se vuoi che proceda separatamente.",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "lscpu"}}},
                            {"function": {"name": "bash", "arguments": {"command": "free -h"}}},
                            {"function": {"name": "bash", "arguments": {"command": "cat /proc/cpuinfo"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "CPU and memory information collected successfully."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            command = arguments.get("command")
            if command == "lscpu && free -h && cat /proc/cpuinfo":
                return {"ok": False, "error": "shell operators are not allowed"}
            return {"ok": True, "command": command, "stdout": "ok\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("verifica le risorse locali di questa macchina")
        self.assertIn("collected successfully", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "lscpu && free -h && cat /proc/cpuinfo"}),
                ("bash", {"command": "lscpu"}),
                ("bash", {"command": "free -h"}),
                ("bash", {"command": "cat /proc/cpuinfo"}),
            ],
        )
        self.assertEqual(len(client.calls), 4)
        retry_messages = client.calls[2]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "retry now using separate safe bash calls" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_redirects_machine_resource_request_away_from_list_files_tool(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "uname -srm"}}},
                            {"function": {"name": "bash", "arguments": {"command": "free -h"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "System and memory information collected successfully."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {"ok": True, "entries": [{"path": "README.md", "type": "file"}]}
            return {"ok": True, "command": arguments.get("command"), "stdout": "ok\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("verifica le risorse locali di questa macchina")
        self.assertIn("collected successfully", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "uname -srm"}),
                ("bash", {"command": "free -h"}),
            ],
        )
        retry_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "do not use list_files or read_file for this turn" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_recovers_from_blocked_bash_operator_with_tool_retry(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "uname -a && free -h"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "uname -srm"}}},
                            {"function": {"name": "bash", "arguments": {"command": "free -h"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "System and memory information collected successfully."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            command = arguments.get("command")
            if command == "uname -a && free -h":
                return {"ok": False, "error": "shell operators are not allowed"}
            return {"ok": True, "command": command, "stdout": "ok\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("verifica le risorse locali di questa macchina")
        self.assertIn("collected successfully", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "uname -a && free -h"}),
                ("bash", {"command": "uname -srm"}),
                ("bash", {"command": "free -h"}),
            ],
        )
        retry_messages = client.calls[1]["messages"]
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "retry now with separate safe bash calls" in str(message.get("content", "")).lower()
                for message in retry_messages
            )
        )

    def test_gemma_model_redirects_machine_resource_hesitation_after_list_files(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 10,
                    "eval_duration": 70_000_000,
                    "total_duration": 210_000_000,
                    "message": {
                        "content": "I have listed the files in the current directory. What specific resources would you like me to verify? For example, source code or documentation?",
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "uname -srm"}}},
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "System information collected successfully."},
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {"ok": True, "entries": [{"path": "README.md", "type": "file"}]}
            return {"ok": True, "command": arguments.get("command"), "stdout": "ok\n", "stderr": "", "returncode": 0}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("verifica le risorse locali di questa macchina")
        self.assertIn("collected successfully", result.content.lower())
        self.assertEqual(
            registry.called,
            [
                ("bash", {"command": "uname -srm"}),
            ],
        )

    def test_gemma_model_first_prompt_describes_tool_capabilities_by_category(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        agent = AgentLoop(client=client, registry=FakeRegistry())

        system_prompt = agent.messages[0]["content"]

        self.assertIn("use bash for machine and environment inspection", system_prompt.lower())
        self.assertIn("use list_files for workspace structure", system_prompt.lower())
        self.assertIn("prefer bash over workspace file tools", system_prompt.lower())
        self.assertIn("use df for filesystem free space at the requested path or mount point, including /", system_prompt.lower())
        self.assertIn("use du only for directory size", system_prompt.lower())
        self.assertIn("never guess file contents", system_prompt.lower())
        self.assertIn("prefer read_file before editing", system_prompt.lower())
        self.assertIn("use bash only for bounded inspection and safe commands", system_prompt.lower())

    def test_gemma_model_first_adds_post_tool_guidance_for_url_inspection(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e2b"
        agent = AgentLoop(client=client, registry=FakeRegistry())

        prompt = agent._model_first_post_tool_prompt(type("Route", (), {"intent_class": "url_inspection", "intent": "url_inspection"})())

        self.assertIsNotNone(prompt)
        self.assertIn("fetched chunk only", prompt.lower())
        self.assertIn("specific entity", prompt.lower())
        self.assertIn("start_char=next_start_char", prompt.lower())

    def test_url_inspection_fetches_next_chunk_when_page_is_truncated(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "fetch_url", "arguments": {"url": "https://example.com"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "fetch_url",
                                    "arguments": {"url": "https://example.com", "start_char": 40},
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {"content": "CERT-AGID is mentioned in the fetched page content."},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "fetch_url":
                return {"ok": False, "error": "unexpected"}
            start_char = arguments.get("start_char", 0)
            if start_char == 0:
                return {
                    "ok": True,
                    "url": arguments["url"],
                    "final_url": arguments["url"],
                    "status_code": 200,
                    "content_type": "text/html",
                    "title": "Example",
                    "text": "A" * 40,
                    "highlights": ["A" * 40],
                    "links": [],
                    "start_char": 0,
                    "end_char": 40,
                    "total_chars": 80,
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "next_start_char": 40,
                    "has_more": True,
                    "truncated": True,
                }
            return {
                "ok": True,
                "url": arguments["url"],
                "final_url": arguments["url"],
                "status_code": 200,
                "content_type": "text/html",
                "title": "Example",
                "text": "B" * 40,
                "highlights": ["B" * 40],
                "links": [],
                "start_char": 40,
                "end_char": 80,
                "total_chars": 80,
                "chunk_index": 2,
                "chunk_count": 2,
                "next_start_char": None,
                "has_more": False,
                "truncated": False,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("summarize this article: https://example.com and tell me what it says about cert-agid")
        self.assertIn("CERT-AGID", result.content)
        self.assertEqual(
            registry.called,
            [
                ("fetch_url", {"url": "https://example.com"}),
                ("fetch_url", {"url": "https://example.com", "start_char": 40}),
            ],
        )

    def test_config_file_read_request_in_italian_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": "config.json",
                "content": '{\n  "mode": "orbit",\n  "enabled": true\n}\n',
                "truncated": False,
                "has_more": False,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Usa lo strumento per leggere il contenuto di un file di configurazione come config.json.")
        self.assertIn('"mode": "orbit"', result.content)
        self.assertEqual(registry.called, [("read_file", {"path": "config.json"})])
        self.assertEqual(client.calls, [])

    def test_config_file_read_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "read_file":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": "config.json",
                "content": '{\n  "mode": "orbit",\n  "enabled": true\n}\n',
                "truncated": False,
                "has_more": False,
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Use the tool to read the contents of a configuration file such as config.json.")
        self.assertIn('"enabled": true', result.content)
        self.assertEqual(registry.called, [("read_file", {"path": "config.json"})])
        self.assertEqual(client.calls, [])

    def test_tool_calls_explanation_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Spiegami in modo semplice cosa sono le tool calls in un LLM.")
        self.assertIn("tool calls", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_tool_calls_explanation_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Explain in simple terms what tool calls are in an LLM.")
        self.assertIn("tool calls", result.content.lower())
        self.assertIn("runtime", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_log_analysis_strategy_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Ragiona passo-passo: cosa faresti se ti chiedessi di analizzare un file di log e trovare errori critici?")
        self.assertIn("chunk bounded", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_tool_message_explanation_mentions_email_or_message(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Descrivi come un modello LLM potrebbe usare uno strumento per inviare una email o un messaggio.")
        lowered = result.content.lower()
        self.assertTrue("email" in lowered or "messaggi" in lowered or "messaggio" in lowered)
        self.assertIn("tool", lowered)
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_simulated_multi_tool_conversation_in_italian_prefers_flow_answer(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn(
            "Prova a simulare una conversazione con me in cui usi più strumenti, come ricerca, lettura file e calcolo, per rispondere a una domanda complessa, spiegando i tuoi passi."
        )
        lowered = result.content.lower()
        self.assertIn("flusso di esempio", lowered)
        self.assertIn("tool web", lowered)
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])

    def test_weather_request_can_seed_web_search_and_model_synthesizes(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "il meteo attuale a firenze",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "A Firenze oggi il tempo è sereno e mite, con temperatura intorno ai 28°C e senza pioggia prevista."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": arguments["query"],
                "results": [
                    {"title": "Meteo Firenze oggi", "url": "https://weather.example/florence", "snippet": "Sunny, 28C."},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Usa lo strumento per ottenere il meteo attuale a Firenze.")
        self.assertIn("firenze", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "il meteo attuale a firenze", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_generic_search_request_with_tool_preface_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "tool calling in modelli open-source italiani",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Tool calling in open-source models is typically implemented through structured tool definitions and a small execution loop."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": arguments["query"],
                "results": [
                    {"title": "Result A", "url": "https://example.com/a", "snippet": "tool calling overview"},
                    {"title": "Result B", "url": "https://example.com/b", "snippet": "open-source models"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Usa lo strumento per cercare informazioni su tool calling in modelli open-source italiani.")
        self.assertIn("tool calling in open-source models", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "tool calling in modelli open-source italiani", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_generic_search_request_with_gemma_model_is_model_driven(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "Mario Nobile",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Mario Nobile appears in public-sector digital leadership roles, including AgID references."
                    },
                },
            ]
        )
        client.model = "gemma4:e2b"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": arguments["query"],
                "results": [
                    {
                        "title": "Mario Nobile - Governo Italiano",
                        "url": "https://www.governo.it/mario-nobile",
                        "snippet": "Direttore generale dell'Agenzia per l'Italia Digitale.",
                    }
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("search online for information about Mario Nobile and summarize it")
        self.assertIn("agid", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "Mario Nobile", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_documentation_search_request_in_english_uses_model_for_summary_after_search(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "tool calling Granite 2 Ollama on-premise",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Tool calling in Ollama is suitable for on-premise use because the runtime and the model can stay local. Granite 2 support should still be validated against the specific Ollama documentation."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": arguments["query"],
                "results": [
                    {
                        "title": "Tool calling - Ollama",
                        "url": "https://docs.ollama.com/capabilities/tool-calling",
                        "snippet": "Official Ollama documentation for tool calling.",
                    }
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn(
            "Search for documentation on tool calling with Granite 2 and Ollama, summarize it, and tell me whether it is suitable for an on-premise environment."
        )
        self.assertIn("suitable for on-premise use", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "tool calling Granite 2 Ollama on-premise", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_search_and_summary_request_in_italian_uses_model_for_summary_after_search(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "Mario Nobile",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Mario Nobile risulta associato all'Agenzia per l'Italia Digitale e a ruoli pubblici nel digitale, secondo i risultati trovati."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": arguments["query"],
                "results": [
                    {
                        "title": "Mario Nobile - Governo Italiano",
                        "url": "https://www.governo.it/mario-nobile",
                        "snippet": "Direttore generale dell'Agenzia per l'Italia Digitale.",
                    },
                    {
                        "title": "Chi è il nuovo direttore dell'Agenzia per l'Italia digitale",
                        "url": "https://www.wired.it/article/agid-direttore-generale-mario-nobile/",
                        "snippet": "Approfondimento sul nuovo direttore generale AgID.",
                    },
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("cerca online informazioni su Mario Nobile e fammi una sintesi")
        self.assertIn("agenzia per l'italia digitale", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "Mario Nobile", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_simple_calculation_request_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Usa lo strumento per calcolare 12345÷345 e poi moltiplicare il risultato per 345.")
        self.assertEqual(result.content, "12345")
        self.assertEqual(client.calls, [])

    def test_simple_calculation_request_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Use the tool to calculate 12345÷345 and then multiply the result by 345.")
        self.assertEqual(result.content, "12345")
        self.assertEqual(client.calls, [])

    def test_email_send_request_without_tool_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Usa lo strumento per inviare una notifica o email a gianni@example.org con oggetto Test strumenti.")
        self.assertIn("non e` disponibile", result.content.lower())
        self.assertEqual(client.calls, [])

    def test_email_send_request_without_tool_in_english_can_finalize_locally(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("Use the tool to send a notification or email to gianni@example.org with subject Test tools.")
        self.assertIn("not available", result.content.lower())
        self.assertEqual(client.calls, [])

    def test_generic_web_search_request_can_finalize_locally_from_search_results(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "Mario Nobile",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Mario Nobile risulta associato a ruoli pubblici nel digitale, in particolare all'Agenzia per l'Italia Digitale."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": "Mario Nobile",
                "results": [
                    {
                        "title": "Mario Nobile - Governo Italiano",
                        "url": "https://www.governo.it/mario-nobile",
                        "snippet": "Direttore generale dell'Agenzia per l'Italia Digitale.",
                    },
                    {
                        "title": "Profilo LinkedIn di Mario Nobile",
                        "url": "https://www.linkedin.com/in/mario-nobile",
                        "snippet": "Esperienze e incarichi pubblici e digitali.",
                    },
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("mi cerchi online informazioni su Mario Nobile?")
        self.assertIn("agenzia per l'italia digitale", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "Mario Nobile", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_entity_lookup_request_can_finalize_locally_from_search_results(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {
                                        "query": "guelfoweb",
                                        "max_results": 3,
                                    },
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "guelfoweb appears to be Gianni Amato's GitHub profile and personal site."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "query": "guelfoweb",
                "results": [
                    {
                        "title": "guelfoweb (Gianni Amato) · GitHub",
                        "url": "https://github.com/guelfoweb",
                        "snippet": "GitHub profile for guelfoweb / Gianni Amato.",
                    },
                    {
                        "title": "Random notes | guelfoweb",
                        "url": "https://guelfoweb.com/",
                        "snippet": "Personal site and notes.",
                    },
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("dimi chi è guelfoweb")
        self.assertIn("github profile", result.content.lower())
        self.assertEqual(registry.called, [("search_web", {"query": "guelfoweb", "max_results": 3})])
        self.assertEqual(len(client.calls), 2)

    def test_current_factual_title_only_can_finalize_locally_from_fetch(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "Example Domain",
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("On the web, fetch this URL and tell me the page title only: https://example.com")
        self.assertEqual(result.content, "Example Domain")
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(client.calls, [])

    def test_explicit_site_check_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Il sito riporta note e articoli di guelfoweb su sicurezza, Linux e sviluppo."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://guelfoweb.com/",
                "final_url": "https://guelfoweb.com/",
                "title": "Random notes | guelfoweb",
                "text": "Random notes and articles by guelfoweb about security, Linux and development.",
                "highlights": ["Random notes and articles by guelfoweb about security, Linux and development."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("controlla cosa riporta il sito: https://guelfoweb.com/")
        self.assertIn("sicurezza, Linux e sviluppo", result.content)
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://guelfoweb.com/", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_explicit_site_check_request_in_english_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "The page says the domain is for documentation examples and usage notes."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("check what this site says: https://example.com")
        self.assertIn("documentation examples", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_open_link_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "La pagina è un dominio di esempio usato per documentazione."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("aprimi questo link: https://example.com")
        self.assertIn("dominio di esempio", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_read_page_request_in_english_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "This page explains that the domain is reserved for documentation examples."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("read this page: https://example.com")
        self.assertIn("documentation examples", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_summarize_site_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Il sito è una pagina di esempio con istruzioni e note di documentazione."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("riassumi questo sito: https://example.com")
        self.assertIn("documentazione", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_colloquial_page_question_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "La pagina descrive un dominio usato per esempi di documentazione."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("cosa c'è scritto qui: https://example.com")
        self.assertIn("documentazione", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_colloquial_page_question_in_english_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "The page is a documentation example site with a short explanatory note."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("what does this page say? https://example.com")
        self.assertIn("documentation example", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_summary_link_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "Il link conduce a una pagina di esempio con note di documentazione."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("dammi un riassunto di questo link: https://example.com")
        self.assertIn("documentazione", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_whats_on_page_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "La pagina contiene una breve descrizione del sito e delle sue note."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("che c'è su questa pagina? https://example.com")
        self.assertIn("descrizione", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_read_site_request_in_italian_routes_to_fetch_url_and_uses_model_to_summarize(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 80_000_000,
                    "total_duration": 250_000_000,
                    "message": {
                        "content": "La pagina offre una breve nota informativa e un riepilogo del dominio."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            return {
                "ok": True,
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "text": "This domain is for use in documentation examples without prior coordination or asking for permission.",
                "highlights": ["This domain is for use in documentation examples without prior coordination or asking for permission."],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("mi leggi questo sito? https://example.com")
        self.assertIn("riepilogo", result.content.lower())
        self.assertEqual(registry.called, [("fetch_url", {"url": "https://example.com", "max_links": 0})])
        self.assertEqual(len(client.calls), 2)

    def test_explicit_image_request_in_english_uses_vision_without_tools(self) -> None:
        client = VisionFakeClient(
            [
                {
                    "prompt_eval_count": 24,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 10,
                    "eval_duration": 80_000_000,
                    "total_duration": 240_000_000,
                    "message": {"content": "The image shows a red warning banner."},
                }
            ]
        )
        registry = FakeRegistry()
        _write_png(registry.workdir / "screen.png", 255, 0, 0)

        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("describe the image screen.png in one short line")

        self.assertEqual(result.content, "The image shows a red warning banner.")
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tools"], [])
        self.assertEqual(client.calls[0]["messages"][1]["images"][0][:8], "iVBORw0K")

    def test_explicit_image_request_in_italian_uses_vision_without_tools(self) -> None:
        client = VisionFakeClient(
            [
                {
                    "prompt_eval_count": 24,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 12,
                    "eval_duration": 90_000_000,
                    "total_duration": 260_000_000,
                    "message": {"content": "L'immagine mostra una finestra con un errore di accesso."},
                }
            ]
        )
        registry = FakeRegistry()
        _write_png(registry.workdir / "schermata.png", 255, 0, 0)

        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("descrivi l'immagine 'schermata.png' in una riga")

        self.assertIn("errore di accesso", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tools"], [])

    def test_compare_two_explicit_images_in_english_attaches_both(self) -> None:
        client = VisionFakeClient(
            [
                {
                    "prompt_eval_count": 28,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 14,
                    "eval_duration": 90_000_000,
                    "total_duration": 260_000_000,
                    "message": {"content": "The first image is blue, while the second image is red."},
                }
            ]
        )
        registry = FakeRegistry()
        _write_png(registry.workdir / "cmp-blue.png", 0, 0, 255)
        _write_png(registry.workdir / "cmp-red.png", 255, 0, 0)

        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("compare cmp-blue.png and cmp-red.png and tell me the differences")

        self.assertIn("first image is blue", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]["messages"][1]["images"]), 2)

    def test_compare_two_explicit_images_in_italian_attaches_both(self) -> None:
        client = VisionFakeClient(
            [
                {
                    "prompt_eval_count": 28,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 14,
                    "eval_duration": 90_000_000,
                    "total_duration": 260_000_000,
                    "message": {"content": "La prima immagine è blu, mentre la seconda è rossa."},
                }
            ]
        )
        registry = FakeRegistry()
        _write_png(registry.workdir / "blu.png", 0, 0, 255)
        _write_png(registry.workdir / "rosso.png", 255, 0, 0)

        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("confronta due immagini: blu.png e rosso.png e dimmi le differenze")

        self.assertIn("prima immagine", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]["messages"][1]["images"]), 2)

    def test_explicit_image_request_without_vision_support_returns_readable_error(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        _write_png(registry.workdir / "screen.png", 255, 0, 0)

        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("describe the image screen.png")

        self.assertIn("does not advertise vision support", result.content.lower())
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls, [])


    def test_directory_request_seeds_list_files_for_current_workdir(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "The current directory contains source files and tests."},
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("what does this directory contain?")
        self.assertEqual(result.content, "The current directory contains source files and tests.")
        self.assertEqual(registry.called[0], ("list_files", {"path": ".", "recursive": False, "max_entries": 12}))

    def test_directory_request_retries_when_model_claims_no_local_access(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "I do not have access to local files or directories. Please provide the path."
                    },
                },
                {
                    "prompt_eval_count": 24,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 6,
                    "eval_duration": 60_000_000,
                    "total_duration": 240_000_000,
                    "message": {"content": "This directory contains the listed entries from the tool result."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("what does this directory contain?")
        self.assertEqual(result.content, "This directory contains the listed entries from the tool result.")
        self.assertEqual(registry.called[0], ("list_files", {"path": ".", "recursive": False, "max_entries": 12}))

    def test_directory_request_can_return_only_directories_in_italian(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "list_files":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": ".",
                "recursive": False,
                "count": 4,
                "dir_count": 2,
                "file_count": 2,
                "truncated": False,
                "entries": [
                    {"path": "src", "type": "dir"},
                    {"path": "tests", "type": "dir"},
                    {"path": "README.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("quali cartelle, non file, ci sono in questa directory di lavoro?")
        self.assertEqual(result.content, "src, tests")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_directory_request_with_cartelle_only_can_return_directories_in_italian(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "list_files":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": ".",
                "recursive": False,
                "count": 4,
                "dir_count": 2,
                "file_count": 2,
                "truncated": False,
                "entries": [
                    {"path": "src", "type": "dir"},
                    {"path": "tests", "type": "dir"},
                    {"path": "README.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("quali cartelle ci sono in questa directory?")
        self.assertEqual(result.content, "src, tests")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_directory_request_can_report_no_subdirectories_in_english(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "list_files":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": ".",
                "recursive": False,
                "count": 2,
                "dir_count": 0,
                "file_count": 2,
                "truncated": False,
                "entries": [
                    {"path": "README.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("which folders, not files, are in this working directory?")
        self.assertEqual(result.content, "There are no subdirectories in the current working directory.")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_directory_request_can_return_mixed_entries_in_italian(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name != "list_files":
                return {"ok": False, "error": "unexpected"}
            return {
                "ok": True,
                "path": ".",
                "recursive": False,
                "count": 5,
                "dir_count": 2,
                "file_count": 3,
                "truncated": False,
                "entries": [
                    {"path": "src", "type": "dir"},
                    {"path": "tests", "type": "dir"},
                    {"path": "README.md", "type": "file"},
                    {"path": "pyproject.toml", "type": "file"},
                    {"path": "AGENTS.md", "type": "file"},
                ],
            }

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=3)
        result = agent.run_turn("cosa contiene questa directory?")
        self.assertEqual(result.content, "src, tests, README.md, pyproject.toml, AGENTS.md")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": False, "max_entries": 12})])
        self.assertEqual(client.calls, [])

    def test_chat_only_mode_ignores_tool_calls(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": '{"name":"read_file","arguments":{"path":"README.md"}}',
                    },
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2, tools_enabled=False)
        result = agent.run_turn("read README.md")
        self.assertEqual(result.content, '{"name":"read_file","arguments":{"path":"README.md"}}')
        self.assertEqual(registry.called, [])
        self.assertEqual(client.calls[0]["tools"], [])
        self.assertEqual(result.status.warning, "chat-only: model does not advertise tool support")

    def test_show_thinking_streams_trace_before_final_answer(self) -> None:
        client = FakeClient(
            [
                [
                    {"message": {"thinking": "step 1 "}},
                    {
                        "message": {"thinking": "step 2", "content": "done"},
                        "prompt_eval_count": 10,
                        "prompt_eval_duration": 100_000_000,
                        "eval_count": 1,
                        "eval_duration": 10_000_000,
                        "total_duration": 200_000_000,
                    },
                ]
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2, think_mode="on", show_thinking=True)
        seen = []
        result = agent.run_turn("hello there", on_event=seen.append)
        self.assertEqual(result.content, "done")
        self.assertEqual(client.calls[0]["think"], True)
        self.assertTrue(client.calls[0]["stream"])
        event_names = [type(item).__name__ for item in seen]
        self.assertIn("ThinkingStartEvent", event_names)
        self.assertIn("ThinkingChunkEvent", event_names)
        self.assertIn("ThinkingEndEvent", event_names)

    def test_falls_back_when_model_does_not_support_thinking(self) -> None:
        class ThinkingFailClient(FakeClient):
            def chat(self, *, messages, tools, options, think=None, stream=False):
                self.calls.append({"messages": messages, "tools": tools, "options": options, "think": think, "stream": stream})
                if stream and think is True:
                    raise OllamaError('"demo-model" does not support thinking (status code: 400)')
                return {
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 1,
                    "eval_duration": 10_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                }

        client = ThinkingFailClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2, think_mode="on", show_thinking=True)
        seen = []
        result = agent.run_turn("hello there", on_event=seen.append)
        self.assertEqual(result.content, "done")
        self.assertEqual(client.calls[0]["think"], True)
        self.assertTrue(client.calls[0]["stream"])
        self.assertEqual(client.calls[1]["think"], False)
        self.assertFalse(client.calls[1]["stream"])
        event_names = [type(item).__name__ for item in seen]
        self.assertIn("ThinkingUnavailableEvent", event_names)

    def test_fallback_json_tool_call(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {"content": '{"name":"read_file","arguments":{"path":"README.md"}}'},
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read it")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("read_file", {"path": "README.md"})])

    def test_fallback_json_tool_call_from_fenced_block(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": (
                            "Per elencare i file puoi usare questo comando:\n\n"
                            "```json\n"
                            '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read it")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("read_file", {"path": "README.md"})])

    def test_fallback_json_tool_call_from_fenced_block_with_multiline_string(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "# REPORT\\n\\nLine one\nLine two"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("write it")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [("write_file", {"path": "REPORT.md", "content": "# REPORT\n\nLine one\nLine two"})],
        )

    def test_file_edit_placeholder_reply_is_repaired_into_real_write(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "# REPORT\\n\\n## README Summary\\n\\n" + "<tool_response.content>"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 160,
                    "prompt_eval_duration": 2_200_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_300_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "README body",
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("read README.md and then update REPORT.md adding a README Summary section")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "REPORT.md", "content": "# REPORT\n\n## README Summary\n\nREADME body"}),
            ],
        )

    def test_file_edit_generic_content_placeholder_is_repaired(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "# REPORT.md\\n\\n## README Summary\\n\\n" + "<readme_content>"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 160,
                    "prompt_eval_duration": 2_200_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_300_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "README body",
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("read README.md and then update REPORT.md adding a README Summary section")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "REPORT.md", "content": "# REPORT.md\n\n## README Summary\n\nREADME body"}),
            ],
        )

    def test_file_edit_placeholder_from_unclosed_json_fence_is_repaired(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "append_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "\\n\\n## README Summary\\n\\n" + "<tool_response.content>"\n'
                            "  }\n"
                            "}\n"
                        )
                    },
                },
                {
                    "prompt_eval_count": 160,
                    "prompt_eval_duration": 2_200_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_300_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file":
                return {
                    "ok": True,
                    "path": arguments["path"],
                    "content": "# Project README\n\nThis project demonstrates a sample workflow.",
                    "truncated": False,
                    "has_more": False,
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("read README.md and then update REPORT.md adding a README Summary section")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("append_file", {"path": "REPORT.md", "content": "\n\n## README Summary\n\n# Project README\n\nThis project demonstrates a sample workflow."}),
            ],
        )

    def test_file_edit_placeholder_after_successful_write_finalizes_locally(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "write_file", "arguments": {"path": "REPORT.md", "content": "done"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "<tool_response.content>"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "write_file":
                return {"ok": True, "path": arguments["path"]}
            return {"ok": True, "content": "hello"}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("update REPORT.md")
        self.assertEqual(result.content, "Updated `REPORT.md` and added the requested follow-up section.")
        self.assertEqual(registry.called, [("write_file", {"path": "REPORT.md", "content": "done"})])

    def test_file_edit_fake_tool_response_after_successful_write_finalizes_locally(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "append_file", "arguments": {"path": "REPORT.md", "content": "done"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true}</tool_response>"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "append_file":
                return {"ok": True, "path": arguments["path"]}
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("update REPORT.md")
        self.assertEqual(result.content, "Updated `REPORT.md` and added the requested follow-up section.")
        self.assertEqual(registry.called, [("append_file", {"path": "REPORT.md", "content": "done"})])

    def test_file_edit_post_write_plain_text_with_unknown_tool_finalizes_locally(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "write_file",
                                    "arguments": {"path": "REPORT2.md", "content": "# Report 2\n\n## Summary\n\nDone."},
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": "Il file `REPORT2.md` è stato creato con successo. Seleziona il comando `open_file` per visualizzarlo e aggiornarlo come necessario."
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "write_file":
                return {"ok": True, "path": arguments["path"]}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Crea un file REPORT2.md con un riassunto di questa cartella, poi riaprilo e aggiungi una sezione finale con i prossimi passi.")
        self.assertEqual(result.content, "Created `REPORT2.md`.")
        self.assertEqual(
            registry.called,
            [("write_file", {"path": "REPORT2.md", "content": "# Report 2\n\n## Summary\n\nDone."})],
        )

    def test_file_edit_post_write_verbose_content_is_collapsed_to_confirmation(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "write_file",
                                    "arguments": {"path": "fibonacci.py", "content": "def fibonacci(n):\n    return n\n"},
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "I have written a Python function for the Fibonacci sequence and saved it to a file named `fibonacci.py`.\n\n"
                            "Here is the content of the file:\n"
                            "```python\n"
                            "def fibonacci(n):\n"
                            "    return n\n"
                            "```"
                        )
                    },
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "write_file":
                return {"ok": True, "path": arguments["path"]}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("write a file fibonacci.py containing a python function")
        self.assertEqual(result.content, "Created `fibonacci.py`.")
        self.assertEqual(registry.called, [("write_file", {"path": "fibonacci.py", "content": "def fibonacci(n):\n    return n\n"})])

    def test_guarded_write_adds_prompt_and_reads_target_file(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "write_file", "arguments": {"path": "REPORT.md", "content": "x"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=3)

        def call_tool(*, registry, name, arguments):
            if name == "write_file":
                return ({"ok": False, "error": "guarded", "_guarded": True, "path": "REPORT.md"}, 0)
            if name == "read_file":
                return (
                    {
                        "ok": True,
                        "path": "REPORT.md",
                        "content": "current report",
                        "truncated": False,
                        "has_more": False,
                    },
                    0,
                )
            return ({"ok": True}, 0)

        agent._tool_policy.call_tool = call_tool
        result = agent.run_turn("update REPORT.md")
        self.assertEqual(result.content, "done")
        system_messages = [item.get("content", "") for item in agent.messages if item.get("role") == "system"]
        self.assertTrue(any("Read `REPORT.md` first" in content for content in system_messages))
        tool_messages = [item for item in agent.messages if item.get("role") == "tool" and item.get("tool_name") == "read_file"]
        self.assertTrue(any("current report" in str(item.get("content", "")) for item in tool_messages))

    def test_file_edit_placeholder_repair_prefers_source_read_over_guarded_target_read(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "# REPORT\\n\\n## README Summary\\n\\n" + "<tool_response.content>"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 145,
                    "prompt_eval_duration": 2_050_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_150_000_000,
                    "message": {
                        "content": (
                            "```json\n"
                            "{\n"
                            '  "name": "write_file",\n'
                            '  "arguments": {\n'
                            '    "path": "REPORT.md",\n'
                            '    "content": "# REPORT\\n\\n## README Summary\\n\\n" + "<tool_response.content>"\n'
                            "  }\n"
                            "}\n"
                            "```"
                        )
                    },
                },
                {
                    "prompt_eval_count": 160,
                    "prompt_eval_duration": 2_200_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_300_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=6)

        call_counts = {"write_file": 0}

        def call_tool(*, registry, name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file" and arguments["path"] == "README.md":
                return (
                    {
                        "ok": True,
                        "path": "README.md",
                        "content": "README body",
                        "truncated": False,
                        "has_more": False,
                    },
                    0,
                )
            if name == "read_file" and arguments["path"] == "REPORT.md":
                return (
                    {
                        "ok": True,
                        "path": "REPORT.md",
                        "content": "# REPORT\n\n## Existing\n\nBase report.",
                        "truncated": False,
                        "has_more": False,
                    },
                    0,
                )
            if name == "write_file":
                call_counts["write_file"] += 1
                if call_counts["write_file"] == 1:
                    return ({"ok": False, "error": "guarded", "_guarded": True, "path": "REPORT.md"}, 0)
                return ({"ok": True, "path": "REPORT.md"}, 0)
            return ({"ok": True}, 0)

        agent._tool_policy.call_tool = call_tool
        result = agent.run_turn("read README.md and then update REPORT.md adding a README Summary section")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "REPORT.md", "content": "# REPORT\n\n## README Summary\n\nREADME body"}),
                ("read_file", {"path": "REPORT.md"}),
                ("write_file", {"path": "REPORT.md", "content": "# REPORT\n\n## README Summary\n\nREADME body"}),
            ],
        )

    def test_fallback_json_tool_call_relaxed_name_value(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": '{"name": list_files, "arguments": {"path": ".", "recursive": true}}',
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("list it")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": True})])

    def test_fallback_json_tool_call_relaxed_keys_and_python_literals(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 5,
                    "eval_duration": 500_000_000,
                    "total_duration": 1_600_000_000,
                    "message": {
                        "content": '{name: "list_files", arguments: {path: ".", recursive: True,},}',
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 3_100_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("list it")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("list_files", {"path": ".", "recursive": True})])

    def test_routes_web_requests_to_web_tools_subset(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("cerca online informazioni su Mario Nobile")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.routed[0], ("web",))
        self.assertEqual(client.calls[0]["tools"][0]["function"]["name"], "search_web")
        self.assertEqual(route_intent("cerca online informazioni su Mario Nobile").intent, INTENT_CURRENT_FACTUAL_LOOKUP)

    def test_routes_weather_queries_to_web_tools_subset(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=2)
        result = agent.run_turn("com'è il tempo oggi a Roma?")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.routed[0], ("web",))
        self.assertEqual(client.calls[0]["tools"][0]["function"]["name"], "search_web")

    def test_current_factual_lookup_retries_from_existing_page_after_redundant_web_tool(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "search_web", "arguments": {"query": "tempo oggi roma"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "search_web", "arguments": {"query": "tempo oggi roma"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "fetch_url", "arguments": {"url": "https://weather.example/rome"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "Oggi a Roma piove con massima di 18°C."},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "search_web":
                return {
                    "ok": True,
                    "query": arguments["query"],
                    "results": [{"title": "Meteo Roma", "url": "https://weather.example/rome", "snippet": "Pioggia, 18°C"}],
                }
            if name == "fetch_url":
                return {
                    "ok": True,
                    "url": arguments["url"],
                    "final_url": arguments["url"],
                    "status_code": 200,
                    "content_type": "text/html",
                    "title": "Meteo Roma",
                    "text": "Pioggia con massima di 18°C e minima di 12°C.",
                    "links": [],
                    "truncated": False,
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("com'è il tempo oggi a Roma?")
        self.assertIn("18°C", result.content)
        self.assertEqual(
            registry.called,
            [
                ("search_web", {"query": "tempo oggi roma"}),
                ("fetch_url", {"url": "https://weather.example/rome"}),
            ],
        )

    def test_current_factual_lookup_fake_tool_response_finalizes_from_fetched_page(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "search_web", "arguments": {"query": "tempo oggi roma"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "fetch_url", "arguments": {"url": "https://weather.example/rome"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true}</tool_response>"},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "search_web":
                return {
                    "ok": True,
                    "query": arguments["query"],
                    "results": [{"title": "Meteo Roma", "url": "https://weather.example/rome", "snippet": "Pioggia, 18°C"}],
                }
            if name == "fetch_url":
                return {
                    "ok": True,
                    "url": arguments["url"],
                    "final_url": arguments["url"],
                    "status_code": 200,
                    "content_type": "text/html",
                    "title": "Meteo Roma",
                    "text": "Pioggia con massima di 18°C e minima di 12°C.",
                    "highlights": ["Pioggia con massima di 18°C e minima di 12°C."],
                    "links": [],
                    "truncated": False,
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("com'è il tempo oggi a Roma?")
        self.assertIn("Meteo Roma", result.content)
        self.assertIn("18°C", result.content)

    def test_intent_router_detects_codebase_inspection(self) -> None:
        route = route_intent("analizza il codice del progetto in questa cartella")
        self.assertEqual(route.intent, INTENT_CODEBASE_INSPECTION)

    def test_intent_router_does_not_confuse_report_with_repo(self) -> None:
        route = route_intent("mostrami il contenuto di REPORT.md")
        self.assertEqual(route.intent, "text_document_analysis")

    def test_intent_router_detects_binary_or_pdf_analysis(self) -> None:
        route = route_intent("analizza questo file pdf con pdftotext o strings")
        self.assertEqual(route.intent, INTENT_BINARY_OR_PDF_ANALYSIS)

    def test_intent_router_detects_bare_binary_format_tokens(self) -> None:
        route = route_intent("analizza il file apk dentro questa cartella di lavoro")
        self.assertEqual(route.intent, INTENT_BINARY_OR_PDF_ANALYSIS)

    def test_intent_router_detects_binary_stem_analysis(self) -> None:
        route = route_intent("nel binario che trovi in questa directory, prova prima il metodo più adatto")
        self.assertEqual(route.intent, INTENT_BINARY_OR_PDF_ANALYSIS)

    def test_intent_router_detects_general_knowledge_without_web(self) -> None:
        route = route_intent("perchè il cielo è blu?")
        self.assertEqual(route.intent, INTENT_GENERAL_KNOWLEDGE)

    def test_general_knowledge_blocks_unneeded_tool_calls_when_no_tools_are_allowed(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "search_web", "arguments": {"query": "why is the sky blue"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "The sky is blue because shorter blue wavelengths are scattered more strongly by the atmosphere."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("perchè il cielo è blu?")
        self.assertIn("shorter blue wavelengths", result.content)
        self.assertEqual(registry.called, [])
        self.assertTrue(
            any("Do not use any tools for this turn" in str(item.get("content", "")) for item in client.calls[1]["messages"])
        )

    def test_rejects_tool_calls_outside_routed_subset(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "search_web", "arguments": {"query": "how to analyze binary files"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Use list_files first and then a binary-aware shell command."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("nel binario che trovi in questa directory, prova prima il metodo più adatto")
        self.assertIn("Use list_files first", result.content)
        self.assertEqual(registry.called[0][0], "list_files")
        self.assertTrue(
            any("unsupported tools" in str(item.get("content", "")) for item in client.calls[1]["messages"])
        )

    def test_binary_analysis_blocks_guessed_read_file_until_candidate_is_listed(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "./binario"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "List files first, then inspect the real binary candidate with bash."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the binary you find in this directory using the most appropriate method first")
        self.assertIn("List files first", result.content)
        self.assertEqual(registry.called[0][0], "list_files")
        self.assertTrue(
            any("do not guess a read_file path" in str(item.get("content", "")) for item in client.calls[1]["messages"])
        )

    def test_binary_analysis_blocks_read_file_on_archive_container(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "app.apk"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Use unzip -l first, then inspect embedded files."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("Use unzip -l first", result.content)
        self.assertEqual(registry.called[0][0], "list_files")
        self.assertTrue(
            any("archive/container format" in str(item.get("content", "")) for item in client.calls[1]["messages"])
        )

    def test_binary_analysis_blocks_strings_on_archive_container(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "strings app.apk | head -20"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Use unzip -l first, then inspect embedded files."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("Use unzip -l first", result.content)
        self.assertTrue(
            any(
                "Do not start archive/container analysis with strings" in str(item.get("content", ""))
                for item in client.calls[1]["messages"]
            )
        )

    def test_binary_analysis_blocks_read_file_on_archive_member_before_extraction(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "unzip -l app.apk | grep 'classes.dex'"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 180_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "classes.dex"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 50,
                    "prompt_eval_duration": 140_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Use unzip -p or extract the embedded member first."},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [{"path": "app.apk", "type": "file"}],
                }
            return {"ok": True, "command": arguments["command"], "stdout": "  1234  classes.dex\n"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("extract the embedded member first", result.content)
        self.assertTrue(
            any("inside the archive/container app.apk" in str(item.get("content", "")) for item in client.calls[2]["messages"])
        )

    def test_binary_analysis_retries_when_model_answers_with_guessed_missing_path_text(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 50_000_000,
                    "total_duration": 170_000_000,
                    "message": {
                        "content": "The file `./binario` was not found. Fallback: strings ./binario | head -n 20",
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 8,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Use list_files first, then inspect the discovered binary candidate."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the binary you find in this directory using the most appropriate method first")
        self.assertIn("Use list_files first", result.content)
        self.assertTrue(
            any("Do not answer with a guessed binary path" in str(item.get("content", "")) for item in client.calls[1]["messages"])
        )

    def test_binary_analysis_finalizes_locally_after_second_guessed_missing_path_text(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 12,
                    "eval_duration": 50_000_000,
                    "total_duration": 170_000_000,
                    "message": {
                        "content": "The file `./binario` was not found. Fallback: strings ./binario | head -n 20",
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 12,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {
                        "content": "The file `./binario` was not found. Fallback: strings ./binario | head -n 20",
                    },
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the binary you find in this directory using the most appropriate method first")
        self.assertIn("could not identify a real binary or PDF candidate", result.content)

    def test_binary_analysis_seeds_list_files_before_model_guessing(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 50_000_000,
                    "total_duration": 170_000_000,
                    "message": {"content": "Use the discovered candidate from the listing."},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the binary you find in this directory using the most appropriate method first")
        self.assertIn("Use the discovered candidate", result.content)
        self.assertEqual(registry.called[0][0], "list_files")
        self.assertEqual(registry.called[0][1]["path"], ".")

    def test_binary_analysis_seeds_file_probe_after_listing_candidate(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 8,
                    "eval_duration": 50_000_000,
                    "total_duration": 170_000_000,
                    "message": {"content": "Use the discovered candidate from the listing."},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 2,
                    "dir_count": 0,
                    "file_count": 2,
                    "truncated": False,
                    "summary": "sample.apk",
                    "entries": [
                        {"path": "sample.apk", "type": "file"},
                        {"path": "README.md", "type": "file"},
                    ],
                }
            if name == "bash":
                return {"ok": True, "command": arguments["command"], "stdout": "sample.apk: Zip archive data"}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("Use the discovered candidate", result.content)
        self.assertEqual(registry.called[0], ("list_files", {"path": ".", "recursive": False, "max_entries": 12}))
        self.assertEqual(registry.called[1], ("bash", {"command": "file sample.apk"}))
        self.assertEqual(registry.called[2], ("bash", {"command": "unzip -l sample.apk | head -n 20"}))

    def test_binary_analysis_repeated_list_files_is_redirected_to_real_candidates(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 30,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 120_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 180_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "unzip -l Questionario_BNL.apk"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 50,
                    "prompt_eval_duration": 140_000_000,
                    "eval_count": 6,
                    "eval_duration": 60_000_000,
                    "total_duration": 220_000_000,
                    "message": {"content": "Found the APK candidate and started archive-aware inspection."},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "entries": [
                        {"path": "Questionario_BNL.apk", "type": "file"},
                        {"path": "notes.txt", "type": "file"},
                    ],
                }
            return {"ok": True, "stdout": "archive listing"}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("started archive-aware inspection", result.content)
        self.assertEqual(registry.called[0][0], "list_files")
        self.assertEqual(registry.called[1][0], "bash")
        system_messages = [item.get("content", "") for item in agent.messages if item.get("role") == "system"]
        self.assertTrue(
            any("Likely candidate paths from that listing: Questionario_BNL.apk" in content for content in system_messages)
        )

    def test_tokens_per_second_helper(self) -> None:
        value = AgentLoop._tokens_per_second(500, 2_000_000_000)
        self.assertEqual(value, 250.0)

    def test_retries_once_on_empty_reply(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                    "total_duration": 1_200_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 8,
                    "eval_duration": 400_000_000,
                    "total_duration": 1_800_000_000,
                    "message": {"content": "done after retry"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read README.md")
        self.assertEqual(result.content, "done after retry")
        self.assertEqual(len(client.calls), 2)

    def test_reports_double_empty_reply(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                    "total_duration": 1_100_000_000,
                    "message": {"content": ""},
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 2,
                    "eval_duration": 200_000_000,
                    "total_duration": 1_400_000_000,
                    "message": {"content": ""},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze app.apk")
        self.assertIn("empty reply twice", result.content)

    def test_retries_once_before_aborting_repeated_tool_loop(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                    "total_duration": 1_100_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 2,
                    "eval_duration": 200_000_000,
                    "total_duration": 1_400_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 1_400_000_000,
                    "eval_count": 3,
                    "eval_duration": 300_000_000,
                    "total_duration": 1_700_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("read README.md")
        self.assertIn("repeating the same call pattern", result.content)
        self.assertIn("read_file.path=README.md", result.content)
        self.assertEqual(len(client.calls), 3)

    def test_recovers_after_repeated_tool_nudge(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                    "total_duration": 1_100_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 2,
                    "eval_duration": 200_000_000,
                    "total_duration": 1_400_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": {"path": "README.md"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 1_400_000_000,
                    "eval_count": 8,
                    "eval_duration": 500_000_000,
                    "total_duration": 2_000_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("read README.md")
        self.assertEqual(result.content, "done")
        self.assertEqual(registry.called, [("read_file", {"path": "README.md"})])

    def test_aborts_repeated_tool_loop(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 100,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 1,
                    "eval_duration": 100_000_000,
                    "total_duration": 1_100_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": {"command": "unzip -l app.apk | grep -i manifest"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 1_200_000_000,
                    "eval_count": 2,
                    "eval_duration": 200_000_000,
                    "total_duration": 1_400_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bash",
                                    "arguments": {"command": "unzip -l app.apk | grep -i manifest"},
                                }
                            }
                        ],
                    },
                },
                {
                    "prompt_eval_count": 140,
                    "prompt_eval_duration": 1_400_000_000,
                    "eval_count": 3,
                    "eval_duration": 300_000_000,
                    "total_duration": 1_700_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bash",
                                    "arguments": {"command": "unzip -l app.apk | grep -i manifest"},
                                }
                            }
                        ],
                    },
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze app.apk")
        self.assertIn("same call", result.content)
        self.assertIn("bash.command=unzip -l app.apk | grep -i manifest", result.content)

    def test_context_pressure_reports_soft_level(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(29):
            agent.messages.append({"role": "user", "content": f"user {idx}"})
        pressure = agent.context_pressure()
        self.assertTrue(pressure.should_compact)
        self.assertEqual(pressure.level, "soft")
        self.assertGreaterEqual(pressure.score, 1.0)

    def test_context_pressure_reports_hard_level(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(71):
            agent.messages.append({"role": "user", "content": f"user {idx}"})
        pressure = agent.context_pressure()
        self.assertTrue(pressure.should_compact)
        self.assertEqual(pressure.level, "hard")
        self.assertGreaterEqual(pressure.score, 1.5)

    def test_context_pressure_projects_pending_user_input(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(35):
            agent.messages.append({"role": "user", "content": "x" * 250})
        resting = agent.context_pressure()
        projected = agent.context_pressure(pending_user_input="y" * 4000)
        self.assertGreater(projected.estimated_prompt_tokens, resting.estimated_prompt_tokens)
        self.assertGreaterEqual(projected.overflow_tokens, resting.overflow_tokens)

    def test_context_pressure_scales_soft_budget_to_real_context_window(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e4b-fast",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        for idx in range(18):
            agent.messages.append({"role": "user", "content": "x" * 1000})
        pressure = agent.context_pressure()
        self.assertTrue(pressure.should_compact)
        self.assertEqual(pressure.level, "soft")

    def test_model_first_runtime_applies_to_gemma4_family(self) -> None:
        client = FakeClient([])
        client.model = "gemma4:e4b-fast"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e4b-fast",
            context_window=131072,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        self.assertTrue(agent._prefers_model_first_runtime())

    def test_model_first_code_generation_does_not_expose_file_tools(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 12,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 2_000_000_000,
                    "message": {"content": "```python\ndef fibonacci(n):\n    return n\n```"},
                }
            ]
        )
        client.model = "gemma4:e2b-fast"
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b-fast",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        result = agent.run_turn("write a fibonacci python function")
        self.assertIn("fibonacci", result.content)
        self.assertEqual(registry.routed, [()])
        self.assertEqual(client.calls[0]["tools"], [])
        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertIn("concise local assistant", system_prompt)
        self.assertNotIn("Use write_file", system_prompt)

    def test_model_first_explicit_file_creation_still_exposes_write_tools(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 40,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 0,
                    "eval_duration": 1,
                    "total_duration": 1_000_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "write_file", "arguments": {"path": "fibonacci.py", "content": "def fibonacci(n):\n    return n\n"}}}
                        ],
                    },
                },
                {
                    "prompt_eval_count": 45,
                    "prompt_eval_duration": 1_000_000_000,
                    "eval_count": 4,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 2_000_000_000,
                    "message": {"content": "Created `fibonacci.py`."},
                },
            ]
        )
        client.model = "gemma4:e2b-fast"
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "write_file":
                return {"ok": True, "path": arguments["path"]}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b-fast",
            context_window=8192,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        result = agent.run_turn("write a file fibonacci.py containing a python function")
        self.assertEqual(result.content, "Created `fibonacci.py`.")
        self.assertIn(("write", "filesystem"), registry.routed)
        self.assertEqual(registry.called, [("write_file", {"path": "fibonacci.py", "content": "def fibonacci(n):\n    return n\n"})])

    def test_context_budget_uses_single_internal_profile(self) -> None:
        self.assertEqual(profile_for_model("gemma4:e2b").soft_tokens, 9000)
        self.assertEqual(profile_for_model("unknown-model").soft_tokens, 9000)

    def test_compact_uses_model_refinement_when_available(self) -> None:
        client = FakeClient(
            [
                {
                    "message": {
                        "content": (
                            "Working memory:\n"
                            "Current objective:\n- analyze the file\n"
                            "Next step:\n- inspect the remaining code path\n\n"
                            "Durable memory:\n"
                            "Confirmed facts:\n- file already read"
                        )
                    }
                }
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(20):
            agent.messages.append({"role": "user", "content": f"user {idx}"})
        changed = agent.compact()
        self.assertTrue(changed)
        self.assertEqual(agent.messages[1]["role"], "system")
        self.assertIn("SESSION MEMORY SUMMARY", agent.messages[1]["content"])
        self.assertIn("Working memory:", agent.messages[1]["content"])
        self.assertIn("Current objective:", agent.messages[1]["content"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tools"], [])

    def test_compact_falls_back_to_local_summary_on_empty_model_reply(self) -> None:
        client = FakeClient([{"message": {"content": ""}}])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(20):
            agent.messages.append({"role": "user", "content": f"user {idx}"})
        changed = agent.compact()
        self.assertTrue(changed)
        self.assertIn("SESSION MEMORY SUMMARY", agent.messages[1]["content"])
        self.assertIn("Working memory:", agent.messages[1]["content"])
        self.assertIn("Current objective:", agent.messages[1]["content"])

    def test_compact_falls_back_when_model_refinement_drops_structure(self) -> None:
        client = FakeClient([{"message": {"content": "Very short summary"}}])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        for idx in range(20):
            agent.messages.append({"role": "user", "content": f"user {idx}"})
        changed = agent.compact()
        self.assertTrue(changed)
        self.assertIn("Working memory:", agent.messages[1]["content"])
        self.assertNotIn("Very short summary", agent.messages[1]["content"])

    def test_hard_compact_runs_before_large_tool_append_and_preserves_tool_result(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        client.model = "gemma4:e2b"
        agent._model_metadata = ModelMetadata(
            active_model="gemma4:e2b",
            context_window=3000,
            capabilities=("completion", "tools"),
            tools_supported=True,
        )
        for idx in range(18):
            agent.messages.append({"role": "user", "content": f"user {idx} " + ("x" * 200)})
            agent.messages.append({"role": "assistant", "content": f"assistant {idx} " + ("y" * 200)})
        tool_message = {
            "role": "tool",
            "tool_name": "list_files",
            "content": json.dumps(
                {
                    "ok": True,
                    "path": ".",
                    "count": 120,
                    "dir_count": 10,
                    "file_count": 110,
                    "truncated": True,
                    "summary": "many entries",
                    "entries": [{"path": f"file_{idx}.txt", "type": "file"} for idx in range(120)],
                }
            ),
        }
        events = []
        agent._append_tool_message_with_compaction(tool_message, tool_name="list_files", on_event=events.append)
        self.assertTrue(any(item.get("role") == "system" and "SESSION MEMORY SUMMARY" in str(item.get("content", "")) for item in agent.messages))
        self.assertTrue(any(item.get("role") == "tool" and item.get("tool_name") == "list_files" for item in agent.messages))
        self.assertTrue(any(isinstance(event, ToolResultCompactEvent) for event in events))
        compact_event = next(event for event in events if isinstance(event, ToolResultCompactEvent))
        self.assertEqual(compact_event.tool_name, "list_files")
        self.assertEqual(agent.messages[-1]["tool_name"], "list_files")

    def test_restore_messages_refreshes_system_prompt(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        agent.restore_messages(
            [
                {"role": "system", "content": "old prompt"},
                {"role": "user", "content": "hello"},
            ]
        )
        self.assertNotEqual(agent.messages[0]["content"], "old prompt")
        self.assertIn("Use tools only when needed", agent.messages[0]["content"])
        self.assertEqual(agent.messages[1]["content"], "hello")

    def test_system_prompt_does_not_leak_skill_path(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        skill = type("Skill", (), {"name": "demo-skill", "path": "/tmp/demo/SKILL.md", "content": "Use tools carefully."})()
        agent = AgentLoop(client=client, registry=registry, max_loops=4, skill=skill)
        prompt = agent.messages[0]["content"]
        self.assertIn("Active skill: demo-skill", prompt)
        self.assertNotIn("Skill path:", prompt)
        self.assertNotIn("/tmp/demo/SKILL.md", prompt)

    def test_default_skill_does_not_add_active_skill_header_to_full_prompt(self) -> None:
        client = FakeClient([])
        registry = FakeRegistry()
        skill = type(
            "Skill",
            (),
            {
                "name": "orbit-default",
                "path": "/tmp/orbit-default/SKILL.md",
                "content": "Reuse exact paths returned by tools.\nStop once there is enough evidence.\n",
            },
        )()
        agent = AgentLoop(client=client, registry=registry, max_loops=4, skill=skill)
        prompt = agent.messages[0]["content"]
        self.assertNotIn("Active skill:", prompt)
        self.assertIn("Reuse exact paths returned by tools.", prompt)

    def test_compact_message_for_model_strips_assistant_thinking(self) -> None:
        message = {
            "role": "assistant",
            "content": "Done.",
            "thinking": "private reasoning",
        }
        compact = _compact_message_for_model(message)
        self.assertEqual(compact["content"], "Done.")
        self.assertNotIn("thinking", compact)

    def test_compact_message_for_model_keeps_larger_fetch_url_excerpt(self) -> None:
        payload = {
            "ok": True,
            "url": "https://example.com",
            "text": "x" * 5000,
            "has_more": False,
        }
        message = {
            "role": "tool",
            "tool_name": "fetch_url",
            "content": json.dumps(payload),
        }
        compact = _compact_message_for_model(message)
        compact_payload = json.loads(compact["content"])
        self.assertEqual(compact_payload["url"], "https://example.com")
        self.assertEqual(len(compact_payload["text"]), 4000)

    def test_compact_message_for_model_keeps_list_files_summary(self) -> None:
        payload = {
            "ok": True,
            "path": ".",
            "count": 3,
            "dir_count": 1,
            "file_count": 2,
            "truncated": False,
            "summary": "dirs: src | files: README.md, pyproject.toml",
            "entries": [{"path": "src", "type": "dir"}, {"path": "README.md", "type": "file"}],
        }
        message = {
            "role": "tool",
            "tool_name": "list_files",
            "content": json.dumps(payload),
        }
        compact = _compact_message_for_model(message)
        compact_payload = json.loads(compact["content"])
        self.assertEqual(compact_payload["summary"], "dirs: src | files: README.md, pyproject.toml")
        self.assertEqual(compact_payload["entries"][0]["path"], "src")

    def test_compact_message_for_model_bounds_stat_path_entries(self) -> None:
        payload = {
            "ok": True,
            "path": ".",
            "type": "dir",
            "size_bytes": 4096,
            "modified_at": "2026-05-27T10:00:00+00:00",
            "mode": "0o755",
            "recursive": True,
            "count": 20,
            "total_entries": 20,
            "file_count": 20,
            "dir_count": 0,
            "truncated": True,
            "entries": [
                {
                    "path": f"file-{index}.txt",
                    "type": "file",
                    "size_bytes": index,
                    "modified_at": "2026-05-27T10:00:00+00:00",
                    "mode": "0o644",
                    "modified_ts": 1_700_000_000 + index,
                }
                for index in range(20)
            ],
        }
        message = {
            "role": "tool",
            "tool_name": "stat_path",
            "content": json.dumps(payload),
        }
        compact = _compact_message_for_model(message)
        compact_payload = json.loads(compact["content"])
        self.assertEqual(compact_payload["path"], ".")
        self.assertEqual(compact_payload["file_count"], 20)
        self.assertEqual(len(compact_payload["entries"]), 12)
        self.assertNotIn("modified_ts", compact_payload["entries"][0])
        self.assertNotIn("mode", compact_payload["entries"][0])

    def test_repeated_list_files_prompt_tells_model_to_reuse_paths(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="list_files",
                signature="list_files:{\"path\": \".\"}",
                detail="list_files.path=.",
            ),
            TurnPolicyState(),
        )
        self.assertIn("paths already returned", prompt)
        self.assertIn("recursive listing", prompt)
        self.assertIn("Do not call list_files again for the same location", prompt)

    def test_codebase_inspection_blocks_read_file_on_guessed_or_directory_paths_after_listing(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "note.txt"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "docs"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 4,
                    "dir_count": 1,
                    "file_count": 3,
                    "truncated": False,
                    "summary": "REPORT.md, README.md, docs/note.txt",
                    "entries": [
                        {"path": "REPORT.md", "type": "file"},
                        {"path": "README.md", "type": "file"},
                        {"path": "docs", "type": "dir"},
                        {"path": "docs/note.txt", "type": "file"},
                    ],
                }
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=6)
        result = agent.run_turn("Analizza il codice di questo progetto e dimmi solo i 5 file più importanti da leggere prima di rispondere.")
        self.assertEqual(result.content, "done")
        system_messages = [
            item.get("content", "")
            for call in client.calls
            for item in call["messages"]
            if item.get("role") == "system"
        ]
        self.assertTrue(any("was not returned exactly" in msg for msg in system_messages))
        self.assertTrue(any("is a directory" in msg for msg in system_messages))
        self.assertEqual(
            registry.called,
            [
                ("list_files", {"path": ".", "recursive": True, "max_entries": 80}),
                ("list_files", {"path": ".", "recursive": False}),
            ],
        )

    def test_fake_tool_response_text_retries_then_finalizes(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true, \"path\": \"note.txt\"}</tool_response>"},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true, \"path\": \"note.txt\"}</tool_response>"},
                },
            ]
        )
        registry = FakeRegistry()
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Analizza il codice di questo progetto e dimmi solo i 5 file più importanti da leggere prima di rispondere.")
        self.assertIn("fabricating tool results", result.content)

    def test_codebase_fake_tool_response_finalizes_from_real_listing(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true, \"path\": \"note.txt\"}</tool_response>"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 3,
                    "dir_count": 1,
                    "file_count": 3,
                    "truncated": False,
                    "summary": "REPORT.md, README.md, docs/note.txt",
                    "entries": [
                        {"path": "REPORT.md", "type": "file"},
                        {"path": "README.md", "type": "file"},
                        {"path": "docs/note.txt", "type": "file"},
                    ],
                }
            if name == "read_file":
                return {"ok": True, "path": "README.md", "content": "readme", "truncated": False, "has_more": False}
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("Analizza il codice di questo progetto e dimmi solo i 5 file più importanti da leggere prima di rispondere.")
        self.assertIn("README.md", result.content)
        self.assertIn("REPORT.md", result.content)

    def test_binary_fake_tool_response_finalizes_with_seeded_triage_summary(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": ".", "recursive": False}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true}</tool_response>"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 1,
                    "dir_count": 0,
                    "file_count": 1,
                    "truncated": False,
                    "summary": "sample.apk",
                    "entries": [
                        {"path": "sample.apk", "type": "file"},
                    ],
                }
            if name == "bash":
                return {"ok": True, "command": arguments["command"], "stdout": "sample.apk: Zip archive data"}
            return {"ok": True, "path": arguments.get("path")}
        registry.call = call
        skill = Skill(name="analysis-skill", path=Path("/tmp/demo/SKILL.md"), content="Create or read `AGENTS.md` and `REPORT.md`.\nCreate or reuse a case directory.\n")
        agent = AgentLoop(client=client, registry=registry, max_loops=4, skill=skill)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("Initial APK triage completed", result.content)
        self.assertIn("sample.apk", result.content)

    def test_binary_analysis_max_loops_falls_back_to_seeded_triage_summary(self) -> None:
        responses = []
        for _ in range(4):
            responses.append(
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": "."}}}],
                    },
                }
            )
        client = FakeClient(responses)
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 1,
                    "dir_count": 0,
                    "file_count": 1,
                    "truncated": False,
                    "summary": "sample.apk",
                    "entries": [{"path": "sample.apk", "type": "file"}],
                }
            if name == "bash":
                return {"ok": True, "command": arguments["command"], "stdout": "sample.apk: Zip archive data"}
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("analyze the apk in this directory")
        self.assertIn("Initial APK triage completed", result.content)
        self.assertIn("sample.apk", result.content)

    def test_binary_analysis_max_loops_falls_back_to_seeded_pdf_summary(self) -> None:
        responses = []
        for _ in range(4):
            responses.append(
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 0,
                    "eval_duration": 0,
                    "total_duration": 150_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": "."}}}],
                    },
                }
            )
        client = FakeClient(responses)
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "list_files":
                return {
                    "ok": True,
                    "path": ".",
                    "recursive": False,
                    "count": 1,
                    "dir_count": 0,
                    "file_count": 1,
                    "truncated": False,
                    "summary": "FASE 1 ABSTRACT.pdf",
                    "entries": [{"path": "FASE 1 ABSTRACT.pdf", "type": "file"}],
                }
            if name == "bash":
                if str(arguments["command"]).startswith("pdftotext "):
                    return {"ok": False, "command": arguments["command"], "stderr": "I/O Error: Couldn't open file"}
                if str(arguments["command"]).startswith("strings "):
                    return {"ok": False, "command": arguments["command"], "stderr": "strings produced no useful text"}
                return {"ok": True, "command": arguments["command"], "stdout": "FASE 1 ABSTRACT.pdf: PDF document"}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=4)
        result = agent.run_turn("leggi il contenuto di PLO ABSTRACT.pdf")
        self.assertIn("Initial PDF triage completed", result.content)
        self.assertIn("FASE 1 ABSTRACT.pdf", result.content)
        self.assertIn("pdftotext", result.content)

    def test_file_edit_fake_tool_response_retries_when_source_and_target_were_read(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "<tool_response>{\"ok\": true}</tool_response>"},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            return {"ok": True, "path": arguments.get("path"), "content": "body", "truncated": False, "has_more": False}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("leggi README.md e poi aggiorna REPORT.md aggiungendo una sezione README Summary")
        self.assertEqual(result.content, "done")
        system_messages = [
            item.get("content", "")
            for call in client.calls
            for item in call["messages"]
            if item.get("role") == "system"
        ]
        self.assertTrue(any("emit one real edit tool call only" in msg for msg in system_messages))

    def test_file_edit_can_infer_append_section_after_source_and_target_reads(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "I will now update the report."},
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()
        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file" and arguments["path"] == "README.md":
                return {"ok": True, "path": "README.md", "content": "# Project README\n\nThis project demonstrates a sample workflow.", "truncated": False, "has_more": False}
            if name == "read_file" and arguments["path"] == "REPORT.md":
                return {"ok": True, "path": "REPORT.md", "content": "# REPORT.md\n\n## Existing\n\nBase report.", "truncated": False, "has_more": False}
            if name == "append_file":
                return {"ok": True, "path": "REPORT.md"}
            return {"ok": True}
        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=5)
        result = agent.run_turn("leggi README.md e poi aggiorna REPORT.md aggiungendo una sezione README Summary")
        self.assertEqual(result.content, "done")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("read_file", {"path": "REPORT.md"}),
                ("append_file", {"path": "REPORT.md", "content": "\n\n## README Summary\n\n# Project README\n\nThis project demonstrates a sample workflow."}),
            ],
        )

    def test_file_edit_can_infer_append_after_repeated_target_read(self) -> None:
        client = FakeClient(
            [
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "REPORT.md", "content": "x"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
                    },
                },
                {
                    "prompt_eval_count": 20,
                    "prompt_eval_duration": 100_000_000,
                    "eval_count": 4,
                    "eval_duration": 50_000_000,
                    "total_duration": 200_000_000,
                    "message": {"content": "done"},
                },
            ]
        )
        registry = FakeRegistry()

        def call(name, arguments):
            registry.called.append((name, arguments))
            if name == "read_file" and arguments["path"] == "README.md":
                return {"ok": True, "path": "README.md", "content": "# Project README\n\nThis project demonstrates a sample workflow.", "truncated": False, "has_more": False}
            if name == "read_file" and arguments["path"] == "REPORT.md":
                return {"ok": True, "path": "REPORT.md", "content": "# REPORT.md\n\n## Existing\n\nBase report.", "truncated": False, "has_more": False}
            if name == "append_file":
                return {"ok": True, "path": "REPORT.md"}
            if name == "write_file":
                return {"ok": False, "error": "refused", "_guarded": True}
            return {"ok": True}

        registry.call = call
        agent = AgentLoop(client=client, registry=registry, max_loops=6)
        result = agent.run_turn("leggi README.md e poi aggiorna REPORT.md aggiungendo una sezione README Summary")
        self.assertEqual(result.content, "Updated `REPORT.md` and added the requested follow-up section.")
        self.assertEqual(
            registry.called,
            [
                ("read_file", {"path": "README.md"}),
                ("write_file", {"path": "REPORT.md", "content": "x"}),
                ("read_file", {"path": "REPORT.md"}),
                ("append_file", {"path": "REPORT.md", "content": "\n\n## README Summary\n\n# Project README\n\nThis project demonstrates a sample workflow."}),
            ],
        )

    def test_repeated_dunder_read_file_prompt_pushes_real_modules(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="read_file",
                signature="read_file:{\"path\": \"src/orbit/__main__.py\"}",
                detail="read_file.path=src/orbit/__main__.py",
            ),
            TurnPolicyState(),
        )
        self.assertIn("__main__.py", prompt)
        self.assertIn("agent.py", prompt)
        self.assertIn("matching tests", prompt)

    def test_repeated_read_file_prompt_can_tell_model_to_stop_exploring(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="read_file",
                signature="read_file:{\"path\": \"src/orbit/core/compact.py\"}",
                detail="read_file.path=src/orbit/core/compact.py",
            ),
            TurnPolicyState(
                tool_history=[
                    ToolCallRecord(name="read_file", signature="read_file:{\"path\": \"README.md\"}", detail="read_file.path=README.md"),
                    ToolCallRecord(name="read_file", signature="read_file:{\"path\": \"src/orbit/core/agent.py\"}", detail="read_file.path=src/orbit/core/agent.py"),
                    ToolCallRecord(name="read_file", signature="read_file:{\"path\": \"src/orbit/core/compact.py\"}", detail="read_file.path=src/orbit/core/compact.py"),
                ],
            ),
        )
        self.assertIn("README.md", prompt)
        self.assertIn("src/orbit/core/agent.py", prompt)
        self.assertIn("already sampled these files", prompt)
        self.assertIn("synthesize the architecture", prompt)
        self.assertIn("Do not keep drilling into the same file", prompt)

    def test_repeated_read_file_same_path_is_blocked_after_enough_unique_samples(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="read_file", signature='read_file:{"path":"a.py"}', detail="read_file.path=a.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"b.py"}', detail="read_file.path=b.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"c.py"}', detail="read_file.path=c.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"d.py"}', detail="read_file.path=d.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"e.py"}', detail="read_file.path=e.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"compact.py","start_line":1}', detail="read_file.path=compact.py"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "compact.py", "start_line": 120}}}],
            state=state,
        )
        self.assertEqual(decision.action, "retry_repeated_tool")
        self.assertIn("Do not keep drilling into the same file", decision.content)

    def test_codebase_inspection_stops_reading_after_sufficient_sample(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="read_file", signature='read_file:{"path":"a.py"}', detail="read_file.path=a.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"b.py"}', detail="read_file.path=b.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"c.py"}', detail="read_file.path=c.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"d.py"}', detail="read_file.path=d.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"e.py"}', detail="read_file.path=e.py"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "f.py"}}}],
            state=state,
            intent=INTENT_CODEBASE_INSPECTION,
        )
        self.assertEqual(decision.action, "retry_repeated_tool")
        self.assertIn("Stop reading more files and answer now", decision.content)

    def test_codebase_inspection_stops_after_three_unique_files_by_default(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="read_file", signature='read_file:{"path":"a.py"}', detail="read_file.path=a.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"b.py"}', detail="read_file.path=b.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"c.py"}', detail="read_file.path=c.py"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "d.py"}}}],
            state=state,
            intent=INTENT_CODEBASE_INSPECTION,
        )
        self.assertEqual(decision.action, "retry_repeated_tool")
        self.assertIn("the 3 most important files", decision.content)

    def test_codebase_inspection_uses_requested_sample_count_when_prompt_is_bounded(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="read_file", signature='read_file:{"path":"a.py"}', detail="read_file.path=a.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"b.py"}', detail="read_file.path=b.py"),
                ToolCallRecord(name="read_file", signature='read_file:{"path":"c.py"}', detail="read_file.path=c.py"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "d.py"}}}],
            state=state,
            intent=INTENT_CODEBASE_INSPECTION,
            user_input="Inspect the workspace, identify the three most relevant files, read them, and give me a concise technical assessment.",
        )
        self.assertEqual(decision.action, "retry_repeated_tool")
        self.assertIn("the 3 most important files", decision.content)

    def test_repeated_fetch_url_prompt_tells_model_not_to_guess_search_urls(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="fetch_url",
                signature='fetch_url:{"url": "https://www.google.com/search?q=Mario+Nobile"}',
                detail="fetch_url.url=https://www.google.com/search?q=Mario+Nobile",
            ),
            TurnPolicyState(),
        )
        self.assertIn("use search_web first", prompt)
        self.assertIn("Do not guess Google, Bing, Wikipedia", prompt)

    def test_repeated_search_web_prompt_tells_model_to_reuse_results(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="search_web",
                signature='search_web:{"query": "Mario Nobile"}',
                detail="search_web.query=Mario Nobile",
            ),
            TurnPolicyState(),
        )
        self.assertIn("Reuse the structured search results", prompt)
        self.assertIn("fetch_url", prompt)

    def test_repeated_append_file_prompt_tells_model_to_stop_editing(self) -> None:
        prompt = _repeated_tool_retry_prompt(
            ToolCallRecord(
                name="append_file",
                signature='append_file:{"path": "REPORT.md"}',
                detail="append_file.path=REPORT.md",
            ),
            TurnPolicyState(),
        )
        self.assertIn("already updated REPORT.md", prompt)
        self.assertIn("Answer now with a short confirmation", prompt)

    def test_file_edit_stops_after_requested_update_completed(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="write_file", signature='write_file:{"path":"REPORT.md"}', detail="write_file.path=REPORT.md"),
                ToolCallRecord(name="append_file", signature='append_file:{"path":"REPORT.md"}', detail="append_file.path=REPORT.md"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "append_file", "arguments": {"path": "REPORT.md", "content": "More"}}}],
            state=state,
            intent=INTENT_FILE_EDIT,
        )
        self.assertEqual(decision.action, "retry_repeated_tool")
        self.assertIn("Stop editing now", decision.content)

    def test_file_edit_does_not_stop_after_only_write(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="write_file", signature='write_file:{"path":"REPORT.md"}', detail="write_file.path=REPORT.md"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "append_file", "arguments": {"path": "REPORT.md", "content": "Next"}}}],
            state=state,
            intent=INTENT_FILE_EDIT,
        )
        self.assertEqual(decision.action, "tool_phase")

    def test_file_edit_finalizes_locally_after_retry_was_already_used(self) -> None:
        state = TurnPolicyState(
            synthesis_retries=1,
            tool_history=[
                ToolCallRecord(name="write_file", signature='write_file:{"path":"REPORT.md"}', detail="write_file.path=REPORT.md"),
                ToolCallRecord(name="append_file", signature='append_file:{"path":"REPORT.md"}', detail="append_file.path=REPORT.md"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "append_file", "arguments": {"path": "REPORT.md", "content": "More"}}}],
            state=state,
            intent=INTENT_FILE_EDIT,
        )
        self.assertEqual(decision.action, "final_text")
        self.assertIn("Updated `REPORT.md`", decision.content)

    def test_file_edit_finalizes_when_model_re_reads_completed_target(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="write_file", signature='write_file:{"path":"REPORT.md"}', detail="write_file.path=REPORT.md"),
                ToolCallRecord(name="append_file", signature='append_file:{"path":"REPORT.md"}', detail="append_file.path=REPORT.md"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
            state=state,
            intent=INTENT_FILE_EDIT,
        )
        self.assertEqual(decision.action, "final_text")
        self.assertIn("Updated `REPORT.md`", decision.content)

    def test_file_edit_finalizes_when_model_re_reads_after_single_write(self) -> None:
        state = TurnPolicyState(
            tool_history=[
                ToolCallRecord(name="write_file", signature='write_file:{"path":"REPORT.md"}', detail="write_file.path=REPORT.md"),
            ],
        )
        decision = classify_model_reply(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "REPORT.md"}}}],
            state=state,
            intent=INTENT_FILE_EDIT,
        )
        self.assertEqual(decision.action, "final_text")
        self.assertIn("Updated `REPORT.md`", decision.content)

    def test_signatures_match_requires_same_normalized_signature(self) -> None:
        first = ToolCallRecord(
            name="search_web",
            signature='search_web:{"query": "Mario Nobile architetto"}',
            detail="search_web.query=Mario Nobile architetto",
        )
        second = ToolCallRecord(
            name="search_web",
            signature='search_web:{"query": "Mario Nobile architect Italy"}',
            detail="search_web.query=Mario Nobile architect Italy",
        )
        same = ToolCallRecord(
            name="search_web",
            signature='search_web:{"query": "Mario Nobile architetto"}',
            detail="search_web.query=Mario Nobile architetto",
        )
        self.assertFalse(signatures_match(first, second))
        self.assertTrue(signatures_match(first, same))

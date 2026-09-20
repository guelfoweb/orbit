"""Client configuration reaches real ANALYSIS dispatch; only HTTP is scripted."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.backend.llama_server import (
    LlamaServerBackend, LlamaServerError, LlamaServerToolCallParseError,
    _parse_chat_result,
)
from orbit.native_server.protocol import parse_chat_request
from orbit.runtime.analysis_runtime import (
    ContextAdmissionError, FINISH_TOOL_SCHEMA, PLAN_TOOL_SCHEMA,
)
from orbit.runtime.sessions import SessionStore
from orbit.terminal import cli
from orbit.terminal.config import load_app_config
from orbit.terminal.repl import Repl


class FinishConstraintCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "sample.txt").write_bytes(b"Safe fixture.\r\n")
        self.config_path = self.root / "config.json"
        self.env = mock.patch.dict(os.environ, {"ORBIT_ANALYSIS_AUTONOMOUS": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.counts = []
        self.generations = []
        self.repair_once = False

    def config(self, *flags, values=None):
        if values is not None:
            self.config_path.write_text(json.dumps(values))
        return load_app_config(cli.build_parser().parse_args([
            "--config", str(self.config_path), "--workdir", str(self.root), *flags,
        ]))

    def backend(self, *, supported=True):
        backend = LlamaServerBackend(base_url="http://test-native", timeout=1)

        def get(path):
            if path == "/health":
                return {"status": "ok"}
            if path == "/props":
                return {"backend": "orbit-native", "required_tool_decoding": supported,
                        "n_ctx": 4096}
            if path == "/v1/models":
                return {"data": [{"id": "fixture"}]}
            self.fail(f"unexpected GET {path}")

        def post(path, payload):
            self.assertEqual(path, "/tokens/count")
            parse_chat_request(payload)
            self.counts.append(copy.deepcopy(payload))
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            return {"tokens": 200, "context_tokens": 4096,
                    "rendered_hash": digest, "token_hash": digest}

        def stream(path, payload, **kwargs):
            self.assertEqual(path, "/chat/stream")
            request = parse_chat_request(payload)
            self.generations.append(copy.deepcopy(payload))
            name = request.tools[0]["function"]["name"]
            if name == "finish_analysis_question" and self.repair_once:
                self.repair_once = False
                raise LlamaServerToolCallParseError("fixture malformed control")
            args = {"status": "still_open", "answer_summary": "No conclusion established."}
            calls = []
            if name == "submit_analysis_plan":
                args = {"questions": []}
            if name != "execute_analysis":
                calls = [{"id": "control", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                }}]
            return _parse_chat_result({"model": "fixture", "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant",
                "content": "", "tool_calls": calls},
            }], "usage": {"prompt_tokens": 200, "completion_tokens": 20}})

        for attr, fn in (("_get_json", get), ("_post_json", post),
                         ("_post_native_stream", stream)):
            patcher = mock.patch.object(backend, attr, side_effect=fn)
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend

    def client(self, *flags, backend=None):
        """Use main's real config/runtime/REPL construction, replace its input loop."""
        backend = backend or self.backend()
        created = []
        def input_loop(repl):
            repl.autonomous_analysis = False
            created.append(repl)
            return 0
        with (mock.patch.object(cli, "LlamaServerBackend", return_value=backend),
              mock.patch.object(cli, "select_interactive_session",
                                return_value=SessionStore(self.root / "session.json")),
              mock.patch.object(Repl, "run", input_loop)):
            self.assertEqual(cli.main([
                "--config", str(self.config_path), "--workdir", str(self.root),
                "--context-tokens", "4096", *flags,
            ]), 0)
        repl = created[0]
        self.addCleanup(repl._close_analysis)
        return repl

    def open(self, repl, *, routed=False):
        with contextlib.redirect_stdout(io.StringIO()):
            if routed:
                self.assertTrue(repl._enter_analysis_from_route("sample.txt"))
            else:
                self.assertTrue(repl._handle_command("/analysis sample.txt"))
        self.assertIsNotNone(repl.analysis)
        return repl.analysis

    def finish(self, runtime):
        return runtime._control_call(runtime.messages, FINISH_TOOL_SCHEMA, repair_budget=1)

    def status(self, repl):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            repl._handle_command("/status")
        return out.getvalue()

    def test_default_off_and_explicit_config_overrides(self):
        self.assertFalse(self.config().constrain_finish)
        self.assertTrue(self.config(values={"constrain_finish": True}).constrain_finish)
        self.assertFalse(self.config("--no-constrain-finish").constrain_finish)
        self.assertTrue(self.config("--constrain-finish", values={"constrain_finish": False}).constrain_finish)
        for value in ("true", 1, None, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "constrain_finish"):
                self.config(values={"constrain_finish": value})

    def test_cli_option_reaches_both_real_analysis_factories(self):
        for flags, expected in (((), "auto"), (("--constrain-finish",), "required")):
            for routed in (False, True):
                with self.subTest(flags=flags, routed=routed):
                    repl = self.client(*flags)
                    rt = self.open(repl, routed=routed)
                    self.counts.clear()
                    self.assertEqual(self.finish(rt)[0]["status"], "still_open")
                    self.assertEqual(self.generations[-1].get("tool_choice", "auto"), expected)
                    self.assertTrue(self.counts)
                    self.assertEqual({p.get("tool_choice", "auto") for p in self.counts}, {expected})
                    self.assertRegex(self.status(repl), r"FINISH constraint\s+" + ("on" if flags else "off"))
                    repl._close_analysis()

    def test_json_setting_reaches_finish_and_cli_off_wins(self):
        self.config_path.write_text(json.dumps({"constrain_finish": True}))
        on = self.client()
        off = self.client("--no-constrain-finish")
        for repl, expected in ((on, "required"), (off, "auto")):
            self.finish(self.open(repl))
            self.assertEqual(self.generations[-1].get("tool_choice", "auto"), expected)

    def test_repair_keeps_contract_without_changing_other_phases(self):
        rt = self.open(self.client("--constrain-finish"))
        rt._control_dispatch(rt.messages, PLAN_TOOL_SCHEMA)
        rt.step("Inspect the fixture without changing it.")
        self.assertTrue(all(p.get("tool_choice", "auto") == "auto" for p in self.generations))
        self.repair_once = True
        self.counts.clear()
        start = len(self.generations)
        self.assertEqual(self.finish(rt)[0]["status"], "still_open")
        calls = self.generations[start:]
        self.assertEqual(len(calls), 2)
        self.assertEqual(rt.control_repairs, 1)
        self.assertTrue(all(p["tool_choice"] == "required" for p in calls))
        self.assertTrue(all(p["tool_choice"] == "required" for p in self.counts))
        self.assertIn("previous control response", calls[1]["messages"][-1]["content"])
        rt.step("Record remaining uncertainty.")
        self.assertEqual(self.generations[-1].get("tool_choice", "auto"), "auto")

    def test_clients_and_replaced_sessions_do_not_share_the_setting(self):
        first, second = self.client("--constrain-finish"), self.client()
        a, b = self.open(first), self.open(second)
        for rt, expected in ((a, "required"), (b, "auto"), (a, "required"), (b, "auto")):
            self.finish(rt)
            self.assertEqual(self.generations[-1].get("tool_choice", "auto"), expected)
        with contextlib.redirect_stdout(io.StringIO()):
            first._handle_command("/reset")
        self.finish(self.open(first))
        self.assertEqual(self.generations[-1]["tool_choice"], "required")
        self.assertRegex(self.status(second), r"FINISH constraint\s+off")
        # Actual instance policy wins over the startup setting in /status.
        second.analysis.constrain_finish = True
        self.assertRegex(self.status(second), r"FINISH constraint\s+on")

    def test_unavailable_constraint_is_an_error_before_generation(self):
        repl = self.client("--constrain-finish", backend=self.backend(supported=False))
        rt = self.open(repl)
        # Exact admission rejects the unavailable count before dispatch; the
        # existing backend capability guard also rejects direct generation.
        with self.assertRaisesRegex(ContextAdmissionError, "context admission failed"):
            self.finish(rt)
        self.assertEqual(self.generations, [])
        self.assertEqual(rt.control_repairs, 0)
        with self.assertRaisesRegex(LlamaServerError, "required tool decoding"):
            repl.backend.chat_stream(rt.messages, tools=[FINISH_TOOL_SCHEMA],
                                     tool_choice="required", max_tokens=2048,
                                     temperature=0, on_delta=lambda _: None)
        self.assertEqual(self.generations, [])

    def test_choice_does_not_rewrite_messages_schemas_or_output_budget(self):
        on, off = self.client("--constrain-finish"), self.client()
        for repl in (on, off):
            self.finish(self.open(repl))
        selected, default = self.generations[-2:]
        for key in ("messages", "tools", "max_tokens", "thinking", "temperature"):
            self.assertEqual(selected[key], default[key], key)

    def test_status_before_analysis_and_after_reset_shows_client_policy(self):
        repl = self.client("--constrain-finish")
        self.assertRegex(self.status(repl), r"FINISH constraint\s+on")
        self.open(repl)
        with contextlib.redirect_stdout(io.StringIO()):
            repl._handle_command("/reset")
        self.assertIsNone(repl.analysis)
        self.assertRegex(self.status(repl), r"FINISH constraint\s+on")


if __name__ == "__main__":
    unittest.main()

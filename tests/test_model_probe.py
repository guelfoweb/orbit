from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orbit.backend.base import ChatResult
from orbit.native_llama.model_contract import ReasoningBehavior
from orbit.native_llama.research_admission import ResearchAdmission
from orbit.qualification import model_probe as probe


def admission():
    return ResearchAdmission("a"*64, 100, "b"*64, "Candidate", "test",
                             (("general.architecture", "test"), ("general.name", "Candidate")),
                             ReasoningBehavior.ALWAYS, True)


def props(identity=None):
    identity = identity or admission()
    return {"backend": "orbit-native", "in_flight": False, "ctx_size": 8192,
            "model_compatibility": {"compatibility_profile": identity.profile_id,
                "verified": False, "research_admission": {"pid": 123,
                    "artifact_sha256": identity.artifact_sha256,
                    "template_sha256": identity.template_sha256, "production_enabled": False}}}


def chat_result(**changes):
    return ChatResult(**(dict(content="answer", model="Candidate", finish_reason="stop", tool_calls=[],
                             prompt_tokens=20, completion_tokens=2, cached_tokens=0,
                             prompt_tokens_per_second=10, generation_tokens_per_second=10) | changes))


class ProbeTests(unittest.TestCase):
    def test_disabled_parent_canonical_gate_is_refused_before_server_start(self):
        with patch.dict("os.environ", {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"}), \
             patch.object(probe, "managed_server") as server:
            with self.assertRaisesRegex(ValueError, "parent process"):
                probe.run_probe(SimpleNamespace(), script=Path("script.py"), root=Path("."))
            server.assert_not_called()

    def test_foreign_verified_busy_or_other_process_server_is_not_reset(self):
        for change in ("verified", "pid", "busy", "artifact", "template", "enabled", "profile"):
            value = deepcopy(props())
            compat = value["model_compatibility"]
            research = compat["research_admission"]
            if change == "verified": compat["verified"] = True
            elif change == "pid": research["pid"] = 456
            elif change == "busy": value["in_flight"] = True
            elif change == "artifact": research["artifact_sha256"] = "c"*64
            elif change == "template": research["template_sha256"] = "c"*64
            elif change == "enabled": research["production_enabled"] = True
            elif change == "profile": compat["compatibility_profile"] = "production"
            with self.subTest(change=change), patch.object(probe, "LlamaServerBackend") as backend:
                backend.return_value.backend_props.return_value = value
                with self.assertRaises(ValueError):
                    probe._fresh_backend("http://unused", admission(), 123)
                backend.return_value.reset_static_analysis_session.assert_not_called()

    def test_good_server_reset_error_is_not_ignored(self):
        with patch.object(probe, "LlamaServerBackend") as backend:
            backend.return_value.backend_props.return_value = props()
            backend.return_value.reset_static_analysis_session.return_value = "busy"
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                probe._fresh_backend("http://unused", admission(), 123)

    def test_completed_chat_is_manual_check_even_when_answer_is_wrong(self):
        rows = [{"result": asdict(chat_result(content="deliberately incorrect answer"))} for _ in range(5)]
        self.assertEqual(probe.chat_status(rows), "MANUAL_CHECK")
        for change in ({"content": ""}, {"finish_reason": "length"},
                       {"tool_calls": [{"function": {"name": "shell"}}]}):
            values = deepcopy(rows)
            values[0]["result"].update(change)
            self.assertEqual(probe.chat_status(values), "FAIL")
        self.assertEqual(probe.chat_status(rows[:-1]), "FAIL")

    def test_chat_preserves_history_without_executing_tools(self):
        backend = Mock()
        backend.chat.return_value = chat_result()
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe, "_fresh_backend", return_value=backend):
            rows = probe.chat_observations("http://unused", admission(), 123, output=Path(tmp))
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(rows[-2]["messages"]), 1)
            self.assertEqual(rows[-2]["messages"][0]["role"], "user")
            self.assertEqual([m["role"] for m in rows[-1]["messages"]], ["user", "assistant", "user"])
            for call in backend.chat.call_args_list:
                self.assertNotIn("tools", call.kwargs)
                self.assertEqual(call.kwargs["temperature"], 0)
            self.assertEqual(len(json.loads((Path(tmp)/"chat-progress.json").read_text())), 5)

    def test_incomplete_evidence_cannot_be_written_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"evidence.json"
            probe._write(target, {"previous": "complete"})
            with self.assertRaises(ValueError):
                probe._write(target, {"invalid": float("nan")})
            self.assertEqual(json.loads(target.read_text()), {"previous": "complete"})

    def run_mock_probe(self, tmp, *, gate_status="PASS", fail_load=False):
        output = Path(tmp)/"bundle"
        identity = Path(tmp)/"identity.json"
        identity.write_text("{}")
        args = SimpleNamespace(identity=identity, model=Path(tmp)/"candidate.gguf", output=output,
                               ctx=8192, threads=6, threads_batch=6, batch=256, ubatch=128)
        @contextmanager
        def server(*a, **k):
            if fail_load:
                raise RuntimeError("load failed")
            yield "http://unused", 123, props()
        with patch.object(probe.ResearchAdmission, "read", return_value=admission()), \
             patch.object(probe, "resolve_legacy_paths", return_value=SimpleNamespace()), \
             patch.object(probe, "_backend_identity", return_value={"lib.so": "c"*64}), \
             patch.object(probe, "_source_identity", return_value={"sha256": "d"*64, "files": {}}), \
             patch.object(probe, "managed_server", side_effect=server), \
             patch.object(probe, "chat_observations", return_value=[{"result": asdict(chat_result())}]*5), \
             patch.object(probe, "tool_intent_observations", return_value={"status": gate_status, "scope": "first intercepted handoff"}):
            rc = probe.run_probe(args, script=Path(tmp)/"script.py", root=Path(tmp))
        return rc, json.loads((output/"qualification.json").read_text())

    def test_gate_pass_does_not_promote_chat_or_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result = self.run_mock_probe(tmp)
            self.assertEqual(rc, 2)
            self.assertFalse(result["production_enabled"])
            self.assertEqual(result["results"]["load"]["status"], "PASS")
            self.assertEqual(result["results"]["chat"]["status"], "MANUAL_CHECK")
            self.assertEqual(result["results"]["tool_intent"]["status"], "PASS")
            self.assertEqual(result["results"]["analysis_semantics"]["status"], "NOT_RUN")

    def test_failed_gate_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result = self.run_mock_probe(tmp, gate_status="FAIL")
            self.assertEqual(rc, 1)
            self.assertEqual(result["results"]["tool_intent"]["status"], "FAIL")

    def test_load_error_keeps_later_capabilities_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result = self.run_mock_probe(tmp, fail_load=True)
            self.assertEqual(rc, 1)
            self.assertEqual(result["results"]["load"]["status"], "FAIL")
            self.assertEqual(result["results"]["chat"]["status"], "NOT_RUN")
            self.assertEqual(result["results"]["tool_intent"]["status"], "NOT_RUN")

    def test_manifest_and_explicit_loopback_are_required_by_server_entry(self):
        from orbit.native_server.app import run_server
        with patch("orbit.native_server.app.NativeLlamaClient") as client:
            for argv in ([], ["--model", "/tmp/model", "--host", "0.0.0.0"],
                         ["--model", "/tmp/model", "--model-id", "gemma4-26b-a4b-it-q40"]):
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    run_server(argv, research_admission=admission())
            client.assert_not_called()

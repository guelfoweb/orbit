from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from orbit.backend.base import ChatResult
from orbit.runtime import ChatRuntime
from orbit.runtime.artifacts import ARTIFACT_VERIFICATION_PROMPT


def _result(
    content: str,
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict] | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        model="fake",
        finish_reason=finish_reason,
        tool_calls=tool_calls or [],
        prompt_tokens=10,
        completion_tokens=3,
        cached_tokens=0,
        prompt_tokens_per_second=None,
        generation_tokens_per_second=None,
    )


def _tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class _ArtifactWorkflowBackend:
    def __init__(
        self,
        *,
        artifact_result: ChatResult | None = None,
        request_path: str = "samples/game.js",
        initial_result: ChatResult | None = None,
        verification_call: dict | None = None,
        verification_result: ChatResult | None = None,
    ) -> None:
        self.calls = 0
        self.chat_messages = []
        self.tools_seen = []
        self.artifact_calls = []
        self.artifact_result = artifact_result or _result("console.log('playable');\n")
        self.request_path = request_path
        self.initial_result = initial_result
        self.verification_call = verification_call or _tool_call(
            "verify_artifact",
            {"path": "samples/game.js", "check": "text_integrity"},
            "verify-1",
        )
        self.verification_result = verification_result

    def chat(self, messages, *, temperature, max_tokens, tools=None):
        del temperature, max_tokens
        self.calls += 1
        self.chat_messages.append(messages)
        self.tools_seen.append(tools)
        if self.calls == 1:
            if self.initial_result is not None:
                return self.initial_result
            return _result(
                json.dumps(
                    {
                        "tool": "write_artifact",
                        "arguments": {
                            "path": self.request_path,
                            "overwrite": False,
                            "create_parents": True,
                        },
                    }
                )
            )
        if tools is not None and [tool["function"]["name"] for tool in tools] == ["verify_artifact"]:
            if self.verification_result is not None:
                return self.verification_result
            return _result("", finish_reason="tool_calls", tool_calls=[self.verification_call])
        visible = "\n".join(str(message.get("content", "")) for message in messages)
        if "artifact_verification: complete" in visible:
            return _result("Created samples/game.js and verified its UTF-8 content, byte count, and hash.")
        if "finish_reason=length" in visible:
            return _result("The artifact was not created because content generation reached its limit.")
        if any(marker in visible.lower() for marker in ("no-mutation", "read-only", "mutation constraint")):
            return _result("The artifact was not created because the explicit no-mutation policy denied it.")
        return _result("The artifact workflow did not complete.")

    def chat_stream(self, messages, *, temperature, max_tokens, tools=None, on_delta, on_progress=None):
        del on_progress
        result = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        if result.content:
            on_delta(result.content)
        return result

    def artifact_content_stream(self, messages, **kwargs):
        self.artifact_calls.append((messages, kwargs))
        return self.artifact_result


class ArtifactToolLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._post_tool_reuse = mock.patch.dict(
            os.environ,
            {"ORBIT_POST_TOOL_FINAL_REUSE": "1"},
        )
        self._post_tool_reuse.start()
        self.addCleanup(self._post_tool_reuse.stop)

    def test_model_selected_artifact_is_published_then_verified_read_only(self) -> None:
        backend = _ArtifactWorkflowBackend()
        tool_calls = []
        tool_results = []
        phases = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "write a small JavaScript game and save it in samples",
                temperature=0,
                max_tokens=256,
                workdir=root,
                on_tool_call=lambda name, arguments: tool_calls.append((name, arguments)),
                on_tool_result=lambda name, chars, source, content: tool_results.append(
                    (name, chars, source, content)
                ),
                on_phase_start=lambda phase: phases.append(phase.phase),
            )

            artifact = root / "samples/game.js"
            self.assertEqual(artifact.read_text(encoding="utf-8"), "console.log('playable');\n")
            self.assertEqual(os.stat(root / "samples").st_mode & 0o777, 0o755)

        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("Created samples/game.js", result.content)
        self.assertEqual([name for name, _ in tool_calls], ["write_artifact", "verify_artifact"])
        self.assertEqual([item[0] for item in tool_results], ["write_artifact", "verify_artifact"])
        self.assertEqual(len(backend.artifact_calls), 1)
        self.assertEqual(
            [tool["function"]["name"] for tool in backend.tools_seen[1]],
            ["verify_artifact"],
        )
        self.assertIn(ARTIFACT_VERIFICATION_PROMPT, backend.chat_messages[1][-1]["content"])
        self.assertIn("artifact_content", phases)
        self.assertEqual(runtime.mutation_verifications, 1)
        self.assertEqual(runtime.mutation_semantic_repairs, 0)

    def test_artifact_final_keeps_publication_policy_when_content_evidence_is_large(self) -> None:
        backend = _ArtifactWorkflowBackend(
            artifact_result=_result("x" * 2_000),
            verification_call=_tool_call(
                "verify_artifact",
                {"path": "samples/game.js", "check": "content"},
                "verify-content",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertTrue((root / "samples/game.js").is_file())
            final_messages = backend.chat_messages[-1]
            self.assertTrue(any(message.get("name") == "write_artifact" for message in final_messages))
            self.assertTrue(any(message.get("name") == "verify_artifact" for message in final_messages))
            self.assertIn("Created samples/game.js", result.content)

    def test_noncanonical_safe_path_uses_canonical_pending_verification_path(self) -> None:
        backend = _ArtifactWorkflowBackend(request_path="./samples/game.js")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)

            result = runtime.ask_auto(
                "write a small JavaScript game and save it in samples",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertEqual((root / "samples/game.js").read_text(), "console.log('playable');\n")
            self.assertEqual(result.finish_reason, "stop")
            verification_messages = next(
                messages
                for messages, tools in zip(backend.chat_messages, backend.tools_seen)
                if tools is not None
                and [tool["function"]["name"] for tool in tools] == ["verify_artifact"]
            )
            self.assertIn("Exact artifact path: samples/game.js", verification_messages[-1]["content"])
            self.assertNotIn("Exact artifact path: ./samples/game.js", verification_messages[-1]["content"])

    def test_wrong_verification_tool_is_rejected_without_execution(self) -> None:
        for canonical_gate in ("0", "1"):
            with self.subTest(canonical_gate=canonical_gate), mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_CANONICAL_GATE": canonical_gate},
            ):
                backend = _ArtifactWorkflowBackend(
                    verification_call=_tool_call(
                        "exec_shell_full_command",
                        {"command": "printf bad >> samples/game.js"},
                        "verify-mutation",
                    )
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_auto(
                        "write a small JavaScript game and save it in samples",
                        temperature=0,
                        max_tokens=256,
                        workdir=root,
                    )
                    artifact_exists = (root / "samples/game.js").exists()
                    parent_exists = (root / "samples").exists()

                self.assertFalse(artifact_exists)
                self.assertFalse(parent_exists)
                self.assertIn("no artifact was published", result.content)
                self.assertEqual(backend.calls, 2)

    def test_malformed_artifact_call_cannot_fall_through_to_shell(self) -> None:
        malformed = _tool_call(
            "write_artifact",
            {
                "path": "samples/game.js",
                "command": "mkdir -p samples && printf bypassed > samples/game.js",
            },
            "artifact-malformed",
        )
        backend = _ArtifactWorkflowBackend(
            initial_result=_result("", finish_reason="tool_calls", tool_calls=[malformed])
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertFalse((root / "samples/game.js").exists())
            self.assertEqual(backend.artifact_calls, [])
            self.assertNotIn("Created samples/game.js", result.content)

    def test_cancelled_artifact_generation_terminates_without_another_model_call(self) -> None:
        backend = _ArtifactWorkflowBackend(
            artifact_result=_result("partial", finish_reason="cancelled")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertEqual(result.finish_reason, "cancelled")
            self.assertEqual(backend.calls, 1)
            self.assertFalse((root / "samples").exists())
            self.assertEqual([message["role"] for message in runtime.messages], ["user", "assistant", "tool"])
            self.assertEqual(runtime.messages[-1]["name"], "write_artifact")
            self.assertIn("cancelled", runtime.messages[-1]["content"])

    def test_duplicate_write_path_is_rejected_with_canonical_gate_off(self) -> None:
        duplicate = {
            "id": "write-duplicate",
            "type": "function",
            "function": {
                "name": "write_artifact",
                "arguments": (
                    '{"path":"safe.txt","path":"other.txt",'
                    '"overwrite":false,"create_parents":false}'
                ),
            },
        }
        backend = _ArtifactWorkflowBackend(
            initial_result=_result("", finish_reason="tool_calls", tool_calls=[duplicate])
        )
        with mock.patch.dict(
            os.environ,
            {"ORBIT_TOOL_CALL_CANONICAL_GATE": "0"},
        ), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                "create safe.txt",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertFalse((root / "safe.txt").exists())
            self.assertFalse((root / "other.txt").exists())
            self.assertEqual(backend.artifact_calls, [])
            self.assertNotIn("Created", result.content)

    def test_rejected_verification_discards_pending_evidence(self) -> None:
        backend = _ArtifactWorkflowBackend(
            verification_call=_tool_call(
                "exec_shell_full_command",
                {"command": "printf bad >> samples/game.js"},
                "invalid-verifier",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertIn("no artifact was published", result.content)
            self.assertNotIn(
                "artifact_pending",
                [record.kind for record in runtime.evidence_store.recent_records(10)],
            )
            route_context = "\n".join(
                str(message.get("content", "")) for message in runtime._route_messages()
            )
            self.assertNotIn("artifact_pending", route_context)

    def test_two_empty_verifier_results_fail_without_chat_fallback(self) -> None:
        backend = _ArtifactWorkflowBackend(
            verification_result=_result("", finish_reason="stop")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertEqual(backend.calls, 3)
            self.assertFalse((root / "samples").exists())
            self.assertIn("no artifact was published", result.content)
            self.assertNotIn(
                "artifact_pending",
                [record.kind for record in runtime.evidence_store.recent_records(10)],
            )

    def test_artifact_generation_timeout_propagates_without_another_model_call(self) -> None:
        class TimeoutBackend(_ArtifactWorkflowBackend):
            def artifact_content_stream(self, messages, **kwargs):
                del messages, kwargs
                raise TimeoutError("artifact timeout")

        backend = TimeoutBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(TimeoutError, "artifact timeout"):
                ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                    "create samples/game.js",
                    temperature=0,
                    max_tokens=256,
                    workdir=root,
                )

            self.assertEqual(backend.calls, 1)
            self.assertFalse((root / "samples").exists())

    def test_verification_timeout_discards_pending_evidence(self) -> None:
        class VerificationTimeoutBackend(_ArtifactWorkflowBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                if self.calls == 1:
                    self.calls += 1
                    raise TimeoutError("verification timeout")
                return super().chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )

        backend = VerificationTimeoutBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            with self.assertRaisesRegex(TimeoutError, "verification timeout"):
                runtime.ask_auto(
                    "create samples/game.js",
                    temperature=0,
                    max_tokens=256,
                    workdir=root,
                )

            self.assertFalse((root / "samples").exists())
            self.assertNotIn(
                "artifact_pending",
                [record.kind for record in runtime.evidence_store.recent_records(10)],
            )
            self.assertNotIn(
                "artifact_pending",
                "\n".join(str(message.get("content", "")) for message in runtime._route_messages()),
            )

    def test_loop_bound_before_verification_aborts_pending_artifact(self) -> None:
        backend = _ArtifactWorkflowBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_with_tools(
                "create samples/game.js",
                temperature=0,
                max_tokens=256,
                max_loops=1,
                workdir=root,
                tool_names=("write_artifact",),
            )

            self.assertFalse((root / "samples").exists())
            self.assertIn("no artifact was published", result.content)
            self.assertNotIn(
                "artifact_pending",
                [record.kind for record in runtime.evidence_store.recent_records(10)],
            )

    def test_successful_verification_finalizes_before_second_artifact(self) -> None:
        class TwoArtifactBackend(_ArtifactWorkflowBackend):
            def chat(self, messages, *, temperature, max_tokens, tools=None):
                if self.calls == 2 and tools is not None:
                    self.calls += 1
                    return _result(
                        "",
                        finish_reason="tool_calls",
                        tool_calls=[
                            _tool_call(
                                "write_artifact",
                                {
                                    "path": "samples/second.js",
                                    "overwrite": False,
                                    "create_parents": True,
                                },
                                "write-second",
                            )
                        ],
                    )
                return super().chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )

        backend = TwoArtifactBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                "create one JavaScript file",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertTrue((root / "samples/game.js").is_file())
            self.assertFalse((root / "samples/second.js").exists())
            self.assertEqual(len(backend.artifact_calls), 1)
            self.assertEqual(result.finish_reason, "stop")
            self.assertIn("Created samples/game.js", result.content)

    def test_second_write_attempt_during_verification_is_rejected(self) -> None:
        backend = _ArtifactWorkflowBackend(
            verification_call=_tool_call(
                "write_artifact",
                {
                    "path": "samples/second.js",
                    "overwrite": False,
                    "create_parents": True,
                },
                "second-write",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "write one JavaScript game in samples",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertFalse((root / "samples").exists())
            self.assertEqual(len(backend.artifact_calls), 1)
            self.assertIn("no artifact was published", result.content)

    def test_verification_requires_exactly_one_call_with_canonical_gate_on_or_off(self) -> None:
        valid = _tool_call(
            "verify_artifact",
            {"path": "samples/game.js", "check": "text_integrity"},
            "verify",
        )
        second = _tool_call(
            "write_artifact",
            {"path": "samples/second.js", "overwrite": False, "create_parents": True},
            "second",
        )
        for canonical_gate in ("0", "1"):
            with self.subTest(canonical_gate=canonical_gate), mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_CANONICAL_GATE": canonical_gate},
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = _ArtifactWorkflowBackend(
                    verification_result=_result(
                        "",
                        finish_reason="tool_calls",
                        tool_calls=[valid, second],
                    )
                )
                result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                    "write one JavaScript game in samples",
                    temperature=0,
                    max_tokens=256,
                    workdir=root,
                )

                self.assertFalse((root / "samples").exists())
                self.assertEqual(len(backend.artifact_calls), 1)
                self.assertIn("no artifact was published", result.content)

    def test_duplicate_verification_arguments_fail_with_canonical_gate_on_or_off(self) -> None:
        duplicate = {
            "id": "verify-duplicate",
            "type": "function",
            "function": {
                "name": "verify_artifact",
                "arguments": (
                    '{"path":"samples/game.js","path":"samples/game.js",'
                    '"check":"text_integrity"}'
                ),
            },
        }
        for canonical_gate in ("0", "1"):
            with self.subTest(canonical_gate=canonical_gate), mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_CANONICAL_GATE": canonical_gate},
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = _ArtifactWorkflowBackend(verification_call=duplicate)
                result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                    "write one JavaScript game in samples",
                    temperature=0,
                    max_tokens=256,
                    workdir=root,
                )

                self.assertFalse((root / "samples").exists())
                self.assertIn("no artifact was published", result.content)

    def test_internal_verifier_prose_is_not_streamed(self) -> None:
        backend = _ArtifactWorkflowBackend(
            verification_result=_result("I cannot verify.", finish_reason="stop")
        )
        emitted: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ChatRuntime(backend=backend, system_prompt=None).ask_auto(
                "write one JavaScript game in samples",
                temperature=0,
                max_tokens=256,
                workdir=root,
                on_final_delta=emitted.append,
            )

            self.assertFalse((root / "samples").exists())
        self.assertEqual("".join(emitted), result.content)
        self.assertNotIn("I cannot verify", "".join(emitted))

    def test_verification_must_reference_the_model_selected_artifact(self) -> None:
        backend = _ArtifactWorkflowBackend(
            verification_call=_tool_call(
                "verify_artifact",
                {"path": "samples/other.js", "check": "content"},
                "verify-unrelated",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "write a small JavaScript game and save it in samples",
                temperature=0,
                max_tokens=256,
                workdir=Path(tmp),
            )

            self.assertFalse((Path(tmp) / "samples").exists())

        self.assertIn("no artifact was published", result.content)

    def test_verification_extra_arguments_fail_closed_with_canonical_gate_on_or_off(self) -> None:
        for canonical_gate in ("0", "1"):
            with self.subTest(canonical_gate=canonical_gate), mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_CANONICAL_GATE": canonical_gate},
            ):
                backend = _ArtifactWorkflowBackend(
                    verification_call=_tool_call(
                        "verify_artifact",
                        {
                            "path": "samples/game.js",
                            "check": "text_integrity",
                            "command": "printf unexpected",
                        },
                        "verify-extra",
                    )
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_auto(
                        "write one JavaScript game in samples",
                        temperature=0,
                        max_tokens=256,
                        workdir=root,
                    )

                    self.assertFalse((root / "samples").exists())
                    self.assertNotIn("Created samples/game.js", result.content)
                    self.assertIn("verification was invalid", result.content)

    def test_length_artifact_is_not_published(self) -> None:
        backend = _ArtifactWorkflowBackend(
            artifact_result=_result("partial content", finish_reason="length")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "write a small JavaScript game and save it in samples",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertFalse((root / "samples").exists())
            self.assertEqual(list(root.glob(".orbit-artifact-*")), [])

        self.assertEqual(result.finish_reason, "stop")
        self.assertNotIn("Created samples/game.js", result.content)

    def test_cancelled_or_length_verification_aborts_pending_file_and_parents(self) -> None:
        cases = (
            ("cancelled", None),
            ("length", None),
            (
                "length",
                _tool_call(
                    "verify_artifact",
                    {"path": "samples/game.js", "check": "text_integrity"},
                    "truncated-verification",
                ),
            ),
        )
        for finish_reason, tool_call in cases:
            with self.subTest(
                finish_reason=finish_reason,
                tool_call=tool_call is not None,
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = _ArtifactWorkflowBackend(
                    verification_result=_result(
                        "",
                        finish_reason=finish_reason,
                        tool_calls=[tool_call] if tool_call is not None else None,
                    )
                )
                runtime = ChatRuntime(backend=backend, system_prompt=None)
                emitted: list[str] = []
                result = runtime.ask_auto(
                    "write a small JavaScript game and save it in samples",
                    temperature=0,
                    max_tokens=256,
                    workdir=root,
                    on_final_delta=emitted.append,
                )

                self.assertFalse((root / "samples").exists())
                self.assertEqual(list(root.glob(".orbit-artifact-*")), [])
                self.assertEqual(result.finish_reason, "stop")
                self.assertIn("no artifact was published", result.content)
                self.assertEqual("".join(emitted), result.content)
                self.assertEqual(runtime.messages[-1], {"role": "assistant", "content": result.content})

    def test_quoted_no_mutation_phrase_does_not_block_legitimate_artifact(self) -> None:
        backend = _ArtifactWorkflowBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                'Create samples/game.js containing the text "without changing any files".',
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertTrue((root / "samples/game.js").is_file())
            self.assertEqual(result.finish_reason, "stop")

    def test_no_mutation_policy_blocks_artifact_before_content_generation(self) -> None:
        for canonical_gate in ("0", "1"):
            with self.subTest(canonical_gate=canonical_gate), mock.patch.dict(
                os.environ,
                {"ORBIT_TOOL_CALL_CANONICAL_GATE": canonical_gate},
            ):
                backend = _ArtifactWorkflowBackend()
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    runtime = ChatRuntime(backend=backend, system_prompt=None)
                    result = runtime.ask_auto(
                        "Without changing any files, write a JavaScript game in samples.",
                        temperature=0,
                        max_tokens=256,
                        workdir=root,
                    )

                    self.assertFalse((root / "samples").exists())

                self.assertEqual(backend.artifact_calls, [])
                self.assertIn("no-mutation", result.content.lower())

    def test_mixed_no_mutation_constraint_blocks_artifact(self) -> None:
        backend = _ArtifactWorkflowBackend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = ChatRuntime(backend=backend, system_prompt=None)
            result = runtime.ask_auto(
                "Inspect without changing files, then create samples/game.js.",
                temperature=0,
                max_tokens=256,
                workdir=root,
            )

            self.assertFalse((root / "samples").exists())
            self.assertEqual(backend.artifact_calls, [])
            self.assertIn("no-mutation", result.content.lower())


if __name__ == "__main__":
    unittest.main()

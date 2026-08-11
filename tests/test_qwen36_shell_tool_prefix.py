from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.client import (
    NativeClientConfig,
    NativeLlamaClient,
    _Qwen36ShellToolAnchorRuntimePlan,
)
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN36_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.native_llama.qwen_route_prefix import QwenRoutePrefixSpec, hash_text, hash_tokens
from orbit.native_llama.qwen36_shell_tool_prefix import (
    QWEN36_SHELL_TOOL_PREFIX_FORMAT_VERSION,
    QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT,
    QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH,
    QWEN36_SHELL_TOOL_SCHEMA_HASH,
    QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH,
    Qwen36ShellToolPrefixSpec,
    derive_qwen36_shell_tool_prefix_spec,
    exact_qwen36_shell_tool_schema,
    resolve_qwen36_shell_tool_prefix_reuse,
)
from orbit.runtime.shell_guardrails import exec_shell_full_definition


def _profile(profile_id: str = QWEN36_PROFILE_ID) -> NativeModelProfile:
    return NativeModelProfile(
        profile_id=profile_id,
        family="qwen3.6" if profile_id == QWEN36_PROFILE_ID else "other",
        model_name="Qwen3.6-35B-A3B",
        architecture="qwen35moe",
        renderer="llama.cpp-jinja",
        reasoning_protocol="qwen-think",
        tool_call_protocol="qwen3.6-xml",
        history_serialization="qwen-leading-system-only",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256="a" * 64,
        thinking_supported=True,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
        route_prefix_reuse_supported=True,
    )


class _FakeBridge:
    def __init__(self, production_prompt: str) -> None:
        self.production_prompt = production_prompt

    def render(self, _context, messages, _tools, *, thinking: bool):
        assert not thinking
        content = str(messages[0].get("content", "")) if messages else ""
        if content.startswith("A-orbit-"):
            return {"prompt": "a" * 500 + "A"}
        if content.startswith("Z-orbit-"):
            return {"prompt": "a" * 500 + "Z"}
        return {"prompt": self.production_prompt}


class Qwen36ShellToolPrefixConfigTests(unittest.TestCase):
    def test_default_on_explicit_values_and_invalid_fail_closed(self) -> None:
        self.assertTrue(resolve_qwen36_shell_tool_prefix_reuse({}).enabled)
        self.assertTrue(
            resolve_qwen36_shell_tool_prefix_reuse(
                {"ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE": "1"}
            ).enabled
        )
        self.assertFalse(
            resolve_qwen36_shell_tool_prefix_reuse(
                {"ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE": "0"}
            ).enabled
        )
        invalid = resolve_qwen36_shell_tool_prefix_reuse(
            {"ORBIT_QWEN36_SHELL_TOOL_PREFIX_REUSE": "yes"}
        )
        self.assertFalse(invalid.enabled)
        self.assertEqual(
            invalid.validation_error,
            "invalid_qwen36_shell_tool_prefix_reuse_value",
        )

    def test_exact_schema_is_bound_to_qualified_hash(self) -> None:
        tools = [exec_shell_full_definition()]
        self.assertTrue(exact_qwen36_shell_tool_schema(tools))
        self.assertEqual(QWEN36_SHELL_TOOL_SCHEMA_HASH, "a2669f863cd2f569bc6e5b009ef72dd8fb6f31a66d83c29d4861e1f510071a68")

        changed = [exec_shell_full_definition()]
        changed[0]["function"]["parameters"]["properties"]["timeout"]["maximum"] = 16
        self.assertFalse(exact_qwen36_shell_tool_schema(changed))
        self.assertFalse(exact_qwen36_shell_tool_schema(tools + tools))
        self.assertFalse(
            exact_qwen36_shell_tool_schema(
                [{"type": "function", "function": {"name": "verify_artifact"}}]
            )
        )
        self.assertFalse(exact_qwen36_shell_tool_schema([]))
        self.assertFalse(exact_qwen36_shell_tool_schema(None))


class Qwen36ShellToolBoundaryTests(unittest.TestCase):
    def test_qualified_boundary_has_no_user_dependent_token(self) -> None:
        tools = [exec_shell_full_definition()]
        stable = "s" * 439

        def render(user: str) -> str:
            return stable + user

        full = render("actual")
        tokens = [ord(char) for char in full]
        expected_prefix_hash = hash_tokens(tokens[:QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT])
        expected_text_hash = hash_text(stable)
        with mock.patch.multiple(
            "orbit.native_llama.qwen36_shell_tool_prefix",
            QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH=expected_prefix_hash,
            QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH=expected_text_hash,
        ):
            spec, reason = derive_qwen36_shell_tool_prefix_spec(
                tools=tools,
                full_prompt=full,
                full_tokens=tokens,
                render_reference=render,
                tokenize=lambda text: [ord(char) for char in text],
            )

        self.assertIsNone(reason)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), 384)
        self.assertEqual(spec.invariant_token_count, 439)
        self.assertEqual(spec.next_boundary_token, ord("s"))

    def test_boundary_rejects_changed_qualified_identity(self) -> None:
        tools = [exec_shell_full_definition()]

        def render(user: str) -> str:
            return "s" * 439 + user

        full = render("actual")
        spec, reason = derive_qwen36_shell_tool_prefix_spec(
            tools=tools,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=render,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(spec)
        self.assertEqual(reason, "qualified_prefix_token_hash_changed")


class Qwen36ShellToolClientTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen.gguf"),
            model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(
                paths,
                NativeClientConfig(qwen36_shell_tool_prefix_reuse_enabled=True),
            )
        client.model_profile = _profile()
        client._model_metadata_identity = {
            "general.architecture": "qwen35moe",
            "general.name": "Qwen3.6-35B-A3B",
            "general.file_type": "15",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen35",
        }
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        client._qwen_backend_build_identity = lambda: "build"  # type: ignore[method-assign]
        return client

    def _spec(self) -> Qwen36ShellToolPrefixSpec:
        return Qwen36ShellToolPrefixSpec(
            prefix_tokens=tuple(range(QWEN36_SHELL_TOOL_PREFIX_TOKEN_COUNT)),
            prefix_token_hash=QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH,
            invariant_text_hash=QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH,
            tool_schema_hash=QWEN36_SHELL_TOOL_SCHEMA_HASH,
            invariant_token_count=439,
            next_boundary_token=42,
        )

    def _plan(self) -> _Qwen36ShellToolAnchorRuntimePlan:
        spec = self._spec()
        return _Qwen36ShellToolAnchorRuntimePlan(
            prefix_tokens=list(spec.prefix_tokens),
            prefix_hash="key",
            state_kwargs={
                "model_id": "m",
                "template_id": "t",
                "tool_schema_hash": "s",
                "capability_summary_hash": "c",
                "runtime_policy_hash": "p",
                "route_contract_hash": "r",
                "backend_version": "b",
                "native_version": "n",
                "tools_mode": "q",
            },
            spec=spec,
        )

    def test_planner_requires_exact_profile_schema_thinking_and_config(self) -> None:
        client = self._client()
        prompt = "p" * 500
        client.chat_bridge = _FakeBridge(prompt)  # type: ignore[assignment]
        client._chat_bridge_context = 1  # type: ignore[assignment]
        client.tokenize = lambda text: list(range(len(text)))  # type: ignore[method-assign]
        spec = self._spec()
        with mock.patch(
            "orbit.native_llama.client.derive_qwen36_shell_tool_prefix_spec",
            return_value=(spec, None),
        ):
            plan = client._qwen36_shell_tool_anchor_plan_for_prompt(
                [{"role": "user", "content": "run pwd"}],
                tools=[exec_shell_full_definition()],
                thinking=False,
                prompt=prompt,
            )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(len(plan.prefix_tokens), 384)

        client._qwen36_shell_tool_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"shell",
            checkpoint_size=5,
        )
        self.assertIsNone(
            client._qwen36_shell_tool_anchor_plan_for_prompt(
                [{"role": "user", "content": "run pwd"}],
                tools=[{"type": "function", "function": {"name": "other"}}],
                thinking=False,
                prompt=prompt,
            )
        )
        self.assertFalse(client._qwen36_shell_tool_prefix_anchor_state.valid)
        self.assertIsNone(
            client._qwen36_shell_tool_anchor_plan_for_prompt(
                [{"role": "user", "content": "run pwd"}],
                tools=[exec_shell_full_definition()],
                thinking=True,
                prompt=prompt,
            )
        )
        client.model_profile = _profile("orbit-qwen3-coder-native-v1")
        client._qwen36_shell_tool_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"shell",
            checkpoint_size=5,
        )
        self.assertIsNone(
            client._qwen36_shell_tool_anchor_plan_for_prompt(
                [{"role": "user", "content": "run pwd"}],
                tools=[exec_shell_full_definition()],
                thinking=False,
                prompt=prompt,
            )
        )
        self.assertFalse(client._qwen36_shell_tool_prefix_anchor_state.valid)

    def test_identity_binds_template_model_and_runtime_config(self) -> None:
        client = self._client()
        spec = self._spec()
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=10, st_mtime_ns=20)):
            messages = [{"role": "system", "content": "policy"}]
            first = client._qwen36_shell_tool_prefix_state_kwargs(spec, messages=messages)
            client.config = NativeClientConfig(
                context_tokens=16384,
                qwen36_shell_tool_prefix_reuse_enabled=True,
            )
            second = client._qwen36_shell_tool_prefix_state_kwargs(spec, messages=messages)
            client.model_profile = _profile()
            client.model_profile = SimpleNamespace(**{**client.model_profile.__dict__, "template_sha256": "b" * 64})
            third = client._qwen36_shell_tool_prefix_state_kwargs(spec, messages=messages)
            client._model_metadata_identity["general.name"] = "changed"
            fourth = client._qwen36_shell_tool_prefix_state_kwargs(spec, messages=messages)
            changed_policy = client._qwen36_shell_tool_prefix_state_kwargs(
                spec,
                messages=[{"role": "system", "content": "changed policy"}],
            )

        self.assertNotEqual(first["template_id"], second["template_id"])
        self.assertNotEqual(second["template_id"], third["template_id"])
        self.assertNotEqual(third["model_id"], fourth["model_id"])
        self.assertNotEqual(fourth["capability_summary_hash"], changed_policy["capability_summary_hash"])
        self.assertEqual(first["tool_schema_hash"], QWEN36_SHELL_TOOL_SCHEMA_HASH)
        self.assertEqual(first["tools_mode"], "qwen36-shell-tool-call-tools-on-thinking-off")

    def test_capture_restore_and_route_prefix_isolation(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client._decode_prompt_range = mock.Mock(return_value=384)  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        plan = self._plan()
        state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"x",
            checkpoint_size=1,
            token_count=384,
        )
        client._qwen_route_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"route",
            checkpoint_size=5,
        )

        with mock.patch("orbit.native_llama.client.capture_prefix_anchor", return_value=(state, {})):
            self.assertEqual(client._prepare_memory_with_qwen36_shell_tool_anchor(plan), (384, 0))
        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            return_value=(True, state, {"restore_used": True}),
        ):
            self.assertEqual(client._prepare_memory_with_qwen36_shell_tool_anchor(plan), (384, 384))

        status = client.qwen36_shell_tool_prefix_reuse_status()
        self.assertEqual(status["capture_count"], 1)
        self.assertEqual(status["restore_count"], 1)
        self.assertEqual(status["fallback_count"], 0)
        self.assertTrue(client._qwen_route_prefix_anchor_state.valid)

    def test_truncated_restore_has_one_cold_fallback_and_no_loop(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client._decode_prompt_range = mock.Mock(return_value=384)  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        plan = self._plan()
        client._qwen36_shell_tool_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"truncated",
            checkpoint_size=9,
            token_count=384,
        )

        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            return_value=(
                False,
                PrefixAnchorState(invalidation_reason="checkpoint_restore_size_mismatch"),
                {"fallback_reason": "checkpoint_restore_size_mismatch"},
            ),
        ):
            self.assertEqual(client._prepare_memory_with_qwen36_shell_tool_anchor(plan), (0, 0))

        replacement = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"replacement",
            checkpoint_size=11,
            token_count=384,
        )
        with mock.patch("orbit.native_llama.client.capture_prefix_anchor", return_value=(replacement, {})):
            self.assertEqual(client._prepare_memory_with_qwen36_shell_tool_anchor(plan), (384, 0))

        status = client.qwen36_shell_tool_prefix_reuse_status()
        self.assertEqual(status["fallback_count"], 1)
        self.assertEqual(status["capture_count"], 1)
        self.assertTrue(status["initialized"])

    def test_cancel_racing_restore_cannot_republish_checkpoint(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        plan = self._plan()
        state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"checkpoint",
            checkpoint_size=10,
            token_count=384,
        )
        client._qwen36_shell_tool_prefix_anchor_state = state

        def restore_then_cancel(*_args, **_kwargs):
            client.cancel()
            return True, state, {"restore_used": True}

        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            side_effect=restore_then_cancel,
        ):
            self.assertEqual(client._prepare_memory_with_qwen36_shell_tool_anchor(plan), (0, 0))

        status = client.qwen36_shell_tool_prefix_reuse_status()
        self.assertFalse(status["initialized"])
        self.assertEqual(status["checkpoint_size_bytes"], 0)
        self.assertEqual(status["restore_count"], 0)
        self.assertEqual(status["invalidation_count"], 1)

    def test_cancel_after_check_cannot_publish_capture_or_restore(self) -> None:
        class CancelAfterCheck:
            def __init__(self, client, *, trigger_call: int) -> None:
                self.client = client
                self.trigger_call = trigger_call
                self.calls = 0
                self.flag = False

            def is_set(self) -> bool:
                self.calls += 1
                if self.calls == self.trigger_call:
                    self.client.cancel()
                    return False
                return self.flag

            def set(self) -> None:
                self.flag = True

            def clear(self) -> None:
                self.flag = False

        for mode in ("capture", "restore"):
            with self.subTest(mode=mode):
                client = self._client()
                client._session.ctx_tgt = 1  # type: ignore[assignment]
                client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
                client._decode_prompt_range = mock.Mock(return_value=384)  # type: ignore[method-assign]
                client.lib.lib = SimpleNamespace()
                plan = self._plan()
                state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"checkpoint",
                    checkpoint_size=10,
                    token_count=384,
                )
                if mode == "capture":
                    client.cancel_event = CancelAfterCheck(client, trigger_call=2)  # type: ignore[assignment]
                    patcher = mock.patch(
                        "orbit.native_llama.client.capture_prefix_anchor",
                        return_value=(state, {}),
                    )
                else:
                    client._qwen36_shell_tool_prefix_anchor_state = state
                    client.cancel_event = CancelAfterCheck(client, trigger_call=1)  # type: ignore[assignment]
                    patcher = mock.patch(
                        "orbit.native_llama.client.restore_prefix_anchor",
                        return_value=(True, state, {"restore_used": True}),
                    )

                with patcher:
                    self.assertEqual(
                        client._prepare_memory_with_qwen36_shell_tool_anchor(plan),
                        (0, 0),
                    )

                status = client.qwen36_shell_tool_prefix_reuse_status()
                self.assertFalse(status["initialized"])
                self.assertEqual(status["checkpoint_size_bytes"], 0)
                self.assertEqual(status["capture_count"], 0)
                self.assertEqual(status["restore_count"], 0)
                self.assertEqual(status["last_used"], "invalidated")

    def test_reset_cancel_reload_and_close_release_shell_state(self) -> None:
        for operation in ("reset", "cancel", "reload", "close"):
            with self.subTest(operation=operation):
                client = self._client()
                client._qwen36_shell_tool_prefix_anchor_state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"shell",
                    checkpoint_size=5,
                )
                client._qwen_route_prefix_anchor_state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"route",
                    checkpoint_size=5,
                )
                if operation == "reset":
                    client._session.ctx_tgt = 1  # type: ignore[assignment]
                    client.lib.lib.llama_get_memory.return_value = 2
                    client.reset_session_state()
                elif operation == "cancel":
                    client.cancel()
                elif operation == "reload":
                    client._invalidate_qwen36_shell_tool_prefix("model_reload")
                else:
                    client._model = None
                    client.close()
                self.assertFalse(client._qwen36_shell_tool_prefix_anchor_state.valid)
                if operation == "reload":
                    self.assertTrue(client._qwen_route_prefix_anchor_state.valid)
                else:
                    self.assertFalse(client._qwen_route_prefix_anchor_state.valid)

    def test_diagnostics_are_bounded_and_do_not_retain_unsupported_states(self) -> None:
        client = self._client()
        client._qwen36_shell_tool_prefix_spec = self._spec()
        client._qwen_route_prefix_spec = QwenRoutePrefixSpec(tuple(), "route", "text", "system", 0, 0)

        diagnostics = client.qwen36_shell_tool_prefix_reuse_status()

        self.assertEqual(diagnostics["checkpoint_identity"], QWEN36_SHELL_TOOL_PREFIX_FORMAT_VERSION)
        self.assertEqual(diagnostics["prefix_token_hash"], QWEN36_SHELL_TOOL_PREFIX_TOKEN_HASH)
        self.assertEqual(diagnostics["prefix_text_hash"], QWEN36_SHELL_TOOL_RENDERED_PREFIX_HASH)
        self.assertNotIn("prompt", diagnostics)
        self.assertFalse(hasattr(client, "_qwen36_verify_artifact_prefix_anchor_state"))
        self.assertFalse(hasattr(client, "_qwen36_final_from_tool_prefix_anchor_state"))


if __name__ == "__main__":
    unittest.main()

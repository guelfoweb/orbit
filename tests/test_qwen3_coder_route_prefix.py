from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import NativeClientConfig, NativeLlamaClient, _QwenRouteAnchorRuntimePlan
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN3_CODER_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.native_llama.qwen_route_prefix import QwenRoutePrefixSpec, hash_text
from orbit.native_llama.qwen3_coder_route_prefix import (
    QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION,
    QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT,
    QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY,
    derive_qwen3_coder_route_prefix_spec,
    resolve_qwen3_coder_route_prefix_reuse,
)


class Qwen3CoderRoutePrefixConfigTests(unittest.TestCase):
    def test_default_is_on_only_after_exact_profile_qualification(self) -> None:
        self.assertTrue(resolve_qwen3_coder_route_prefix_reuse({}).enabled)

    def test_explicit_on_and_kill_switch(self) -> None:
        self.assertTrue(
            resolve_qwen3_coder_route_prefix_reuse(
                {"ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "1"}
            ).enabled
        )
        self.assertFalse(
            resolve_qwen3_coder_route_prefix_reuse(
                {"ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "0"}
            ).enabled
        )

    def test_invalid_value_disables_safely(self) -> None:
        config = resolve_qwen3_coder_route_prefix_reuse(
            {"ORBIT_QWEN3_CODER_ROUTE_PREFIX_REUSE": "yes"}
        )

        self.assertFalse(config.enabled)
        self.assertEqual(
            config.validation_error,
            "invalid_qwen3_coder_route_prefix_reuse_value",
        )


class Qwen3CoderRoutePrefixBoundaryTests(unittest.TestCase):
    def test_derives_768_token_prefix_before_dynamic_user_content(self) -> None:
        system = "s" * 900

        def render(user: str) -> str:
            return system + "\n<user>" + user + "</user>"

        full = render("actual request")
        spec, reason = derive_qwen3_coder_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=render,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertGreater(spec.invariant_token_count, QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)


class _FakeBridge:
    def render(self, _context, messages, _tools, *, thinking: bool):
        assert not thinking
        return {
            "prompt": str(messages[0]["content"])
            + "\n<user>"
            + str(messages[1]["content"])
            + "</user>"
        }

    def free(self, _context) -> None:
        return None


def _profile() -> NativeModelProfile:
    return NativeModelProfile(
        profile_id=QWEN3_CODER_PROFILE_ID,
        family="qwen3-coder",
        model_name="Qwen3-Coder-30B-A3B-Instruct",
        architecture="qwen3moe",
        renderer="llama.cpp-jinja",
        reasoning_protocol="none",
        tool_call_protocol="qwen3-coder-xml",
        history_serialization="qwen3-coder-chatml",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256="b" * 64,
        thinking_supported=False,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
        route_prefix_reuse_supported=True,
        verified_quantization="Q4_K_M",
    )


class Qwen3CoderRoutePrefixClientTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen3-coder.gguf"),
            model_id="legacy-path",
        )
        config = NativeClientConfig(
            qwen_route_prefix_reuse_enabled=False,
            qwen3_coder_route_prefix_reuse_enabled=True,
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, config)
        client.model_profile = _profile()
        client._model_metadata_identity = {
            "general.architecture": "qwen3moe",
            "general.name": "Qwen3-Coder-30B-A3B-Instruct",
            "general.file_type": "15",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen2",
        }
        client.chat_bridge = _FakeBridge()  # type: ignore[assignment]
        client._chat_bridge_context = 1  # type: ignore[assignment]
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        client.tokenize = lambda text: [ord(char) for char in text]  # type: ignore[method-assign]
        client._qwen_backend_build_identity = lambda: "build"  # type: ignore[method-assign]
        return client

    def _plan(self, client: NativeLlamaClient) -> _QwenRouteAnchorRuntimePlan:
        system = "s" * 900
        plan = client._qwen_route_anchor_plan_for_prompt(
            [{"role": "system", "content": system}, {"role": "user", "content": "hello"}],
            tools=None,
            thinking=False,
            prompt=system + "\n<user>hello</user>",
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        return plan

    def test_plan_uses_coder_identity_and_separate_state(self) -> None:
        client = self._client()

        plan = self._plan(client)

        self.assertEqual(plan.profile_id, QWEN3_CODER_PROFILE_ID)
        self.assertEqual(len(plan.prefix_tokens), QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertEqual(
            plan.state_kwargs["tools_mode"],
            "qwen3-coder-route-tools-on-thinking-off",
        )
        self.assertFalse(client._qwen_route_prefix_anchor_state.valid)
        self.assertIsNone(client._qwen_route_prefix_spec)

    def test_qwen36_switch_does_not_enable_coder_checkpoint(self) -> None:
        client = self._client()
        client.config = NativeClientConfig(
            qwen_route_prefix_reuse_enabled=True,
            qwen3_coder_route_prefix_reuse_enabled=False,
        )
        system = "s" * 900

        plan = client._qwen_route_anchor_plan_for_prompt(
            [{"role": "system", "content": system}, {"role": "user", "content": "hello"}],
            tools=None,
            thinking=False,
            prompt=system + "\n<user>hello</user>",
        )

        self.assertIsNone(plan)
        self.assertEqual(client.qwen3_coder_route_prefix_reuse_status()["fallback_count"], 0)

    def test_first_prefill_captures_and_second_request_restores_separate_state(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client._decode_prompt_range = mock.Mock(
            return_value=QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT
        )  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        spec = QwenRoutePrefixSpec(
            tuple(range(QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)),
            "t",
            "i",
            "s",
            789,
            42,
        )
        plan = _QwenRouteAnchorRuntimePlan(
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
            metadata={},
            profile_id=QWEN3_CODER_PROFILE_ID,
        )
        state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"x",
            checkpoint_size=1,
            token_count=QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT,
        )

        with mock.patch("orbit.native_llama.client.capture_prefix_anchor", return_value=(state, {})):
            self.assertEqual(
                client._prepare_memory_with_qwen_route_anchor(plan),
                (QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT, 0),
            )
        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            return_value=(True, state, {"restore_used": True}),
        ):
            self.assertEqual(
                client._prepare_memory_with_qwen_route_anchor(plan),
                (QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT, QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT),
            )

        status = client.qwen3_coder_route_prefix_reuse_status()
        self.assertEqual(status["capture_count"], 1)
        self.assertEqual(status["restore_count"], 1)
        self.assertFalse(client._qwen_route_prefix_anchor_state.valid)

    def test_restore_failure_falls_back_cold_without_touching_qwen36_state(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        spec = QwenRoutePrefixSpec(
            tuple(range(QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)),
            "t",
            "i",
            "s",
            789,
            42,
        )
        plan = _QwenRouteAnchorRuntimePlan(
            list(spec.prefix_tokens),
            "key",
            {},
            spec,
            {},
            QWEN3_CODER_PROFILE_ID,
        )
        client._qwen3_coder_route_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"coder",
            checkpoint_size=5,
            token_count=QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT,
        )
        client._qwen_route_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"qwen36",
            checkpoint_size=6,
        )

        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            return_value=(
                False,
                PrefixAnchorState(invalidation_reason="restore_failed"),
                {"fallback_reason": "restore_failed"},
            ),
        ):
            processed, reused = client._prepare_memory_with_qwen_route_anchor(plan)

        self.assertEqual((processed, reused), (0, 0))
        self.assertEqual(client._clear_target_memory.call_count, 2)
        self.assertFalse(client._qwen3_coder_route_prefix_anchor_state.valid)
        self.assertTrue(client._qwen_route_prefix_anchor_state.valid)
        status = client.qwen3_coder_route_prefix_reuse_status()
        self.assertEqual(status["fallback_count"], 1)
        self.assertEqual(status["failure_reason"], "restore_failed")

    def test_context_template_and_profile_identity_are_bound(self) -> None:
        client = self._client()
        spec = QwenRoutePrefixSpec(
            tuple(range(QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)),
            "t",
            "i",
            "s",
            789,
            42,
        )
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=10, st_mtime_ns=20)):
            first = client._qwen_route_prefix_state_kwargs(spec, profile_id=QWEN3_CODER_PROFILE_ID)
            client.config = NativeClientConfig(
                context_tokens=16384,
                qwen3_coder_route_prefix_reuse_enabled=True,
            )
            second = client._qwen_route_prefix_state_kwargs(spec, profile_id=QWEN3_CODER_PROFILE_ID)
            client.model_profile = NativeModelProfile(**{**_profile().__dict__, "template_sha256": "c" * 64})
            third = client._qwen_route_prefix_state_kwargs(spec, profile_id=QWEN3_CODER_PROFILE_ID)

        self.assertNotEqual(first["template_id"], second["template_id"])
        self.assertNotEqual(second["template_id"], third["template_id"])
        self.assertEqual(
            first["template_id"],
            hash_text(
                f"{QWEN3_CODER_ROUTE_PREFIX_FORMAT_VERSION}:{'b' * 64}:"
                f"{QWEN3_CODER_ROUTE_TOKENIZER_IDENTITY}:ctx=8192"
            ),
        )

    def test_cancel_reset_and_close_release_only_profile_specific_checkpoint(self) -> None:
        for action in ("cancel", "reset_session_state", "close"):
            with self.subTest(action=action):
                client = self._client()
                client._qwen3_coder_route_prefix_anchor_state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"coder",
                    checkpoint_size=5,
                )
                client._qwen_route_prefix_anchor_state = PrefixAnchorState(
                    valid=True,
                    checkpoint_data=b"qwen36",
                    checkpoint_size=6,
                )
                if action == "reset_session_state":
                    client._session.ctx_tgt = 1  # type: ignore[assignment]
                    client.lib.lib.llama_get_memory.return_value = 2
                    client.reset_session_state()
                elif action == "close":
                    client.close()
                else:
                    client.cancel()

                self.assertFalse(client._qwen3_coder_route_prefix_anchor_state.valid)
                self.assertFalse(client._qwen_route_prefix_anchor_state.valid)

    def test_diagnostics_are_bounded(self) -> None:
        client = self._client()
        client._qwen3_coder_route_prefix_spec = QwenRoutePrefixSpec(
            tuple(range(QWEN3_CODER_ROUTE_PREFIX_TOKEN_COUNT)),
            "t" * 64,
            "i" * 64,
            "s" * 64,
            789,
            42,
        )

        diagnostics = client.qwen3_coder_route_prefix_reuse_status()

        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(diagnostics["prefix_token_hash"], "t" * 64)
        self.assertNotIn("prompt", diagnostics)
        self.assertNotIn("model_path", diagnostics)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from orbit.native_llama.client import (
    NativeClientConfig,
    NativeLlamaClient,
    _QwenRouteAnchorRuntimePlan,
)
from orbit.native_llama.model_profiles import NativeModelProfile, QWEN36_PROFILE_ID
from orbit.native_llama.paths import NativeLlamaPaths
from orbit.native_llama.prefix_anchor import PrefixAnchorState
from orbit.native_llama.qwen_route_prefix import (
    QWEN_ROUTE_PREFIX_TOKEN_COUNT,
    QwenRoutePrefixSpec,
    derive_qwen_route_prefix_spec,
    resolve_qwen_route_prefix_reuse,
)


class QwenRoutePrefixConfigTests(unittest.TestCase):
    def test_default_is_on_after_exact_profile_qualification(self) -> None:
        self.assertTrue(resolve_qwen_route_prefix_reuse({}).enabled)

    def test_explicit_on_and_kill_switch(self) -> None:
        self.assertTrue(resolve_qwen_route_prefix_reuse({"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "1"}).enabled)
        self.assertFalse(resolve_qwen_route_prefix_reuse({"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "0"}).enabled)

    def test_invalid_value_disables_safely(self) -> None:
        config = resolve_qwen_route_prefix_reuse({"ORBIT_QWEN_ROUTE_PREFIX_REUSE": "yes"})

        self.assertFalse(config.enabled)
        self.assertEqual(config.validation_error, "invalid_qwen_route_prefix_reuse_value")


class QwenRoutePrefixBoundaryTests(unittest.TestCase):
    def test_derives_exact_token_aligned_prefix_before_user_content(self) -> None:
        system = "s" * 900

        def render(user: str) -> str:
            return system + "\n<user>" + user + "</user>"

        full = render("actual request")
        spec, reason = derive_qwen_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=render,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), QWEN_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertGreater(spec.invariant_token_count, QWEN_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertEqual(list(spec.prefix_tokens), [ord(char) for char in full[:QWEN_ROUTE_PREFIX_TOKEN_COUNT]])

    def test_rejects_when_dynamic_content_reaches_boundary(self) -> None:
        system = "s" * 700

        def render(user: str) -> str:
            return system + user

        full = render("x" * 100)
        spec, reason = derive_qwen_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=render,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(spec)
        self.assertEqual(reason, "stable_token_boundary_unavailable")


class _FakeBridge:
    def render(self, _context, messages, _tools, *, thinking: bool):
        assert not thinking
        return {"prompt": str(messages[0]["content"]) + "\n<user>" + str(messages[1]["content"]) + "</user>"}


def _profile() -> NativeModelProfile:
    return NativeModelProfile(
        profile_id=QWEN36_PROFILE_ID,
        family="qwen3.6",
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


class QwenRoutePrefixClientTests(unittest.TestCase):
    def _client(self) -> NativeLlamaClient:
        paths = NativeLlamaPaths(
            llama_root=Path("/llama"),
            build_bin=Path("/llama/build/bin"),
            library=Path("/llama/build/bin/libllama.so"),
            model=Path("/models/qwen.gguf"),
            model_id="legacy-path",
        )
        with mock.patch("orbit.native_llama.client.LlamaLibrary"):
            client = NativeLlamaClient(paths, NativeClientConfig(qwen_route_prefix_reuse_enabled=True))
        client.model_profile = _profile()
        client._model_metadata_identity = {
            "general.architecture": "qwen35moe",
            "general.name": "Qwen3.6-35B-A3B",
            "general.file_type": "15",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen35",
        }
        client.chat_bridge = _FakeBridge()  # type: ignore[assignment]
        client._chat_bridge_context = 1  # type: ignore[assignment]
        client._model = 1  # type: ignore[assignment]
        client._vocab = 1  # type: ignore[assignment]
        client.tokenize = lambda text: [ord(char) for char in text]  # type: ignore[method-assign]
        client._qwen_backend_build_identity = lambda: "build"  # type: ignore[method-assign]
        return client

    def test_plan_is_qwen_specific_and_does_not_reuse_gemma_state(self) -> None:
        client = self._client()
        system = "s" * 900
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "hello"}]
        prompt = system + "\n<user>hello</user>"

        plan = client._qwen_route_anchor_plan_for_prompt(messages, tools=None, thinking=False, prompt=prompt)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(len(plan.prefix_tokens), QWEN_ROUTE_PREFIX_TOKEN_COUNT)
        self.assertFalse(client._route_prefix_anchor_state.valid)
        self.assertEqual(plan.state_kwargs["tools_mode"], "qwen-route-tools-on-thinking-off")

    def test_unverified_quantization_is_ineligible(self) -> None:
        client = self._client()
        client._model_metadata_identity["general.file_type"] = "2"
        system = "s" * 900

        plan = client._qwen_route_anchor_plan_for_prompt(
            [{"role": "system", "content": system}, {"role": "user", "content": "hello"}],
            tools=None,
            thinking=False,
            prompt=system + "\n<user>hello</user>",
        )

        self.assertIsNone(plan)
        self.assertEqual(client.qwen_route_prefix_reuse_status()["failure_reason"], "qwen_quantization_unverified")

    def test_thinking_and_mtp_are_ineligible(self) -> None:
        system = "s" * 900
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "hello"}]
        prompt = system + "\n<user>hello</user>"

        thinking_client = self._client()
        self.assertIsNone(
            thinking_client._qwen_route_anchor_plan_for_prompt(
                messages,
                tools=None,
                thinking=True,
                prompt=prompt,
            )
        )
        self.assertEqual(
            thinking_client.qwen_route_prefix_reuse_status()["failure_reason"],
            "structured_route_mode_ineligible",
        )

        mtp_client = self._client()
        mtp_client.config = NativeClientConfig(
            qwen_route_prefix_reuse_enabled=True,
            use_mtp_experimental=True,
        )
        self.assertIsNone(
            mtp_client._qwen_route_anchor_plan_for_prompt(
                messages,
                tools=None,
                thinking=False,
                prompt=prompt,
            )
        )
        self.assertEqual(mtp_client.qwen_route_prefix_reuse_status()["failure_reason"], "mtp_ineligible")

    def test_changed_route_identity_invalidates_existing_checkpoint(self) -> None:
        client = self._client()
        client._qwen_route_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"x",
            checkpoint_size=1,
            token_count=QWEN_ROUTE_PREFIX_TOKEN_COUNT,
        )
        client._qwen_route_prefix_spec = QwenRoutePrefixSpec(
            tuple(range(QWEN_ROUTE_PREFIX_TOKEN_COUNT)),
            "t",
            "i",
            "different-system-hash",
            810,
            42,
        )
        system = "s" * 900

        plan = client._qwen_route_anchor_plan_for_prompt(
            [{"role": "system", "content": system}, {"role": "user", "content": "hello"}],
            tools=None,
            thinking=False,
            prompt=system + "\n<user>hello</user>",
        )

        self.assertIsNotNone(plan)
        status = client.qwen_route_prefix_reuse_status()
        self.assertEqual(status["invalidation_count"], 1)
        self.assertEqual(status["checkpoint_size_bytes"], 0)

    def test_context_and_model_identity_change_checkpoint_key(self) -> None:
        client = self._client()
        spec = QwenRoutePrefixSpec(tuple(range(QWEN_ROUTE_PREFIX_TOKEN_COUNT)), "t", "i", "s", 810, 42)
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=10, st_mtime_ns=20)):
            first = client._qwen_route_prefix_state_kwargs(spec)
            client.config = NativeClientConfig(context_tokens=16384, qwen_route_prefix_reuse_enabled=True)
            second = client._qwen_route_prefix_state_kwargs(spec)
            client._model_metadata_identity["general.name"] = "changed"
            third = client._qwen_route_prefix_state_kwargs(spec)

        self.assertNotEqual(first["template_id"], second["template_id"])
        self.assertNotEqual(second["model_id"], third["model_id"])

    def test_first_prefill_captures_and_second_request_restores(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client._decode_prompt_range = mock.Mock(return_value=QWEN_ROUTE_PREFIX_TOKEN_COUNT)  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        spec = QwenRoutePrefixSpec(tuple(range(QWEN_ROUTE_PREFIX_TOKEN_COUNT)), "t", "i", "s", 810, 42)
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
        )
        state = PrefixAnchorState(valid=True, checkpoint_data=b"x", checkpoint_size=1, token_count=QWEN_ROUTE_PREFIX_TOKEN_COUNT)

        with mock.patch("orbit.native_llama.client.capture_prefix_anchor", return_value=(state, {})):
            processed, reused = client._prepare_memory_with_qwen_route_anchor(plan)

        self.assertEqual((processed, reused), (QWEN_ROUTE_PREFIX_TOKEN_COUNT, 0))
        self.assertEqual(client.qwen_route_prefix_reuse_status()["capture_count"], 1)

        with mock.patch("orbit.native_llama.client.restore_prefix_anchor", return_value=(True, state, {"restore_used": True})):
            processed, reused = client._prepare_memory_with_qwen_route_anchor(plan)

        self.assertEqual((processed, reused), (QWEN_ROUTE_PREFIX_TOKEN_COUNT, QWEN_ROUTE_PREFIX_TOKEN_COUNT))
        self.assertEqual(client.qwen_route_prefix_reuse_status()["restore_count"], 1)
        self.assertFalse(client._route_prefix_anchor_state.valid)

    def test_restore_failure_clears_state_and_uses_one_cold_fallback(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client._clear_target_memory = mock.Mock()  # type: ignore[method-assign]
        client.lib.lib = SimpleNamespace()
        spec = QwenRoutePrefixSpec(tuple(range(QWEN_ROUTE_PREFIX_TOKEN_COUNT)), "t", "i", "s", 810, 42)
        plan = _QwenRouteAnchorRuntimePlan(list(spec.prefix_tokens), "key", {}, spec, {})
        state = PrefixAnchorState(valid=True, checkpoint_data=b"x", checkpoint_size=1, token_count=QWEN_ROUTE_PREFIX_TOKEN_COUNT)
        client._qwen_route_prefix_anchor_state = state

        with mock.patch(
            "orbit.native_llama.client.restore_prefix_anchor",
            return_value=(False, PrefixAnchorState(invalidation_reason="restore_failed"), {"fallback_reason": "restore_failed"}),
        ):
            processed, reused = client._prepare_memory_with_qwen_route_anchor(plan)

        self.assertEqual((processed, reused), (0, 0))
        self.assertEqual(client._clear_target_memory.call_count, 2)
        status = client.qwen_route_prefix_reuse_status()
        self.assertFalse(status["initialized"])
        self.assertEqual(status["checkpoint_size_bytes"], 0)
        self.assertEqual(status["fallback_count"], 1)
        self.assertEqual(status["invalidation_count"], 1)

    def test_cancel_invalidates_only_qwen_checkpoint(self) -> None:
        client = self._client()
        client._qwen_route_prefix_anchor_state = PrefixAnchorState(valid=True, checkpoint_data=b"x", checkpoint_size=1)
        client._route_prefix_anchor_state = PrefixAnchorState(valid=True, checkpoint_data=b"g", checkpoint_size=1)

        client.cancel()

        self.assertFalse(client._qwen_route_prefix_anchor_state.valid)
        self.assertTrue(client._route_prefix_anchor_state.valid)

    def test_reset_releases_checkpoint_and_preserves_separate_gemma_state(self) -> None:
        client = self._client()
        client._session.ctx_tgt = 1  # type: ignore[assignment]
        client.lib.lib.llama_get_memory.return_value = 2
        client._qwen_route_prefix_anchor_state = PrefixAnchorState(
            valid=True,
            checkpoint_data=b"x",
            checkpoint_size=1,
        )
        client._route_prefix_anchor_state = PrefixAnchorState(valid=True, checkpoint_data=b"g", checkpoint_size=1)

        client.reset_session_state()

        self.assertFalse(client._qwen_route_prefix_anchor_state.valid)
        self.assertEqual(client.qwen_route_prefix_reuse_status()["checkpoint_size_bytes"], 0)
        self.assertTrue(client._route_prefix_anchor_state.valid)

    def test_diagnostics_are_bounded_and_do_not_expose_prompt_or_model_path(self) -> None:
        client = self._client()
        client._qwen_route_prefix_spec = QwenRoutePrefixSpec(
            tuple(range(QWEN_ROUTE_PREFIX_TOKEN_COUNT)),
            "t" * 64,
            "i" * 64,
            "s" * 64,
            810,
            42,
        )

        diagnostics = client.qwen_route_prefix_reuse_status()

        self.assertEqual(diagnostics["prefix_tokens"], 0)
        self.assertEqual(diagnostics["prefix_token_hash"], "t" * 64)
        self.assertEqual(diagnostics["prefix_text_hash"], "i" * 64)
        self.assertNotIn("prompt", diagnostics)
        self.assertNotIn("model_path", diagnostics)


if __name__ == "__main__":
    unittest.main()

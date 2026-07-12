from __future__ import annotations

from ctypes import c_float
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from orbit.native_llama.client import (
    FinalPrefixExperimentStatus,
    NativeClientConfig,
    NativeLlamaClient,
    NativeRoutePrefixPrefillResult,
)
from orbit.native_llama.prefix_anchor_probe import (
    PrefixPrefillOnlyProbeResult,
    PrefixAnchorProbeResult,
    probe_route_boundary_token_prefix,
    probe_route_prefix_prefill_only,
    probe_prefix_anchor_equivalence,
    split_prompt_by_token_prefix,
)
from orbit.native_llama.chat_template import render_gemma4_route_prompt_segments
from orbit.native_llama.prefix_anchor import PrefixAnchorState


def _token_value(value: object) -> int:
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, byteorder="little", signed=True)
    raw = getattr(value, "value", value)
    return int(raw)


class _FakeBatch:
    def __init__(self, n_tokens: int) -> None:
        self.n_tokens = n_tokens
        self.token = [0] * n_tokens
        self.pos = [0] * n_tokens
        self.n_seq_id = [0] * n_tokens
        self.seq_id = [None] * n_tokens
        self.logits = [0] * n_tokens


class _FakeLib:
    def __init__(self) -> None:
        self.current_tokens: list[int] = []
        self.state_payload = b""
        self.logits_rows: list[object] = []
        self.clears = 0
        self.time_us = 1000

    def llama_get_memory(self, _ctx):
        return object()

    def llama_memory_clear(self, _mem, _full: bool) -> None:
        self.current_tokens = []
        self.clears += 1

    def llama_batch_init(self, n_tokens: int, _embd: int, _n_seq_max: int):
        return _FakeBatch(n_tokens)

    def llama_batch_free(self, _batch) -> None:
        return None

    def llama_batch_get_one(self, token_ptr, n_tokens: int):
        batch = _FakeBatch(n_tokens)
        for index in range(n_tokens):
            batch.token[index] = token_ptr[index]
        return batch

    def llama_decode(self, _ctx, batch) -> int:
        for index in range(batch.n_tokens):
            self.current_tokens.append(_token_value(batch.token[index]))
        self.time_us += 5000
        return 0

    def llama_synchronize(self, _ctx) -> None:
        return None

    def llama_time_us(self) -> int:
        return self.time_us

    def llama_state_seq_get_size(self, _ctx, _seq_id: int) -> int:
        payload = ",".join(str(token) for token in self.current_tokens).encode("ascii")
        self.state_payload = payload
        return len(payload)

    def llama_state_seq_get_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        if size != len(self.state_payload):
            return 0
        for index, byte in enumerate(self.state_payload):
            buffer[index] = byte
        return len(self.state_payload)

    def llama_state_seq_set_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        payload = bytes(buffer[:size])
        if not payload:
            self.current_tokens = []
            return 0
        self.current_tokens = [int(item) for item in payload.decode("ascii").split(",") if item]
        return size

    def llama_sampler_reset(self, _sampler) -> None:
        return None

    def llama_sampler_sample(self, _sampler, _ctx, _index: int) -> int:
        return sum(self.current_tokens) % 97

    def llama_vocab_n_tokens(self, _vocab) -> int:
        return 4

    def llama_get_logits_ith(self, _ctx, _index: int):
        base = float(sum(self.current_tokens))
        row = (c_float * 4)(base, base + 1.0, base + 2.0, base + 3.0)
        self.logits_rows.append(row)
        return row


class _FailingRestoreLib(_FakeLib):
    def llama_state_seq_set_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        _ = bytes(buffer[:size])
        return size - 1


class _FailingDecodeLib(_FakeLib):
    def llama_decode(self, _ctx, batch) -> int:
        _ = batch
        return 1


def _prefill_hook_client(segments, *, lib=None) -> NativeLlamaClient:
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    client.paths = SimpleNamespace(model=Path("model-alpha.gguf"))
    client.config = NativeClientConfig(batch_size=2, progress_step=2)
    client.lib = SimpleNamespace(lib=lib or _FakeLib())
    client._vocab = object()
    client.cancel_event = threading.Event()
    client._session = SimpleNamespace(
        ctx_tgt=object(),
        cached_prompt_tokens=[],
        continuation_ready=False,
        in_flight=False,
        cancel_requested=False,
    )
    client._route_prefix_anchor_state = PrefixAnchorState()
    client._final_prefix_anchor_state = PrefixAnchorState()
    client._final_prefix_status = FinalPrefixExperimentStatus()
    client._route_prefix_prefill_lock = threading.Lock()
    mapping = {
        segments.stable_prefix_text: [1, 2, 3, 4],
        segments.full_prompt_text: [1, 2, 3, 4, 5],
    }
    client.tokenize = lambda text: mapping[text]
    return client


def _final_prefix_client(segments, *, lib=None) -> NativeLlamaClient:
    client = _prefill_hook_client(segments, lib=lib)
    stable_tokens = list(range(1, 43))
    prefix_tokens = stable_tokens + [43]
    full_tokens = prefix_tokens + [44, 45]
    client.tokenize = lambda text: {
        segments.stable_prefix_text: stable_tokens,
        segments.full_prompt_text: full_tokens,
    }[text]
    return client


class PrefixAnchorProbeTests(unittest.TestCase):
    def test_final_prefix_experiment_is_ineligible_when_native_mtp_is_enabled(self) -> None:
        client = NativeLlamaClient.__new__(NativeLlamaClient)
        client.config = NativeClientConfig(final_prefix_experiment_enabled=True, use_mtp_experimental=True)

        self.assertFalse(client._final_prefix_experiment_eligible(True))

    def test_final_prefix_capture_then_restore_reuses_exact_43_tokens(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": "final policy"},
                {"role": "user", "content": "request"},
                {"role": "system", "content": "evidence"},
            ],
            thinking=False,
        )
        client = _final_prefix_client(segments)
        prompt_tokens = client.tokenize(segments.full_prompt_text)
        plan = client._final_prefix_plan(segments.full_prompt_text, prompt_tokens, segments)

        self.assertIsNotNone(plan)
        first_start, first_reused = client._prepare_memory_with_final_prefix(plan, prompt_tokens)  # type: ignore[arg-type]
        second_start, second_reused = client._prepare_memory_with_final_prefix(plan, prompt_tokens)  # type: ignore[arg-type]

        self.assertEqual((first_start, first_reused), (43, 0))
        self.assertEqual((second_start, second_reused), (43, 43))
        self.assertEqual(client.lib.lib.current_tokens, list(range(1, 44)))
        self.assertEqual(client.final_prefix_experiment_status()["capture_count"], 1)
        self.assertEqual(client.final_prefix_experiment_status()["restore_count"], 1)
        self.assertTrue(client.final_prefix_experiment_status()["last_used"])

    def test_final_prefix_identity_mismatch_falls_back_before_restore(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": "final policy"},
                {"role": "user", "content": "request"},
                {"role": "system", "content": "evidence"},
            ],
            thinking=False,
        )
        client = _final_prefix_client(segments)
        prompt_tokens = client.tokenize(segments.full_prompt_text)
        client._final_prefix_anchor_state = PrefixAnchorState(
            prefix_hash="different",
            token_count=43,
            valid=True,
            checkpoint_size=1,
            checkpoint_data=b"x",
        )

        plan = client._final_prefix_plan(segments.full_prompt_text, prompt_tokens, segments)

        self.assertIsNotNone(plan)
        with patch.object(client, "_prepare_memory_for_prompt", return_value=0) as normal_prefill:
            start, reused = client._prepare_memory_with_final_prefix(plan, prompt_tokens)  # type: ignore[arg-type]

        self.assertEqual((start, reused), (0, 0))
        normal_prefill.assert_called_once_with(prompt_tokens)
        self.assertEqual(client.final_prefix_experiment_status()["failure_reason"], "prefix_hash_changed")
        self.assertEqual(client.final_prefix_experiment_status()["fallback_count"], 1)

    def test_final_prefix_restore_failure_uses_normal_prefill(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": "final policy"},
                {"role": "user", "content": "request"},
                {"role": "system", "content": "evidence"},
            ],
            thinking=False,
        )
        client = _final_prefix_client(segments, lib=_FailingRestoreLib())
        prompt_tokens = client.tokenize(segments.full_prompt_text)
        plan = client._final_prefix_plan(segments.full_prompt_text, prompt_tokens, segments)
        client._prepare_memory_with_final_prefix(plan, prompt_tokens)  # type: ignore[arg-type]

        with patch.object(client, "_prepare_memory_for_prompt", return_value=0) as normal_prefill:
            start, reused = client._prepare_memory_with_final_prefix(plan, prompt_tokens)  # type: ignore[arg-type]

        self.assertEqual((start, reused), (0, 0))
        normal_prefill.assert_called_once_with(prompt_tokens)
        self.assertFalse(client.final_prefix_experiment_status()["initialized"])
        self.assertEqual(client.final_prefix_experiment_status()["fallback_count"], 1)
        self.assertEqual(
            client.final_prefix_experiment_status()["failure_reason"],
            "checkpoint_restore_size_mismatch",
        )
    def test_route_boundary_token_probe_reports_valid_token_prefix(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": "route policy placeholder"},
                {"role": "user", "content": "placeholder task payload"},
            ]
        )
        mapping = {
            segments.stable_prefix_text: [1, 2, 3],
            segments.full_prompt_text: [1, 2, 3, 4, 5],
        }
        result = probe_route_boundary_token_prefix(
            segments=segments,
            tokenize=lambda text: mapping[text],
        )
        metadata = result.to_metadata()

        self.assertTrue(result.token_prefix_ok)
        self.assertIsNone(result.reason)
        self.assertEqual(result.stable_prefix_token_count, 3)
        self.assertEqual(result.full_prompt_token_count, 5)
        self.assertEqual(result.token_lcp_with_stable_prefix, 3)
        self.assertIsNone(result.divergence_index)
        self.assertTrue(metadata["route_boundary_token_prefix_ok"])
        self.assertNotIn("placeholder task payload", str(metadata))
        self.assertNotIn("route policy placeholder", str(metadata))

    def test_route_boundary_token_probe_reports_mismatch_without_raw_tokens(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [
                {"role": "system", "content": "route policy placeholder"},
                {"role": "user", "content": "placeholder task payload"},
            ]
        )
        mapping = {
            segments.stable_prefix_text: [1, 2, 3],
            segments.full_prompt_text: [1, 2, 9, 4],
        }
        result = probe_route_boundary_token_prefix(
            segments=segments,
            tokenize=lambda text: mapping[text],
        )
        metadata = result.to_metadata()

        self.assertFalse(result.token_prefix_ok)
        self.assertEqual(result.reason, "stable_prefix_not_token_prefix")
        self.assertEqual(result.token_lcp_with_stable_prefix, 2)
        self.assertEqual(result.divergence_index, 2)
        self.assertNotIn("[1, 2, 3]", str(metadata))
        self.assertNotIn("placeholder task payload", str(metadata))

    def test_split_prompt_by_token_prefix_rejects_non_prefix_boundary(self) -> None:
        def tokenize(text: str) -> list[int]:
            mapping = {
                "prefix": [10, 11],
                "prefix+suffix": [10, 99, 12],
            }
            return mapping[text]

        prefix_tokens, suffix_tokens, full_tokens, reason = split_prompt_by_token_prefix(
            tokenize=tokenize,
            prefix_text="prefix",
            full_text="prefix+suffix",
        )

        self.assertEqual(prefix_tokens, [10, 11])
        self.assertEqual(full_tokens, [10, 99, 12])
        self.assertEqual(suffix_tokens, [])
        self.assertEqual(reason, "prefix_not_token_prefix")

    def test_prefill_only_probe_captures_checkpoint_without_sampling_or_generation(self) -> None:
        result = probe_route_prefix_prefill_only(
            lib=_FakeLib(),
            ctx=object(),
            tokenize=lambda text: {"prefix": [1, 2, 3]}[text],
            prefix_text="prefix",
            prefix_hash="prefix-hash-alpha",
            model_id="model-alpha",
            template_id="template-alpha",
            tool_schema_hash="tool-hash-alpha",
            capability_summary_hash="caps-hash-alpha",
            backend_version="backend-alpha",
            native_version="native-alpha",
        )
        metadata = result.to_metadata()

        self.assertTrue(result.ok)
        self.assertIsNone(result.reason)
        self.assertEqual(result.prefix_token_count, 3)
        self.assertGreater(result.checkpoint_size, 0)
        self.assertEqual(result.decode_calls, 1)
        self.assertEqual(result.sampled_tokens, 0)
        self.assertEqual(result.generated_tokens, 0)
        self.assertFalse(result.sampler_touched)
        self.assertFalse(result.session_history_touched)
        self.assertNotIn("raw prefix body", str(metadata))
        self.assertNotIn("[1, 2, 3]", str(metadata))

    def test_prefill_only_probe_decodes_prefix_in_chunks(self) -> None:
        result = probe_route_prefix_prefill_only(
            lib=_FakeLib(),
            ctx=object(),
            tokenize=lambda text: {"prefix": [1, 2, 3, 4, 5]}[text],
            prefix_text="prefix",
            prefix_hash="prefix-hash-alpha",
            model_id="model-alpha",
            template_id="template-alpha",
            tool_schema_hash="tool-hash-alpha",
            capability_summary_hash="caps-hash-alpha",
            backend_version="backend-alpha",
            native_version="native-alpha",
            decode_step=2,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.decode_calls, 3)
        self.assertEqual(result.sampled_tokens, 0)
        self.assertEqual(result.generated_tokens, 0)

    def test_prefill_only_probe_metadata_stays_content_free(self) -> None:
        result = PrefixPrefillOnlyProbeResult(
            ok=True,
            reason=None,
            prefix_hash="prefix-hash-alpha",
            prefix_token_count=3,
            checkpoint_size=12,
            prefill_ms=5.0,
            decode_calls=1,
        )
        rendered = str(result.to_metadata())

        self.assertNotIn("prefix text", rendered)
        self.assertNotIn("user content", rendered)
        self.assertNotIn("tool output", rendered)

    def test_prefill_only_probe_script_imports_without_running_probe(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "probe_native_route_prefix_prefill_only.py"
        spec = importlib.util.spec_from_file_location("probe_native_route_prefix_prefill_only", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(callable(module.main))

    def test_probe_reports_equivalent_restore_for_stable_tokens(self) -> None:
        mapping = {
            "prefix": [1, 2],
            "full": [1, 2, 3, 4],
        }
        result = probe_prefix_anchor_equivalence(
            lib=_FakeLib(),
            ctx=object(),
            vocab=object(),
            sampler=object(),
            tokenize=lambda text: mapping[text],
            prefix_text="prefix",
            full_text="full",
            model_id="model-alpha",
            template_id="template-alpha",
            tool_schema_hash="tool-hash-alpha",
            capability_summary_hash="caps-hash-alpha",
            backend_version="backend-alpha",
            native_version="native-alpha",
        )

        self.assertTrue(result.ok)
        self.assertIsNone(result.reason)
        self.assertTrue(result.restore_used)
        self.assertEqual(result.prefix_token_count, 2)
        self.assertEqual(result.suffix_token_count, 2)
        self.assertEqual(result.full_token_count, 4)
        self.assertEqual(result.baseline_next_token, result.restored_next_token)
        self.assertTrue(result.logits_match)

    def test_probe_falls_back_on_restore_failure(self) -> None:
        mapping = {
            "prefix": [5, 6],
            "full": [5, 6, 7],
        }
        result = probe_prefix_anchor_equivalence(
            lib=_FailingRestoreLib(),
            ctx=object(),
            vocab=object(),
            sampler=object(),
            tokenize=lambda text: mapping[text],
            prefix_text="prefix",
            full_text="full",
            model_id="model-alpha",
            template_id="template-alpha",
            tool_schema_hash="tool-hash-alpha",
            capability_summary_hash="caps-hash-alpha",
            backend_version="backend-alpha",
            native_version="native-alpha",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "checkpoint_restore_size_mismatch")
        self.assertFalse(result.restore_used)

    def test_probe_metadata_stays_content_free(self) -> None:
        result = PrefixAnchorProbeResult(
            ok=True,
            reason=None,
            prefix_token_count=3,
            suffix_token_count=2,
            full_token_count=5,
            checkpoint_size=12,
            restore_used=True,
            baseline_next_token=7,
            restored_next_token=7,
            logits_hash_baseline="abc123",
            logits_hash_restored="abc123",
            logits_match=True,
        )
        metadata = result.to_metadata()
        rendered = str(metadata)
        self.assertNotIn("prefix text", rendered)
        self.assertNotIn("full text", rendered)
        self.assertNotIn("placeholder", rendered)

    @patch.dict("os.environ", {}, clear=True)
    def test_native_prefill_hook_captures_checkpoint_without_generation(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        client = _prefill_hook_client(segments)

        result = client.capture_route_prefix_prefill_only(segments)
        metadata = result.to_metadata()

        self.assertTrue(result.attempted)
        self.assertTrue(result.succeeded)
        self.assertFalse(result.skipped)
        self.assertTrue(result.restore_ready)
        self.assertEqual(result.prefix_token_count, 4)
        self.assertGreater(result.checkpoint_size_bytes or 0, 0)
        self.assertEqual(result.decode_calls, 2)
        self.assertEqual(result.sampled_tokens, 0)
        self.assertEqual(result.generated_tokens, 0)
        self.assertFalse(result.sampler_touched)
        self.assertFalse(result.session_history_touched)
        self.assertTrue(client._route_prefix_anchor_state.valid)
        self.assertEqual(client._session.cached_prompt_tokens, [1, 2, 3, 4])
        self.assertNotIn("route policy placeholder", str(metadata))
        self.assertNotIn("[1, 2, 3, 4]", str(metadata))

    @patch.dict("os.environ", {"ORBIT_KV_PREFIX_ANCHOR": "off"}, clear=True)
    def test_native_prefill_hook_skips_when_anchor_disabled(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        lib = _FakeLib()
        client = _prefill_hook_client(segments, lib=lib)

        result = client.capture_route_prefix_prefill_only(segments)

        self.assertFalse(result.attempted)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "anchor_disabled")
        self.assertEqual(lib.current_tokens, [])
        self.assertFalse(client._route_prefix_anchor_state.valid)

    @patch.dict("os.environ", {"ORBIT_KV_PREFIX_ANCHOR": "off", "ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT": "1"}, clear=True)
    def test_native_prefill_hook_off_wins_over_legacy_flag(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        result = _prefill_hook_client(segments).capture_route_prefix_prefill_only(segments)

        self.assertFalse(result.attempted)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "anchor_disabled")

    @patch.dict("os.environ", {"ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT": "1"}, clear=True)
    def test_native_prefill_hook_legacy_flag_enables_when_new_var_unset(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        result = _prefill_hook_client(segments).capture_route_prefix_prefill_only(segments)

        self.assertTrue(result.succeeded)
        self.assertTrue(result.restore_ready)

    @patch.dict("os.environ", {}, clear=True)
    def test_native_prefill_hook_skips_tools_off(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        result = _prefill_hook_client(segments).capture_route_prefix_prefill_only(
            segments,
            tools_mode="off",
        )

        self.assertFalse(result.attempted)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "tools_mode_ineligible")

    @patch.dict("os.environ", {}, clear=True)
    def test_native_prefill_hook_fails_safely_on_decode_error(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        client = _prefill_hook_client(segments, lib=_FailingDecodeLib())

        result = client.capture_route_prefix_prefill_only(segments)

        self.assertTrue(result.attempted)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.failed_reason, "prefix_decode_failed:RuntimeError")
        self.assertFalse(result.restore_ready)
        self.assertFalse(client._route_prefix_anchor_state.valid)
        self.assertEqual(client._session.cached_prompt_tokens, [])

    @patch.dict("os.environ", {}, clear=True)
    def test_native_prefill_hook_skips_when_context_is_active(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        client = _prefill_hook_client(segments)
        client._session.cached_prompt_tokens = [9]

        result = client.capture_route_prefix_prefill_only(segments)

        self.assertFalse(result.attempted)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "active_context_present")

    @patch.dict("os.environ", {}, clear=True)
    def test_native_prefill_hook_skips_concurrent_capture(self) -> None:
        segments = render_gemma4_route_prompt_segments(
            [{"role": "system", "content": "route policy placeholder"}]
        )
        client = _prefill_hook_client(segments)
        self.assertTrue(client._route_prefix_prefill_lock.acquire(blocking=False))
        try:
            result = client.capture_route_prefix_prefill_only(segments)
        finally:
            client._route_prefix_prefill_lock.release()

        self.assertFalse(result.attempted)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "prefill_in_flight")

    def test_native_prefill_result_metadata_stays_content_free(self) -> None:
        result = NativeRoutePrefixPrefillResult(
            attempted=True,
            succeeded=True,
            skipped=False,
            prefix_hash="prefix-hash-alpha",
            prefix_token_count=4,
            checkpoint_size_bytes=12,
            prefill_ms=3.5,
            decode_calls=2,
            restore_ready=True,
        )
        rendered = str(result.to_metadata())

        self.assertNotIn("prompt body", rendered)
        self.assertNotIn("token ids", rendered)
        self.assertNotIn("command result", rendered)


if __name__ == "__main__":
    unittest.main()

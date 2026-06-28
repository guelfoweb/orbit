from __future__ import annotations

from ctypes import c_float
import unittest

from orbit.native_llama.prefix_anchor_probe import (
    PrefixAnchorProbeResult,
    probe_route_boundary_token_prefix,
    probe_prefix_anchor_equivalence,
    split_prompt_by_token_prefix,
)
from orbit.native_llama.chat_template import render_gemma4_route_prompt_segments


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

    def llama_get_memory(self, _ctx):
        return object()

    def llama_memory_clear(self, _mem, _full: bool) -> None:
        self.current_tokens = []
        self.clears += 1

    def llama_batch_init(self, n_tokens: int, _embd: int, _n_seq_max: int):
        return _FakeBatch(n_tokens)

    def llama_batch_free(self, _batch) -> None:
        return None

    def llama_decode(self, _ctx, batch) -> int:
        for index in range(batch.n_tokens):
            self.current_tokens.append(_token_value(batch.token[index]))
        return 0

    def llama_synchronize(self, _ctx) -> None:
        return None

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


class PrefixAnchorProbeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

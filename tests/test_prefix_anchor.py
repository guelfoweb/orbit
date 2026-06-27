from __future__ import annotations

import os
import unittest
from unittest import mock

from orbit.native_llama.prefix_anchor import (
    PrefixAnchorState,
    anchor_metadata,
    can_use_prefix_anchor,
    capture_prefix_anchor,
    compute_prefix_anchor_key,
    invalidate_prefix_anchor,
    prefix_anchor_enabled,
    restore_prefix_anchor,
)


class _FakeLib:
    def __init__(self, payload: bytes = b"checkpoint-bytes") -> None:
        self.payload = payload
        self.restore_payloads: list[bytes] = []

    def llama_state_seq_get_size(self, _ctx, _seq_id: int) -> int:
        return len(self.payload)

    def llama_state_seq_get_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        if size != len(self.payload):
            return 0
        for index, byte in enumerate(self.payload):
            buffer[index] = byte
        return len(self.payload)

    def llama_state_seq_set_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        data = bytes(buffer[:size])
        self.restore_payloads.append(data)
        return size


class _FailingRestoreLib(_FakeLib):
    def llama_state_seq_set_data(self, _ctx, buffer, size: int, _seq_id: int) -> int:
        _ = bytes(buffer[:size])
        return size - 1


class PrefixAnchorTests(unittest.TestCase):
    def _stable_kwargs(self) -> dict[str, str]:
        return {
            "model_id": "model-alpha",
            "template_id": "template-alpha",
            "tool_schema_hash": "tool-hash-alpha",
            "capability_summary_hash": "caps-hash-alpha",
            "runtime_policy_hash": "policy-hash-alpha",
            "route_contract_hash": "route-hash-alpha",
            "backend_version": "backend-alpha",
            "native_version": "native-alpha",
            "tools_mode": "on",
        }

    def test_flag_default_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORBIT_KV_PREFIX_ANCHOR_EXPERIMENT", None)
            self.assertFalse(prefix_anchor_enabled())

    def test_key_stable_for_stable_input(self) -> None:
        kwargs = self._stable_kwargs()
        self.assertEqual(compute_prefix_anchor_key(**kwargs), compute_prefix_anchor_key(**kwargs))

    def test_key_changes_with_tool_schema_hash(self) -> None:
        kwargs = self._stable_kwargs()
        first = compute_prefix_anchor_key(**kwargs)
        second = compute_prefix_anchor_key(**{**kwargs, "tool_schema_hash": "tool-hash-beta"})
        self.assertNotEqual(first, second)

    def test_key_changes_with_capability_summary_hash(self) -> None:
        kwargs = self._stable_kwargs()
        first = compute_prefix_anchor_key(**kwargs)
        second = compute_prefix_anchor_key(**{**kwargs, "capability_summary_hash": "caps-hash-beta"})
        self.assertNotEqual(first, second)

    def test_invalidation_clears_checkpoint(self) -> None:
        state = PrefixAnchorState(prefix_hash="anchor-alpha", token_count=12, checkpoint_size=5, checkpoint_data=b"abcde", valid=True)
        invalidated = invalidate_prefix_anchor(state, "tool_schema_changed")
        self.assertFalse(invalidated.valid)
        self.assertEqual(invalidated.invalidation_reason, "tool_schema_changed")
        self.assertEqual(invalidated.checkpoint_size, 0)
        self.assertIsNone(invalidated.checkpoint_data)

    def test_flag_off_prevents_capture(self) -> None:
        kwargs = self._stable_kwargs()
        key = compute_prefix_anchor_key(**kwargs)
        state, metadata = capture_prefix_anchor(
            lib=_FakeLib(),
            ctx=object(),
            prefix_hash=key,
            token_count=24,
            enabled=False,
            **kwargs,
        )
        self.assertFalse(state.valid)
        self.assertEqual(state.invalidation_reason, "anchor_disabled")
        self.assertFalse(metadata["capture_attempted"])
        self.assertEqual(metadata["fallback_reason"], "anchor_disabled")

    def test_capture_and_restore_round_trip_with_explicit_enable(self) -> None:
        kwargs = self._stable_kwargs()
        key = compute_prefix_anchor_key(**kwargs)
        lib = _FakeLib(b"orbit-anchor")
        state, capture_meta = capture_prefix_anchor(
            lib=lib,
            ctx=object(),
            prefix_hash=key,
            token_count=24,
            enabled=True,
            **kwargs,
        )
        self.assertTrue(state.valid)
        self.assertEqual(state.checkpoint_size, len(b"orbit-anchor"))
        self.assertTrue(capture_meta["capture_attempted"])
        ok, restored_state, restore_meta = restore_prefix_anchor(
            state,
            lib=lib,
            ctx=object(),
            prefix_hash=key,
            enabled=True,
            **kwargs,
        )
        self.assertTrue(ok)
        self.assertTrue(restored_state.valid)
        self.assertTrue(restore_meta["restore_used"])
        self.assertEqual(lib.restore_payloads, [b"orbit-anchor"])

    def test_restore_failure_falls_back_safely(self) -> None:
        kwargs = self._stable_kwargs()
        key = compute_prefix_anchor_key(**kwargs)
        state = PrefixAnchorState(
            prefix_hash=key,
            token_count=24,
            model_id=kwargs["model_id"],
            template_id=kwargs["template_id"],
            tool_schema_hash=kwargs["tool_schema_hash"],
            capability_summary_hash=kwargs["capability_summary_hash"],
            runtime_policy_hash=kwargs["runtime_policy_hash"],
            route_contract_hash=kwargs["route_contract_hash"],
            backend_version=kwargs["backend_version"],
            native_version=kwargs["native_version"],
            tools_mode=kwargs["tools_mode"],
            checkpoint_size=4,
            checkpoint_data=b"abcd",
            valid=True,
        )
        ok, restored_state, metadata = restore_prefix_anchor(
            state,
            lib=_FailingRestoreLib(b"abcd"),
            ctx=object(),
            prefix_hash=key,
            enabled=True,
            **kwargs,
        )
        self.assertFalse(ok)
        self.assertFalse(restored_state.valid)
        self.assertEqual(restored_state.invalidation_reason, "checkpoint_restore_size_mismatch")
        self.assertEqual(metadata["fallback_reason"], "checkpoint_restore_size_mismatch")

    def test_can_use_prefix_anchor_rejects_changed_capabilities(self) -> None:
        kwargs = self._stable_kwargs()
        key = compute_prefix_anchor_key(**kwargs)
        state = PrefixAnchorState(
            prefix_hash=key,
            token_count=24,
            model_id=kwargs["model_id"],
            template_id=kwargs["template_id"],
            tool_schema_hash=kwargs["tool_schema_hash"],
            capability_summary_hash=kwargs["capability_summary_hash"],
            runtime_policy_hash=kwargs["runtime_policy_hash"],
            route_contract_hash=kwargs["route_contract_hash"],
            backend_version=kwargs["backend_version"],
            native_version=kwargs["native_version"],
            tools_mode=kwargs["tools_mode"],
            checkpoint_size=4,
            checkpoint_data=b"abcd",
            valid=True,
        )
        ok, reason = can_use_prefix_anchor(
            state,
            prefix_hash=key,
            capability_summary_hash="caps-hash-beta",
            enabled=True,
            **{k: v for k, v in kwargs.items() if k != "capability_summary_hash"},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "capability_summary_changed")

    def test_anchor_metadata_is_hash_only(self) -> None:
        state = PrefixAnchorState(
            prefix_hash="anchor-key-alpha",
            token_count=24,
            checkpoint_size=12,
            valid=True,
            invalidation_reason=None,
        )
        metadata = anchor_metadata(state, enabled=True, anchor_hit=True, restore_attempted=True, restore_used=True)
        self.assertTrue(metadata["anchor_enabled"])
        self.assertEqual(metadata["anchor_key_hash"], "anchor-key-alpha")
        self.assertTrue(metadata["anchor_hit"])
        self.assertTrue(metadata["restore_attempted"])
        self.assertTrue(metadata["restore_used"])
        serialized = str(metadata)
        self.assertNotIn("checkpoint-bytes", serialized)
        self.assertNotIn("placeholder", serialized)


if __name__ == "__main__":
    unittest.main()

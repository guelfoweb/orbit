from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orbit.native_server.app import _mtp_config_payload, _mtp_last_completion_payload, _mtp_last_timing_payload
from orbit.native_llama.mtp_completion import MtpCompletionResult


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "scripts" / "bench_mtp_throughput.py"
SPEC = importlib.util.spec_from_file_location("bench_mtp_throughput", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
bench_mtp = importlib.util.module_from_spec(SPEC)
sys.modules["bench_mtp_throughput"] = bench_mtp
SPEC.loader.exec_module(bench_mtp)


class NativeServerMtpPropsTests(unittest.TestCase):
    def test_mtp_config_payload_is_none_when_mtp_experimental_is_disabled(self) -> None:
        client = SimpleNamespace(
            config=SimpleNamespace(use_mtp_experimental=False),
            _session=SimpleNamespace(ctx_tgt=None, ctx_dft=None),
        )

        self.assertIsNone(_mtp_config_payload(client))

    def test_mtp_config_payload_exposes_available_effective_values(self) -> None:
        client = SimpleNamespace(
            config=SimpleNamespace(use_mtp_experimental=True),
            _session=SimpleNamespace(ctx_tgt=object(), ctx_dft=None),
        )

        payload = _mtp_config_payload(client)

        assert payload is not None
        self.assertEqual(payload["n_max"], 3)
        self.assertIsNone(payload["n_min"])
        self.assertIsNone(payload["p_min"])
        self.assertIsNone(payload["backend_sampling"])
        self.assertTrue(payload["ctx_tgt"])
        self.assertFalse(payload["ctx_dft"])

    def test_mtp_last_completion_payload_is_none_when_completion_is_disabled(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=False,
                success=False,
                error=None,
            )
        )

        self.assertIsNone(_mtp_last_completion_payload(client))

    def test_mtp_last_completion_payload_preserves_zero_values(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                output_tokens=0,
                draft_tokens_total=0,
                accepted_tokens_total=0,
                rejected_tokens_total=0,
                acceptance_ratio=0.0,
                fresh_acceptance_ratio=0.0,
                consumed_acceptance_ratio=0.0,
                reused_draft_tokens_total=0,
                reused_accepted_tokens_total=0,
                reused_rejected_tokens_total=0,
                target_decode_calls=0,
                draft_decode_calls=0,
                elapsed_ms=0.0,
                tokens_per_second=0.0,
                full_accept_steps=0,
                replay_steps=0,
                partial_accept_steps=0,
                partial_no_replay_steps=0,
                replay_fallback_steps=0,
                rollback_tokens_total=0,
                checkpoint_count=0,
                restore_count=0,
            )
        )

        payload = _mtp_last_completion_payload(client)

        assert payload is not None
        self.assertEqual(payload["output_tokens"], 0)
        self.assertEqual(payload["draft_tokens_total"], 0)
        self.assertEqual(payload["accepted_tokens_total"], 0)
        self.assertEqual(payload["rejected_tokens_total"], 0)
        self.assertEqual(payload["acceptance_ratio"], 0.0)
        self.assertEqual(payload["target_decode_calls"], 0)
        self.assertEqual(payload["draft_decode_calls"], 0)
        self.assertEqual(payload["rollback_tokens_total"], 0)
        self.assertEqual(payload["checkpoint_count"], 0)
        self.assertEqual(payload["restore_count"], 0)

    def test_mtp_last_timing_payload_is_none_when_completion_is_disabled(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=False,
                success=False,
                error=None,
            )
        )

        self.assertIsNone(_mtp_last_timing_payload(client))

    def test_mtp_last_timing_payload_is_none_when_timing_json_is_absent(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                timing_json=None,
            )
        )

        self.assertIsNone(_mtp_last_timing_payload(client))

    def test_mtp_last_timing_payload_is_none_when_timing_json_is_malformed(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                timing_json="{",
            )
        )

        self.assertIsNone(_mtp_last_timing_payload(client))

    def test_mtp_last_timing_payload_extracts_summary_and_preserves_zero(self) -> None:
        client = SimpleNamespace(
            last_mtp_completion=MtpCompletionResult(
                enabled=True,
                success=True,
                error=None,
                timing_json=json.dumps(
                    {
                        "summary": {
                            "total_wall_ms": 10.0,
                            "suffix_target_prefill_ms": 0.0,
                            "speculative_loop_ms": 6.0,
                            "speculative_loop_including_suffix_ms": 6.5,
                            "target_validate_ms": 2.0,
                            "draft_generation_ms": 1.5,
                            "checkpoint_restore_ms": 0.0,
                            "sampler_ms": 0.5,
                            "seq_rm_ms": 0.0,
                            "non_loop_overhead_ms": 1.0,
                        }
                    }
                ),
            )
        )

        payload = _mtp_last_timing_payload(client)

        assert payload is not None
        self.assertEqual(payload["total_ms"], 10.0)
        self.assertEqual(payload["suffix_target_prefill_ms"], 0.0)
        self.assertEqual(payload["checkpoint_restore_ms"], 0.0)
        self.assertEqual(payload["seq_rm_ms"], 0.0)
        self.assertEqual(payload["other_ms"], 0.0)


class BenchMtpThroughputMarkdownTests(unittest.TestCase):
    def test_markdown_mtp_diagnostics_uses_measured_ok_rows_only(self) -> None:
        rows = [
            {
                "timestamp": "t0",
                "server_mode": "mtp_on",
                "scenario": "short_chat",
                "phase": "warmup",
                "repetition": 1,
                "prompt": "p",
                "exit_kind": "ok",
                "finish_reason": "stop",
                "wall_ms": 1000.0,
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "cached_tokens": 4,
                "evaluated_tokens": 6,
                "prompt_tokens_per_second": None,
                "generation_tokens_per_second": 2.0,
                "backend_mode": "mtp",
                "mtp_enabled": True,
                "mtp_initialized": True,
                "mtp_failure_reason": None,
                "mtp_last_completion": {"output_tokens": 999, "draft_tokens_total": 999, "accepted_tokens_total": 999, "rejected_tokens_total": 999, "acceptance_ratio": 0.9, "target_decode_calls": 999, "draft_decode_calls": 999, "full_accept_steps": 999, "partial_accept_steps": 999, "partial_no_replay_steps": 999, "rollback_tokens_total": 999, "checkpoint_count": 999, "restore_count": 999, "reused_draft_tokens_total": 999, "reused_accepted_tokens_total": 999, "reused_rejected_tokens_total": 999},
                "multimodal_available": True,
                "in_flight": False,
                "backend_still_in_flight": False,
                "raw_error": None,
            },
            {
                "timestamp": "t1",
                "server_mode": "mtp_on",
                "scenario": "short_chat",
                "phase": "measured",
                "repetition": 1,
                "prompt": "p",
                "exit_kind": "ok",
                "finish_reason": "stop",
                "wall_ms": 1000.0,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 4,
                "evaluated_tokens": 6,
                "prompt_tokens_per_second": None,
                "generation_tokens_per_second": 2.0,
                "backend_mode": "mtp",
                "mtp_enabled": True,
                "mtp_initialized": True,
                "mtp_failure_reason": None,
                "mtp_last_completion": {"output_tokens": 20, "draft_tokens_total": 24, "accepted_tokens_total": 14, "rejected_tokens_total": 10, "acceptance_ratio": 0.5833333333333334, "target_decode_calls": 9, "draft_decode_calls": 8, "full_accept_steps": 4, "partial_accept_steps": 4, "partial_no_replay_steps": 4, "rollback_tokens_total": 10, "checkpoint_count": 9, "restore_count": 0, "reused_draft_tokens_total": 0, "reused_accepted_tokens_total": 0, "reused_rejected_tokens_total": 0},
                "multimodal_available": True,
                "in_flight": False,
                "backend_still_in_flight": False,
                "raw_error": None,
            },
            {
                "timestamp": "t2",
                "server_mode": "mtp_on",
                "scenario": "short_chat",
                "phase": "measured",
                "repetition": 2,
                "prompt": "p",
                "exit_kind": "timeout",
                "finish_reason": "timeout",
                "wall_ms": 2000.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
                "evaluated_tokens": None,
                "prompt_tokens_per_second": None,
                "generation_tokens_per_second": None,
                "backend_mode": "mtp",
                "mtp_enabled": True,
                "mtp_initialized": True,
                "mtp_failure_reason": None,
                "mtp_last_completion": {"output_tokens": 500, "draft_tokens_total": 500, "accepted_tokens_total": 500, "rejected_tokens_total": 500, "acceptance_ratio": 0.5, "target_decode_calls": 500, "draft_decode_calls": 500, "full_accept_steps": 500, "partial_accept_steps": 500, "partial_no_replay_steps": 500, "rollback_tokens_total": 500, "checkpoint_count": 500, "restore_count": 500, "reused_draft_tokens_total": 500, "reused_accepted_tokens_total": 500, "reused_rejected_tokens_total": 500},
                "multimodal_available": True,
                "in_flight": False,
                "backend_still_in_flight": False,
                "raw_error": "timeout",
            },
            {
                "timestamp": "t3",
                "server_mode": "mtp_off",
                "scenario": "short_chat",
                "phase": "measured",
                "repetition": 1,
                "prompt": "p",
                "exit_kind": "ok",
                "finish_reason": "stop",
                "wall_ms": 900.0,
                "prompt_tokens": 10,
                "completion_tokens": 17,
                "cached_tokens": 4,
                "evaluated_tokens": 6,
                "prompt_tokens_per_second": 10.0,
                "generation_tokens_per_second": 3.0,
                "backend_mode": "no-mtp",
                "mtp_enabled": False,
                "mtp_initialized": False,
                "mtp_failure_reason": None,
                "mtp_last_completion": None,
                "multimodal_available": True,
                "in_flight": False,
                "backend_still_in_flight": False,
                "raw_error": None,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            bench_mtp.write_markdown(path, rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn("## MTP diagnostics measured-only", text)
        self.assertIn("| mtp_off | short_chat | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |", text)
        self.assertIn("| mtp_on | short_chat | 20.0 | 24.0 | 14.0 | 10.0 | 0.583 | 9.0 | 8.0 | 4.0 | 4.0 | 4.0 | 10.0 | 9.0 | 0.0 | 0.0 | 0.0 | 0.0 |", text)
        self.assertNotIn("999", text)
        self.assertNotIn("| 500.0 | 500.0 | 500.0 |", text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.backend.payloads import ChatPayloadOptions, build_chat_payload


class PayloadTests(unittest.TestCase):
    def test_build_chat_payload_strips_internal_message_metadata(self) -> None:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "read_file",
                        "content": "compact",
                        "_orbit_original_tool_content": "verbatim",
                        "_orbit_tool_compaction": {"before_tokens": 100},
                    }
                ],
                temperature=0,
                max_tokens=32,
            )
        )

        self.assertEqual(
            payload["messages"],
            [{"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "compact"}],
        )

    def test_build_chat_payload_includes_thinking_flag(self) -> None:
        payload = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
                thinking=True,
            )
        )

        self.assertTrue(payload["thinking"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_build_chat_payload_includes_route_prefix_anchor_only_when_requested(self) -> None:
        baseline = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
            )
        )
        experimental = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
                route_prefix_anchor=True,
            )
        )

        self.assertNotIn("route_prefix_anchor", baseline)
        self.assertTrue(experimental["route_prefix_anchor"])

    def test_build_chat_payload_includes_allow_mtp_only_when_explicit(self) -> None:
        baseline = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
            )
        )
        disabled = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
                allow_mtp_experimental=False,
            )
        )

        self.assertNotIn("allow_mtp_experimental", baseline)
        self.assertFalse(disabled["allow_mtp_experimental"])

    def test_build_chat_payload_includes_final_prefix_experiment_only_when_requested(self) -> None:
        baseline = build_chat_payload(
            ChatPayloadOptions(model="m", messages=[{"role": "user", "content": "hello"}], temperature=0, max_tokens=32)
        )
        experimental = build_chat_payload(
            ChatPayloadOptions(
                model="m",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=32,
                final_prefix_experiment=True,
            )
        )

        self.assertNotIn("final_prefix_experiment", baseline)
        self.assertTrue(experimental["final_prefix_experiment"])

if __name__ == "__main__":
    unittest.main()

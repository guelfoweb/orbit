from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbit.runtime.history_serialization import (
    LEADING_SYSTEM_ONLY,
    serialize_profile_messages,
)


class SharedHistorySerializationTests(unittest.TestCase):
    def test_leading_system_preserved_trailing_demoted(self) -> None:
        msgs = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "evidence card"},
        ]
        out = serialize_profile_messages(msgs, history_serialization=LEADING_SYSTEM_ONLY)
        self.assertEqual([m["role"] for m in out], ["system", "user", "user"])
        self.assertEqual(out[2]["content"], "evidence card")

    def test_multiple_trailing_system_messages(self) -> None:
        msgs = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "citation policy"},
            {"role": "system", "content": "full document"},
        ]
        out = serialize_profile_messages(msgs, history_serialization=LEADING_SYSTEM_ONLY)
        self.assertEqual([m["role"] for m in out], ["system", "user", "user", "user"])

    def test_tool_and_assistant_ordering_unchanged(self) -> None:
        msgs = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "name": "t", "content": "r"},
        ]
        out = serialize_profile_messages(msgs, history_serialization=LEADING_SYSTEM_ONLY)
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "tool"])
        self.assertEqual(out[2]["tool_calls"], [{"id": "1"}])
        self.assertEqual(out[3]["tool_call_id"], "1")

    def test_unknown_contract_is_byte_identical(self) -> None:
        msgs = [
            {"role": "system", "content": "lead"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "trailing"},
        ]
        for contract in (None, "", "other-contract"):
            with self.subTest(contract=contract):
                out = serialize_profile_messages(msgs, history_serialization=contract)
                self.assertEqual(out, [dict(m) for m in msgs])

    def test_input_is_not_mutated(self) -> None:
        msgs = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}]
        serialize_profile_messages(msgs, history_serialization=LEADING_SYSTEM_ONLY)
        self.assertEqual([m["role"] for m in msgs], ["system", "system"])

    def test_empty_message_list(self) -> None:
        self.assertEqual(serialize_profile_messages([], history_serialization=LEADING_SYSTEM_ONLY), [])


class TemplateContractLookupTests(unittest.TestCase):
    def test_verified_templates_resolve_contract(self) -> None:
        from orbit.native_llama.model_profiles import (
            QWEN36_OFFICIAL_TEMPLATE_SHA256,
            QWEN38_OFFICIAL_TEMPLATE_SHA256,
            _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION,
        )

        for digest in (QWEN36_OFFICIAL_TEMPLATE_SHA256, QWEN38_OFFICIAL_TEMPLATE_SHA256):
            self.assertEqual(
                _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION[digest], LEADING_SYSTEM_ONLY
            )

    def test_unknown_template_returns_none(self) -> None:
        from orbit.native_llama.model_profiles import history_serialization_for_template

        self.assertIsNone(history_serialization_for_template("unreviewed template"))
        self.assertIsNone(history_serialization_for_template(""))


if __name__ == "__main__":
    unittest.main()


class ExternalBackendSerializationTests(unittest.TestCase):
    """External llama-server must honour the contract without claiming native."""

    def _backend(self, props):
        from orbit.backend.llama_server import LlamaServerBackend

        b = LlamaServerBackend(base_url="http://127.0.0.1:1", timeout=1)
        b._props_cache = props
        b._props_discovery_status = "ok"
        return b

    def test_upstream_template_hash_resolves_contract(self) -> None:
        from orbit.native_llama.model_profiles import QWEN38_OFFICIAL_TEMPLATE_SHA256

        import hashlib

        # Build a template whose digest is pinned by using the real one.
        tpl = None
        import struct

        path = "models/unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
        try:
            f = open(path, "rb")
        except OSError:
            self.skipTest("model not present")
        with f:
            f.read(4); struct.unpack("<I", f.read(4)); struct.unpack("<Q", f.read(8))
            nkv, = struct.unpack("<Q", f.read(8))
            def rs():
                n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
            S = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            def sk(t):
                if t == 8: rs(); return
                if t == 9:
                    et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
                    for _ in range(n): sk(et)
                    return
                f.read(S[t])
            for _ in range(nkv):
                k = rs(); t, = struct.unpack("<I", f.read(4))
                if k == "tokenizer.chat_template" and t == 8: tpl = rs()
                else: sk(t)
        self.assertEqual(hashlib.sha256(tpl.encode()).hexdigest(), QWEN38_OFFICIAL_TEMPLATE_SHA256)
        b = self._backend({"chat_template": tpl})
        self.assertEqual(b._verified_history_serialization(), LEADING_SYSTEM_ONLY)
        out = b._serialize_for_profile(
            [{"role": "system", "content": "lead"},
             {"role": "user", "content": "q"},
             {"role": "system", "content": "card"}]
        )
        self.assertEqual([m["role"] for m in out], ["system", "user", "user"])

    def test_unknown_template_gets_no_normalization(self) -> None:
        b = self._backend({"chat_template": "some unreviewed template"})
        self.assertIsNone(b._verified_history_serialization())
        msgs = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}]
        self.assertEqual(b._serialize_for_profile(msgs), msgs)

    def test_absent_props_gets_no_normalization(self) -> None:
        b = self._backend({})
        self.assertIsNone(b._verified_history_serialization())

    def test_native_compatibility_block_is_preferred(self) -> None:
        b = self._backend({
            "model_compatibility": {"verified": True, "history_serialization": LEADING_SYSTEM_ONLY},
        })
        self.assertEqual(b._verified_history_serialization(), LEADING_SYSTEM_ONLY)

    def test_unverified_compatibility_block_is_ignored(self) -> None:
        b = self._backend({
            "model_compatibility": {"verified": False, "history_serialization": LEADING_SYSTEM_ONLY},
        })
        self.assertIsNone(b._verified_history_serialization())

    def test_recognized_template_does_not_enable_exact_token_counting(self) -> None:
        """Item-6 invariant: normalization must never imply exact admission.

        A non-native server carrying a recognized template digest resolves the
        message-shape contract, but exact token counting must stay unavailable
        so counting and generation cannot desynchronize.
        """
        b = self._backend({"chat_template": "x", "model_compatibility": {
            "verified": True, "history_serialization": LEADING_SYSTEM_ONLY}})
        self.assertEqual(b._verified_history_serialization(), LEADING_SYSTEM_ONLY)
        self.assertFalse(b._is_orbit_native_backend())
        self.assertIsNone(b.count_chat_tokens([{"role": "user", "content": "q"}]))
        self.assertFalse(b.supports_exact_context_admission())

    def test_external_backend_remains_non_native(self) -> None:
        b = self._backend({"chat_template": "x"})
        self.assertFalse(b._is_orbit_native_backend())

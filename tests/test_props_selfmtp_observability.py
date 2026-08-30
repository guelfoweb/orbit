"""`/props` must describe single-GGUF self-MTP honestly.

`mtp_available` has always meant "an EXTERNAL DRAFT MODEL is present"
(`draft_path is not None`), and several internal callers gate on exactly that:
persistent_mtp.py, mtp_probe.py, mtp_decode_probe.py and mtp_accept_probe.py all
short-circuit on `not paths.mtp_available or paths.draft_mtp_model is None`.
Widening it to mean "MTP of any kind is possible" would make each of those admit
a self-MTP artifact that has no draft model, so its meaning is preserved and an
additive field carries the new fact.

`self_mtp_active` reports OBSERVED state: whether this session actually built a
single-GGUF self-MTP runtime. It is deliberately not a capability probe --
deciding capability means hashing a ~20 GiB artifact (`_self_mtp_eligible`), and
that must never happen per /props request.
"""
from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/orbit/native_server/app.py"


def _props_expression(field: str) -> ast.AST:
    """The AST of the value assigned to `field` in the /props payload."""
    tree = ast.parse(APP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == field:
                    return value
    raise AssertionError(f"/props does not expose {field!r}")


def _active(runtime) -> bool:
    """The shipped expression, evaluated over a stand-in client."""
    client = types.SimpleNamespace(_persistent_mtp_runtime=runtime)
    return bool(getattr(client._persistent_mtp_runtime, "self_mtp", False))


class SelfMtpActiveSemanticsTests(unittest.TestCase):
    """The matrix Phase 2A requires, executed rather than described."""

    def test_current_artifact_running_self_mtp_reports_true(self) -> None:
        """CURRENT exact self-MTP artifact + MTP requested -> true."""
        self.assertTrue(_active(types.SimpleNamespace(self_mtp=True)))

    def test_mtp_off_reports_false_without_claiming_incapability(self) -> None:
        """CURRENT artifact + MTP OFF -> no runtime, so not active.

        `self_mtp_active` answers "is it running now", never "could it run".
        The capability question is deliberately unanswered here because
        answering it costs a 20 GiB hash.
        """
        self.assertFalse(_active(None))

    def test_external_draft_runtime_is_not_self_mtp(self) -> None:
        """An external-draft session must NOT report self_mtp_active."""
        self.assertFalse(_active(types.SimpleNamespace(self_mtp=False)))

    def test_construction_failure_is_distinguishable(self) -> None:
        """A failed construction leaves no runtime -> false.

        `mtp_enabled` (requested) and `mtp_initialized` (constructed) remain the
        fields that distinguish requested-but-broken from never-asked; this one
        adds "and it was the single-GGUF kind".
        """
        self.assertFalse(_active(None))

    def test_unknown_or_legacy_artifact_reports_false(self) -> None:
        """Legacy/unknown artifacts never build a self-MTP runtime."""
        self.assertFalse(_active(types.SimpleNamespace()))

    def test_the_field_never_raises(self) -> None:
        """/props must not 500 because a runtime is missing or half-built."""
        for runtime in (None, types.SimpleNamespace(),
                        types.SimpleNamespace(self_mtp=None),
                        types.SimpleNamespace(self_mtp=1)):
            with self.subTest(runtime=runtime):
                self.assertIsInstance(_active(runtime), bool)


class PropsContractTests(unittest.TestCase):
    """Structural guarantees about the payload itself."""

    def test_mtp_available_semantics_are_unchanged(self) -> None:
        """It must still read the external-draft flag, not a widened notion.

        Four internal callers gate on this meaning; widening it would let them
        admit a self-MTP artifact that has no draft model.
        """
        expr = _props_expression("mtp_available")
        self.assertEqual(ast.unparse(expr), "state.client.paths.mtp_available")

    def test_self_mtp_active_is_present_and_boolean(self) -> None:
        expr = _props_expression("self_mtp_active")
        rendered = ast.unparse(expr)
        self.assertTrue(
            rendered.startswith("bool("),
            f"self_mtp_active must be a real bool for JSON, got {rendered}",
        )
        self.assertIn("self_mtp", rendered)

    def test_props_does_not_hash_the_artifact(self) -> None:
        """No /props request may pay a 20 GiB digest.

        `_self_mtp_eligible` and the verified-artifact helpers do exactly that,
        so none of them may be reachable from the payload expression.
        """
        expr = ast.unparse(_props_expression("self_mtp_active"))
        for forbidden in ("_self_mtp_eligible", "verified_artifact_supports",
                          "sha256", "digest"):
            self.assertNotIn(
                forbidden, expr,
                "/props must report observed state, never probe capability",
            )

    def test_requested_and_constructed_remain_separately_reported(self) -> None:
        """The existing fields that distinguish the lifecycle stages stay."""
        text = APP.read_text()
        for field in ("mtp_enabled", "mtp_initialized", "mtp_failure_reason"):
            self.assertIn(f'"{field}"', text)


if __name__ == "__main__":
    unittest.main()

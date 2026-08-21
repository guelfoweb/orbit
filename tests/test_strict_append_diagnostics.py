"""Diagnostics for the first strict-append cache miss.

A miss turns a cheap turn into a full cold prefill, and the cost hinges on a
single divergence the logs never named. These cover what is reported and,
just as importantly, that reporting changes nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from orbit.native_llama import kv_diag
from orbit.native_llama.kv_diag import (
    STRICT_APPEND_WINDOW_TOKENS,
    emit_strict_append_miss,
    strict_append_miss,
)


class StrictAppendMissTests(unittest.TestCase):
    def test_exact_prefix_reports_no_miss(self) -> None:
        self.assertIsNone(strict_append_miss(committed=[1, 2, 3], prompt=[1, 2, 3, 4, 5]))

    def test_mismatch_reports_exact_index(self) -> None:
        out = strict_append_miss(committed=[1, 2, 3, 4], prompt=[1, 2, 9, 4, 5])
        assert out is not None
        self.assertEqual(out["first_mismatch_index"], 2)
        self.assertEqual(out["expected_token"], 3)
        self.assertEqual(out["actual_token"], 9)
        self.assertEqual(out["reason"], "prefix_mismatch_at_token_2")

    def test_mismatch_at_first_token(self) -> None:
        out = strict_append_miss(committed=[7, 8, 9], prompt=[1, 8, 9, 10])
        assert out is not None
        self.assertEqual(out["first_mismatch_index"], 0)
        self.assertEqual(out["expected_token"], 7)
        self.assertEqual(out["actual_token"], 1)
        self.assertEqual(out["window_start"], 0)

    def test_mismatch_at_last_committed_token(self) -> None:
        committed = [1, 2, 3, 4, 5]
        out = strict_append_miss(committed=committed, prompt=[1, 2, 3, 4, 99, 6])
        assert out is not None
        self.assertEqual(out["first_mismatch_index"], 4)
        self.assertEqual(out["expected_token"], 5)
        self.assertEqual(out["actual_token"], 99)

    def test_prompt_shorter_than_committed_is_a_miss(self) -> None:
        out = strict_append_miss(committed=[1, 2, 3, 4], prompt=[1, 2])
        assert out is not None
        self.assertEqual(out["first_mismatch_index"], None)
        self.assertEqual(out["reason"], "prompt_not_longer_than_committed")

    def test_equal_sequences_are_a_miss_not_an_append(self) -> None:
        """Nothing new to evaluate is a distinct case from a divergence."""
        out = strict_append_miss(committed=[1, 2, 3], prompt=[1, 2, 3])
        assert out is not None
        self.assertEqual(out["reason"], "prompt_not_longer_than_committed")

    def test_empty_committed_is_named_not_crashed(self) -> None:
        out = strict_append_miss(committed=[], prompt=[1, 2, 3])
        assert out is not None
        self.assertEqual(out["reason"], "no_committed_sequence")

    def test_hashes_distinguish_sequences(self) -> None:
        a = strict_append_miss(committed=[1, 2, 3, 4], prompt=[1, 2, 9])
        b = strict_append_miss(committed=[1, 2, 3, 4], prompt=[1, 2, 8])
        assert a is not None and b is not None
        self.assertEqual(a["committed_hash"], b["committed_hash"])
        self.assertNotEqual(a["prompt_hash"], b["prompt_hash"])

    def test_identity_fields_are_carried(self) -> None:
        out = strict_append_miss(
            committed=[1, 2], prompt=[9, 9, 9],
            session_id="sess-1", profile_id="prof-1", lifecycle="phase-1",
        )
        assert out is not None
        self.assertEqual(out["session_id"], "sess-1")
        self.assertEqual(out["profile_id"], "prof-1")
        self.assertEqual(out["lifecycle"], "phase-1")


class BoundednessTests(unittest.TestCase):
    """Bounded output is the privacy property, not a formatting preference."""

    def test_window_is_bounded_regardless_of_sequence_length(self) -> None:
        committed = list(range(10_000))
        prompt = list(range(10_000))
        prompt[5000] = -1
        out = strict_append_miss(committed=committed, prompt=prompt)
        assert out is not None
        self.assertEqual(out["first_mismatch_index"], 5000)
        self.assertLessEqual(len(out["committed_window"]), STRICT_APPEND_WINDOW_TOKENS * 2)
        self.assertLessEqual(len(out["prompt_window"]), STRICT_APPEND_WINDOW_TOKENS * 2)

    # Every key the payload is allowed to carry, and what shape its value may
    # take. An allowlist rather than a size limit: the point is that source or
    # model text cannot appear at all, and a short string is text just the same
    # as a long one. A new field has to be added here deliberately, which is
    # the moment to ask whether it can carry content.
    ALLOWED_FIELDS = {
        "reason": "identifier",
        "committed_tokens": int,
        "prompt_tokens": int,
        "committed_hash": "hash",
        "prompt_hash": "hash",
        "first_mismatch_index": int,
        "expected_token": int,
        "actual_token": int,
        "window_start": int,
        "committed_window": "tokens",
        "prompt_window": "tokens",
        "session_id": "identifier",
        "profile_id": "identifier",
        "lifecycle": "identifier",
    }

    def _assert_payload_shape(self, out: dict) -> None:
        unknown = set(out) - set(self.ALLOWED_FIELDS)
        self.assertEqual(unknown, set(), f"unexpected field(s) in payload: {unknown}")
        for key, value in out.items():
            if value is None:
                continue
            shape = self.ALLOWED_FIELDS[key]
            if shape is int:
                self.assertIsInstance(value, int, f"{key} is not an integer")
            elif shape == "hash":
                self.assertRegex(value, r"^[0-9a-f]{16}$", f"{key} is not a digest")
            elif shape == "tokens":
                self.assertIsInstance(value, list, f"{key} is not a token list")
                for token in value:
                    self.assertIsInstance(token, int, f"{key} holds a non-integer")
            else:  # identifier: a short machine-generated label, never content
                self.assertIsInstance(value, str)
                self.assertRegex(
                    value, r"^[A-Za-z0-9_.:-]+$", f"{key} carries free text"
                )

    def test_payload_carries_no_text(self) -> None:
        """Only known machine-generated fields may appear, whatever their size.

        The earlier version of this test allowed any string under 80 chars, so
        a realistic leak -- a line of source, a credential, a sentence of model
        output -- passed unnoticed. Shape and membership are what actually
        distinguish an identifier from content.
        """
        out = strict_append_miss(
            committed=[1, 2, 3], prompt=[1, 5, 6, 7],
            session_id="sess-1", profile_id="prof-1", lifecycle="phase-1",
        )
        assert out is not None
        self._assert_payload_shape(out)

    def test_payload_shape_holds_for_every_outcome(self) -> None:
        """The allowlist must hold on all branches, not just the common one."""
        for committed, prompt in (
            ([1, 2, 3], [1, 5, 6, 7]),   # internal mismatch
            ([7, 8], [1, 8, 9]),         # first-token mismatch
            ([1, 2, 3, 4], [1, 2]),      # prompt shorter
            ([1, 2, 3], [1, 2, 3]),      # equal length
            ([], [1, 2, 3]),             # no committed sequence
        ):
            out = strict_append_miss(committed=committed, prompt=prompt)
            assert out is not None
            self._assert_payload_shape(out)

    def test_full_sequences_are_never_included(self) -> None:
        committed = list(range(500))
        prompt = list(range(500)) + [999]
        prompt[100] = -5
        out = strict_append_miss(committed=committed, prompt=prompt)
        assert out is not None
        serialised = json.dumps(out)
        self.assertNotIn(str(committed[300]), serialised.split('"window_start"')[0][:200])
        self.assertLess(len(serialised), 1000, "payload is not bounded")


class EmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        kv_diag.reset_diagnostics_for_tests()

    def tearDown(self) -> None:
        os.environ.pop("ORBIT_KV_DIAG", None)
        os.environ.pop("ORBIT_KV_DIAG_FILE", None)
        kv_diag.reset_diagnostics_for_tests()

    def test_disabled_diagnostics_emit_nothing(self) -> None:
        os.environ.pop("ORBIT_KV_DIAG", None)
        kv_diag.reset_diagnostics_for_tests()
        self.assertIsNone(
            emit_strict_append_miss(committed=[1, 2], prompt=[9, 9, 9])
        )

    def test_enabled_emission_records_authoritative_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag.jsonl"
            os.environ["ORBIT_KV_DIAG"] = "1"
            os.environ["ORBIT_KV_DIAG_FILE"] = str(path)
            kv_diag.reset_diagnostics_for_tests()
            out = emit_strict_append_miss(
                committed=[1, 2, 3], prompt=[1, 9, 3, 4],
                seq_rm_result=False, memory_cleared=True,
                reused_prompt_tokens=0, evaluated_prompt_tokens=4,
            )
            self.assertIsNotNone(out)
            written = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(written), 1)
            record = written[0]
            self.assertEqual(record["event"], "kv_diag_strict_append_miss")
            self.assertEqual(record["first_mismatch_index"], 1)
            self.assertIs(record["seq_rm_result"], False)
            self.assertIs(record["memory_cleared"], True)
            self.assertEqual(record["reused_prompt_tokens"], 0)
            self.assertEqual(record["evaluated_prompt_tokens"], 4)

    def test_exact_prefix_emits_nothing_even_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag.jsonl"
            os.environ["ORBIT_KV_DIAG"] = "1"
            os.environ["ORBIT_KV_DIAG_FILE"] = str(path)
            kv_diag.reset_diagnostics_for_tests()
            self.assertIsNone(
                emit_strict_append_miss(committed=[1, 2], prompt=[1, 2, 3])
            )
            self.assertFalse(path.exists() and path.read_text().strip())


class ObservationalOnlyTests(unittest.TestCase):
    """The diagnostic must not be able to change what it observes."""

    def test_inputs_are_not_mutated(self) -> None:
        committed = [1, 2, 3, 4]
        prompt = [1, 2, 9, 4]
        before_c, before_p = list(committed), list(prompt)
        strict_append_miss(committed=committed, prompt=prompt)
        self.assertEqual(committed, before_c)
        self.assertEqual(prompt, before_p)

    def test_repeated_calls_are_deterministic(self) -> None:
        args = dict(committed=[1, 2, 3, 4], prompt=[1, 2, 9, 4, 5])
        self.assertEqual(strict_append_miss(**args), strict_append_miss(**args))


if __name__ == "__main__":
    unittest.main()

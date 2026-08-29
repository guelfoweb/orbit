"""The generated-token accessor contract, and the ban on retokenized identity.

Committed identity must describe the exact token ids the model decoded into KV.
Reconstructing them by retokenizing the final text is not equivalent: a text
round trip is not guaranteed to reproduce the same ids, so the identity could
claim a prefix that is not physically resident -- a false cache hit, which is a
correctness bug rather than a slow path.

These drive the real reader and the real publisher.
"""
from __future__ import annotations

import ctypes
import unittest
from unittest.mock import MagicMock

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.mtp_completion import MtpCompletionResult
from orbit.native_llama.persistent_mtp import _read_generated_tokens


class _Lib:
    """A stand-in for the native library implementing the documented contract."""

    def __init__(self, ids, *, short_write=False, absent=False):
        self._ids = list(ids)
        self._short_write = short_write
        if absent:
            del self.orbit_mtp_session_last_generated_token_count

    def orbit_mtp_session_last_generated_token_count(self, _handle):
        return len(self._ids)

    def orbit_mtp_session_last_generated_tokens(self, _handle, out, capacity):
        count = len(self._ids)
        if capacity < count:
            return count          # reports the required size, writes nothing
        written = count - 1 if self._short_write else count
        for i in range(written):
            out[i] = self._ids[i]
        return written


class GeneratedTokenReaderTests(unittest.TestCase):
    def test_reads_the_exact_ids(self) -> None:
        self.assertEqual(_read_generated_tokens(_Lib([5, 6, 7]), None), (5, 6, 7))

    def test_count_matches_length(self) -> None:
        lib = _Lib([1, 2, 3, 4])
        ids = _read_generated_tokens(lib, None)
        self.assertEqual(len(ids), lib.orbit_mtp_session_last_generated_token_count(None))

    def test_empty_generation_yields_empty(self) -> None:
        self.assertEqual(_read_generated_tokens(_Lib([]), None), ())

    def test_short_write_fails_closed(self) -> None:
        """A partial copy must yield nothing, never a truncated identity."""
        self.assertEqual(_read_generated_tokens(_Lib([1, 2, 3], short_write=True), None), ())

    def test_absent_accessor_fails_closed(self) -> None:
        """An older shim yields no ids rather than a wrong answer."""
        class Bare:
            pass
        self.assertEqual(_read_generated_tokens(Bare(), None), ())

    def test_large_ids_survive_the_round_trip(self) -> None:
        """No signed/unsigned truncation on realistic vocabulary ids."""
        ids = [0, 1, 32000, 151643, 262143]
        self.assertEqual(_read_generated_tokens(_Lib(ids), None), tuple(ids))

    def test_buffer_is_sized_from_the_native_count(self) -> None:
        captured = {}
        class Recorder(_Lib):
            def orbit_mtp_session_last_generated_tokens(self, handle, out, capacity):
                captured["capacity"] = capacity
                return super().orbit_mtp_session_last_generated_tokens(handle, out, capacity)
        _read_generated_tokens(Recorder([9, 9, 9]), None)
        self.assertEqual(captured["capacity"], 3)


class NoRetokenizationTests(unittest.TestCase):
    """Identity must come from native ids, never from the decoded text."""

    def test_native_ids_win_over_a_divergent_retokenization(self) -> None:
        """The decisive case: text would retokenize to something different.

        Native decoded [10, 11, 12]. If the publisher retokenized the final text
        it would get [10, 99, 12] -- a plausible-looking but WRONG identity that
        claims a prefix the model never made resident.
        """
        c = object.__new__(NativeLlamaClient)
        c._session = MagicMock()
        c._session.committed_sequence_tokens = []
        # Any retokenization of the text would produce the corrupted sequence.
        c.tokenize = lambda text: [1, 2] if text == "prompt" else [10, 99, 12]
        c._qwen3_coder_native_protocol = lambda: False
        c._invalidate_committed_sequence = (
            lambda: setattr(c._session, "committed_sequence_tokens", [])
        )

        result = MtpCompletionResult(
            enabled=True, success=True, error=None,
            pair_canonical=True, generated_tokens=(10, 11, 12),
            resident_tokens=(1, 2, 10, 11),
            content="whatever the text happens to be",
        )
        c._publish_mtp_committed_identity(result, "prompt")

        published = c._session.committed_sequence_tokens
        self.assertEqual(published, [1, 2, 10, 11],
                         "identity must be the natively measured residency")
        self.assertNotIn(99, published,
                         "a retokenized id must never reach committed identity")

    def test_publisher_never_touches_result_content(self) -> None:
        """There must be no hidden text round trip in the publication path.

        Asserted over names the code actually references, not raw source text:
        the explanatory comment in this helper names the rejected approaches, so
        a substring check would bind to prose rather than behaviour.
        """
        import ast
        import inspect

        fn = ast.parse(
            inspect.getsource(
                NativeLlamaClient._publish_mtp_committed_identity
            ).lstrip()
        ).body[0]
        # Attribute nodes plus string constants: fields are reached through
        # getattr(result, "...") as well as direct attribute access.
        names = {
            node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)
        } | {
            node.value for node in ast.walk(fn)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("content", names,
                         "publication must not read decoded text")
        self.assertIn("resident_tokens", names,
                      "identity must come from the measured resident sequence")


if __name__ == "__main__":
    unittest.main()

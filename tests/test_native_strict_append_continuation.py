from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.native_llama.client import NativeLlamaClient
from orbit.native_llama.session_state import NativeSessionState


class _Memory:
    def __init__(self) -> None:
        self.cleared = 0
        self.seq_rm_calls: list[tuple[int, int, int]] = []
        self.seq_rm_result = False


class _Lib:
    def __init__(self, memory: _Memory) -> None:
        self._memory = memory

    def llama_get_memory(self, _ctx):
        return self._memory

    def llama_memory_clear(self, memory, _data):
        memory.cleared += 1

    def llama_memory_seq_rm(self, memory, seq, p0, p1):
        memory.seq_rm_calls.append((seq, p0, p1))
        return memory.seq_rm_result


def _client(committed: list[int], *, profile_id: str = "orbit-ornith15-native-v1"):
    client = NativeLlamaClient.__new__(NativeLlamaClient)
    memory = _Memory()
    client.lib = types.SimpleNamespace(lib=_Lib(memory))
    client._session = NativeSessionState(session_id="test")
    client._session.ctx_tgt = object()
    client._session.committed_sequence_tokens = list(committed)
    client.model_profile = types.SimpleNamespace(verified=True, profile_id=profile_id)
    return client, memory


class StrictAppendContinuationTests(unittest.TestCase):
    def test_exact_append_reuses_committed_sequence(self) -> None:
        client, memory = _client([1, 2, 3, 4])

        common = client._prepare_memory_for_prompt([1, 2, 3, 4, 5, 6])

        self.assertEqual(common, 4)
        # The resident sequence must be left intact: no trim, no clear.
        self.assertEqual(memory.seq_rm_calls, [])
        self.assertEqual(memory.cleared, 0)

    def test_one_token_divergence_is_rejected(self) -> None:
        client, memory = _client([1, 2, 3, 4])

        common = client._prepare_memory_for_prompt([1, 2, 9, 4, 5, 6])

        self.assertNotEqual(common, 4)
        self.assertTrue(memory.seq_rm_calls or memory.cleared)

    def test_shorter_prompt_takes_the_fallback_path(self) -> None:
        client, memory = _client([1, 2, 3, 4, 5])

        client._prepare_memory_for_prompt([1, 2, 3])

        # Must reach the pre-existing logic, not the fast path.
        self.assertTrue(memory.seq_rm_calls or memory.cleared)

    def test_equal_length_prompt_is_rejected(self) -> None:
        # Strict continuation requires a genuine suffix to prefill.
        client, memory = _client([1, 2, 3, 4])

        client._prepare_memory_for_prompt([1, 2, 3, 4])

        self.assertTrue(memory.seq_rm_calls or memory.cleared)

    def test_different_generated_history_is_rejected(self) -> None:
        # Same prompt prefix, different committed generation -> not an extension.
        client, _ = _client([1, 2, 3, 99])

        common = client._prepare_memory_for_prompt([1, 2, 3, 4, 5])

        self.assertNotEqual(common, 4)

    def test_empty_committed_state_falls_back(self) -> None:
        client, memory = _client([])

        client._prepare_memory_for_prompt([1, 2, 3])

        self.assertEqual(memory.cleared, 1)

    def test_multiple_consecutive_appends_advance_state(self) -> None:
        client, memory = _client([1, 2])
        self.assertEqual(client._prepare_memory_for_prompt([1, 2, 3]), 2)

        client._session.committed_sequence_tokens = [1, 2, 3, 4]
        self.assertEqual(client._prepare_memory_for_prompt([1, 2, 3, 4, 5]), 4)
        self.assertEqual(memory.seq_rm_calls, [])

    def test_qwen3_coder_never_uses_strict_continuation(self) -> None:
        client, memory = _client([1, 2, 3], profile_id="orbit-qwen3-coder-native-v1")

        common = client._prepare_memory_for_prompt([1, 2, 3, 4])

        # Qwen3-Coder keeps its full-prefill correctness policy.
        self.assertEqual(common, 0)
        self.assertEqual(memory.cleared, 1)

    def test_strict_path_never_requires_partial_seq_rm(self) -> None:
        # iSWA refuses partial removal; the fast path must not depend on it.
        client, memory = _client([1, 2, 3, 4])
        memory.seq_rm_result = False

        common = client._prepare_memory_for_prompt([1, 2, 3, 4, 5])

        self.assertEqual(common, 4)
        self.assertEqual(memory.seq_rm_calls, [])

    def test_fallback_path_invalidates_committed_sequence(self) -> None:
        # The fallback rewrites KV; identity must not survive it.
        client, _ = _client([1, 2, 3, 4])

        client._prepare_memory_for_prompt([9, 9, 9])

        self.assertEqual(client._session.committed_sequence_tokens, [])

    def test_clear_target_memory_invalidates_committed_sequence(self) -> None:
        # _clear_target_memory is the repo's main KV wipe: ~25 call sites,
        # including every anchor restore/fallback branch.
        client, _ = _client([1, 2, 3, 4])
        client._clear_target_memory()

        self.assertEqual(client._session.committed_sequence_tokens, [])

    def test_real_reset_session_state_invalidates_committed_sequence(self) -> None:
        client, _ = _client([1, 2, 3, 4])
        client.reset_cancel = lambda: None
        client._invalidate_final_prefix = lambda _r: None
        client._invalidate_qwen_route_prefix = lambda _r: None
        client._invalidate_qwen36_shell_tool_prefix = lambda _r: None
        client._invalidate_qwen3_coder_route_prefix = lambda _r, **k: None

        try:
            client.reset_session_state()
        except Exception:
            # Any unstubbed collaborator is irrelevant: the invalidation runs
            # before them, which is what this asserts.
            pass

        self.assertEqual(client._session.committed_sequence_tokens, [])


class CommittedSequenceRecordingTests(unittest.TestCase):
    """The recorded set must contain exactly the tokens decoded into KV.

    Driving the full generation loop needs a large ctypes surface, so these
    assert the two structural guarantees directly on the source: an EOG token
    breaks BEFORE llama_decode, and the append is gated on decode_rc == 0.
    A regression in either ordering silently corrupts the committed identity.
    """

    SOURCE = (SRC / "orbit" / "native_llama" / "client.py").read_text(encoding="utf-8")

    def _generation_loop(self) -> str:
        marker = "    def _generate_from_current_context("
        start = self.SOURCE.index(marker)
        end = self.SOURCE.index("\n    def ", start + len(marker))
        return self.SOURCE[start:end]

    def test_eog_breaks_before_decode_so_it_is_never_committed(self) -> None:
        body = self._generation_loop()
        eog = body.index("llama_vocab_is_eog")
        decode = body.index("llama_decode(")
        append = body.index("last_committed_generated_tokens.append")

        self.assertLess(eog, decode, "EOG check must precede llama_decode")
        self.assertLess(decode, append, "append must follow the decode call")
        # The break belongs to the EOG branch, before any decode.
        self.assertIn("break", body[eog:decode])

    def test_append_is_gated_on_successful_decode(self) -> None:
        body = self._generation_loop()
        append_at = body.index("last_committed_generated_tokens.append")
        preceding = body[:append_at]

        self.assertIn("if decode_rc == 0:", preceding[-120:])

    def test_commit_requires_complete_prefill_and_no_cancel(self) -> None:
        self.assertIn("if cancelled or processed < n_prompt:", self.SOURCE)
        self.assertIn("self._commit_sequence(", self.SOURCE)


class InvalidationCoverageTests(unittest.TestCase):
    """Every KV-mutating path must drop committed identity.

    These assert POSITION via AST, not mere substring presence. A substring
    test cannot tell a first statement from an unreachable branch, and that is
    precisely the misplacement bug class this feature has already suffered
    twice.
    """

    SOURCE = (SRC / "orbit" / "native_llama" / "client.py").read_text(encoding="utf-8")
    TREE = ast.parse(SOURCE)

    def _function(self, name: str) -> ast.FunctionDef:
        for node in ast.walk(self.TREE):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"function {name} not found")

    @staticmethod
    def _is_invalidation(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, "attr", None) == "_invalidate_committed_sequence"
        )

    def _body_after_docstring(self, fn: ast.FunctionDef) -> list[ast.stmt]:
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant
        ):
            body = body[1:]
        return body

    def _top_level_invalidation_line(self, name: str) -> int | None:
        for stmt in self._body_after_docstring(self._function(name)):
            if self._is_invalidation(stmt):
                return stmt.lineno
        return None

    def test_mtp_invalidates_as_its_first_statement(self) -> None:
        # Must run before every early return, not inside a conditional branch.
        body = self._body_after_docstring(self._function("_try_complete_with_mtp_experimental"))
        self.assertTrue(self._is_invalidation(body[0]))

    def test_multimodal_invalidates_before_clearing_kv(self) -> None:
        fn = self._function("_complete_prompt_multimodal")
        line = self._top_level_invalidation_line("_complete_prompt_multimodal")
        self.assertIsNotNone(line, "multimodal path must invalidate at top level")
        clears = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "llama_memory_clear"
        ]
        self.assertTrue(clears)
        self.assertLess(line, min(clears), "invalidation must precede the KV clear")

    def test_clear_target_memory_invalidates_at_top_level(self) -> None:
        self.assertIsNotNone(self._top_level_invalidation_line("_clear_target_memory"))

    def test_load_invalidates_at_top_level(self) -> None:
        self.assertIsNotNone(self._top_level_invalidation_line("load"))

    def test_reset_session_state_invalidates_at_top_level(self) -> None:
        self.assertIsNotNone(self._top_level_invalidation_line("reset_session_state"))

    def test_complete_prompt_exception_handler_invalidates(self) -> None:
        fn = self._function("complete_prompt")
        handlers = [h for node in ast.walk(fn) if isinstance(node, ast.Try) for h in node.handlers]
        self.assertTrue(handlers)
        self.assertTrue(
            any(any(self._is_invalidation(s) for s in h.body) for h in handlers),
            "an exception must drop committed identity",
        )


if __name__ == "__main__":
    unittest.main()

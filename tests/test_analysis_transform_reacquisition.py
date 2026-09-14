"""A re-derivation of a deterministic stage costs no useful action.

IBAN-EVIDENCE-GROUNDING-CLOSURE-1, defect A, at the runtime seam. The transform
preflight decodes the artifact's stages and records each as authoritative
evidence before any model call. An action whose output only reproduces one of
those stages re-derives a fact the session already holds exactly -- the
source-reacquisition case, one seam inward. These tests prove what the runtime
does with that: the program still runs and its output is still recorded, but it
does not consume an action slot, it does not become new useful evidence, and it
feeds the NO_PROGRESS path.

The suppression is byte-exact (or the source-plus-recomputable-property form)
against a recorded stage. It deliberately does NOT depend on the observed
failure mode (a re-derivation whose arithmetic is wrong): a CORRECT
re-derivation, which fails nothing, is exactly what must be suppressed, and a
failed one reaches the model as the failure it is.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.backend.base import ChatResult, TokenCount  # noqa: E402
from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_deobfuscate import TransformStage  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    TRANSFORM_REACQUISITION,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.analysis_sandbox import AnalysisResult, DerivedArtifact  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

CTX = 8192
# A plain artifact; the stage below is what the action re-derives, not this.
SOURCE = "var x = 1;\n"
# The decoded stage the deterministic pass is standing in for -- the shape of
# the real IBAN MMGCLZ decode: a script the model might try to reproduce.
STAGE_OUTPUT = (
    'var fso = new ActiveXObject("Scripting.FileSystemObject");\n'
    'var filepath = fso.GetSpecialFolder(2) + "/TKFSIK.exe";\n'
)


def _stage(output: str) -> TransformStage:
    digest = hashlib.sha256(output.encode()).hexdigest()
    return TransformStage(
        kind="js_fromcharcode_offset", key=0, delimiter="MMGCLZ", line=2,
        offset=0, depth=0, encoded="<encoded>", output=output,
        input_sha256="e" * 64, output_sha256=digest,
    )


class _Backend:
    thinking = False

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.code = "import orbit_tools\nprint(orbit_tools.read_file(0, 4096))"

    def supports_exact_context_admission(self) -> bool:
        return True

    def model_info(self):
        class _Info:
            context_length = CTX
        return _Info()

    def count_chat_tokens(self, messages, *, tools=None, thinking=False):
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return TokenCount(
            tokens=int(40 + chars * 0.25), context_tokens=CTX,
            rendered_hash="a" * 64, token_hash="b" * 64,
        )

    def chat_stream(self, messages, **kwargs):
        self.chat_calls.append({"messages": list(messages)})
        if not kwargs.get("tools"):
            return self._result("noted", [])
        import json

        return self._result("running", [{
            "id": f"call_{len(self.chat_calls)}", "type": "function",
            "function": {"name": "execute_analysis",
                         "arguments": json.dumps({"code": self.code})},
        }])

    def _result(self, content, calls):
        return ChatResult(
            content=content, model="m", finish_reason="stop", tool_calls=calls,
            prompt_tokens=1, completion_tokens=1, cached_tokens=0,
            prompt_tokens_per_second=None, generation_tokens_per_second=None,
        )

    def chat(self, messages, **kwargs):
        return self.chat_stream(messages, **kwargs)


def _result(stdout="", stderr="", status="ok", artifacts=(), **kw):
    return AnalysisResult(
        status=status, code_sha256="c" * 64, input_sha256="i" * 64,
        stdout=stdout, stderr=stderr, exit_status=0, duration_seconds=0.1,
        artifacts=tuple(artifacts), **kw,
    )


class _Case(unittest.TestCase):
    def _runtime(self, *, stage_output: str | None = STAGE_OUTPUT) -> AnalysisRuntime:
        data = SOURCE.encode()
        self.backend = _Backend()
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.js"
        path.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=self.backend,
            source=AnalysisSource(
                snapshot_path=path, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        if stage_output is not None:
            stage = _stage(stage_output)
            record = runtime.evidence_store.add(
                "execute_analysis", stage.output,
                metadata={"produced_by_phase": "analysis_transform"},
            )
            runtime.transform_stages.append((stage, record))
            self.stage_evidence_id = record.evidence_id
        return runtime

    def _step(self, runtime, result):
        with mock.patch.object(module, "execute_analysis", lambda **kw: result):
            return runtime.step("go")


class SuppressionTests(_Case):
    def test_exact_stage_reacquisition_is_suppressed(self) -> None:
        runtime = self._runtime()
        before = runtime.actions_executed
        step = self._step(runtime, _result(stdout=STAGE_OUTPUT))
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertEqual(step.suppressed_duplicate_of, self.stage_evidence_id)
        self.assertFalse(step.action_executed)
        # Not counted as a useful action, but the model call it cost is bounded
        # elsewhere; the run stays finite.
        self.assertEqual(runtime.actions_executed, before)

    def test_suppression_points_the_model_at_the_stage_evidence(self) -> None:
        runtime = self._runtime()
        self._step(runtime, _result(stdout=STAGE_OUTPUT))
        tool_msgs = [m for m in runtime.messages if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        note = tool_msgs[-1]["content"]
        self.assertIn(TRANSFORM_REACQUISITION.upper(), note)
        self.assertIn(self.stage_evidence_id, note)

    def test_stage_plus_recomputable_property_is_suppressed(self) -> None:
        """The source-dominated form, one seam inward: the stage plus a value
        Orbit can recompute from it (its length) establishes nothing new."""
        runtime = self._runtime()
        step = self._step(
            runtime, _result(stdout=f"{STAGE_OUTPUT}\nLEN: {len(STAGE_OUTPUT)}")
        )
        self.assertIsNotNone(step.suppressed_duplicate_of)
        self.assertFalse(step.action_executed)


class FailClosedTests(_Case):
    def _assert_useful(self, runtime, result) -> None:
        before = runtime.actions_executed
        step = self._step(runtime, result)
        self.assertIsNone(step.suppressed_duplicate_of)
        self.assertTrue(step.action_executed)
        self.assertEqual(runtime.actions_executed, before + 1)

    def test_no_stage_never_suppresses(self) -> None:
        """Without a recorded stage there is no authority, so behaviour is
        byte-identical to a run before this existed."""
        runtime = self._runtime(stage_output=None)
        self._assert_useful(runtime, _result(stdout=STAGE_OUTPUT))

    def test_a_failing_re_derivation_reaches_the_model_as_a_failure(self) -> None:
        """The observed redundant runs failed on bad decode arithmetic. A
        failure must reach the model as one, not be suppressed -- and the fix
        does NOT rely on the arithmetic being wrong (see the exact-stage test)."""
        runtime = self._runtime()
        self._assert_useful(
            runtime,
            _result(stdout="", stderr="ValueError: invalid literal for int()"),
        )

    def test_partial_stage_stays_useful(self) -> None:
        runtime = self._runtime()
        self._assert_useful(runtime, _result(stdout=STAGE_OUTPUT[:30]))

    def test_one_byte_difference_stays_useful(self) -> None:
        runtime = self._runtime()
        self._assert_useful(
            runtime, _result(stdout=STAGE_OUTPUT.replace("TKFSIK", "TKFSIL", 1))
        )

    def test_stage_plus_a_semantic_line_stays_useful(self) -> None:
        """A value Orbit cannot recompute (a parse of the decoded program) is
        real new work, however trivially true."""
        runtime = self._runtime()
        self._assert_useful(
            runtime, _result(stdout=f"{STAGE_OUTPUT}\nACTIVEX_OBJECTS: 1")
        )

    def test_truncated_output_is_never_suppressed(self) -> None:
        runtime = self._runtime()
        self._assert_useful(runtime, _result(stdout=STAGE_OUTPUT, truncated=True))

    def test_replaced_output_is_never_suppressed(self) -> None:
        runtime = self._runtime()
        self._assert_useful(
            runtime, _result(stdout=STAGE_OUTPUT, output_replaced=True)
        )

    def test_stderr_defeats_the_proof(self) -> None:
        runtime = self._runtime()
        self._assert_useful(
            runtime, _result(stdout=STAGE_OUTPUT, stderr="warning: deprecated")
        )

    def test_an_artifact_alongside_the_output_defeats_the_proof(self) -> None:
        runtime = self._runtime()
        copy = DerivedArtifact(
            name="drop.exe", size_bytes=4, sha256="d" * 64,
        )
        self._assert_useful(runtime, _result(stdout=STAGE_OUTPUT, artifacts=[copy]))


if __name__ == "__main__":
    unittest.main()

"""WSH GetSpecialFolder(n) constants are resolved by the runtime, not memory.

IBAN-EVIDENCE-GROUNDING-CLOSURE-1, defect B. The decoded IBAN payload drops to
`fso.GetSpecialFolder(2) + "/TKFSIK.exe"`. A model that resolves that constant
from memory renders it as the Windows folder; it is the temporary folder. The
runtime knows the mapping exactly (a closed, documented WSH enum), so it states
it as an authoritative deterministic fact and flags a report that contradicts
it -- the same discipline as an unsupported network indicator.

Nothing here is fuzzy or sample-specific: the mapping is keyed on the exact
`GetSpecialFolder(n)` call form and the closed enum {0,1,2}; an undefined index
is not surfaced.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime import analysis_runtime as module  # noqa: E402
from orbit.runtime.analysis_deobfuscate import TransformStage, deobfuscate  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

IBAN = ROOT / "workdir" / "samples" / "IBAN.js"
ORACLE_STAGE_SHA = "5d51e7659955a754d55a83bce9157d8999864ee30a4f3cd5dc752ed0191a7de0"
C2 = "https://productoslili.cl/cv/cr2.exe"


class PureHelperTests(unittest.TestCase):
    def test_defined_constant_is_recovered(self) -> None:
        got = module._special_folder_constants(['fso.GetSpecialFolder(2) + "/x"'])
        self.assertEqual(
            got, [(2, "TemporaryFolder", "the per-user temporary directory (%TEMP%)")]
        )

    def test_all_three_defined_constants(self) -> None:
        text = "GetSpecialFolder(0) GetSpecialFolder(1) GetSpecialFolder(2)"
        got = {index: name for index, name, _ in module._special_folder_constants([text])}
        self.assertEqual(got, {0: "WindowsFolder", 1: "SystemFolder", 2: "TemporaryFolder"})

    def test_two_is_never_the_windows_folder(self) -> None:
        got = module._special_folder_constants(["GetSpecialFolder(2)"])
        self.assertEqual(got[0][1], "TemporaryFolder")
        self.assertNotEqual(got[0][1], "WindowsFolder")

    def test_undefined_index_fails_closed(self) -> None:
        self.assertEqual(module._special_folder_constants(["GetSpecialFolder(9)"]), [])

    def test_bare_integer_is_not_a_constant(self) -> None:
        self.assertEqual(module._special_folder_constants(["the value 2 appears"]), [])

    def test_contradiction_on_wrong_enum_name(self) -> None:
        constants = module._special_folder_constants(["GetSpecialFolder(2)"])
        contra = module._special_folder_contradictions(
            "drops via GetSpecialFolder(2) (the WindowsFolder) to disk", constants
        )
        self.assertEqual(contra, [(2, "TemporaryFolder",
                                   "the per-user temporary directory (%TEMP%)")])

    def test_no_contradiction_on_correct_enum_name(self) -> None:
        constants = module._special_folder_constants(["GetSpecialFolder(2)"])
        self.assertEqual(
            module._special_folder_contradictions(
                "GetSpecialFolder(2) is the TemporaryFolder", constants
            ),
            [],
        )

    def test_no_contradiction_on_contrastive_prose(self) -> None:
        """A correct label that also names another folder for contrast is not a
        mismatch: the right name is present, so the call is correctly labelled."""
        constants = module._special_folder_constants(["GetSpecialFolder(2)"])
        self.assertEqual(
            module._special_folder_contradictions(
                "GetSpecialFolder(2), the TemporaryFolder, not the SystemFolder "
                "used by other droppers.",
                constants,
            ),
            [],
        )

    def test_no_contradiction_across_two_correct_constants(self) -> None:
        """Two calls each labelled correctly: a later constant's name must not
        bleed into an earlier call's window and read as its mapping."""
        constants = module._special_folder_constants(
            ["GetSpecialFolder(2) GetSpecialFolder(1)"]
        )
        self.assertEqual(
            module._special_folder_contradictions(
                "GetSpecialFolder(2) (TemporaryFolder) and "
                "GetSpecialFolder(1) (SystemFolder).",
                constants,
            ),
            [],
        )


class RuntimeSurfacingTests(unittest.TestCase):
    def _runtime_with_stage(self, output: str) -> AnalysisRuntime:
        data = b"var x = 1;\n"
        workspace = AnalysisWorkspace.create()
        path = workspace.source_root / "artifact.js"
        path.write_bytes(data)
        runtime = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=path, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(path),
            ),
            evidence_store=EvidenceStore(root=workspace.root / "evidence"),
            workspace=workspace,
        )
        self.addCleanup(runtime.close)
        digest = hashlib.sha256(output.encode()).hexdigest()
        stage = TransformStage(
            kind="js_fromcharcode_offset", key=0, delimiter="MMGCLZ", line=2,
            offset=0, depth=0, encoded="<e>", output=output,
            input_sha256="e" * 64, output_sha256=digest,
        )
        record = runtime.evidence_store.add(
            "execute_analysis", output,
            metadata={"produced_by_phase": "analysis_transform"},
        )
        runtime.transform_stages.append((stage, record))
        return runtime

    def test_folder_semantics_states_the_temp_mapping(self) -> None:
        runtime = self._runtime_with_stage(
            'fso.GetSpecialFolder(2) + "/TKFSIK.exe"'
        )
        text = runtime.folder_semantics()
        self.assertIn("GetSpecialFolder(2) = TemporaryFolder", text)
        self.assertIn("%TEMP%", text)
        # The wrong mapping must never be rendered for index 2.
        self.assertNotIn("GetSpecialFolder(2) = WindowsFolder", text)

    def test_empty_when_no_special_folder(self) -> None:
        runtime = self._runtime_with_stage('var a = "hello";')
        self.assertEqual(runtime.folder_semantics(), "")

    def test_fact_reaches_the_report_grounding_prompt(self) -> None:
        runtime = self._runtime_with_stage(
            'fso.GetSpecialFolder(2) + "/TKFSIK.exe"'
        )
        self.assertIn("GetSpecialFolder(2) = TemporaryFolder",
                      runtime.deterministic_sections())
        messages = runtime._report_messages("Report.", runtime._reportable_records())
        joined = "\n".join(str(m.get("content") or "") for m in messages)
        self.assertIn("GetSpecialFolder(2) = TemporaryFolder", joined)

    def test_report_grounding_flags_a_wrong_mapping(self) -> None:
        runtime = self._runtime_with_stage(
            'fso.GetSpecialFolder(2) + "/TKFSIK.exe"'
        )
        wrong = "The payload is saved via GetSpecialFolder(2) (the WindowsFolder)."
        grounded = runtime._flag_special_folder_contradictions(wrong)
        self.assertIn(module.WSH_SPECIAL_FOLDER_CONTRADICTION_NOTICE, grounded)
        self.assertIn("GetSpecialFolder(2) = TemporaryFolder", grounded)
        # The model's prose is preserved below the correction, not rewritten.
        self.assertIn(wrong, grounded)

    def test_report_grounding_leaves_a_correct_mapping_untouched(self) -> None:
        runtime = self._runtime_with_stage(
            'fso.GetSpecialFolder(2) + "/TKFSIK.exe"'
        )
        right = "Saved to GetSpecialFolder(2), the temporary folder (%TEMP%)."
        self.assertEqual(runtime._flag_special_folder_contradictions(right), right)


@unittest.skipUnless(IBAN.exists(), "IBAN.js sample required")
class RealSampleTests(unittest.TestCase):
    """T6: the decode identity is unchanged, and the folder fact rides on it."""

    def test_stage_sha_and_c2_unchanged(self) -> None:
        src = IBAN.read_text(encoding="utf-8", errors="surrogatepass")
        stages = deobfuscate(src)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].output_sha256, ORACLE_STAGE_SHA)
        self.assertIn(C2, stages[0].output)

    def test_folder_semantics_from_the_real_decode(self) -> None:
        src = IBAN.read_text(encoding="utf-8", errors="surrogatepass")
        stages = deobfuscate(src)
        constants = module._special_folder_constants([s.output for s in stages])
        self.assertEqual(
            constants,
            [(2, "TemporaryFolder", "the per-user temporary directory (%TEMP%)")],
        )


if __name__ == "__main__":
    unittest.main()

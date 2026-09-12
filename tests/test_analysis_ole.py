"""Static OLE2/CFB + MS-OVBA VBA extraction.

The extractor turns a binary Office document into VBA module source, executing
nothing and depending on no external library. These tests exercise the MS-OVBA
decompressor directly (against an independent compressor), the CFB parser and
`extract_office_vba` against the frozen real sample and hostile mutations of it,
and the honest classification of non-Office and malformed inputs. The frozen
sample's expected module SHA is an independently-established witness
(olevba/olefile), used as a validation expectation, never a recognition rule.
"""

from __future__ import annotations

import hashlib
import struct
import unittest

from orbit.runtime.analysis_ole import (
    OleError,
    OleFile,
    decompress_vba,
    extract_office_vba,
    is_ole_container,
)

SAMPLE = "workdir/samples/99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"
# Independently established (olevba) — validation expectation only.
WITNESS_MODULE = "ThisDocument"
WITNESS_SOURCE_SHA = "d034bd8381f4663af80a7f516cf104ae9fd2a49581379d6a7683bfdf1c8713f0"
WITNESS_SOURCE_CHARS = 55039


def _mkchunk_raw(data: bytes) -> bytes:
    """A single MS-OVBA RAW chunk (flag 0): exactly 4096 data bytes, header
    encodes size-1 (0x0FFF) with the raw flag clear."""
    assert len(data) == 4096
    header = 0x0FFF  # chunk_size-3 == 4093 -> (0x0FFF) ; flag 0 = raw
    return struct.pack("<H", header) + data


class DecompressorTests(unittest.TestCase):
    def test_signature_required(self):
        with self.assertRaises(OleError):
            decompress_vba(b"\x00abc")  # wrong signature byte
        with self.assertRaises(OleError):
            decompress_vba(b"")

    def test_raw_chunk_roundtrip(self):
        # A raw (uncompressed) chunk decompresses to its 4096 payload verbatim.
        payload = bytes((i * 7) & 0xFF for i in range(4096))
        container = b"\x01" + _mkchunk_raw(payload)
        self.assertEqual(decompress_vba(container), payload)

    def test_compressed_chunk_against_independent_compressor(self):
        # Compress a known string with an INDEPENDENT MS-OVBA compressor, then
        # require the production decompressor to recover it exactly.
        text = (b"Attribute VB_Name = \"M\"\r\nSub Foo()\r\n"
                b"  x = 1 + 2 + 1 + 2 + 1 + 2 + 1 + 2\r\nEnd Sub\r\n")
        self.assertEqual(decompress_vba(_compress_ovba(text)), text)

    def test_repeated_data_uses_copy_tokens(self):
        text = b"ABCD" * 500  # highly repetitive -> copy tokens exercised
        self.assertEqual(decompress_vba(_compress_ovba(text)), text)

    def test_decompression_bomb_bounded(self):
        # Repetitive input -> copy tokens -> trips the INNER per-copy bound.
        text = b"A" * 4096
        container = _compress_ovba(text)
        with self.assertRaises(OleError):
            decompress_vba(container, max_out=100)  # inner copy-token cap

    def test_decompression_bomb_outer_bound(self):
        # Two raw chunks so the accumulated total from a PRIOR chunk trips the
        # OUTER per-chunk bound (a distinct guard from the inner copy bound).
        chunk = _mkchunk_raw(bytes(4096))
        container = b"\x01" + chunk + chunk  # 8192 bytes of output
        # After chunk 1, out==4096; the outer guard trips at chunk 2's top.
        with self.assertRaises(OleError):
            decompress_vba(container, max_out=3000)  # outer per-chunk cap


class ContainerRecognitionTests(unittest.TestCase):
    def test_non_ole_is_not_container(self):
        self.assertFalse(is_ole_container(b"var x=1;"))
        self.assertFalse(is_ole_container(b""))
        self.assertFalse(is_ole_container(b"MZ\x90\x00"))  # PE, not OLE

    def test_ole_magic_recognized(self):
        raw = open(SAMPLE, "rb").read()
        self.assertTrue(is_ole_container(raw))

    def test_extract_returns_none_for_non_ole(self):
        self.assertIsNone(extract_office_vba(b"var x = String.fromCharCode(65);"))
        self.assertIsNone(extract_office_vba(b"# a powershell script\n"))

    def test_bad_magic_but_ole_length_still_none(self):
        self.assertIsNone(extract_office_vba(b"\x00" * 1024))


class RealSampleExtractionTests(unittest.TestCase):
    def setUp(self):
        self.raw = open(SAMPLE, "rb").read()

    def test_stream_inventory_parses(self):
        ole = OleFile(self.raw)
        paths = {p for p, _ in ole.paths()}
        self.assertIn("Macros/VBA/ThisDocument", paths)
        self.assertIn("Macros/VBA/dir", paths)
        self.assertEqual(len(paths), 11)

    def test_vba_module_source_matches_witness(self):
        ex = extract_office_vba(self.raw)
        self.assertIsNotNone(ex)
        self.assertEqual(ex.container, "ole2-cfb")
        self.assertEqual(ex.codepage, 1252)
        self.assertEqual(len(ex.modules), 1)
        m = ex.modules[0]
        self.assertEqual(m.name, WITNESS_MODULE)
        self.assertEqual(m.stream_path, "Macros/VBA/ThisDocument")
        self.assertEqual(len(m.source), WITNESS_SOURCE_CHARS)
        self.assertEqual(m.source_sha256, WITNESS_SOURCE_SHA)

    def test_provenance_complete(self):
        m = extract_office_vba(self.raw).modules[0]
        for attr in ("stream_sha256", "compressed_sha256", "source_sha256"):
            self.assertRegex(getattr(m, attr), r"^[0-9a-f]{64}$")
        self.assertEqual(m.codepage, 1252)


class MalformedInputTests(unittest.TestCase):
    """Hostile mutations of the real container must fail closed, never crash the
    process, and never fabricate a module."""

    def setUp(self):
        self.raw = open(SAMPLE, "rb").read()

    def test_truncated_file(self):
        # Cut the file mid-way: parsing must raise OleError or return an honest
        # note, never a fabricated module and never an unhandled exception.
        truncated = self.raw[:5000]
        try:
            ex = extract_office_vba(truncated)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"raised {type(exc).__name__} instead of failing closed")
        self.assertTrue(ex is None or not ex.modules)

    def test_corrupted_header_bom(self):
        bad = bytearray(self.raw)
        bad[28] = 0x00  # break the byte-order mark
        ex = extract_office_vba(bytes(bad))
        self.assertTrue(ex is not None and not ex.modules and ex.note)

    def test_corrupted_sector_shift(self):
        bad = bytearray(self.raw)
        bad[30] = 0x07  # invalid sector shift
        ex = extract_office_vba(bytes(bad))
        self.assertTrue(ex is not None and not ex.modules and ex.note)

    def test_ole_magic_only(self):
        # Magic + zeros: recognized as OLE, but not parseable -> honest note.
        raw = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1016
        ex = extract_office_vba(raw)
        self.assertTrue(ex is not None and not ex.modules and ex.note)

    def test_oversized_file_refused(self):
        from orbit.runtime.analysis_ole import MAX_FILE
        raw = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * (MAX_FILE + 1)
        ex = extract_office_vba(raw)
        self.assertTrue(ex is not None and not ex.modules and ex.note)

    def test_bad_magic_length_ole_but_unparseable(self):
        # 1024 bytes that are NOT the OLE magic: is_ole_container False -> None.
        self.assertIsNone(extract_office_vba(b"NOTOLE" + b"\x00" * 1018))

    def test_directory_first_sector_out_of_bounds(self):
        # Point the first directory sector past EOF: the OOB sector guard must
        # fire (OleError), surfaced as an honest note, never a fabricated read.
        bad = bytearray(self.raw)
        struct.pack_into("<I", bad, 48, 0x7FFFFFF0)  # first_dir_sector huge
        ex = extract_office_vba(bytes(bad))
        self.assertTrue(ex is not None and not ex.modules and ex.note)

    def test_valid_ole_without_vba_yields_honest_note(self):
        # A structurally valid OLE that has no discoverable VBA/dir stream must
        # return an extraction with NO modules and a non-empty honest note (a
        # silent empty result would be a lie about coverage). We rename the
        # `dir` directory entry so it is no longer discoverable, leaving the
        # container otherwise intact and parseable.
        raw = bytearray(self.raw)
        name = "dir".encode("utf-16-le") + b"\x00\x00"  # entry name is "dir\0"
        pos = raw.find(name)
        self.assertNotEqual(pos, -1, "expected a 'dir' directory entry")
        # Overwrite the leading 'd' (UTF-16LE) so the name becomes "xir".
        raw[pos] = ord("x")
        ex = extract_office_vba(bytes(raw))
        self.assertIsNotNone(ex)
        self.assertEqual(ex.modules, [])
        self.assertTrue(ex.note, "no-VBA extraction must carry an honest note")

    def test_bad_magic_rejected_by_parser(self):
        # OleFile itself must reject a bad magic BEFORE any header interpretation.
        # Asserting the *reason* (not merely that some OleError is raised) keeps
        # the magic guard load-bearing: a downstream guard raising for another
        # reason would not satisfy this.
        with self.assertRaisesRegex(OleError, "bad magic"):
            OleFile(b"\x00" * 4096)

    def test_fat_cycle_detected(self):
        # A self-referential FAT chain must be rejected *as a cycle* — not merely
        # bounded away by the length backstop. The message pins the cycle guard.
        ole = OleFile(self.raw)
        ole.fat = [0]  # sector 0 -> sector 0 forever
        with self.assertRaisesRegex(OleError, "cycle"):
            ole._chain(0, 1_000_000)

    def test_oob_sector_read_raises(self):
        # Reading a sector id past max_sector must be rejected *as out of bounds*
        # by the sector-offset guard, not by an incidental past-EOF slice check.
        ole = OleFile(self.raw)
        with self.assertRaisesRegex(OleError, "out of bounds"):
            ole._read_sector(ole.max_sector + 100)


class PreflightIntegrationTests(unittest.TestCase):
    """The Office preflight as ANALYSIS runs it: derived VBA evidence + a PLAN
    bootstrap, produced deterministically before any model call, without marking
    the raw binary covered and without touching the text-source path."""

    def _runtime(self, sample_bytes: bytes, name: str = "in.doc"):
        import tempfile
        from pathlib import Path

        from orbit.runtime.analysis_runtime import (
            AnalysisRuntime, acquire_analysis_source,
        )
        from orbit.runtime.evidence import EvidenceStore

        tmp = tempfile.mkdtemp(prefix="orbit-ole-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        p = Path(tmp) / name
        p.write_bytes(sample_bytes)
        source = acquire_analysis_source(p, Path(tmp) / "owned")
        store = EvidenceStore(root=Path(tmp) / "evidence")
        rt = AnalysisRuntime(backend=None, source=source, evidence_store=store)
        self.addCleanup(rt.close)
        return rt

    def test_office_preflight_extracts_module_evidence(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertEqual(len(rt.office_modules), 1)
        module, record = rt.office_modules[0]
        self.assertEqual(module.name, WITNESS_MODULE)
        self.assertEqual(module.source_sha256, WITNESS_SOURCE_SHA)
        # the module source is retrievable from the store as its evidence
        self.assertTrue(record.evidence_id)

    def test_plan_receives_office_bootstrap(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        boot = [m for m in rt.messages if "Office document" in m.get("content", "")]
        self.assertEqual(len(boot), 1)
        text = boot[0]["content"]
        self.assertIn("ThisDocument", text)
        self.assertIn("evidence:", text)
        # structure only — no behavioural conclusion in the bootstrap
        for banned in ("PowerShell", "C2", "download", "GandCrab", "malware"):
            self.assertNotIn(banned, text)

    def test_raw_binary_not_marked_covered(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertFalse(rt.source_covered)

    def test_text_sample_does_not_trigger_office_path(self):
        rt = self._runtime(b"var x = String.fromCharCode(65,66);\n", name="a.js")
        self.assertEqual(len(rt.office_modules), 0)
        self.assertFalse(
            any("Office document" in m.get("content", "") for m in rt.messages)
        )

    def test_malformed_ole_yields_honest_bootstrap_not_silence(self):
        raw = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1016
        rt = self._runtime(raw)
        self.assertEqual(len(rt.office_modules), 0)
        boot = [m for m in rt.messages if "Office document" in m.get("content", "")]
        self.assertEqual(len(boot), 1)  # honest structural note, not blind silence

    def test_office_records_excluded_from_citation_budget(self):
        # A crafted document's module count is file-controlled; office records
        # must not occupy the bounded report citation budget (they would crowd
        # out action findings). They are rendered by the office appendix instead.
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertEqual(len(rt.office_modules), 1)
        reportable_ids = {r.evidence_id for r in rt._reportable_records()}
        office_ids = {rec.evidence_id for _m, rec in rt.office_modules}
        self.assertTrue(office_ids)
        self.assertTrue(office_ids.isdisjoint(reportable_ids))

    def test_office_appendix_renders_module_source(self):
        # The macro source reaches the report deterministically, not via the
        # model's choice to cite it: the appendix names the module, its evidence
        # id, and its source sha.
        rt = self._runtime(open(SAMPLE, "rb").read())
        appendix = rt.office_appendix()
        self.assertIn("Extracted VBA modules", appendix)
        self.assertIn(WITNESS_MODULE, appendix)
        self.assertIn(WITNESS_SOURCE_SHA, appendix)
        module, record = rt.office_modules[0]
        self.assertIn(record.evidence_id, appendix)

    def test_office_appendix_in_deterministic_sections(self):
        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertIn("Extracted VBA modules", rt.deterministic_sections())

    def test_macro_content_cannot_forge_card_kind_or_status(self):
        # A macro author controls the source bytes. A line like `status: ok` in
        # the source must NOT make the evidence card read `kind: fetch` -- nothing
        # was fetched. The kind is decided by the producing phase, not the bytes.
        from orbit.runtime.evidence import (
            ANALYSIS_OFFICE_PHASE,
            VBA_SOURCE_KIND,
            build_evidence_record,
        )

        hostile = (
            "Attribute VB_Name = \"M\"\r\n"
            "Sub Demo()\r\n"
            "  ' status: ok\r\n"          # would trip the fetch sniffer
            "  ' shell_command_failed: true\r\n"  # would trip the error sniffer
            "End Sub\r\n"
        )
        rec = build_evidence_record(
            "execute_analysis",
            hostile,
            metadata={
                "tool_call_id": "office_vba_1",
                "user_turn_id": "turn_0",
                "produced_by_phase": ANALYSIS_OFFICE_PHASE,
            },
        )
        self.assertEqual(rec.kind, VBA_SOURCE_KIND)
        self.assertEqual(rec.status, "ok")
        self.assertNotEqual(rec.kind, "fetch")

    def test_no_action_office_run_report_is_not_false_no_evidence(self):
        # The common macro-document outcome: the office preflight extracted VBA
        # source but no execute_analysis action ran (the model concluded from the
        # source). The closing report must NOT headline "No analysis evidence has
        # been collected yet." while rendering the macro source right below it --
        # that is a self-contradiction. It must acknowledge the recovered source.
        from orbit.runtime.analysis_runtime import (
            NO_EVIDENCE_REPORT,
            OFFICE_ONLY_REPORT,
        )

        rt = self._runtime(open(SAMPLE, "rb").read())
        self.assertTrue(rt.office_modules)
        self.assertEqual(rt._reportable_records(), [])  # no action records
        # backend=None is safe: this path returns deterministically, no model call
        report = rt.report("What does this do?")
        self.assertEqual(report.model_calls, 0)
        self.assertFalse(report.text.lstrip().startswith(NO_EVIDENCE_REPORT))
        self.assertTrue(
            report.text.startswith(OFFICE_ONLY_REPORT.split("{", 1)[0][:40])
        )
        # ...and the macro source is present in the same report.
        self.assertIn("Extracted VBA modules", report.text)
        self.assertIn(WITNESS_MODULE, report.text)

    def test_office_appendix_bounds_many_modules(self):
        # A crafted document can declare hundreds of modules. The appendix must
        # cap the identity lines it prints and account for the remainder, rather
        # than expand without bound into the report/prompt.
        from orbit.runtime.analysis_ole import VbaModule
        from orbit.runtime.analysis_runtime import OFFICE_APPENDIX_MAX_MODULES

        rt = self._runtime(b"var x=1;\n", name="a.js")  # empty office_modules
        n = OFFICE_APPENDIX_MAX_MODULES + 20

        class _Rec:
            def __init__(self, i):
                self.evidence_id = f"ev_fake_{i:04d}"

        for i in range(n):
            mod = VbaModule(
                name=f"M{i}", stream_path=f"VBA/M{i}", codepage=1252,
                stream_sha256="0" * 64, compressed_sha256="0" * 64,
                source_sha256=f"{i:064d}", source="x" * 10,
            )
            rt.office_modules.append((mod, _Rec(i)))
        appendix = rt.office_appendix()
        # Only the capped number of identity lines are printed...
        self.assertEqual(appendix.count("evidence: ev_fake_"), OFFICE_APPENDIX_MAX_MODULES)
        # ...and the remainder is accounted for, not silently dropped.
        self.assertIn("further module(s) held as evidence", appendix)
        self.assertIn(str(n - OFFICE_APPENDIX_MAX_MODULES), appendix)

    def test_module_stream_name_rejects_path_separators(self):
        from orbit.runtime.analysis_ole import _module_stream_names, _ID_MODULESTREAMNAME
        # A MODULESTREAMNAME record whose "name" contains a path separator must
        # be rejected, not turned into a `../X` path fragment.
        def rec(name: bytes) -> bytes:
            return struct.pack("<HI", _ID_MODULESTREAMNAME, len(name)) + name
        blob = rec(b"../evil") + rec(b"good") + rec(b"bad\\name")
        self.assertEqual(_module_stream_names(blob), ["good"])


# --- independent MS-OVBA compressor (test-only oracle) ----------------------
def _compress_ovba(data: bytes) -> bytes:
    """A correct-but-simple MS-OVBA compressor: emits compressed chunks using
    literal and copy tokens per MS-OVBA 2.4.1. Independent of the production
    decompressor; used only to generate test vectors."""
    out = bytearray(b"\x01")
    pos = 0
    n = len(data)
    while pos < n:
        chunk = data[pos:pos + 4096]
        comp = _compress_chunk(chunk)
        # If compression didn't help, emit raw (must be exactly 4096).
        if len(comp) >= 4096 and len(chunk) == 4096:
            out += struct.pack("<H", 0x0FFF) + chunk
        else:
            size = len(comp) + 2
            header = ((size - 3) & 0x0FFF) | 0x8000  # compressed flag
            out += struct.pack("<H", header) + comp
        pos += 4096
    return bytes(out)


def _compress_chunk(chunk: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(chunk)
    while i < n:
        flag_pos = len(out)
        out.append(0)
        flags = 0
        for bit in range(8):
            if i >= n:
                break
            # try to find a back-reference
            best_len, best_off = 0, 0
            diff = i
            bit_count = max(4, (diff - 1).bit_length()) if diff > 1 else 4
            max_len = (0xFFFF >> bit_count) + 3
            max_off = (0xFFFF >> (16 - bit_count)) + 1 if False else (1 << bit_count)
            start = max(0, i - max_off)
            for j in range(start, i):
                l = 0
                while i + l < n and l < max_len and chunk[j + l] == chunk[i + l]:
                    l += 1
                if l > best_len:
                    best_len, best_off = l, i - j
            if best_len >= 3:
                length_mask = 0xFFFF >> bit_count
                token = ((best_off - 1) << (16 - bit_count)) | (best_len - 3)
                out += struct.pack("<H", token & 0xFFFF)
                flags |= (1 << bit)
                i += best_len
            else:
                out.append(chunk[i])
                i += 1
        out[flag_pos] = flags
    return bytes(out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

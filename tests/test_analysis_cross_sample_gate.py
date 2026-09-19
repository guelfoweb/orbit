"""Cross-sample deterministic regression gate for the analysis pipeline.

ANALYSIS-CROSS-SAMPLE-REGRESSION-GATE-1. One cheap, model-free gate over the
whole qualified corpus, so a cross-cutting change to evidence / admission /
indicator / relationship / report-grounding logic cannot silently improve one
sample while regressing another. Level 1 of the two-tier policy in AGENTS.md:
this runs on every cross-cutting ANALYSIS PR; live Ornith (Level 2) is reserved
for release and prompt/controller/model-facing changes.

It exercises the deterministic seams end to end through the REAL runtime entry
points -- preflight, deterministic transforms, source/evidence authority,
indicator admission, static-relationship extraction, and report-grounding
inputs / contradiction guards -- and asserts each sample's frozen contract. No
production algorithm is reimplemented here; the expected values are the
qualified facts (AGENTS.md corpus table, per-mission oracles, and the current
deterministic runtime output they were derived from). Nothing here runs a model
or touches the network, and a regression in ANY sample fails the gate.

Each contract is the OUTER artifact SHA plus only runtime-owned facts:
transform-stage output SHAs, the exact verified-indicator set, indicators that
must NOT be promoted (namespace/schema metadata), extracted-module identity,
static entrypoints and execution reach, deterministic self-invocation, decoded
platform-constant semantics, and report-grounding facts. No model prose is
encoded as truth.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.runtime import analysis_runtime as AR  # noqa: E402
from orbit.runtime.analysis_runtime import (  # noqa: E402
    AnalysisController,
    AnalysisRuntime,
    AnalysisSource,
    AnalysisWorkspace,
    OPEN,
    RESOLVED,
    ANSWERED_UNVERIFIED,
    _invocation_contradictions,
    _special_folder_constants,
    _special_folder_contradictions,
    _stage_entry_invocations,
)
from orbit.runtime.evidence import EvidenceStore  # noqa: E402

SAMPLES = ROOT / "workdir" / "samples"

# --- frozen per-sample contracts -------------------------------------------
# Keys used below:
#   sha              outer artifact sha256 (current frozen sample)
#   stage_shas       every deterministic transform-stage output sha256 (a set;
#                    empty when the sample decodes to no stage)
#   iocs             the EXACT authoritative network-indicator set (missing or
#                    extra/fabricated indicators both fail; a namespace/schema
#                    URI appearing here is a promoted-metadata regression)
#   forbidden_iocs   substrings that must never appear in the indicator set or
#                    the rendered Verified indicators (namespace metadata, etc.)
#   grounding        substrings that MUST appear in deterministic_sections()
#   office           (module_name, module_sha, [(proc, event)], [(proc, sink,
#                    line)]) for an OLE/Office document; None otherwise
#   invocations      expected _stage_entry_invocations across all stages
#   folder_consts    expected _special_folder_constants across all stages

C2_FATTURA = "http://smartmaket.com/1.php?s=AA1789FF-522F-4D9A-94E9-C9BE2BA3A1D3"
C2_YPS = "http://gibuzuy37v2v.top/1.php?s=mints13"
C2_MINE = "https://wall5tghf6fdg.api.opensourcesaas.org/ZOdcfNuo/myxwr5cli.bat"
C2_IBAN = "https://productoslili.cl/cv/cr2.exe"
C2_4B = "https://stylegeneration.ma/sirdee.ps1"
C2_OFFICE = "http://185.189.58.222/x.exe"

CONTRACTS: "dict[str, dict]" = {
    "Fattura981033956.js": {
        "sha": "b7cfd5fdeb16d7b5ecea1063419bdad6ad280ed9b73c636707874c3f4001dc0c",
        "stage_shas": {
            "6a892b624a2a237ca8b6fdd64d2dc02a30aa1afc08db9b9a21f269d9fb67eac2",
            "cafa14b8acab599aeb5425fabb3bdb519935cd39947e653dd89619db19b1b3fe",
            "8716f5d4559ed0f5dbfdb7bfb1717abcc9cfa1171caeea548aaf9c38fccd04db",
            "ec8ccda0cbdce79a76748c0e32c1fb788276c762abc5fd8c6f77609a0c8f58f1",
            "6a4277aa4ae872f43b368c35fcee79ea2ee40824822ae215cf52483f8faa48a3",
        },
        "iocs": {C2_FATTURA},
        "forbidden_iocs": (),
        "grounding": (C2_FATTURA,),
        "office": None,
        "invocations": [],
        "folder_consts": [],
    },
    "peXF7I6W.ps1": {
        "sha": "5eba3e4538cffbde5d39ba81eb4ed85e9c9cc6065e036503073a43a9478f405d",
        "stage_shas": set(),  # the C2 is present in the raw source, not a stage
        "iocs": {C2_YPS},
        "forbidden_iocs": (),
        "grounding": (C2_YPS,),
        "office": None,
        "invocations": [],
        "folder_consts": [],
    },
    "mine.hta": {
        "sha": "6840b6d84f7c7190424fd465e466e2477e7c8a781457e2c6dcd523df498cea3d",
        "stage_shas": {
            "0c6b4253cbd1eb8b4a2b0a58f0cad9a846860aa49579b2eea91090b8a2d627c1",
            "95e52b2ec509e63e7bf30d9daea95b9a491e04ec4b9c62a425b232bc5c38e9e0",
            "81907992dd4830c96203709899fd1854e98fb60afd2efaa58df72583207ae53d",
            "c7e6e3becb343cfec4d3732c04ba6b65d9d4291af3b8080974a2a4c4c95f0e3a",
            "460d8e1b1739d05839f305bb9b71a5433c4c34ef02f822bc191bc9c4d9dd66fc",
            "e2214909b2e7c67123563b4c0aada2530430ed333665a08c3ae832ca1d4706ca",
        },
        "iocs": {C2_MINE},
        # The XHTML namespace is literally present in the .hta source; it must
        # never be promoted to a verified indicator.
        "forbidden_iocs": ("http://www.w3.org/1999/xhtml", "w3.org"),
        # The needle is the invocation FACT phrase, not the bare name -- the
        # name alone appears in the raw decoded stage, so it would not catch a
        # regression that drops the self-invocation grounding.
        "grounding": (C2_MINE, "defines function ROmYsTcn and invokes it"),
        "office": None,
        "invocations": [("ROmYsTcn", "ROmYsTcn;")],
        "folder_consts": [],
    },
    "IBAN.js": {
        "sha": "86e23fa673271308578daf61e783a00662351bab66d74f5f16e16302ad40d8b8",
        "stage_shas": {
            "5d51e7659955a754d55a83bce9157d8999864ee30a4f3cd5dc752ed0191a7de0",
        },
        "iocs": {C2_IBAN},
        "forbidden_iocs": (),
        "grounding": (C2_IBAN, "GetSpecialFolder(2) = TemporaryFolder"),
        "office": None,
        "invocations": [],
        "folder_consts": [(2, "TemporaryFolder", "the per-user temporary directory (%TEMP%)")],
    },
    "4b863c7be268f87e12981204cb0ffc6e.js": {
        "sha": "e1a3a8937909e56d86692fda412312603951a3ea20abf730d538d2e07fda06a3",
        "stage_shas": {
            "576dea76a6e561b4a138a046b3770ad5df83903497cf628448772cf306ae8443",
            "080715940b3982f860e5c87f11970d656e79ba835284dd3287c8463a2d60c543",
            "9e477cf708ab2c590a3736962fb5021bbc40941a141e2715a322e9ac4827a52f",
        },
        "iocs": {C2_4B},
        "forbidden_iocs": (),
        "grounding": (C2_4B,),
        "office": None,
        "invocations": [],
        "folder_consts": [],
    },
    "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc": {
        "sha": "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809",
        "stage_shas": {
            "f1fa67e3f558443e9604b0f0b26715dc8065c53c92db7acbab2e6d32874e5c21",
        },
        "iocs": {C2_OFFICE},
        "forbidden_iocs": (),
        "grounding": (
            C2_OFFICE, "Document_Open", "reaches a Shell execution call", "PHfW.exe",
        ),
        "office": {
            "module_name": "ThisDocument",
            "module_sha": "d034bd8381f4663af80a7f516cf104ae9fd2a49581379d6a7683bfdf1c8713f0",
            "events": [("Document_Open", "document-open")],
            "exec_reach": [("Document_Open", "Shell", 750)],
        },
        "invocations": [],
        "folder_consts": [],
    },
}


def _module_sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()


# The pinned corpus is deliberately NOT committed (live malware). On the
# qualification machine it is present, and the gate is mandatory there. A
# checkout that genuinely lacks the corpus must say so explicitly, so an absent
# corpus becomes a HARD FAILURE (not a silent green) unless this is set -- which
# is the whole point of a regression gate: "I could not test the pinned corpus"
# must never read as "the corpus passed."
ALLOW_MISSING_CORPUS = os.environ.get("ORBIT_ALLOW_MISSING_CORPUS") == "1"


def _sample_state(name: str, expected_sha: str) -> str:
    """`present`, `absent`, or `drifted` for a pinned sample."""
    path = SAMPLES / name
    if not path.exists():
        return "absent"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return "present" if actual == expected_sha else "drifted"


class _GateBase(unittest.TestCase):
    def _runtime(self, name: str, contract: dict) -> AnalysisRuntime:
        path = SAMPLES / name
        state = _sample_state(name, contract["sha"])
        if state == "absent":
            # Absence is the one skip: a checkout without the (uncommitted)
            # corpus cannot test it. The CorpusPresenceTests floor turns a
            # missing corpus into a hard failure unless explicitly opted out, so
            # this skip cannot hide a regression on the qualification machine.
            self.skipTest(f"{name} not present (corpus floor enforces presence)")
        if state == "drifted":
            # PRESENT but the pinned identity changed: never a silent skip. The
            # frozen artifact was modified (a re-save, a swap); re-pinning is a
            # deliberate, reviewed act that must re-confirm the decode facts,
            # not something a mandatory gate glosses over.
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.fail(
                f"{name} present but sha drifted ({actual[:16]} != pinned "
                f"{contract['sha'][:16]}): re-pin deliberately after re-checking "
                "the decode contract"
            )
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        ws = AnalysisWorkspace.create()
        sp = ws.source_root / name
        sp.write_bytes(data)
        rt = AnalysisRuntime(
            backend=None,
            source=AnalysisSource(
                snapshot_path=sp, sha256=actual, size_bytes=len(data),
                original_path=str(sp),
            ),
            evidence_store=EvidenceStore(root=ws.root / "evidence"),
            workspace=ws,
        )
        self.addCleanup(rt.close)
        return rt

    def _assert_contract(self, name: str, contract: dict) -> None:
        rt = self._runtime(name, contract)

        # -- inference-free by construction: preflight decodes and grounds
        # without a model call or a sandbox action. A pin, not a network proof
        # (nothing on this path can reach the network regardless); it fails only
        # if preflight is ever changed to call a model or run an action.
        self.assertEqual(rt.model_calls, 0, f"{name}: preflight made a model call")
        self.assertEqual(rt.actions_executed, 0, f"{name}: preflight ran an action")

        # -- deterministic transforms: every expected stage sha present, exactly.
        stage_shas = {s.output_sha256 for s, _ in rt.transform_stages}
        self.assertEqual(
            stage_shas, contract["stage_shas"],
            f"{name}: transform stage SHAs changed",
        )

        # -- indicator admission: the exact verified set. A shrink is a missing
        # C2; a grow is a fabricated / promoted-metadata IOC.
        self.assertEqual(
            set(rt.authoritative_indicators()), contract["iocs"],
            f"{name}: verified indicator set changed",
        )
        verified = rt.verified_indicators()
        for ioc in contract["iocs"]:
            self.assertIn(ioc, verified, f"{name}: {ioc} absent from Verified indicators")
        for forbidden in contract["forbidden_iocs"]:
            self.assertNotIn(
                forbidden, verified,
                f"{name}: forbidden metadata {forbidden!r} promoted to Verified indicators",
            )
            self.assertFalse(
                any(forbidden in a for a in rt.authoritative_indicators()),
                f"{name}: forbidden metadata {forbidden!r} in authoritative set",
            )

        # -- report-grounding inputs: the deterministic sections the report is
        # built on must carry each required fact (so a report cannot omit or
        # contradict them silently).
        grounding = rt.deterministic_sections()
        for needle in contract["grounding"]:
            self.assertIn(needle, grounding, f"{name}: grounding missing {needle!r}")

        # -- static self-invocation (deterministic).
        found_inv: list = []
        for stage, _ in rt.transform_stages:
            found_inv += _stage_entry_invocations(stage.output)
        self.assertEqual(
            found_inv, contract["invocations"],
            f"{name}: deterministic stage-invocation facts changed",
        )

        # -- decoded platform-constant semantics (deterministic).
        found_consts: list = []
        for stage, _ in rt.transform_stages:
            found_consts += _special_folder_constants([stage.output])
        self.assertEqual(
            found_consts, contract["folder_consts"],
            f"{name}: GetSpecialFolder semantics changed",
        )

        # -- extracted-source authority + static relationships (Office).
        office = contract["office"]
        if office is None:
            self.assertEqual(rt.office_modules, [], f"{name}: unexpected Office modules")
            self.assertEqual(rt.office_exec_reach, [], f"{name}: unexpected exec reach")
        else:
            self.assertEqual(len(rt.office_modules), 1, f"{name}: extracted-module count")
            module, _rec = rt.office_modules[0]
            self.assertEqual(module.name, office["module_name"])
            self.assertEqual(
                _module_sha(module.source), office["module_sha"],
                f"{name}: extracted module source SHA changed (lost source authority)",
            )
            self.assertEqual(
                [(r.procedure, r.event) for r in rt.office_events], office["events"],
                f"{name}: autoexec relationship changed",
            )
            self.assertEqual(
                [(r.procedure, r.sink, r.sink_line) for r in rt.office_exec_reach],
                office["exec_reach"],
                f"{name}: static execution reach changed",
            )

        # -- digest-provenance invariant, per sample: an invented remote-payload
        # digest (bytes Orbit never fetched -- network deny) must be flagged,
        # while the sample's real artifact digest, correctly subjected, must not.
        # This proves the guard is wired for every frozen sample without
        # depending on whether a sample's URI-string hash happens to coincide
        # with a decoded-stage digest (for some samples the decoded stage IS the
        # URL string, so its hash is legitimately a digest of bytes Orbit holds).
        invented = "1234567890abcdef" * 4  # 64 hex, not any real digest
        relabel = f"The downloaded payload x.exe has sha256 {invented}."
        self.assertNotEqual(
            rt._flag_fabricated_digest_claims(relabel), relabel,
            f"{name}: an invented remote-payload digest was not flagged",
        )
        legit = f"The artifact sha256 is {rt.source.sha256}."
        self.assertEqual(
            rt._flag_fabricated_digest_claims(legit), legit,
            f"{name}: a correct artifact-digest claim was flagged",
        )


class CrossSampleContractTests(_GateBase):
    """One test per sample; a regression in any one fails independently."""

    def test_fattura(self) -> None:
        self._assert_contract("Fattura981033956.js", CONTRACTS["Fattura981033956.js"])

    def test_yps(self) -> None:
        self._assert_contract("peXF7I6W.ps1", CONTRACTS["peXF7I6W.ps1"])

    def test_mine_hta(self) -> None:
        self._assert_contract("mine.hta", CONTRACTS["mine.hta"])

    def test_iban(self) -> None:
        self._assert_contract("IBAN.js", CONTRACTS["IBAN.js"])

    def test_4b(self) -> None:
        self._assert_contract(
            "4b863c7be268f87e12981204cb0ffc6e.js",
            CONTRACTS["4b863c7be268f87e12981204cb0ffc6e.js"],
        )

    def test_office_doc(self) -> None:
        self._assert_contract(
            "99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc",
            CONTRACTS["99eb1d90eb5f0d012f35fcc2a7dedd2229312794354843637ebb7f40b74d0809.doc"],
        )


class CorpusPresenceTests(unittest.TestCase):
    """The floor that keeps this mandatory gate honest: because the corpus is
    uncommitted, a machine that lacks it must FAIL rather than silently pass a
    gate that tested nothing. Set ORBIT_ALLOW_MISSING_CORPUS=1 only on a
    checkout that deliberately has no corpus (then the gate's corpus coverage is
    knowingly waived)."""

    def test_the_full_pinned_corpus_is_present_and_unchanged(self) -> None:
        states = {name: _sample_state(name, c["sha"]) for name, c in CONTRACTS.items()}
        present = [n for n, s in states.items() if s == "present"]
        drifted = [n for n, s in states.items() if s == "drifted"]
        absent = [n for n, s in states.items() if s == "absent"]
        # A drifted sample is always a failure -- it silently loses coverage.
        self.assertEqual(drifted, [], f"pinned samples drifted: {drifted}")
        if ALLOW_MISSING_CORPUS:
            # Explicit waiver: at least record that the corpus is intentionally
            # absent; do not assert full presence.
            self.assertTrue(True)
            return
        self.assertEqual(
            absent, [],
            "the mandatory cross-sample gate cannot run without its pinned "
            f"corpus (missing: {absent}). Populate workdir/samples/ on this "
            "machine, or set ORBIT_ALLOW_MISSING_CORPUS=1 to waive corpus "
            "coverage deliberately.",
        )
        self.assertEqual(len(present), len(CONTRACTS), "corpus incomplete")


class ContradictionGuardTests(unittest.TestCase):
    """Report-grounding contradiction guards must flag a claim that contradicts
    deterministic evidence and leave a correct one alone. Proven on the two
    samples whose guards are live, so a change that silently defeats a guard
    fails here even though the deterministic facts still parse."""

    def test_mine_hta_invocation_guard(self) -> None:
        inv = [("ROmYsTcn", "ROmYsTcn;", "ev_x")]
        self.assertEqual(
            _invocation_contradictions(
                "The ROmYsTcn routine is defined but not invoked.", inv
            ),
            inv,
        )
        self.assertEqual(
            _invocation_contradictions("The stage invokes ROmYsTcn (ROmYsTcn;).", inv),
            [],
        )

    def test_iban_special_folder_guard(self) -> None:
        consts = _special_folder_constants(["GetSpecialFolder(2)"])
        self.assertTrue(
            _special_folder_contradictions(
                "dropped via GetSpecialFolder(2) (the WindowsFolder)", consts
            )
        )
        self.assertEqual(
            _special_folder_contradictions(
                "GetSpecialFolder(2) is the TemporaryFolder", consts
            ),
            [],
        )


class NoFalseResolvedInvariantTests(unittest.TestCase):
    """No false RESOLVED, where deterministically testable: the controller never
    marks a question resolved without a completion witness. Model-free."""

    def test_resolved_without_witness_downgrades_to_open(self) -> None:
        c = AnalysisController()
        c.adopt_plan([{"question": "Q?", "missing_fact": "M"}])
        c.activate_next()
        c.close_active(RESOLVED, evidence_ids=(), summary="")
        self.assertEqual(c.states[c.order[0]].status, OPEN)

    def test_resolved_with_witness_is_honoured(self) -> None:
        c = AnalysisController()
        c.adopt_plan([{"question": "Q?", "missing_fact": "M"}])
        c.activate_next()
        c.close_active(RESOLVED, evidence_ids=("ev_1",), summary="the answer")
        self.assertEqual(c.states[c.order[0]].status, ANSWERED_UNVERIFIED)


if __name__ == "__main__":
    unittest.main()

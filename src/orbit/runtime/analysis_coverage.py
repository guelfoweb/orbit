"""Single-shot source coverage: supply the whole artifact, once, or not at all.

The failure this addresses is not that the model reasons badly. It is that it
spends actions *acquiring* text Orbit already holds -- reading the artifact
raw, then numbered, then as a repr, then in slices -- because nothing ever
told it the source was already available. Prompt guidance did not change that.
So the source stops being something to fetch and becomes something supplied.

What this guarantees is narrow and stated precisely:

    SOURCE_COVERED -- the complete artifact source was presented to the model,
                      exactly, in a single call.

It is emphatically **not** ANALYSIS_COMPLETE. Coverage says the model has seen
the bytes; it says nothing about whether the model understood them, whether the
behaviour is resolved, or whether the analysis may stop. Any code that reads
coverage as proof that analysis is finished is a defect.

**Chunked coverage of large artifacts is not supported, and this is
architectural rather than unimplemented.** ANALYSIS history is append-only:
every turn stays resident, so the final call of a multi-part coverage would
carry the whole artifact anyway, plus one header per part. Splitting bounds
what a single *message* adds while making the resident total strictly worse --
it cannot get an artifact under a budget its bytes already exceed. An artifact
too large for one admitted call is therefore not covered at all; the ordinary
autonomous workflow runs, exactly as it did before this existed.

Eligibility is decided by exact tokenizer admission of the message that will
actually be sent, never by file size or a character estimate -- this repo
measured 1.123 chars/token on the obfuscated content it would need to bound,
and rare codepoints run four times the other way, so a character cap cannot
bound tokens in either direction.

Admission of the COVER call alone is not sufficient and is not what is
checked. Because the source stays resident, the calls that follow inherit it:
coverage must also leave room for the RESOLVE step that acts on it and the
REPORT that concludes. That headroom is demanded up front, so an artifact that
would fit but leave the analysis unable to proceed is refused here rather than
discovered at the first action.
"""

from __future__ import annotations

from dataclasses import dataclass


# Coverage status. `COVERAGE_COMPLETE` is about bytes presented, never about
# analysis being finished -- see the module docstring.
COVERAGE_COMPLETE = "complete"
COVERAGE_NOT_ELIGIBLE = "not_eligible"
COVERAGE_TOO_LARGE = "too_large"
COVERAGE_UNADMISSIBLE = "unadmissible"


@dataclass(frozen=True)
class SourceCoverage:
    """The whole artifact, ready to supply -- or a refusal and its reason.

    There is no partial coverage. Presenting some of an artifact and calling it
    coverage would be worse than not covering it at all: it would tell the
    model it had seen everything when it had not. When coverage cannot be
    offered, `text` is empty and `status` says why, and the caller falls back
    to the ordinary autonomous workflow.
    """

    text: str
    status: str
    sha256: str = ""
    size_bytes: int = 0

    @property
    def covered(self) -> bool:
        return self.status == COVERAGE_COMPLETE

    @property
    def covered_bytes(self) -> int:
        return len(self.text.encode("utf-8")) if self.text else 0

    def attest(self) -> "CoverageAttestation":
        return attest_coverage(self)


@dataclass(frozen=True)
class CoverageAttestation:
    """What the runtime can prove, computed from the artifact and the text.

    Never inferred from model output: the model is not asked whether it read
    the source, because an answer to that question is not evidence.
    """

    sha256: str
    size_bytes: int
    covered_bytes: int
    status: str

    @property
    def complete(self) -> bool:
        """Every byte accounted for, and the coverage says so."""
        return (
            self.status == COVERAGE_COMPLETE
            and self.size_bytes > 0
            and self.covered_bytes == self.size_bytes
        )


def decode_artifact(raw: bytes) -> str | None:
    """The artifact as text, or None when it is not text this can cover.

    Strict UTF-8 and only strict. Decoding with `errors="replace"` would put
    U+FFFD into text then presented as the artifact's own bytes, which is a
    character the artifact does not contain -- the same reasoning that governs
    indicator extraction. An embedded NUL means this is not the encoding it
    appears to be, and a binary artifact simply is not covered: existing
    behaviour stands for it.
    """
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    if text.encode("utf-8") != raw:
        # Defensive: strict UTF-8 round-trips by construction. If it ever does
        # not, the text is not the bytes and must not be presented as them.
        return None
    return text


def plan_coverage(raw: bytes, *, fits: "callable", sha256: str) -> SourceCoverage:
    """Offer the whole artifact, or refuse with the reason.

    `fits(text)` is the admission oracle and must answer for the message that
    would actually be sent, with the headroom the following RESOLVE and REPORT
    calls need already reserved. It is asked exactly once, about the complete
    source: there is no smaller thing to fall back to, because a partial
    presentation is not coverage.
    """
    text = decode_artifact(raw)
    if text is None:
        return SourceCoverage("", COVERAGE_NOT_ELIGIBLE, sha256, len(raw))
    if not fits(text):
        return SourceCoverage("", COVERAGE_TOO_LARGE, sha256, len(raw))
    coverage = SourceCoverage(text, COVERAGE_COMPLETE, sha256, len(raw))
    # Checked rather than trusted: coverage that does not actually account for
    # the artifact must never be reported as coverage.
    if not coverage.attest().complete:
        return SourceCoverage("", COVERAGE_UNADMISSIBLE, sha256, len(raw))
    return coverage


def attest_coverage(coverage: SourceCoverage) -> CoverageAttestation:
    """Prove what is covered, from the artifact identity and the bytes held."""
    return CoverageAttestation(
        sha256=coverage.sha256,
        size_bytes=coverage.size_bytes,
        covered_bytes=coverage.covered_bytes,
        status=coverage.status,
    )

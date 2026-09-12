"""Deterministic Office/VBA auto-execution entrypoint recognizer.

STATIC source classification only. Nothing here executes VBA, opens a document,
emulates an Office event, or runs a macro. It reads the ALREADY-EXTRACTED VBA
module source (and the container's stream inventory, for the host) and marks a
procedure as a documented Office auto-execution entrypoint ONLY when the
procedure name AND its module/host context together match a verified
event-handler contract (see workdir/diag/vba_autoexec/SEMANTICS.md).

The fact asserted is narrow: the procedure is the handler Office invokes on the
corresponding event WHEN macros are permitted. It is NOT a claim that a victim
opened the file or enabled macros, and it establishes NO behaviour (network,
persistence, execution) by itself -- that comes from the body's own evidence.

A name alone is never enough: `Document_Open` in a standard module, or
`Workbook_Open` in a Word container, or a name inside a comment/string, or a
longer name (`FooDocument_Open`), is refused (fail closed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Semantic-contract version, recorded in every relationship for provenance.
AUTOEXEC_CONTRACT = "office-vba-autoexec-v1"

# Hosts, from the container's stream inventory (no OLE re-parse).
HOST_WORD = "word"
HOST_EXCEL = "excel"

# Event kinds (the documented trigger), kept descriptive and small.
EV_OPEN = "document-open"
EV_CLOSE = "document-close"
EV_NEW = "document-new"
EV_STARTUP = "application-startup"
EV_WORKBOOK_OPEN = "workbook-open"
EV_WORKBOOK_CLOSE = "workbook-close"


@dataclass(frozen=True)
class OfficeEventRelationship:
    """A static Office event-handler relationship recovered from exact source.

    `host`/`module`/`procedure` locate it; `event` is the documented trigger;
    `module_evidence_id` references the extracted-module evidence the body lives
    in (no duplication); `line` is 1-based into the module source; `contract`
    is the semantic-contract version."""

    host: str
    module: str
    procedure: str
    event: str
    description: str
    module_evidence_id: str
    line: int
    contract: str = AUTOEXEC_CONTRACT


# --- entrypoint taxonomy (small, verified) ----------------------------------
# Each entry: canonical procedure name (lower) -> (event, human description,
# module-context predicate). The predicate receives (host, module_name_lower)
# and decides whether THIS module/host is a valid context for the name.


def _is_word_document_module(host: str, module_lower: str) -> bool:
    # Word document-class events live in the document module `ThisDocument`.
    return host == HOST_WORD and module_lower == "thisdocument"


def _is_word_host(host: str, module_lower: str) -> bool:
    # Word auto-macros run by name regardless of module, so only the host gates.
    return host == HOST_WORD


def _is_excel_workbook_module(host: str, module_lower: str) -> bool:
    return host == HOST_EXCEL and module_lower == "thisworkbook"


def _is_excel_host(host: str, module_lower: str) -> bool:
    return host == HOST_EXCEL


_TAXONOMY: "dict[str, tuple[str, str, object]]" = {
    # Word document-class events (require the ThisDocument module).
    "document_open": (EV_OPEN, "Word document-open event handler", _is_word_document_module),
    "document_close": (EV_CLOSE, "Word document-close event handler", _is_word_document_module),
    "document_new": (EV_NEW, "Word document-new event handler", _is_word_document_module),
    # Word auto-macros (standard module; Word host).
    "autoopen": (EV_OPEN, "Word AutoOpen auto-macro (runs on document open)", _is_word_host),
    "autoclose": (EV_CLOSE, "Word AutoClose auto-macro (runs on document close)", _is_word_host),
    "autoexec": (EV_STARTUP, "Word AutoExec auto-macro (runs on Word startup)", _is_word_host),
    # Excel workbook-class events (require the ThisWorkbook module).
    "workbook_open": (EV_WORKBOOK_OPEN, "Excel workbook-open event handler", _is_excel_workbook_module),
    "workbook_close": (EV_WORKBOOK_CLOSE, "Excel workbook-close event handler", _is_excel_workbook_module),
    # Excel auto-macro (standard module; Excel host).
    "auto_open": (EV_WORKBOOK_OPEN, "Excel Auto_Open auto-macro (runs on workbook open)", _is_excel_host),
}


def office_host(stream_inventory: "list[tuple[str, int]]") -> str | None:
    """The Office host application, from the container's stream inventory only.

    Word documents carry a `WordDocument` stream; Excel workbooks a `Workbook`
    (legacy `Book`) stream. Returns None when the host cannot be determined --
    in which case no relationship is emitted (fail closed). If BOTH appear
    (malformed/polyglot), the host is ambiguous and None is returned. Total: a
    malformed inventory entry yields None rather than raising."""
    names: set[str] = set()
    try:
        for entry in stream_inventory:
            path = entry[0]
            if isinstance(path, str):
                names.add(path.split("/")[-1].lower())
    except (TypeError, IndexError):
        return None
    is_word = "worddocument" in names
    is_excel = "workbook" in names or "book" in names
    if is_word and not is_excel:
        return HOST_WORD
    if is_excel and not is_word:
        return HOST_EXCEL
    return None


# A VBA procedure declaration at the start of a line (after optional scope
# keywords): `[Public|Private|Friend|Static] Sub|Function <name>(`. Case
# insensitive. Anchored at line start (after whitespace) so a name embedded in
# an expression, a comment, or a string cannot match.
_PROC_DECL = re.compile(
    r"^[ \t]*(?:Public[ \t]+|Private[ \t]+|Friend[ \t]+|Static[ \t]+)*"
    # Office event handlers and auto-macros are invoked only as `Sub`s -- Office
    # does not wire a `Function Document_Open` as the handler. Matching only
    # `Sub` keeps a `Function` of the same name from being mislabelled an
    # entrypoint (a narrow but real false positive).
    r"Sub[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(",
    re.IGNORECASE,
)
# A full-line comment (optionally indented). A trailing comment cannot hide a
# declaration because the declaration must START the line.
_COMMENT_LINE = re.compile(r"^[ \t]*'")


def find_office_event_relationships(
    module_name: str,
    module_source: str,
    stream_inventory: "list[tuple[str, int]]",
    module_evidence_id: str,
) -> "list[OfficeEventRelationship]":
    """Every Office auto-execution entrypoint declared in this module, given its
    host/module context. Exact declaration matching (not substring): the
    procedure name must be the whole name of a `Sub` declared at the start of a
    line. A match in a comment or string, a longer name, or a wrong module/host
    context yields nothing (fail closed).

    De-duplicated by (procedure lower-name): a module that declares the same
    entrypoint twice yields one relationship, conservatively.

    Total by construction: a malformed input (non-string source/name, a
    non-(path,size) inventory) yields no relationships rather than raising, so a
    future change upstream can never turn this static enrichment into a crash."""
    if not isinstance(module_name, str) or not isinstance(module_source, str):
        return []
    host = office_host(stream_inventory)
    if host is None:
        return []
    module_lower = module_name.lower()
    results: "list[OfficeEventRelationship]" = []
    seen: set[str] = set()
    for idx, raw in enumerate(module_source.splitlines(), 1):
        if _COMMENT_LINE.match(raw):
            continue  # a full-line comment declares nothing
        m = _PROC_DECL.match(raw)
        if not m:
            continue
        proc_name = m.group(1)
        key = proc_name.lower()
        entry = _TAXONOMY.get(key)
        if entry is None:
            continue
        event, description, context_ok = entry
        if not context_ok(host, module_lower):
            continue  # right name, wrong module/host -> refuse
        if key in seen:
            continue
        seen.add(key)
        results.append(
            OfficeEventRelationship(
                host=host,
                module=module_name,
                procedure=proc_name,
                event=event,
                description=description,
                module_evidence_id=module_evidence_id,
                line=idx,
            )
        )
    return results
